#!/usr/bin/env tsx
/** Determinism exception: pure request-shape translation (Codex Responses ->
 * ClaudeRequest) proven against fixed fixture bodies; a live model cannot be
 * made to emit a specific malformed/duck-typed request on demand, and cache-
 * breakpoint/tool-order determinism needs byte-identical repeated runs over
 * the same input, which a live call cannot guarantee either. Imports only the
 * isolated `src/` module graph, mirroring
 * `test/continuous-test-suite-vertex-anthropic-fallback.ts`. */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import {
  parseCodexNativeRequest,
  translateCodexRequestToClaude,
} from "../src/lib/proxy/codexOutboundFallback.js";
import { applyClaudeRequestCacheBreakpoints } from "../src/lib/utils/anthropicCacheBreakpoints.js";
import type {
  ClaudeTextBlock,
  ClaudeToolResultBlock,
  ClaudeToolUseBlock,
  CodexNativeRequest,
  CodexReasoningEffort,
} from "../src/lib/types/index.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES_DIR = path.join(__dirname, "fixtures");

const FIXTURE_FILES = [
  "codex-request-exec-mode.json",
  "codex-request-interactive-mode.json",
  "codex-request-resumed-session.json",
  "codex-request-tool-result-turn.json",
  "codex-request-parallel-tool-calls.json",
] as const;

function readFixtureBody(file: string): unknown {
  const raw = readFileSync(path.join(FIXTURES_DIR, file), "utf8");
  const parsed = JSON.parse(raw) as { body: unknown };
  return parsed.body;
}

const TARGET = { provider: "anthropic" as const, model: "claude-sonnet-5" };

function assertOkTranslate(body: unknown) {
  const parsed = parseCodexNativeRequest(body);
  if (!parsed.ok) {
    throw new Error(
      `parse failed: ${parsed.error.code} ${parsed.error.message}`,
    );
  }
  const translated = translateCodexRequestToClaude(parsed.value, TARGET);
  if (!translated.ok) {
    throw new Error(
      `translate failed: ${translated.error.code} ${translated.error.message}`,
    );
  }
  return translated.value;
}

function minimalRequest(
  overrides: Partial<CodexNativeRequest> = {},
): CodexNativeRequest {
  return {
    model: "gpt-5-codex",
    stream: true,
    store: false,
    input: [
      {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: "hi" }],
      },
    ],
    ...overrides,
  };
}

let passed = 0;
function test(name: string, run: () => void) {
  run();
  passed++;
  console.log(`PASS ${name}`);
}

test("parseCodexNativeRequest rejects a body missing input", () => {
  const result = parseCodexNativeRequest({
    model: "gpt-5-codex",
    stream: true,
    store: false,
  });
  assert.equal(result.ok, false);
  if (!result.ok) {
    assert.equal(result.error.code, "MALFORMED_REQUEST");
  }
});

test("parseCodexNativeRequest accepts all 5 fixture bodies", () => {
  for (const file of FIXTURE_FILES) {
    const body = readFixtureBody(file);
    const result = parseCodexNativeRequest(body);
    assert.equal(result.ok, true, `${file} should parse ok`);
  }
});

test("translateCodexRequestToClaude is deterministic across repeated runs", () => {
  for (const file of FIXTURE_FILES) {
    const body = readFixtureBody(file);
    const parsed = parseCodexNativeRequest(body);
    assert.equal(parsed.ok, true, `${file} should parse ok`);
    if (!parsed.ok) {
      continue;
    }
    const first = translateCodexRequestToClaude(parsed.value, TARGET);
    const second = translateCodexRequestToClaude(parsed.value, TARGET);
    assert.equal(first.ok, true, `${file} should translate ok`);
    assert.equal(second.ok, true, `${file} should translate ok`);
    assert.equal(
      JSON.stringify(first),
      JSON.stringify(second),
      `${file} translation must be deterministic`,
    );
  }
});

test("system is always ClaudeTextBlock[], never a string", () => {
  for (const file of FIXTURE_FILES) {
    const claudeRequest = assertOkTranslate(readFixtureBody(file));
    if (claudeRequest.system !== undefined) {
      assert.equal(
        Array.isArray(claudeRequest.system),
        true,
        `${file}'s system must be an array`,
      );
    }
  }
});

test("interactive-mode: consecutive user items coalesce; mixed-kind tools split correctly", () => {
  const rawBody = readFixtureBody("codex-request-interactive-mode.json") as {
    input: Array<{
      type: string;
      role?: string;
      content?: unknown[];
      tools?: Array<{ tools: Array<Record<string, unknown>> }>;
    }>;
  };
  const claudeRequest = assertOkTranslate(rawBody);

  const userMessages = claudeRequest.messages.filter((m) => m.role === "user");
  assert.equal(
    userMessages.length,
    1,
    "the two consecutive user items must coalesce into one message",
  );
  const content = userMessages[0]!.content;
  assert.equal(Array.isArray(content), true);
  assert.equal(
    (content as unknown[]).length,
    2,
    "coalesced user message must carry both text blocks",
  );

  const additionalTools = rawBody.input.find(
    (i) => i.type === "additional_tools",
  );
  assert.ok(additionalTools, "fixture must carry an additional_tools item");
  const rawCustom = additionalTools!
    .tools!.flatMap((ns) => ns.tools)
    .find((t) => t.type === "custom") as { name: string };
  const rawFunction = additionalTools!
    .tools!.flatMap((ns) => ns.tools)
    .find((t) => t.type === "function") as {
    name: string;
    parameters: Record<string, unknown>;
  };

  const claudeTools = claudeRequest.tools ?? [];
  const customTool = claudeTools.find((t) => t.name === rawCustom.name);
  const functionTool = claudeTools.find((t) => t.name === rawFunction.name);
  assert.ok(customTool, "custom-kind tool must be mapped");
  assert.ok(functionTool, "function-kind tool must be mapped");
  assert.deepEqual(
    customTool!.input_schema,
    {
      type: "object",
      properties: {
        input: {
          type: "string",
          description:
            "Raw command text, constrained by the grammar in this tool's description.",
        },
      },
      required: ["input"],
    },
    "custom tool must use the single-input fallback schema",
  );
  assert.deepEqual(
    functionTool!.input_schema,
    rawFunction.parameters,
    "function tool's input_schema must equal its original parameters byte-for-byte",
  );
});

test("parallel-tool-calls: 2 function_calls group into one assistant message with 2 tool_use blocks", () => {
  const claudeRequest = assertOkTranslate(
    readFixtureBody("codex-request-parallel-tool-calls.json"),
  );
  const assistantMessages = claudeRequest.messages.filter(
    (m) => m.role === "assistant",
  );
  assert.equal(assistantMessages.length, 1);
  const toolUseBlocks = (
    assistantMessages[0]!.content as ClaudeToolUseBlock[]
  ).filter((b) => b.type === "tool_use");
  assert.equal(toolUseBlocks.length, 2);
  assert.equal(toolUseBlocks[0]!.id, "call_synthetic_0001");
  assert.equal(toolUseBlocks[1]!.id, "call_synthetic_0002");
  assert.deepEqual(toolUseBlocks[0]!.input, { input: "ls -la" });
  assert.deepEqual(toolUseBlocks[1]!.input, { path: "src/index.ts" });

  const userMessages = claudeRequest.messages.filter((m) => m.role === "user");
  const toolResultMessage = userMessages[userMessages.length - 1]!;
  const toolResultBlocks = (
    toolResultMessage.content as ClaudeToolResultBlock[]
  ).filter((b) => b.type === "tool_result");
  assert.equal(toolResultBlocks.length, 2);
  assert.equal(toolResultBlocks[0]!.tool_use_id, "call_synthetic_0001");
  assert.equal(toolResultBlocks[1]!.tool_use_id, "call_synthetic_0002");
});

test("tool-result-turn: non-JSON exec arguments translate without throwing", () => {
  const claudeRequest = assertOkTranslate(
    readFixtureBody("codex-request-tool-result-turn.json"),
  );
  const assistantMessage = claudeRequest.messages.find(
    (m) => m.role === "assistant",
  )!;
  const toolUse = (assistantMessage.content as ClaudeToolUseBlock[]).find(
    (b) => b.type === "tool_use",
  )!;
  assert.deepEqual(toolUse.input, { input: "ls -la" });
});

test("input_image content parts: data URI maps to base64, http(s) maps to url", () => {
  const dataUriRequest = minimalRequest({
    input: [
      {
        type: "message",
        role: "user",
        content: [
          { type: "input_image", image_url: "data:image/png;base64,QUJD" },
        ],
      },
    ],
  });
  const dataUriResult = translateCodexRequestToClaude(dataUriRequest, TARGET);
  assert.equal(dataUriResult.ok, true);
  if (dataUriResult.ok) {
    assert.deepEqual(dataUriResult.value.messages[0]!.content, [
      {
        type: "image",
        source: { type: "base64", media_type: "image/png", data: "QUJD" },
      },
    ]);
  }

  const urlRequest = minimalRequest({
    input: [
      {
        type: "message",
        role: "user",
        content: [
          { type: "input_image", image_url: "https://example.com/x.png" },
        ],
      },
    ],
  });
  const urlResult = translateCodexRequestToClaude(urlRequest, TARGET);
  assert.equal(urlResult.ok, true);
  if (urlResult.ok) {
    assert.deepEqual(urlResult.value.messages[0]!.content, [
      {
        type: "image",
        source: { type: "url", url: "https://example.com/x.png" },
      },
    ]);
  }
});

test("tool_choice table produces the exact §3.3 shape for each case", () => {
  const cases: Array<[CodexNativeRequest["tool_choice"], unknown]> = [
    ["required", { type: "any" }],
    [
      { type: "function", name: "x" },
      { type: "tool", name: "x" },
    ],
    ["auto", { type: "auto" }],
    ["none", { type: "none" }],
    [undefined, undefined],
  ];
  for (const [choice, expected] of cases) {
    const result = translateCodexRequestToClaude(
      minimalRequest({ tool_choice: choice }),
      TARGET,
    );
    assert.equal(result.ok, true);
    if (result.ok) {
      assert.deepEqual(
        result.value.tool_choice,
        expected,
        `choice=${JSON.stringify(choice)}`,
      );
    }
  }
});

test("reasoning.effort table produces the exact §3.5 shape or omission", () => {
  const cases: Array<[string, unknown]> = [
    ["none", undefined],
    ["minimal", undefined],
    ["low", { type: "enabled", budget_tokens: 4096 }],
    ["medium", { type: "enabled", budget_tokens: 8192 }],
    ["high", { type: "enabled", budget_tokens: 16384 }],
    ["xhigh", { type: "enabled", budget_tokens: 24576 }],
    ["max", { type: "enabled", budget_tokens: 32768 }],
  ];
  for (const [effort, expected] of cases) {
    const result = translateCodexRequestToClaude(
      minimalRequest({
        reasoning: { effort: effort as CodexReasoningEffort },
      }),
      TARGET,
    );
    assert.equal(result.ok, true);
    if (result.ok) {
      assert.deepEqual(result.value.thinking, expected, `effort=${effort}`);
    }
  }
});

test("reasoning budget stays within [1024, ceiling - 1024] under the smallest real ceiling", () => {
  // "claude-3-5-haiku" resolves to getClaudeMaxOutputTokens's 8192 ceiling —
  // the smallest any real model name reaches — so resolvedMaxTokens - 1024
  // = 7168, and the "max" effort's 32768 table entry gets clamped down to
  // it. No real model name drives the ceiling low enough to hit the
  // Math.max(1024, ...) floor itself (the lowest path, 4096, still leaves
  // 3072 of headroom) — the design's own §3.5 notes this floor is "not a
  // guarantee this formula enforced" by today's model table, only insurance
  // against a future one. This asserts the reachable half of the clamp
  // formula through the public surface; the floor's own arithmetic
  // (Math.max(1024, ...)) is exercised structurally by the code, not by a
  // reachable end-to-end case.
  const result = translateCodexRequestToClaude(
    minimalRequest({ reasoning: { effort: "max" } }),
    { provider: "anthropic", model: "claude-3-5-haiku" },
  );
  assert.equal(result.ok, true);
  if (result.ok) {
    const thinking = result.value.thinking as
      | { type: string; budget_tokens: number }
      | undefined;
    assert.ok(thinking);
    assert.equal(thinking!.budget_tokens, 7168);
    assert.ok(
      thinking!.budget_tokens >= 1024,
      "budget_tokens must never drop below 1024",
    );
    assert.ok(
      thinking!.budget_tokens <= result.value.max_tokens - 1024,
      "budget_tokens must leave at least 1024 tokens of headroom under max_tokens",
    );
  }
});

test("resumed session missing the fresh-session preamble fails loud, not silent", () => {
  const rawBody = readFixtureBody("codex-request-resumed-session.json") as {
    input: unknown[];
  };
  const parsed = parseCodexNativeRequest(rawBody);
  assert.equal(parsed.ok, true);
  if (!parsed.ok) {
    return;
  }

  const intact = translateCodexRequestToClaude(parsed.value, TARGET);
  assert.equal(
    intact.ok,
    true,
    "unmodified fixture (with preamble) must translate ok",
  );

  const withoutPreamble = {
    ...parsed.value,
    input: parsed.value.input.slice(1),
  };
  const mutated = translateCodexRequestToClaude(withoutPreamble, TARGET);
  assert.equal(mutated.ok, false);
  if (!mutated.ok) {
    assert.equal(mutated.error.code, "SUSPECTED_PARTIAL_HISTORY");
  }
});

test("cache breakpoint lands on the last system block, not the last tool", () => {
  const claudeRequest = assertOkTranslate(
    readFixtureBody("codex-request-interactive-mode.json"),
  );
  const withBreakpoints = applyClaudeRequestCacheBreakpoints(claudeRequest);
  const system = withBreakpoints.system as ClaudeTextBlock[];
  assert.ok(Array.isArray(system) && system.length > 0);
  const lastSystemBlock = system[system.length - 1]!;
  assert.ok(
    lastSystemBlock.cache_control,
    "the last system block must carry the cache breakpoint marker",
  );
  const tools = withBreakpoints.tools ?? [];
  for (const tool of tools) {
    assert.equal(
      tool.cache_control,
      undefined,
      "no tool should carry a cache breakpoint when system is present",
    );
  }
});

console.log(`Passed: ${passed}; Failed: 0; RESULT: PASS`);

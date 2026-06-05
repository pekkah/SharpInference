# SharpInference.Sample.ToolCall

Demonstrates **agentic tool calling** using the SharpInference library directly:

1. Tool schemas are passed to the model's Jinja chat template as a `tools` list
2. The model outputs `<tool_call>{"name":"…","arguments":{…}}</tool_call>` blocks
3. The sample parses those blocks, executes the tools locally, and feeds the results back as `role="tool"` messages
4. A second model turn uses the tool results to produce a natural-language final answer

Works with any model whose chat template supports tool calling (Qwen3, Mistral-Nemo, etc.).

## Run

```bash
# CPU (default)
dotnet run --project samples/SharpInference.Sample.ToolCall -c Release -- \
    -m models/Qwen3-8B-Q4_K_M.gguf

# CUDA — auto-detect how many layers fit in VRAM
dotnet run --project samples/SharpInference.Sample.ToolCall -c Release -- \
    -m models/Qwen3-35B-A3B-Q4_K_M.gguf --backend cuda -ngl -1

# CUDA — force all layers on GPU (will OOM if model doesn't fit)
dotnet run --project samples/SharpInference.Sample.ToolCall -c Release -- \
    -m models/Qwen3-35B-A3B-Q4_K_M.gguf --backend cuda -ngl 999
```

Custom question:

```bash
dotnet run --project samples/SharpInference.Sample.ToolCall -c Release -- \
    -m models/Qwen3-8B-Q4_K_M.gguf \
    -p "What's the weather in Berlin and what is (100 - 37) * 2?"
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-m` / `--model` | `SHARPI_MODEL` env | Path to GGUF model file |
| `-p` / `--prompt` | built-in demo | User question to ask |
| `--temp` | `0.6` | Sampling temperature |
| `--backend` | `cpu` | Compute backend: `cpu` or `cuda` |
| `-g` | `0` | GPU layer count: `0`=CPU-only, `-1`=auto (profile VRAM), `N`=N layers on GPU |

## Available tools (demo)

| Tool | Description |
|------|-------------|
| `get_weather` | Returns weather for a named city (canned data for demo) |
| `calculate` | Evaluates arithmetic expressions via a built-in recursive-descent parser |

## Message flow

```
Turn 1 ──────────────────────────────────────────────────────────
  User:   "What is the weather in Paris and 42 * 17?"
  Prompt: Jinja template with tool schemas injected into system prompt

  Model → "<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>
           <tool_call>{"name":"calculate","arguments":{"expression":"42 * 17"}}</tool_call>"

Tool execution ───────────────────────────────────────────────────
  get_weather("Paris")  → "Partly cloudy, 18 °C."
  calculate("42 * 17")  → "714"

Turn 2 ──────────────────────────────────────────────────────────
  Messages now contain the assistant tool_call + two role="tool" results
  Model → "The weather in Paris is partly cloudy at 18 °C, and 42 × 17 = 714."
```

## Extending

Add a new tool in two steps:

1. **Schema** — add another `MakeFunctionTool(...)` entry to `toolDefinitions`
2. **Implementation** — add a case to `ExecuteTool(...)` and a handler function

For GPU inference, swap `CpuBackend` + `ForwardPass` for `VulkanBackend` + `GpuForwardPass` (add the `SharpInference.Vulkan` project reference) — the rest of the sample is unchanged.

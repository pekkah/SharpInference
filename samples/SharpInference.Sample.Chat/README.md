# SharpInference.Sample.Chat

Minimal example of using the SharpInference library directly (no CLI wrapper). It demonstrates:

- Loading a GGUF model with `GgufModel.Open` and parsing hyperparameters / tokenizer
- Constructing a CPU `InferenceEngine` and disposing it cleanly
- Rendering prompts with the model's own Jinja2 chat template (read from GGUF metadata)
- **Streaming output** — `await foreach` over `IAsyncEnumerable<string>` from `GenerateAsync`
- **Streaming input** — `Console.In.ReadLineAsync(cts.Token)` so stdin lines or piped EOF advance the loop without blocking on cancellation
- Cooperative Ctrl+C via `CancellationTokenSource`
- Multi-turn history that lets the engine's prefix cache amortise repeated prompts

## Run

```bash
dotnet run --project samples/SharpInference.Sample.Chat -c Release -- \
    -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf
```

Pipe input instead of typing it:

```bash
echo "Write a haiku about C# generics." | \
    dotnet run --project samples/SharpInference.Sample.Chat -c Release -- \
        -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf
```

Flags: `-m <path>` (or `SHARPI_MODEL` env var), `--system <prompt>`, `--temp <0..1>`.

For GPU inference, swap `CpuBackend` + `ForwardPass` for `VulkanBackend` + `GpuForwardPass` (or `HybridForwardPass`) — the rest of the sample is unchanged.

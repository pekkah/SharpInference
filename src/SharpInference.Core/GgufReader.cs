using System.IO.Pipelines;

namespace SharpInference.Core;

/// <summary>
/// Streaming GGUF file parser.
/// Reads header, metadata key-value pairs, and tensor descriptors
/// from a GGUF v1/v2/v3 file without loading weights into memory.
/// </summary>
public sealed class GgufReader : IDisposable
{
    private readonly Stream _stream;
    private readonly PipeReader _pipe;

    public GgufReader(Stream stream)
    {
        _stream = stream;
        _pipe = PipeReader.Create(stream);
    }

    public GgufHeader Header { get; private set; }
    public IReadOnlyList<GgufTensorInfo> TensorInfos { get; private set; } = [];
    public IReadOnlyDictionary<string, object> Metadata { get; private set; } =
        new Dictionary<string, object>();

    public ValueTask ReadAsync(CancellationToken ct = default)
    {
        // TODO: implement GGUF binary format parsing
        throw new NotImplementedException();
    }

    public void Dispose() => _stream.Dispose();
}

public readonly record struct GgufHeader(uint Magic, uint Version, ulong TensorCount, ulong MetadataKvCount);
public readonly record struct GgufTensorInfo(string Name, TensorShape Shape, DType DType, ulong Offset);

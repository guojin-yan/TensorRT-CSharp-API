using System;
using System.IO;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtHostMemory : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;

    internal TensorRtHostMemory(TensorRtApiLine line, SafeTensorRtObjectHandle handle)
    {
        Line = line;
        _handle = handle;
        SizeInBytes = NativeBridgeApi.GetHostMemorySize(line, _handle);
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    public TensorRtApiLine Line { get; }

    public ulong SizeInBytes { get; }

    public byte[] ToArray()
    {
        return NativeBridgeApi.CopyHostMemoryToArray(Line, _handle);
    }

    public void SaveToFile(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path must not be null or empty.", nameof(filePath));
        }

        string? directory = Path.GetDirectoryName(Path.GetFullPath(filePath));
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllBytes(filePath, ToArray());
    }

    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}

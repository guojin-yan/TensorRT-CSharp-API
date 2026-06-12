using System;
using System.IO;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtRuntime : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;

    public TensorRtRuntime(TensorRtLogger logger)
    {
        if (logger == null)
        {
            throw new ArgumentNullException(nameof(logger));
        }

        Line = logger.Line;
        _handle = NativeBridgeApi.CreateRuntime(Line, logger.Handle);
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    public TensorRtApiLine Line { get; }

    public TensorRtEngine Deserialize(TensorRtHostMemory hostMemory)
    {
        if (hostMemory == null)
        {
            throw new ArgumentNullException(nameof(hostMemory));
        }

        if (hostMemory.Line != Line)
        {
            throw new ArgumentException("Host memory belongs to a different TensorRT API line.", nameof(hostMemory));
        }

        return new TensorRtEngine(Line, NativeBridgeApi.DeserializeHostMemory(Line, _handle, hostMemory.Handle));
    }

    public TensorRtEngine Deserialize(byte[] serializedEngine)
    {
        if (serializedEngine == null)
        {
            throw new ArgumentNullException(nameof(serializedEngine));
        }

        return new TensorRtEngine(Line, NativeBridgeApi.DeserializeEngineData(Line, _handle, serializedEngine));
    }

    public TensorRtEngine DeserializeFromFile(string filePath)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException("Serialized TensorRT engine file was not found.", filePath);
        }

        return Deserialize(File.ReadAllBytes(filePath));
    }

    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}

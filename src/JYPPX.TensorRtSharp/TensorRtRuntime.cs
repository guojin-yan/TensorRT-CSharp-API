using System;
using System.IO;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Managed wrapper around a TensorRT runtime.
/// TensorRT runtime 的托管封装。
/// </summary>
public sealed partial class TensorRtRuntime : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;

    /// <summary>
    /// Creates a TensorRT runtime from a logger.
    /// 使用 logger 创建一个 TensorRT runtime。
    /// </summary>
    /// <param name="logger">The TensorRT logger used by the runtime. runtime 使用的 TensorRT logger。</param>
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

    /// <summary>
    /// Gets the TensorRT API line used by this runtime.
    /// 获取当前 runtime 使用的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Deserializes an engine from TensorRT host memory.
    /// 从 TensorRT host memory 反序列化一个 engine。
    /// </summary>
    /// <param name="hostMemory">The serialized engine memory. 序列化 engine 内存。</param>
    /// <returns>A TensorRT engine wrapper. TensorRT engine 封装。</returns>
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

    /// <summary>
    /// Deserializes an engine from a managed byte buffer.
    /// 从托管字节缓冲区反序列化一个 engine。
    /// </summary>
    /// <param name="serializedEngine">The serialized engine bytes. 序列化 engine 字节数组。</param>
    /// <returns>A TensorRT engine wrapper. TensorRT engine 封装。</returns>
    public TensorRtEngine Deserialize(byte[] serializedEngine)
    {
        if (serializedEngine == null)
        {
            throw new ArgumentNullException(nameof(serializedEngine));
        }

        return new TensorRtEngine(Line, NativeBridgeApi.DeserializeEngineData(Line, _handle, serializedEngine));
    }

    /// <summary>
    /// Deserializes an engine from a serialized engine file.
    /// 从序列化 engine 文件中反序列化一个 engine。
    /// </summary>
    /// <param name="filePath">The path to the serialized engine file. 序列化 engine 文件路径。</param>
    /// <returns>A TensorRT engine wrapper. TensorRT engine 封装。</returns>
    public TensorRtEngine DeserializeFromFile(string filePath)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException("Serialized TensorRT engine file was not found.", filePath);
        }

        return Deserialize(File.ReadAllBytes(filePath));
    }

    /// <summary>
    /// Releases the TensorRT runtime handle.
    /// 释放 TensorRT runtime 句柄。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}

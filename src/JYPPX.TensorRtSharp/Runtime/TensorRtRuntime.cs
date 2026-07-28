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
    private readonly TensorRtLogger _loggerKeepAlive;
    private bool _disposed;

    /// <summary>
    /// Creates a TensorRT runtime from a logger.
    /// 使用 logger 创建一个 TensorRT runtime。
    /// </summary>
    /// <param name="logger">The TensorRT logger used by the runtime. runtime 使用的 TensorRT logger。</param>
    /// <remarks>
    /// TensorRT borrows the logger pointer. This runtime keeps the managed logger attached until the runtime is disposed.
    /// TensorRT 只借用 logger 指针；当前 runtime 会保持托管 logger 借用关系直到 runtime 释放。
    /// </remarks>
    public TensorRtRuntime(TensorRtLogger logger)
    {
        if (logger == null)
        {
            throw new ArgumentNullException(nameof(logger));
        }

        Line = logger.Line;
        _loggerKeepAlive = logger;
        _loggerKeepAlive.AttachBorrower(Line);
        try
        {
            _handle = NativeBridgeApi.CreateRuntime(Line, logger.Handle);
        }
        catch
        {
            _loggerKeepAlive.DetachBorrower();
            throw;
        }
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
    /// Deserializes an engine from a managed byte-array segment.
    /// 从托管字节数组片段反序列化一个 engine。
    /// </summary>
    /// <param name="serializedEngine">The serialized engine byte segment. 序列化 engine 字节数组片段。</param>
    /// <returns>A TensorRT engine wrapper. TensorRT engine 封装。</returns>
    /// <remarks>
    /// The segment is copied into an exact managed byte array before the native deserialize call, so TensorRT only observes a pinned managed buffer for the duration of the call.
    /// 调用 native 反序列化前会将片段复制为精确长度的托管字节数组，因此 TensorRT 只会在调用期间看到 pinned 托管缓冲区。
    /// </remarks>
    public TensorRtEngine Deserialize(ArraySegment<byte> serializedEngine)
    {
        if (serializedEngine.Array == null)
        {
            throw new ArgumentException("Serialized engine segment must reference a byte array.", nameof(serializedEngine));
        }

        byte[] buffer = new byte[serializedEngine.Count];
        Buffer.BlockCopy(serializedEngine.Array, serializedEngine.Offset, buffer, 0, serializedEngine.Count);
        return Deserialize(buffer);
    }

#if NETCOREAPP3_1_OR_GREATER || NET5_0_OR_GREATER || NET6_0_OR_GREATER || NET7_0_OR_GREATER || NET8_0_OR_GREATER || NET9_0_OR_GREATER || NET10_0_OR_GREATER
    /// <summary>
    /// Deserializes an engine from a managed read-only byte span.
    /// 从托管只读字节 span 反序列化一个 engine。
    /// </summary>
    /// <param name="serializedEngine">The serialized engine bytes. 序列化 engine 字节。</param>
    /// <returns>A TensorRT engine wrapper. TensorRT engine 封装。</returns>
    /// <remarks>
    /// The span is copied into a managed byte array before native interop so no caller-owned span is retained by TensorRT.
    /// 调用 native interop 前会将 span 复制到托管字节数组，因此 TensorRT 不会保留调用方拥有的 span。
    /// </remarks>
    public TensorRtEngine Deserialize(ReadOnlySpan<byte> serializedEngine)
    {
        return Deserialize(serializedEngine.ToArray());
    }

#endif
    /// <summary>
    /// Deserializes an engine from a managed stream.
    /// 从托管 stream 反序列化一个 engine。
    /// </summary>
    /// <param name="serializedEngineStream">The readable stream containing serialized engine bytes. 包含序列化 engine 字节的可读 stream。</param>
    /// <returns>A TensorRT engine wrapper. TensorRT engine 封装。</returns>
    /// <remarks>
    /// The stream is copied into managed memory before native interop; this is not a TensorRT <c>IStreamReader</c> callback bridge.
    /// 调用 native interop 前会将 stream 内容复制到托管内存；该入口不是 TensorRT <c>IStreamReader</c> 回调桥。
    /// </remarks>
    public TensorRtEngine Deserialize(Stream serializedEngineStream)
    {
        if (serializedEngineStream == null)
        {
            throw new ArgumentNullException(nameof(serializedEngineStream));
        }

        if (!serializedEngineStream.CanRead)
        {
            throw new ArgumentException("Serialized engine stream must be readable.", nameof(serializedEngineStream));
        }

        using MemoryStream copy = new MemoryStream();
        serializedEngineStream.CopyTo(copy);
        return Deserialize(copy.ToArray());
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
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _handle.Dispose();
        GC.KeepAlive(_loggerKeepAlive);
        _loggerKeepAlive.DetachBorrower();
        GC.SuppressFinalize(this);
    }
}

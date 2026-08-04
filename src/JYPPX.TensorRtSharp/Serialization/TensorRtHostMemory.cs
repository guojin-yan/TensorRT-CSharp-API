using System;
using System.IO;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Managed wrapper around TensorRT host memory.
/// TensorRT host memory 的托管封装。
/// </summary>
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

    /// <summary>
    /// Gets the TensorRT API line that produced this host memory.
    /// 获取生成当前 host memory 的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets the serialized payload size in bytes.
    /// 获取序列化负载大小，单位为字节。
    /// </summary>
    public ulong SizeInBytes { get; }

    /// <summary>
    /// Copies the TensorRT host-memory payload into a managed byte array.
    /// 将 TensorRT host memory 负载复制到托管字节数组中。
    /// </summary>
    /// <returns>A managed byte-array copy of the payload. 负载的托管字节数组副本。</returns>
    public byte[] ToArray()
    {
        return NativeBridgeApi.CopyHostMemoryToArray(Line, _handle);
    }

    /// <summary>
    /// Copies the TensorRT host-memory payload into a managed stream.
    /// 将 TensorRT host memory 负载复制到托管 stream。
    /// </summary>
    /// <param name="destination">The writable destination stream. 可写目标 stream。</param>
    /// <remarks>
    /// The payload is first copied into a managed byte array, so no borrowed TensorRT pointer escapes this wrapper.
    /// 负载会先复制到托管字节数组，因此不会从当前封装泄露 borrowed TensorRT 指针。
    /// </remarks>
    public void CopyTo(Stream destination)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        if (!destination.CanWrite)
        {
            throw new ArgumentException("Destination stream must be writable.", nameof(destination));
        }

        byte[] buffer = ToArray();
        destination.Write(buffer, 0, buffer.Length);
    }

    /// <summary>
    /// Opens a read-only managed stream over a copied TensorRT host-memory payload.
    /// 基于 TensorRT host memory 负载副本打开只读托管 stream。
    /// </summary>
    /// <returns>A read-only managed memory stream containing a copied payload. 包含负载副本的只读托管内存流。</returns>
    /// <remarks>
    /// The returned stream owns an independent managed copy and remains valid after this host-memory object is disposed.
    /// 返回的 stream 持有独立托管副本，因此当前 host memory 对象释放后仍然有效。
    /// </remarks>
    public MemoryStream OpenRead()
    {
        return new MemoryStream(ToArray(), writable: false);
    }

    /// <summary>
    /// Saves the TensorRT host-memory payload to a file.
    /// 将 TensorRT host memory 负载保存到文件。
    /// </summary>
    /// <param name="filePath">The destination file path. 目标文件路径。</param>
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

    /// <summary>
    /// Releases the TensorRT host-memory handle.
    /// 释放 TensorRT host memory 句柄。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}

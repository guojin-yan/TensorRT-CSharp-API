using System;
using System.IO;
using JYPPX.Shared.Interop;
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

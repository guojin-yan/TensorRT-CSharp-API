using System;
using System.IO;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>Owns a CUDA Driver module loaded from copied code bytes. 拥有从复制型 code bytes 加载的 CUDA Driver module。</summary>
public sealed class CudaDriverModule : IDisposable
{
    private readonly SafeCudaDriverModuleHandle _handle;

    private CudaDriverModule(SafeCudaDriverModuleHandle handle, int deviceOrdinal)
    {
        _handle = handle;
        DeviceOrdinal = deviceOrdinal;
    }

    internal SafeCudaDriverModuleHandle Handle => _handle;

    /// <summary>Gets the device ordinal whose primary context owns this module. 获取拥有此 module 的 device ordinal。</summary>
    public int DeviceOrdinal { get; }

    /// <summary>Loads copied PTX or CUBIN bytes into the selected device primary context. 将复制型 PTX 或 CUBIN bytes 加载到指定 device primary context。</summary>
    public static CudaDriverModule Load(byte[] code, int deviceOrdinal = 0)
    {
        if (code == null) throw new ArgumentNullException(nameof(code));
        if (code.Length == 0) throw new ArgumentException("CUDA Driver module code must not be empty.", nameof(code));
        if (deviceOrdinal < 0) throw new ArgumentOutOfRangeException(nameof(deviceOrdinal));

        NativeBridgeLoader.EnsureInitialized();
        return new CudaDriverModule(NativeCudaApi.LoadDriverModule(code, deviceOrdinal), deviceOrdinal);
    }

    /// <summary>Loads copied PTX or CUBIN bytes from a file. 从文件加载复制型 PTX 或 CUBIN bytes。</summary>
    public static CudaDriverModule LoadFromFile(string path, int deviceOrdinal = 0)
    {
        if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("CUDA Driver module path must not be null or empty.", nameof(path));
        if (deviceOrdinal < 0) throw new ArgumentOutOfRangeException(nameof(deviceOrdinal));

        string fullPath = Path.GetFullPath(path);
        if (!File.Exists(fullPath)) throw new FileNotFoundException("CUDA Driver module file was not found.", fullPath);
        return Load(File.ReadAllBytes(fullPath), deviceOrdinal);
    }

    /// <summary>Launches a named function with copied scalar and owner-bound device-memory arguments. 使用复制型标量和 owner-bound device-memory 参数启动 named function。</summary>
    /// <remarks>The returned owner leases this module, stream, and every device-memory argument until disposal. 返回 owner 会在释放前租住 module、stream 和所有 device-memory 参数。</remarks>
    public CudaDriverKernelLaunch Launch(
        string kernelName,
        CudaKernelLaunchConfiguration configuration,
        CudaStream stream,
        params CudaKernelArgument[] arguments)
    {
        ValidateName(kernelName, nameof(kernelName));
        if (stream == null) throw new ArgumentNullException(nameof(stream));
        if (arguments == null) throw new ArgumentNullException(nameof(arguments));
        return CudaDriverKernelLaunch.Create(_handle, kernelName, configuration, stream, arguments);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }

    private static void ValidateName(string name, string parameterName)
    {
        if (string.IsNullOrEmpty(name)) throw new ArgumentException("CUDA Driver function name must not be null or empty.", parameterName);
        if (name.IndexOf('\0') >= 0) throw new ArgumentException("CUDA Driver names must not contain embedded NUL characters.", parameterName);
    }
}

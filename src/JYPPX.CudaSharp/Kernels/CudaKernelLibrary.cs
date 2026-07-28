using System;
using System.IO;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>Owns a CUDA runtime kernel library and exposes only copied metadata. 拥有 CUDA runtime kernel library，并且只公开复制型元数据。</summary>
public sealed class CudaKernelLibrary : IDisposable
{
    private readonly SafeCudaKernelLibraryHandle _handle;

    private CudaKernelLibrary(SafeCudaKernelLibraryHandle handle)
    {
        _handle = handle;
    }

    internal SafeCudaKernelLibraryHandle Handle => _handle;

    /// <summary>Loads a CUDA cubin, fatbin, or null-terminated PTX file. 加载 CUDA cubin、fatbin 或以 null 结尾的 PTX 文件。</summary>
    public static CudaKernelLibrary LoadFromFile(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("CUDA kernel library path must not be null or empty.", nameof(path));
        }

        string fullPath = Path.GetFullPath(path);
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException("CUDA kernel library file was not found.", fullPath);
        }

        NativeBridgeLoader.EnsureInitialized();
        return new CudaKernelLibrary(NativeCudaApi.LoadKernelLibrary(fullPath));
    }

    /// <summary>Loads CUDA cubin, fatbin, or PTX bytes while the native owner retains its own copy. 加载 CUDA cubin、fatbin 或 PTX 字节，native owner 会保留独立副本。</summary>
    public static CudaKernelLibrary Load(byte[] code)
    {
        if (code == null)
        {
            throw new ArgumentNullException(nameof(code));
        }
        if (code.Length == 0)
        {
            throw new ArgumentException("CUDA kernel library code must not be empty.", nameof(code));
        }

        NativeBridgeLoader.EnsureInitialized();
        return new CudaKernelLibrary(NativeCudaApi.LoadKernelLibrary(code));
    }

    /// <summary>Gets the CUDA-reported kernel count. 获取 CUDA 报告的 kernel 数量。</summary>
    public uint KernelCount => NativeCudaApi.GetKernelLibraryCount(_handle);

    /// <summary>Gets a copied validation snapshot produced by count/enumerate calls. 获取由 count/enumerate 调用产生的复制型校验快照。</summary>
    public CudaKernelLibraryInventorySnapshot Inventory =>
        new CudaKernelLibraryInventorySnapshot(NativeCudaApi.GetKernelLibraryInventory(_handle));

    /// <summary>Determines whether the library contains a named kernel without exposing its borrowed handle. 判断 library 是否包含指定 kernel，不暴露 borrowed handle。</summary>
    public bool ContainsKernel(string name)
    {
        ValidateName(name, nameof(name), "CUDA kernel name must not be null or empty.");
        return NativeCudaApi.KernelLibraryContains(_handle, name);
    }

    /// <summary>
    /// Tries to copy the size of a named global symbol without retrieving its device pointer.
    /// 尝试复制指定 global symbol 的大小，不获取其 device pointer。
    /// </summary>
    public bool TryGetGlobalSymbolSize(string name, out ulong sizeInBytes)
    {
        ValidateName(name, nameof(name), "CUDA global symbol name must not be null or empty.");
        return NativeCudaApi.TryGetKernelLibraryGlobalSize(_handle, name, out sizeInBytes);
    }

    /// <summary>
    /// Tries to copy the size of a named managed symbol without retrieving its pointer.
    /// 尝试复制指定 managed symbol 的大小，不获取其 pointer。
    /// </summary>
    public bool TryGetManagedSymbolSize(string name, out ulong sizeInBytes)
    {
        ValidateName(name, nameof(name), "CUDA managed symbol name must not be null or empty.");
        return NativeCudaApi.TryGetKernelLibraryManagedSize(_handle, name, out sizeInBytes);
    }

    /// <summary>
    /// Determines whether a named unified function exists without exposing its function pointer.
    /// 判断指定 unified function 是否存在，不暴露其 function pointer。
    /// </summary>
    public bool ContainsUnifiedFunction(string name)
    {
        ValidateName(name, nameof(name), "CUDA unified-function name must not be null or empty.");
        return NativeCudaApi.KernelLibraryContainsUnifiedFunction(_handle, name);
    }

    /// <summary>
    /// Sets a mutable attribute on a named kernel for one device without exposing the borrowed kernel handle.
    /// 为指定设备上的 named kernel 设置可变属性，不暴露 borrowed kernel handle。
    /// </summary>
    public void SetAttributeForDevice(string kernelName, CudaKernelAttribute attribute, int value, int deviceOrdinal)
    {
        ValidateName(kernelName, nameof(kernelName), "CUDA kernel name must not be null or empty.");
        if (!Enum.IsDefined(typeof(CudaKernelAttribute), attribute))
        {
            throw new ArgumentOutOfRangeException(nameof(attribute));
        }
        if (deviceOrdinal < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(deviceOrdinal));
        }

        NativeCudaApi.SetKernelLibraryKernelAttributeForDevice(_handle, kernelName, attribute, value, deviceOrdinal);
    }

    /// <summary>
    /// Launches a named kernel with copied scalar arguments and owner-bound device-memory arguments.
    /// 使用复制型标量参数和 owner-bound device-memory 参数启动 named kernel。
    /// </summary>
    /// <remarks>
    /// The returned owner retains this library, the stream, and every device-memory allocation until disposal.
    /// Dispose or synchronize the launch before reading results. 返回 owner 会租用 library、stream 与全部 device-memory allocation；读取结果前必须同步或释放 launch。
    /// </remarks>
    public CudaKernelLaunch Launch(
        string kernelName,
        CudaKernelLaunchConfiguration configuration,
        CudaStream stream,
        params CudaKernelArgument[] arguments)
    {
        ValidateName(kernelName, nameof(kernelName), "CUDA kernel name must not be null or empty.");
        if (stream == null) throw new ArgumentNullException(nameof(stream));
        if (arguments == null) throw new ArgumentNullException(nameof(arguments));
        return CudaKernelLaunch.Create(_handle, kernelName, configuration, stream, arguments);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _handle.Dispose();
    }

    private static void ValidateName(string name, string parameterName, string message)
    {
        if (string.IsNullOrEmpty(name))
        {
            throw new ArgumentException(message, parameterName);
        }
        if (name.IndexOf('\0') >= 0)
        {
            throw new ArgumentException("CUDA names must not contain embedded NUL characters.", parameterName);
        }
    }
}

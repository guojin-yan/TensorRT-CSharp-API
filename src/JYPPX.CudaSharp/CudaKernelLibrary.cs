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
        if (string.IsNullOrEmpty(name))
        {
            throw new ArgumentException("CUDA kernel name must not be null or empty.", nameof(name));
        }

        return NativeCudaApi.KernelLibraryContains(_handle, name);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _handle.Dispose();
    }
}

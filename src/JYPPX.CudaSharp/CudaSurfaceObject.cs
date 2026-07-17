using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>Owns a CUDA surface object backed by a managed CUDA array owner. 拥有由托管 CUDA array owner 支撑的 CUDA surface object。</summary>
public sealed class CudaSurfaceObject : IDisposable
{
    private readonly SafeCudaSurfaceObjectHandle _handle;

    /// <summary>Creates a CUDA surface object for an array owner. 为 array owner 创建 CUDA surface object。</summary>
    public CudaSurfaceObject(CudaArray array)
    {
        OwnerArray = array ?? throw new ArgumentNullException(nameof(array));
        NativeBridgeLoader.EnsureInitialized();
        _handle = NativeCudaApi.CreateSurfaceObject(array.Handle);
    }

    /// <summary>Gets the managed array owner retained for this surface. 获取此 surface 保留的托管 array owner。</summary>
    public CudaArray OwnerArray { get; }

    /// <summary>Gets a pointer-free resource descriptor snapshot. 获取无指针的 resource descriptor 快照。</summary>
    public CudaResourceDescriptorSnapshot Resource =>
        new CudaResourceDescriptorSnapshot(NativeCudaApi.GetSurfaceResourceSnapshot(_handle));

    /// <inheritdoc />
    public void Dispose()
    {
        _handle.Dispose();
    }
}

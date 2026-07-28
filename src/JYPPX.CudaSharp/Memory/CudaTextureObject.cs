using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>Owns a CUDA texture object backed by a managed CUDA array owner. 拥有由托管 CUDA array owner 支撑的 CUDA texture object。</summary>
public sealed class CudaTextureObject : IDisposable
{
    private readonly SafeCudaTextureObjectHandle _handle;
    private readonly bool _usesCuda11Version2;

    /// <summary>Creates a texture object with the standard CUDA descriptor ABI. 使用标准 CUDA descriptor ABI 创建 texture object。</summary>
    public CudaTextureObject(CudaArray array, CudaTextureDescriptor descriptor)
        : this(array, descriptor, false)
    {
    }

    /// <summary>Creates a texture object with conservative default sampling. 使用保守的默认采样参数创建 texture object。</summary>
    public CudaTextureObject(CudaArray array)
        : this(array, CudaTextureDescriptor.Default, false)
    {
    }

    private CudaTextureObject(CudaArray array, CudaTextureDescriptor descriptor, bool useCuda11Version2)
    {
        OwnerArray = array ?? throw new ArgumentNullException(nameof(array));
        NativeBridgeLoader.EnsureInitialized();
        _handle = NativeCudaApi.CreateTextureObject(array.Handle, descriptor, useCuda11Version2);
        _usesCuda11Version2 = useCuda11Version2;
    }

    /// <summary>Creates a texture object through the CUDA 11.8 v2 descriptor ABI. 通过 CUDA 11.8 v2 descriptor ABI 创建 texture object。</summary>
    public static CudaTextureObject CreateCuda11Version2(CudaArray array, CudaTextureDescriptor descriptor)
    {
        return new CudaTextureObject(array, descriptor, true);
    }

    /// <summary>Gets the managed array owner retained for this texture. 获取此 texture 保留的托管 array owner。</summary>
    public CudaArray OwnerArray { get; }

    /// <summary>Gets the descriptor through the ABI used to create this texture. 通过创建该 texture 时使用的 ABI 获取 descriptor。</summary>
    public CudaTextureDescriptor Descriptor =>
        CudaTextureDescriptor.FromNative(NativeCudaApi.GetTextureDescriptor(_handle, _usesCuda11Version2));

    /// <summary>Gets the standard CUDA texture descriptor. 获取标准 CUDA texture descriptor。</summary>
    public CudaTextureDescriptor GetStandardDescriptor() =>
        CudaTextureDescriptor.FromNative(NativeCudaApi.GetTextureDescriptor(_handle, false));

    /// <summary>Gets the CUDA 11.8 v2 texture descriptor. 获取 CUDA 11.8 v2 texture descriptor。</summary>
    public CudaTextureDescriptor GetCuda11Version2Descriptor() =>
        CudaTextureDescriptor.FromNative(NativeCudaApi.GetTextureDescriptor(_handle, true));

    /// <summary>Gets a pointer-free resource descriptor snapshot. 获取无指针的 resource descriptor 快照。</summary>
    public CudaResourceDescriptorSnapshot Resource =>
        new CudaResourceDescriptorSnapshot(NativeCudaApi.GetTextureResourceSnapshot(_handle));

    /// <summary>Gets a copied resource-view descriptor. 获取复制型 resource-view descriptor。</summary>
    public CudaTextureResourceViewSnapshot ResourceView =>
        new CudaTextureResourceViewSnapshot(NativeCudaApi.GetTextureResourceViewSnapshot(_handle));

    /// <inheritdoc />
    public void Dispose()
    {
        _handle.Dispose();
    }
}

using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>
/// Managed wrapper around a device memory allocation.
/// CUDA 设备内存分配的托管封装。
/// </summary>
public partial class CudaMemory : IDisposable
{
    private readonly SafeCudaMemoryHandle _handle;

    /// <summary>
    /// Allocates CUDA device memory.
    /// 分配 CUDA 设备内存。
    /// </summary>
    /// <param name="sizeInBytes">The allocation size in bytes. 分配大小，单位为字节。</param>
    public CudaMemory(int sizeInBytes)
        : this(AllocateDeviceMemory(sizeInBytes))
    {
    }

    internal CudaMemory(SafeCudaMemoryHandle handle)
    {
        _handle = handle ?? throw new ArgumentNullException(nameof(handle));
        SizeInBytes = checked((int)NativeCudaApi.GetMemorySize(_handle));
    }

    /// <summary>
    /// Gets the allocation size in bytes.
    /// 获取分配大小，单位为字节。
    /// </summary>
    public int SizeInBytes { get; }

    internal SafeCudaMemoryHandle Handle => _handle;

    /// <summary>
    /// Copies an opaque CUDA IPC export token for this synchronous device allocation.
    /// 复制当前同步 device allocation 的 opaque CUDA IPC 导出 token。
    /// </summary>
    /// <remarks>
    /// Only allocations created by the public <see cref="CudaMemory(int)"/> constructor are exportable.
    /// Managed, asynchronous, and pool allocations are rejected. Keep this allocation alive while another
    /// process uses the token. This method never exposes the device pointer.
    /// 只有通过公开构造函数创建的同步分配可导出；managed、async 与 pool allocation 会被拒绝。
    /// 其他进程使用 token 期间必须保持当前分配存活；此方法绝不公开 device pointer。
    /// </remarks>
    public CudaIpcExportToken ExportIpcToken()
    {
        return NativeCudaApi.ExportMemoryIpcToken(_handle);
    }

    /// <summary>Tries to copy an IPC export token and returns a diagnostic on failure. 尝试复制 IPC 导出 token，失败时返回诊断。</summary>
    public bool TryExportIpcToken(out CudaIpcExportToken? token, out string diagnostic)
    {
        try
        {
            token = ExportIpcToken();
            diagnostic = string.Empty;
            return true;
        }
        catch (CudaException exception)
        {
            token = null;
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Queries CUDA pointer attributes for this device allocation.
    /// 查询当前设备内存分配的 CUDA 指针属性。
    /// </summary>
    /// <returns>Pointer metadata useful for diagnostics and deployment validation. 用于诊断和部署验证的指针元数据。</returns>
    public CudaPointerAttributes GetPointerAttributes()
    {
        NativeCudaPointerAttributes attributes = NativeCudaApi.GetPointerAttributes(_handle);
        return new CudaPointerAttributes(
            (CudaMemoryPointerType)attributes.MemoryType,
            attributes.Device,
            attributes.DevicePointer,
            attributes.HostPointer);
    }

    /// <summary>
    /// Queries one scalar CUDA managed-memory range attribute for this allocation.
    /// 查询当前分配的一个 CUDA managed memory range 标量属性。
    /// </summary>
    /// <param name="attribute">The scalar memory range attribute. 标量 memory range 属性。</param>
    /// <returns>A managed attribute snapshot. 托管属性快照。</returns>
    public CudaMemoryRangeAttributeValue GetRangeAttribute(CudaMemoryRangeAttribute attribute)
    {
        return GetRangeAttribute(attribute, 0, SizeInBytes);
    }

    /// <summary>
    /// Queries one scalar CUDA managed-memory range attribute for a subrange of this allocation.
    /// 查询当前分配子范围的一个 CUDA managed memory range 标量属性。
    /// </summary>
    /// <param name="attribute">The scalar memory range attribute. 标量 memory range 属性。</param>
    /// <param name="offset">The byte offset within this allocation. 当前分配内的字节偏移。</param>
    /// <param name="count">The number of bytes to query. 要查询的字节数。</param>
    /// <returns>A managed attribute snapshot. 托管属性快照。</returns>
    public CudaMemoryRangeAttributeValue GetRangeAttribute(CudaMemoryRangeAttribute attribute, int offset, int count)
    {
        ValidateRange(offset, count, nameof(offset), nameof(count));
        ValidateScalarRangeAttribute(attribute, nameof(attribute));
        int rawValue = NativeCudaApi.GetMemoryRangeAttribute(_handle, offset, count, (int)attribute);
        return new CudaMemoryRangeAttributeValue(attribute, rawValue);
    }

    /// <summary>
    /// Queries multiple scalar CUDA managed-memory range attributes for this allocation.
    /// 查询当前分配的多个 CUDA managed memory range 标量属性。
    /// </summary>
    /// <param name="attributes">The scalar memory range attributes. 标量 memory range 属性列表。</param>
    /// <returns>Managed attribute snapshots in request order. 按请求顺序返回的托管属性快照。</returns>
    public CudaMemoryRangeAttributeValue[] GetRangeAttributes(params CudaMemoryRangeAttribute[] attributes)
    {
        return GetRangeAttributes(0, SizeInBytes, attributes);
    }

    /// <summary>
    /// Queries multiple scalar CUDA managed-memory range attributes for a subrange of this allocation.
    /// 查询当前分配子范围的多个 CUDA managed memory range 标量属性。
    /// </summary>
    /// <param name="offset">The byte offset within this allocation. 当前分配内的字节偏移。</param>
    /// <param name="count">The number of bytes to query. 要查询的字节数。</param>
    /// <param name="attributes">The scalar memory range attributes. 标量 memory range 属性列表。</param>
    /// <returns>Managed attribute snapshots in request order. 按请求顺序返回的托管属性快照。</returns>
    public CudaMemoryRangeAttributeValue[] GetRangeAttributes(int offset, int count, params CudaMemoryRangeAttribute[] attributes)
    {
        if (attributes == null)
        {
            throw new ArgumentNullException(nameof(attributes));
        }

        if (attributes.Length == 0)
        {
            throw new ArgumentException("At least one CUDA memory range attribute must be requested.", nameof(attributes));
        }

        ValidateRange(offset, count, nameof(offset), nameof(count));
        int[] nativeAttributes = new int[attributes.Length];
        for (int index = 0; index < attributes.Length; ++index)
        {
            ValidateScalarRangeAttribute(attributes[index], nameof(attributes));
            nativeAttributes[index] = (int)attributes[index];
        }

        NativeCudaMemRangeAttributeValue[] nativeValues = NativeCudaApi.GetMemoryRangeAttributes(_handle, offset, count, nativeAttributes);
        CudaMemoryRangeAttributeValue[] values = new CudaMemoryRangeAttributeValue[nativeValues.Length];
        for (int index = 0; index < nativeValues.Length; ++index)
        {
            values[index] = new CudaMemoryRangeAttributeValue((CudaMemoryRangeAttribute)nativeValues[index].Attribute, nativeValues[index].Value);
        }

        return values;
    }

    /// <summary>
    /// Queries CUDA device ordinals that have accessed-by advice for this allocation.
    /// 查询对当前分配设置了 accessed-by 建议的 CUDA 设备序号。
    /// </summary>
    /// <returns>A managed copy of device ordinals reported by CUDA. CUDA 返回的设备序号托管副本。</returns>
    public int[] GetRangeAccessedByDevices()
    {
        return GetRangeAccessedByDevices(0, SizeInBytes);
    }

    /// <summary>
    /// Queries CUDA device ordinals that have accessed-by advice for a subrange of this allocation.
    /// 查询当前分配子范围内设置了 accessed-by 建议的 CUDA 设备序号。
    /// </summary>
    /// <param name="offset">The byte offset within this allocation. 当前分配内的字节偏移。</param>
    /// <param name="count">The number of bytes to query. 要查询的字节数。</param>
    /// <returns>A managed copy of device ordinals reported by CUDA. CUDA 返回的设备序号托管副本。</returns>
    public int[] GetRangeAccessedByDevices(int offset, int count)
    {
        ValidateRange(offset, count, nameof(offset), nameof(count));
        return NativeCudaApi.GetMemoryRangeAccessedByDevices(_handle, offset, count);
    }

    /// <summary>
    /// Gets a compact copied diagnostic summary for CUDA managed-memory range queries.
    /// 获取 CUDA managed-memory range 查询的紧凑复制型诊断摘要。
    /// </summary>
    /// <param name="attributes">The scalar attributes to query. 要查询的标量属性。</param>
    /// <returns>A pointer-free copied memory range diagnostic summary. 无指针逃逸的复制型 memory range 诊断摘要。</returns>
    /// <remarks>
    /// This method copies scalar attributes and accessed-by device ids into managed values. It does not expose
    /// native memory pointers and does not promote local diagnostics to runtime or package-consumer proof.
    /// 该方法将标量属性和 accessed-by 设备序号复制到托管值；不暴露原生内存指针，也不会将本地诊断
    /// 晋级为 runtime 或 package-consumer proof。
    /// </remarks>
    public CudaMemoryRangeDiagnosticSummary GetRangeDiagnosticSummary(params CudaMemoryRangeAttribute[] attributes)
    {
        return GetRangeDiagnosticSummary(0, SizeInBytes, adviceControlAttempted: false, prefetchControlAttempted: false, attributes);
    }

    /// <summary>
    /// Gets a compact copied diagnostic summary for a CUDA managed-memory subrange.
    /// 获取 CUDA managed-memory 子范围的紧凑复制型诊断摘要。
    /// </summary>
    /// <param name="offset">The byte offset within this allocation. 当前分配内的字节偏移。</param>
    /// <param name="count">The number of bytes to query. 要查询的字节数。</param>
    /// <param name="adviceControlAttempted">Whether memory advice control was attempted before this summary. 快照前是否尝试过 memory advice 控制。</param>
    /// <param name="prefetchControlAttempted">Whether memory prefetch control was attempted before this summary. 快照前是否尝试过 memory prefetch 控制。</param>
    /// <param name="attributes">The scalar attributes to query. 要查询的标量属性。</param>
    /// <returns>A pointer-free copied memory range diagnostic summary. 无指针逃逸的复制型 memory range 诊断摘要。</returns>
    public CudaMemoryRangeDiagnosticSummary GetRangeDiagnosticSummary(
        int offset,
        int count,
        bool adviceControlAttempted,
        bool prefetchControlAttempted,
        params CudaMemoryRangeAttribute[] attributes)
    {
        CudaMemoryRangeAttributeValue[] scalarAttributes = GetRangeAttributes(offset, count, attributes);
        int[] accessedByDevices = GetRangeAccessedByDevices(offset, count);
        return new CudaMemoryRangeDiagnosticSummary(
            count,
            scalarAttributes.Length,
            accessedByDevices.Length,
            adviceControlAttempted,
            prefetchControlAttempted);
    }

    /// <summary>
    /// Asynchronously allocates device memory on a CUDA stream.
    /// 在 CUDA stream 上异步分配设备内存。
    /// </summary>
    /// <param name="sizeInBytes">The allocation size in bytes. 分配大小，单位为字节。</param>
    /// <param name="stream">The CUDA stream that orders the allocation. 用于排序分配操作的 CUDA stream。</param>
    /// <returns>A managed device-memory wrapper. 设备内存的托管封装。</returns>
    public static CudaMemory AllocateAsync(int sizeInBytes, CudaStream stream)
    {
        if (sizeInBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sizeInBytes));
        }

        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        NativeBridgeLoader.EnsureInitialized();
        return new CudaMemory(NativeCudaApi.AllocateMemoryAsync(sizeInBytes, stream.Handle));
    }

    /// <summary>
    /// Asynchronously allocates device memory from a specific CUDA memory pool.
    /// 从指定 CUDA memory pool 中异步分配设备内存。
    /// </summary>
    /// <param name="sizeInBytes">The allocation size in bytes. 分配大小，单位为字节。</param>
    /// <param name="memoryPool">The CUDA memory pool used for allocation. 用于分配的 CUDA memory pool。</param>
    /// <param name="stream">The CUDA stream that orders the allocation. 用于排序分配操作的 CUDA stream。</param>
    /// <returns>A managed CUDA memory wrapper. CUDA 设备内存托管封装。</returns>
    public static CudaMemory AllocateFromPoolAsync(int sizeInBytes, CudaMemoryPool memoryPool, CudaStream stream)
    {
        if (sizeInBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sizeInBytes));
        }

        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        NativeBridgeLoader.EnsureInitialized();
        return new CudaMemory(NativeCudaApi.AllocateMemoryFromPoolAsync(sizeInBytes, memoryPool.Handle, stream.Handle));
    }

    /// <summary>
    /// Fills part of the allocation with a byte value.
    /// 使用一个字节值填充部分分配区域。
    /// </summary>
    /// <param name="value">The fill byte value. 填充值。</param>
    /// <param name="count">The number of bytes to fill. 要填充的字节数。</param>
    public void Fill(byte value, int count)
    {
        if (count < 0 || count > SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.FillMemory(_handle, value, count);
    }

    /// <summary>
    /// Fills the entire allocation with a byte value.
    /// 使用一个字节值填充整个设备内存分配。
    /// </summary>
    /// <param name="value">The fill byte value. 填充值。</param>
    public void Fill(byte value)
    {
        Fill(value, SizeInBytes);
    }

    /// <summary>
    /// Asynchronously fills part of the allocation with a byte value.
    /// 使用一个字节值异步填充分配的一部分区域。
    /// </summary>
    /// <param name="value">The fill byte value. 填充值。</param>
    /// <param name="count">The number of bytes to fill. 要填充的字节数。</param>
    /// <param name="stream">The CUDA stream that orders the fill. 用于排序填充操作的 CUDA stream。</param>
    public void FillAsync(byte value, int count, CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        ValidateCount(count, nameof(count));
        NativeCudaApi.FillMemoryAsync(_handle, value, count, stream.Handle);
    }

    /// <summary>
    /// Asynchronously fills the entire allocation with a byte value.
    /// 异步使用一个字节值填充整个设备内存分配。
    /// </summary>
    /// <param name="value">The fill byte value. 填充值。</param>
    /// <param name="stream">The CUDA stream that orders the fill. 用于排序填充操作的 CUDA stream。</param>
    public void FillAsync(byte value, CudaStream stream)
    {
        FillAsync(value, SizeInBytes, stream);
    }

    /// <summary>
    /// Asynchronously prefetches part of the allocation to a target device.
    /// 异步将分配的一部分预取到目标设备。
    /// </summary>
    /// <param name="offset">The byte offset within this allocation. 当前分配内的字节偏移。</param>
    /// <param name="count">The number of bytes to prefetch. 要预取的字节数。</param>
    /// <param name="destinationDevice">The target CUDA device ordinal. 目标 CUDA 设备序号。</param>
    /// <param name="stream">The CUDA stream that orders the prefetch. 用于排序预取操作的 CUDA stream。</param>
    public void PrefetchAsync(int offset, int count, int destinationDevice, CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        ValidateRange(offset, count, nameof(offset), nameof(count));
        NativeCudaApi.PrefetchMemoryRangeAsync(_handle, offset, count, destinationDevice, stream.Handle);
    }

    /// <summary>
    /// Asynchronously prefetches the first bytes of the allocation to a target device.
    /// 异步将分配起始处的一部分预取到目标设备。
    /// </summary>
    /// <param name="count">The number of bytes to prefetch. 要预取的字节数。</param>
    /// <param name="destinationDevice">The target CUDA device ordinal. 目标 CUDA 设备序号。</param>
    /// <param name="stream">The CUDA stream that orders the prefetch. 用于排序预取操作的 CUDA stream。</param>
    public void PrefetchAsync(int count, int destinationDevice, CudaStream stream)
    {
        PrefetchAsync(0, count, destinationDevice, stream);
    }

    /// <summary>
    /// Asynchronously prefetches the entire allocation to a target CUDA device.
    /// 异步将整个分配预取到目标 CUDA 设备。
    /// </summary>
    /// <param name="destinationDevice">The target CUDA device ordinal. 目标 CUDA 设备序号。</param>
    /// <param name="stream">The CUDA stream that orders the prefetch. 用于排序预取操作的 CUDA stream。</param>
    public void PrefetchAsync(int destinationDevice, CudaStream stream)
    {
        PrefetchAsync(SizeInBytes, destinationDevice, stream);
    }

    /// <summary>
    /// Applies CUDA memory advice to part of the allocation.
    /// 对分配的一部分应用 CUDA memory advice。
    /// </summary>
    /// <param name="offset">The byte offset within this allocation. 当前分配内的字节偏移。</param>
    /// <param name="count">The number of bytes covered by the advice. advice 覆盖的字节数。</param>
    /// <param name="advice">The CUDA memory advice. CUDA 内存建议。</param>
    /// <param name="device">The device ordinal associated with the advice. 与 advice 关联的设备序号。</param>
    public void Advise(int offset, int count, CudaMemoryAdvice advice, int device)
    {
        ValidateRange(offset, count, nameof(offset), nameof(count));
        ValidateMemoryAdvice(advice, nameof(advice));
        NativeCudaApi.AdviseMemoryRange(_handle, offset, count, (int)advice, device);
    }

    /// <summary>
    /// Applies CUDA memory advice to the first bytes of the allocation.
    /// 对分配起始处的一部分应用 CUDA memory advice。
    /// </summary>
    /// <param name="count">The number of bytes covered by the advice. advice 覆盖的字节数。</param>
    /// <param name="advice">The CUDA memory advice. CUDA 内存建议。</param>
    /// <param name="device">The device ordinal associated with the advice. 与 advice 关联的设备序号。</param>
    public void Advise(int count, CudaMemoryAdvice advice, int device)
    {
        Advise(0, count, advice, device);
    }

    /// <summary>
    /// Applies a CUDA memory advice to the entire allocation.
    /// 对整个分配应用 CUDA memory advice。
    /// </summary>
    /// <param name="advice">The CUDA memory advice. CUDA 内存建议。</param>
    /// <param name="device">The device ordinal associated with the advice. 与 advice 关联的设备序号。</param>
    public void Advise(CudaMemoryAdvice advice, int device)
    {
        Advise(SizeInBytes, advice, device);
    }

    /// <summary>
    /// Copies a byte array into device memory.
    /// 将字节数组复制到设备内存中。
    /// </summary>
    /// <param name="source">The source byte array. 源字节数组。</param>
    public void CopyFrom(byte[] source)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        if (source.Length > SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(source));
        }

        NativeCudaApi.CopyFromHost(_handle, source, source.Length);
    }

    /// <summary>
    /// Copies a float array into device memory.
    /// 将浮点数组复制到设备内存中。
    /// </summary>
    /// <param name="source">The source float array. 源浮点数组。</param>
    public void CopyFrom(float[] source)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        int byteCount = checked(source.Length * sizeof(float));
        if (byteCount > SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(source));
        }

        byte[] bytes = new byte[byteCount];
        Buffer.BlockCopy(source, 0, bytes, 0, byteCount);
        CopyFrom(bytes);
    }

    /// <summary>
    /// Asynchronously copies a pinned host buffer into device memory.
    /// 将 pinned host 缓冲区异步复制到设备内存中。
    /// </summary>
    /// <param name="source">The source pinned host buffer. 源 pinned host 缓冲区。</param>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyFromAsync(CudaPinnedMemory source, int count, CudaStream stream)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        ValidateCount(count, nameof(count));
        if (count > source.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.CopyFromPinnedHostAsync(_handle, source.Handle, count, stream.Handle);
    }

    /// <summary>
    /// Asynchronously copies the full pinned-host buffer into this device allocation.
    /// 异步将整个 pinned host 缓冲区复制到当前设备内存。
    /// </summary>
    /// <param name="source">The source pinned host buffer. 源 pinned host 缓冲区。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyFromAsync(CudaPinnedMemory source, CudaStream stream)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        CopyFromAsync(source, Math.Min(source.SizeInBytes, SizeInBytes), stream);
    }

    /// <summary>
    /// Copies device memory into a byte array.
    /// 将设备内存复制到字节数组中。
    /// </summary>
    /// <param name="destination">The destination byte array. 目标字节数组。</param>
    public void CopyTo(byte[] destination)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        if (destination.Length > SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(destination));
        }

        NativeCudaApi.CopyToHost(_handle, destination, destination.Length);
    }

    /// <summary>
    /// Copies device memory into a float array.
    /// 将设备内存复制到浮点数组中。
    /// </summary>
    /// <param name="destination">The destination float array. 目标浮点数组。</param>
    public void CopyTo(float[] destination)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        int byteCount = checked(destination.Length * sizeof(float));
        if (byteCount > SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(destination));
        }

        byte[] bytes = new byte[byteCount];
        CopyTo(bytes);
        Buffer.BlockCopy(bytes, 0, destination, 0, byteCount);
    }

    /// <summary>
    /// Asynchronously copies device memory into a pinned host buffer.
    /// 将设备内存异步复制到 pinned host 缓冲区中。
    /// </summary>
    /// <param name="destination">The destination pinned host buffer. 目标 pinned host 缓冲区。</param>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyToAsync(CudaPinnedMemory destination, int count, CudaStream stream)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        ValidateCount(count, nameof(count));
        if (count > destination.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.CopyToPinnedHostAsync(_handle, destination.Handle, count, stream.Handle);
    }

    /// <summary>
    /// Asynchronously copies this allocation into a full pinned-host buffer.
    /// 异步将当前设备内存复制到整个 pinned host 缓冲区。
    /// </summary>
    /// <param name="destination">The destination pinned host buffer. 目标 pinned host 缓冲区。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyToAsync(CudaPinnedMemory destination, CudaStream stream)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        CopyToAsync(destination, Math.Min(destination.SizeInBytes, SizeInBytes), stream);
    }

    /// <summary>
    /// Copies bytes from this allocation to another device allocation.
    /// 将当前分配中的字节复制到另一个设备分配中。
    /// </summary>
    /// <param name="destination">The destination device allocation. 目标设备内存分配。</param>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    public void CopyTo(CudaMemory destination, int count)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        ValidateCount(count, nameof(count));
        if (count > destination.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.CopyDeviceToDevice(destination._handle, _handle, count);
    }

    /// <summary>
    /// Copies bytes from this allocation to another allocation using the smaller allocation size.
    /// 按两个分配中较小的大小，将当前分配复制到另一个分配。
    /// </summary>
    /// <param name="destination">The destination device allocation. 目标设备内存分配。</param>
    public void CopyTo(CudaMemory destination)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        CopyTo(destination, Math.Min(SizeInBytes, destination.SizeInBytes));
    }

    /// <summary>
    /// Asynchronously copies bytes to another device allocation.
    /// 异步将字节复制到另一个设备分配中。
    /// </summary>
    /// <param name="destination">The destination device allocation. 目标设备内存分配。</param>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyToAsync(CudaMemory destination, int count, CudaStream stream)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        ValidateCount(count, nameof(count));
        if (count > destination.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.CopyDeviceToDeviceAsync(destination._handle, _handle, count, stream.Handle);
    }

    /// <summary>
    /// Asynchronously copies bytes to another allocation using the smaller allocation size.
    /// 按两个分配中较小的大小，异步将当前分配复制到另一个分配。
    /// </summary>
    /// <param name="destination">The destination device allocation. 目标设备内存分配。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyToAsync(CudaMemory destination, CudaStream stream)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        CopyToAsync(destination, Math.Min(SizeInBytes, destination.SizeInBytes), stream);
    }

    /// <summary>
    /// Copies bytes to another CUDA allocation and lets CUDA infer the copy direction.
    /// 将字节复制到另一个 CUDA 分配，并让 CUDA 自动推断复制方向。
    /// </summary>
    /// <param name="destination">The destination allocation. 目标设备内存分配。</param>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    public void CopyToAuto(CudaMemory destination, int count)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        ValidateCount(count, nameof(count));
        if (count > destination.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.CopyDefault(destination._handle, _handle, count);
    }

    /// <summary>
    /// Asynchronously copies bytes to another CUDA allocation and lets CUDA infer the copy direction.
    /// 异步将字节复制到另一个 CUDA 分配，并让 CUDA 自动推断复制方向。
    /// </summary>
    /// <param name="destination">The destination allocation. 目标设备内存分配。</param>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序复制操作的 CUDA stream。</param>
    public void CopyToAutoAsync(CudaMemory destination, int count, CudaStream stream)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        ValidateCount(count, nameof(count));
        if (count > destination.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.CopyDefaultAsync(destination._handle, _handle, count, stream.Handle);
    }

    /// <summary>
    /// Copies bytes from this allocation to a destination allocation on another CUDA device.
    /// 将当前分配中的字节复制到另一个 CUDA 设备上的目标分配。
    /// </summary>
    /// <param name="destination">The destination allocation. 目标设备内存分配。</param>
    /// <param name="sourceDevice">The CUDA device ordinal that owns this source allocation. 拥有当前源分配的 CUDA 设备序号。</param>
    /// <param name="destinationDevice">The CUDA device ordinal that owns the destination allocation. 拥有目标分配的 CUDA 设备序号。</param>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    public void CopyToPeer(CudaMemory destination, int sourceDevice, int destinationDevice, int count)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        ValidateCount(count, nameof(count));
        if (count > destination.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.CopyPeer(destination._handle, destinationDevice, _handle, sourceDevice, count);
    }

    /// <summary>
    /// Asynchronously copies bytes from this allocation to another CUDA device using a stream.
    /// 使用 stream 将当前分配中的字节异步复制到另一个 CUDA 设备。
    /// </summary>
    /// <param name="destination">The destination allocation. 目标设备内存分配。</param>
    /// <param name="sourceDevice">The CUDA device ordinal that owns this source allocation. 拥有当前源分配的 CUDA 设备序号。</param>
    /// <param name="destinationDevice">The CUDA device ordinal that owns the destination allocation. 拥有目标分配的 CUDA 设备序号。</param>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    /// <param name="stream">The CUDA stream that orders the copy. 用于排序该复制操作的 CUDA stream。</param>
    public void CopyToPeerAsync(CudaMemory destination, int sourceDevice, int destinationDevice, int count, CudaStream stream)
    {
        if (destination == null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        ValidateCount(count, nameof(count));
        if (count > destination.SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        NativeCudaApi.CopyPeerAsync(destination._handle, destinationDevice, _handle, sourceDevice, count, stream.Handle);
    }

    /// <summary>
    /// Asynchronously frees this allocation on the supplied CUDA stream and invalidates the managed handle.
    /// 在指定 CUDA stream 上异步释放当前分配，并使托管句柄失效。
    /// </summary>
    /// <param name="stream">The CUDA stream that orders the free operation. 用于排序释放操作的 CUDA stream。</param>
    public void FreeAsync(CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        NativeCudaApi.FreeMemoryAsync(_handle, stream.Handle);
        _handle.MarkReleased();
    }

    /// <summary>
    /// Materializes a byte array from device memory.
    /// 从设备内存中生成字节数组。
    /// </summary>
    /// <param name="count">The number of bytes to copy. 要复制的字节数。</param>
    /// <returns>A managed byte array copy. 托管字节数组副本。</returns>
    public byte[] ToArray(int count)
    {
        if (count < 0 || count > SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        byte[] data = new byte[count];
        CopyTo(data);
        return data;
    }

    /// <summary>
    /// Materializes a float array from device memory.
    /// 从设备内存中生成浮点数组。
    /// </summary>
    /// <param name="elementCount">The number of single-precision elements to copy. 要复制的单精度元素数量。</param>
    /// <returns>A managed float array copy. 托管浮点数组副本。</returns>
    public float[] ToSingleArray(int elementCount)
    {
        if (elementCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(elementCount));
        }

        int byteCount = checked(elementCount * sizeof(float));
        ValidateCount(byteCount, nameof(elementCount));
        float[] data = new float[elementCount];
        CopyTo(data);
        return data;
    }

    /// <summary>
    /// Releases the device allocation.
    /// 释放设备内存分配。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ValidateCount(int count, string parameterName)
    {
        if (count < 0 || count > SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(parameterName);
        }
    }

    protected void ValidateRange(int offset, int count, string offsetParameterName, string countParameterName)
    {
        if (offset < 0 || offset > SizeInBytes)
        {
            throw new ArgumentOutOfRangeException(offsetParameterName);
        }

        if (count <= 0 || count > SizeInBytes - offset)
        {
            throw new ArgumentOutOfRangeException(countParameterName);
        }
    }

    private static void ValidateScalarRangeAttribute(CudaMemoryRangeAttribute attribute, string parameterName)
    {
        if (!Enum.IsDefined(typeof(CudaMemoryRangeAttribute), attribute))
        {
            throw new ArgumentOutOfRangeException(parameterName, attribute, "Unsupported CUDA memory range attribute.");
        }

        if (attribute == CudaMemoryRangeAttribute.AccessedBy)
        {
            throw new ArgumentException("cudaMemRangeAttributeAccessedBy returns a device-id array and is not exposed by the scalar range attribute helpers.", parameterName);
        }
    }

    protected static void ValidateMemoryAdvice(CudaMemoryAdvice advice, string parameterName)
    {
        if (!Enum.IsDefined(typeof(CudaMemoryAdvice), advice))
        {
            throw new ArgumentOutOfRangeException(parameterName, advice, "Unsupported CUDA memory advice.");
        }
    }

    private static SafeCudaMemoryHandle AllocateDeviceMemory(int sizeInBytes)
    {
        if (sizeInBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sizeInBytes));
        }

        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.AllocateMemory(sizeInBytes);
    }
}

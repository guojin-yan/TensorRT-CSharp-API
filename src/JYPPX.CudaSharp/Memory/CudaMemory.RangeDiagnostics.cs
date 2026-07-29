using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public partial class CudaMemory
{
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

}

using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>Describes a copied CUDA kernel-library inventory without exposing borrowed kernel handles. 描述不暴露 borrowed kernel handle 的 CUDA kernel-library 复制型清单。</summary>
public readonly struct CudaKernelLibraryInventorySnapshot
{
    internal CudaKernelLibraryInventorySnapshot(NativeCudaKernelLibraryInventory native)
    {
        ReportedKernelCount = native.ReportedKernelCount;
        EnumeratedKernelCount = native.EnumeratedKernelCount;
        NullKernelCount = native.NullKernelCount;
        IsComplete = native.IsComplete != 0;
    }

    /// <summary>Gets the count reported by CUDA. 获取 CUDA 报告的 kernel 数量。</summary>
    public uint ReportedKernelCount { get; }

    /// <summary>Gets the number of non-null borrowed handles observed inside the native call. 获取 native 调用栈内观察到的非空 borrowed handle 数量。</summary>
    public uint EnumeratedKernelCount { get; }

    /// <summary>Gets the number of null entries observed while validating the inventory. 获取校验清单时观察到的空条目数量。</summary>
    public uint NullKernelCount { get; }

    /// <summary>Gets whether the copied inventory was complete. 获取复制型清单是否完整。</summary>
    public bool IsComplete { get; }

    /// <inheritdoc />
    public override string ToString() =>
        $"Reported={ReportedKernelCount}, Enumerated={EnumeratedKernelCount}, Null={NullKernelCount}, Complete={IsComplete}";
}

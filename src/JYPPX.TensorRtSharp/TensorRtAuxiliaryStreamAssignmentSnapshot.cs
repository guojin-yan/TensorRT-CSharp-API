using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Describes the managed lifetime state of execution-context auxiliary CUDA streams without exposing native pointers.
/// 描述 execution context 辅助 CUDA stream 的托管生命周期状态，且不暴露原生指针。
/// </summary>
public sealed class TensorRtAuxiliaryStreamAssignmentSnapshot
{
    internal TensorRtAuxiliaryStreamAssignmentSnapshot(
        TensorRtApiLine line,
        int assignedStreamCount,
        bool isCleared,
        bool managedHandleLeaseActive,
        string diagnostic)
    {
        Line = line;
        AssignedStreamCount = assignedStreamCount;
        IsCleared = isCleared;
        ManagedHandleLeaseActive = managedHandleLeaseActive;
        Diagnostic = diagnostic;
    }

    /// <summary>Gets the TensorRT API line. 获取 TensorRT API 版本线。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the number of assigned auxiliary streams. 获取已分配辅助 stream 数量。</summary>
    public int AssignedStreamCount { get; }

    /// <summary>Gets whether user-provided auxiliary streams are cleared. 获取是否已清除用户提供的辅助 stream。</summary>
    public bool IsCleared { get; }

    /// <summary>Gets whether managed SafeHandle leases are active. 获取托管 SafeHandle lease 是否有效。</summary>
    public bool ManagedHandleLeaseActive { get; }

    /// <summary>Always reports that no native CUDA stream pointer is exposed. 始终表示未暴露原生 CUDA stream 指针。</summary>
    public bool NativeStreamPointerExposed => false;

    /// <summary>Always reports that no borrowed handle escapes the wrapper. 始终表示没有借用句柄逃逸出封装。</summary>
    public bool BorrowedHandleEscaped => false;

    /// <summary>Gets a pointer-free lifetime diagnostic. 获取不含指针的生命周期诊断。</summary>
    public string Diagnostic { get; }
}

using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a compact pointer-free summary of a copied TensorRT error-recorder snapshot.
/// 表示 TensorRT error-recorder 已复制快照的紧凑无指针摘要。
/// </summary>
public sealed class TensorRtErrorRecorderSummary
{
    internal TensorRtErrorRecorderSummary(
        TensorRtApiLine line,
        bool hasRecorder,
        int errorCount,
        int copiedErrorRecordCount,
        bool hasOverflowed,
        bool interfaceInfoAvailable,
        string interfaceName,
        int interfaceMajor,
        int interfaceMinor,
        int? firstErrorCode,
        int firstErrorDescriptionLength)
    {
        Line = line;
        HasRecorder = hasRecorder;
        ErrorCount = errorCount < 0 ? 0 : errorCount;
        CopiedErrorRecordCount = copiedErrorRecordCount < 0 ? 0 : copiedErrorRecordCount;
        HasOverflowed = hasOverflowed;
        InterfaceInfoAvailable = interfaceInfoAvailable;
        InterfaceName = interfaceName ?? string.Empty;
        InterfaceMajor = interfaceMajor;
        InterfaceMinor = interfaceMinor;
        FirstErrorCode = firstErrorCode;
        FirstErrorDescriptionLength = firstErrorDescriptionLength < 0 ? 0 : firstErrorDescriptionLength;
    }

    /// <summary>Gets the TensorRT API line used to collect this summary. 获取采集该摘要的 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets whether the source object had an attached error recorder. 获取来源对象是否附加 error recorder。</summary>
    public bool HasRecorder { get; }

    /// <summary>Gets the error count reported by TensorRT. 获取 TensorRT 报告的错误数量。</summary>
    public int ErrorCount { get; }

    /// <summary>Gets the number of copied managed error records. 获取已复制到托管侧的错误记录数量。</summary>
    public int CopiedErrorRecordCount { get; }

    /// <summary>Gets whether the recorder reported overflow. 获取 recorder 是否报告溢出。</summary>
    public bool HasOverflowed { get; }

    /// <summary>Gets whether copied interface metadata was available. 获取复制型 interface 元数据是否可用。</summary>
    public bool InterfaceInfoAvailable { get; }

    /// <summary>Gets the copied interface name, if available. 获取已复制 interface 名称；不可用时为空。</summary>
    public string InterfaceName { get; }

    /// <summary>Gets the copied interface major version, if available. 获取已复制 interface 主版本；不可用时为 0。</summary>
    public int InterfaceMajor { get; }

    /// <summary>Gets the copied interface minor version, if available. 获取已复制 interface 次版本；不可用时为 0。</summary>
    public int InterfaceMinor { get; }

    /// <summary>Gets the first copied error code, if any. 获取第一条已复制错误码；无记录时为空。</summary>
    public int? FirstErrorCode { get; }

    /// <summary>Gets the first copied error-description length without retaining native memory. 获取第一条已复制错误描述长度，不持有原生内存。</summary>
    public int FirstErrorDescriptionLength { get; }

    /// <summary>Gets whether all reported errors were copied into managed records. 获取报告的错误是否都复制到了托管记录。</summary>
    public bool CopiedRecordCountMatchesErrorCount => ErrorCount == CopiedErrorRecordCount;

    /// <summary>Gets the runtime evidence kind represented by this copied summary. 获取该 copied summary 表示的 runtime evidence 类型。</summary>
    public string RuntimeEvidenceKind => "copied-readonly-summary";

    /// <summary>Gets whether this summary is runtime execution evidence. 获取该摘要是否为 runtime execution evidence。</summary>
    public bool IsRuntimeExecutionEvidence => false;

    /// <summary>Gets whether this summary is runtime execution proof. 获取该摘要是否为 runtime execution proof。</summary>
    public bool IsRuntimeExecutionProof => false;

    /// <summary>Gets whether this summary is copied and pointer-free. 获取该摘要是否为复制型且不暴露指针。</summary>
    public bool PointerFreeCopiedSummary => true;

    /// <summary>Gets whether this summary can be promoted as runtime proof. 获取该摘要是否可晋级为 runtime proof。</summary>
    public bool CanPromoteRuntimeProof => false;

    /// <summary>Gets whether this summary can promote public release proof. 获取该摘要是否可晋级为 public release proof。</summary>
    public bool CanPromoteReleaseProof => false;

    /// <summary>Gets whether this summary allows deleting deferred records. 获取该摘要是否允许删除 deferred 记录。</summary>
    public bool CanDeleteDeferredRecord => false;

    /// <summary>
    /// Converts the summary to a compact diagnostic string.
    /// 转换为紧凑诊断字符串。
    /// </summary>
    public override string ToString()
    {
        string interfaceInfo = InterfaceInfoAvailable
            ? $"{InterfaceName}:{InterfaceMajor}.{InterfaceMinor}"
            : "n/a";
        string firstError = FirstErrorCode.HasValue ? FirstErrorCode.Value.ToString() : "n/a";
        return $"{Line}:hasRecorder={HasRecorder}:errors={ErrorCount}:copied={CopiedErrorRecordCount}:overflow={HasOverflowed}:interface={interfaceInfo}:first={firstError}:firstDescLen={FirstErrorDescriptionLength}:runtimeProof={CanPromoteRuntimeProof}";
    }
}

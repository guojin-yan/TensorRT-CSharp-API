using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Pointer-free aggregate of profiler interface metadata probes.
/// profiler interface metadata probe 的无指针聚合副本。
/// </summary>
public readonly struct TensorRtProfilerInterfaceMetadataSnapshot
{
    internal TensorRtProfilerInterfaceMetadataSnapshot(
        TensorRtApiLine line,
        bool interfaceInfoAvailable,
        TensorRtInterfaceInfo interfaceInfo,
        string interfaceInfoDiagnostic,
        bool apiLanguageAvailable,
        TensorRtApiLanguage apiLanguage,
        string apiLanguageDiagnostic)
    {
        Line = line;
        InterfaceInfoAvailable = interfaceInfoAvailable;
        InterfaceInfo = interfaceInfo;
        InterfaceInfoDiagnostic = interfaceInfoDiagnostic ?? string.Empty;
        ApiLanguageAvailable = apiLanguageAvailable;
        ApiLanguage = apiLanguage;
        ApiLanguageDiagnostic = apiLanguageDiagnostic ?? string.Empty;
    }

    /// <summary>Gets the TensorRT API line probed.</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets whether copied interface info was available.</summary>
    public bool InterfaceInfoAvailable { get; }

    /// <summary>Gets copied interface info when available.</summary>
    public TensorRtInterfaceInfo InterfaceInfo { get; }

    /// <summary>Gets the interface-info diagnostic.</summary>
    public string InterfaceInfoDiagnostic { get; }

    /// <summary>Gets whether copied API-language metadata was available.</summary>
    public bool ApiLanguageAvailable { get; }

    /// <summary>Gets copied API-language metadata when available.</summary>
    public TensorRtApiLanguage ApiLanguage { get; }

    /// <summary>Gets the API-language diagnostic.</summary>
    public string ApiLanguageDiagnostic { get; }

    /// <summary>
    /// Gets whether both metadata queries completed successfully.
    /// 获取两个 metadata 查询是否都成功完成。
    /// </summary>
    public bool IsComplete => InterfaceInfoAvailable && ApiLanguageAvailable;

    /// <summary>
    /// This aggregate is copied diagnostic state, not runtime or package-consumer proof.
    /// 该聚合是复制出的诊断状态，不是 runtime 或 package-consumer proof。
    /// </summary>
    public bool IsRuntimeProof => false;
}

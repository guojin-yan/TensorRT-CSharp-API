using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Copied no-throw diagnostic snapshot for TensorRT 11 runtime creation.
/// TensorRT 11 runtime 创建过程的复制型 no-throw 诊断快照。
/// </summary>
public sealed class TensorRtRuntimeCreateDiagnosticSnapshot
{
    internal TensorRtRuntimeCreateDiagnosticSnapshot(
        TensorRtApiLine line,
        bool diagnosticAvailable,
        bool attempted,
        bool loggerHandlePresent,
        bool loggerPayloadPresent,
        bool createInferRuntimeReturnedNonNull,
        bool createInferRuntimeReturnedNull,
        BridgeStatusCode lastStatus,
        bool tensorRtAvailable,
        int expectedMajor,
        int bridgeBuiltMajor,
        string detectedVersion,
        bool loggerCallbackAvailable,
        uint loggerMessageCount,
        int lastLoggerSeverity,
        string lastLoggerMessage,
        string createRuntimePhase,
        string nativeDetail,
        string diagnostic)
    {
        Line = line;
        DiagnosticAvailable = diagnosticAvailable;
        Attempted = attempted;
        LoggerHandlePresent = loggerHandlePresent;
        LoggerPayloadPresent = loggerPayloadPresent;
        CreateInferRuntimeReturnedNonNull = createInferRuntimeReturnedNonNull;
        CreateInferRuntimeReturnedNull = createInferRuntimeReturnedNull;
        LastStatus = lastStatus;
        TensorRtAvailable = tensorRtAvailable;
        ExpectedMajor = expectedMajor;
        BridgeBuiltMajor = bridgeBuiltMajor;
        DetectedVersion = detectedVersion ?? string.Empty;
        LoggerCallbackAvailable = loggerCallbackAvailable;
        LoggerMessageCount = loggerMessageCount;
        LastLoggerSeverity = lastLoggerSeverity;
        LastLoggerMessage = lastLoggerMessage ?? string.Empty;
        CreateRuntimePhase = createRuntimePhase ?? string.Empty;
        NativeDetail = nativeDetail ?? string.Empty;
        Diagnostic = diagnostic ?? string.Empty;
    }

    /// <summary>
    /// Gets the TensorRT API line used to collect this snapshot.
    /// 获取采集该快照的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets whether the native diagnostic entry was available and returned a copied snapshot.
    /// 获取 native 诊断入口是否可用并返回了复制快照。
    /// </summary>
    public bool DiagnosticAvailable { get; }

    /// <summary>
    /// Gets whether the bridge attempted to call <c>createInferRuntime</c>.
    /// 获取 bridge 是否尝试调用 <c>createInferRuntime</c>。
    /// </summary>
    public bool Attempted { get; }

    /// <summary>
    /// Gets whether a logger handle was passed into the diagnostic entry.
    /// 获取是否向诊断入口传入了 logger handle。
    /// </summary>
    public bool LoggerHandlePresent { get; }

    /// <summary>
    /// Gets whether the logger handle contained a native logger payload.
    /// 获取 logger handle 是否包含 native logger payload。
    /// </summary>
    public bool LoggerPayloadPresent { get; }

    /// <summary>
    /// Gets whether <c>createInferRuntime</c> returned a non-null runtime.
    /// 获取 <c>createInferRuntime</c> 是否返回了非空 runtime。
    /// </summary>
    public bool CreateInferRuntimeReturnedNonNull { get; }

    /// <summary>
    /// Gets whether <c>createInferRuntime</c> returned null.
    /// 获取 <c>createInferRuntime</c> 是否返回 null。
    /// </summary>
    public bool CreateInferRuntimeReturnedNull { get; }

    /// <summary>
    /// Gets the copied native bridge status from the diagnostic attempt.
    /// 获取诊断尝试复制出的 native bridge 状态。
    /// </summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>
    /// Gets whether this native bridge was compiled with TensorRT support.
    /// 获取该 native bridge 是否编译了 TensorRT 支持。
    /// </summary>
    public bool TensorRtAvailable { get; }

    /// <summary>
    /// Gets the expected TensorRT major version for this diagnostic.
    /// 获取该诊断期望的 TensorRT 主版本。
    /// </summary>
    public int ExpectedMajor { get; }

    /// <summary>
    /// Gets the TensorRT major version used to build the native bridge.
    /// 获取 native bridge 构建时使用的 TensorRT 主版本。
    /// </summary>
    public int BridgeBuiltMajor { get; }

    /// <summary>
    /// Gets the copied TensorRT version text compiled into the native bridge.
    /// 获取 native bridge 编译进来的 TensorRT 版本文本副本。
    /// </summary>
    public string DetectedVersion { get; }

    /// <summary>
    /// Gets whether the native logger payload had a managed callback installed.
    /// 获取 native logger payload 是否安装了托管回调。
    /// </summary>
    public bool LoggerCallbackAvailable { get; }

    /// <summary>
    /// Gets the number of TensorRT logger messages observed during diagnostic runtime creation.
    /// 获取诊断 runtime 创建期间捕获到的 TensorRT logger 消息数量。
    /// </summary>
    public uint LoggerMessageCount { get; }

    /// <summary>
    /// Gets the last TensorRT logger severity observed by the native logger.
    /// 获取 native logger 观察到的最后一条 TensorRT logger severity。
    /// </summary>
    public int LastLoggerSeverity { get; }

    /// <summary>
    /// Gets the last copied TensorRT logger message observed by the native logger.
    /// 获取 native logger 复制出的最后一条 TensorRT logger 消息。
    /// </summary>
    public string LastLoggerMessage { get; }

    /// <summary>
    /// Gets the native diagnostic phase reached by the createInferRuntime attempt.
    /// 获取 createInferRuntime 尝试到达的 native 诊断阶段。
    /// </summary>
    public string CreateRuntimePhase { get; }

    /// <summary>
    /// Gets copied native detail about the runtime creation attempt.
    /// 获取 runtime 创建尝试的 native 细节副本。
    /// </summary>
    public string NativeDetail { get; }

    /// <summary>
    /// Gets a copied diagnostic message. No runtime or logger pointer is exposed.
    /// 获取复制出的诊断消息；不会暴露 runtime 或 logger 指针。
    /// </summary>
    public string Diagnostic { get; }

    /// <summary>
    /// Gets whether this diagnostic can promote runtime proof.
    /// 获取该诊断是否能提升 runtime proof。
    /// </summary>
    public bool CanPromoteRuntimeProof => false;

    /// <summary>
    /// Gets whether this diagnostic is public package-consumer runtime proof.
    /// 获取该诊断是否为公开 package-consumer runtime proof。
    /// </summary>
    public bool IsPackageConsumerRuntimeProof => false;

    /// <summary>
    /// Gets whether this diagnostic exposes native pointers.
    /// 获取该诊断是否暴露 native 指针。
    /// </summary>
    public bool ExposesNativePointer => false;

    public override string ToString()
    {
        return $"{Line}:available={DiagnosticAvailable}:attempted={Attempted}:returnedNull={CreateInferRuntimeReturnedNull}:status={LastStatus}:version={DetectedVersion}";
    }
}

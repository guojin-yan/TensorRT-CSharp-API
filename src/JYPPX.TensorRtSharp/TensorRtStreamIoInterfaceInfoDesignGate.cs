using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates the pointer-free TensorRT stream reader/writer interface-info design gate.
/// 评估 TensorRT stream reader/writer interface-info 的无裸指针设计门。
/// </summary>
/// <remarks>
/// Stream reader and writer interfaces are application-provided serialization/deserialization callbacks.
/// This gate allows only copied interface metadata planning and does not expose reader or writer handles.
/// Stream reader/writer 是应用侧提供的序列化/反序列化 callback；该门禁只允许 copied interface metadata
/// 规划，不暴露 reader 或 writer handle。
/// </remarks>
public static class TensorRtStreamIoInterfaceInfoDesignGate
{
    /// <summary>
    /// Evaluates the known public stream IO interface-info design surface for a TensorRT API line.
    /// 基于已知 public stream IO interface-info 设计边界评估指定 TensorRT API line。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <returns>A pointer-free stream IO interface-info design gate result. 无裸指针 stream IO interface-info 设计门结果。</returns>
    public static TensorRtStreamIoInterfaceInfoDesignGateResult EvaluateKnownSurface(TensorRtApiLine line)
    {
        return Evaluate(
            line,
            copiedInterfaceInfoMetadataReady: true,
            streamOwnerLifetimeModeled: false,
            readWriteBufferOwnershipModeled: false,
            seekTellLifetimeModeled: false);
    }

    /// <summary>
    /// Evaluates stream IO interface-info readiness from explicit capability flags.
    /// 根据显式能力标记评估 stream IO interface-info 就绪状态。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <param name="copiedInterfaceInfoMetadataReady">Whether copied interface-info metadata shape is ready. copied interface-info metadata 形态是否就绪。</param>
    /// <param name="streamOwnerLifetimeModeled">Whether stream owner lifetime has been modeled. stream owner lifetime 是否已建模。</param>
    /// <param name="readWriteBufferOwnershipModeled">Whether read/write buffer ownership has been modeled. read/write buffer ownership 是否已建模。</param>
    /// <param name="seekTellLifetimeModeled">Whether seek/tell state lifetime has been modeled. seek/tell state lifetime 是否已建模。</param>
    /// <returns>A pointer-free stream IO interface-info design gate result. 无裸指针 stream IO interface-info 设计门结果。</returns>
    public static TensorRtStreamIoInterfaceInfoDesignGateResult Evaluate(
        TensorRtApiLine line,
        bool copiedInterfaceInfoMetadataReady,
        bool streamOwnerLifetimeModeled,
        bool readWriteBufferOwnershipModeled,
        bool seekTellLifetimeModeled)
    {
        bool lineSupportsStreamIo =
            line == TensorRtApiLine.TensorRt10 ||
            line == TensorRtApiLine.TensorRt11;
        bool lineSupportsStreamWriter = line == TensorRtApiLine.TensorRt11;

        List<string> blockers = new List<string>();
        if (!lineSupportsStreamIo)
        {
            blockers.Add("TensorRT 10 or 11 stream IO line support has not been selected.");
        }

        if (!copiedInterfaceInfoMetadataReady)
        {
            blockers.Add("copied stream IO interface-info metadata shape is not ready.");
        }

        if (!streamOwnerLifetimeModeled)
        {
            blockers.Add("stream reader/writer owner lifetime is not modeled.");
        }

        if (!readWriteBufferOwnershipModeled)
        {
            blockers.Add("stream read/write buffer ownership remains deferred.");
        }

        if (!seekTellLifetimeModeled)
        {
            blockers.Add("stream seek/tell state lifetime remains deferred.");
        }

        blockers.Add("managed-owned stream owner SafeHandle/GCHandle lifetime is not implemented.");
        blockers.Add("native stream owner create/destroy symmetry is not implemented.");
        blockers.Add("no-throw stream read/write/seek vtable and exception-to-status mapping are not implemented.");
        blockers.Add("stream detach-before-release ordering is not implemented.");
        blockers.Add("stream getAPILanguage metadata remains deferred until owner lifetime is closed.");
        blockers.Add("direct IStreamReader, IStreamReaderV2, and IStreamWriter pointer access remains deferred by design.");
        blockers.Add("stream read/seek/write callback invocation remains deferred by design.");
        blockers.Add("full package consumer runtime execution proof has not been promoted from this design gate.");

        return new TensorRtStreamIoInterfaceInfoDesignGateResult(
            line,
            lineSupportsStreamIo,
            lineSupportsStreamWriter,
            copiedInterfaceInfoMetadataReady,
            streamOwnerLifetimeModeled,
            readWriteBufferOwnershipModeled,
            seekTellLifetimeModeled,
            blockers.ToArray());
    }
}

/// <summary>
/// Reports stream IO interface-info design status without exposing native stream handles.
/// 报告 stream IO interface-info 设计状态，不暴露原生 stream handle。
/// </summary>
public readonly struct TensorRtStreamIoInterfaceInfoDesignGateResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtStreamIoInterfaceInfoDesignGateResult(
        TensorRtApiLine line,
        bool lineSupportsStreamIo,
        bool lineSupportsStreamWriter,
        bool copiedInterfaceInfoMetadataReady,
        bool streamOwnerLifetimeModeled,
        bool readWriteBufferOwnershipModeled,
        bool seekTellLifetimeModeled,
        string[] blockedPrerequisites)
    {
        Line = line;
        LineSupportsStreamIo = lineSupportsStreamIo;
        LineSupportsStreamWriter = lineSupportsStreamWriter;
        CopiedInterfaceInfoMetadataReady = copiedInterfaceInfoMetadataReady;
        StreamOwnerLifetimeModeled = streamOwnerLifetimeModeled;
        ReadWriteBufferOwnershipModeled = readWriteBufferOwnershipModeled;
        SeekTellLifetimeModeled = seekTellLifetimeModeled;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this gate. 获取该设计门的证据标记。</summary>
    public string EvidenceKind => "stream-io-interface-info-design-gate";

    /// <summary>Gets the diagnostics kind represented by this gate. 获取该设计门代表的诊断类型。</summary>
    public string DiagnosticsKind => "stream-io-interface-info";

    /// <summary>Gets the runtime evidence kind. 获取 runtime evidence 类型。</summary>
    public string RuntimeEvidenceKind => "design-gate";

    /// <summary>Gets whether this gate is runtime execution evidence. 获取该设计门是否为 runtime execution evidence。</summary>
    public bool IsRuntimeExecutionEvidence => false;

    /// <summary>Gets whether this gate may be promoted as runtime execution proof. 获取该设计门是否可晋级为 runtime execution proof。</summary>
    public bool IsRuntimeExecutionProof => false;

    /// <summary>Gets the TensorRT API line. 获取 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets whether the selected TensorRT line supports stream reader metadata. 获取当前 TensorRT line 是否支持 stream reader metadata。</summary>
    public bool LineSupportsStreamIo { get; }

    /// <summary>Gets whether the selected TensorRT line supports stream writer metadata. 获取当前 TensorRT line 是否支持 stream writer metadata。</summary>
    public bool LineSupportsStreamWriter { get; }

    /// <summary>Gets whether copied interface-info metadata shape is ready. 获取 copied interface-info metadata 形态是否就绪。</summary>
    public bool CopiedInterfaceInfoMetadataReady { get; }

    /// <summary>Gets whether stream owner lifetime has been modeled. 获取 stream owner lifetime 是否已建模。</summary>
    public bool StreamOwnerLifetimeModeled { get; }

    /// <summary>Gets whether read/write buffer ownership has been modeled. 获取 read/write buffer ownership 是否已建模。</summary>
    public bool ReadWriteBufferOwnershipModeled { get; }

    /// <summary>Gets whether seek/tell state lifetime has been modeled. 获取 seek/tell state lifetime 是否已建模。</summary>
    public bool SeekTellLifetimeModeled { get; }

    /// <summary>Gets whether public APIs expose stream reader pointers. 获取 public API 是否暴露 stream reader 指针。</summary>
    public bool StreamReaderPointerExposed => false;

    /// <summary>Gets whether public APIs expose stream writer pointers. 获取 public API 是否暴露 stream writer 指针。</summary>
    public bool StreamWriterPointerExposed => false;

    /// <summary>Gets whether public APIs can invoke stream callbacks. 获取 public API 是否可调用 stream callback。</summary>
    public bool StreamCallbackInvocationEnabled => false;

    /// <summary>Gets whether public APIs expose stream buffers. 获取 public API 是否暴露 stream buffer。</summary>
    public bool StreamBufferExposed => false;

    /// <summary>Gets whether direct stream callback rows intentionally remain deferred. 获取 direct stream callback 行是否继续 deferred。</summary>
    public bool DirectStreamCallbackRowsDeferred => true;

    /// <summary>Gets whether the managed-owned stream owner ledger is present. 获取 managed-owned stream owner 台账是否存在。</summary>
    public bool ManagedOwnedStreamOwnerLedgerReady => true;

    /// <summary>Gets whether a managed SafeHandle/GCHandle stream owner lifetime is implemented. 获取托管 SafeHandle/GCHandle stream owner 生命周期是否已实现。</summary>
    public bool ManagedOwnerLifetimeReady => false;

    /// <summary>Gets whether native stream owner create/destroy symmetry is implemented. 获取 native stream owner 创建/销毁对称性是否已实现。</summary>
    public bool NativeOwnerCreateDestroySymmetric => false;

    /// <summary>Gets whether a no-throw native stream callback vtable is implemented. 获取 no-throw native stream callback vtable 是否已实现。</summary>
    public bool NoThrowVTableReady => false;

    /// <summary>Gets whether callback exception-to-status mapping is implemented. 获取 callback exception 到 status/diagnostic 映射是否已实现。</summary>
    public bool ExceptionToStatusMappingReady => false;

    /// <summary>Gets whether detach-before-release ordering is implemented. 获取 detach-before-release 顺序是否已实现。</summary>
    public bool DetachBeforeReleaseReady => false;

    /// <summary>Gets whether stream getInterfaceInfo/getAPILanguage can be promoted now. 获取 stream getInterfaceInfo/getAPILanguage 当前是否可提升。</summary>
    public bool CanImplementStreamMetadataNow =>
        ManagedOwnerLifetimeReady &&
        NativeOwnerCreateDestroySymmetric &&
        NoThrowVTableReady &&
        ExceptionToStatusMappingReady &&
        DetachBeforeReleaseReady;

    /// <summary>Gets whether stream read/seek/write callback bridge is ready. 获取 stream read/seek/write callback bridge 是否就绪。</summary>
    public bool StreamCallbackBridgeReady => false;

    /// <summary>Gets whether copied stream metadata shape is ready. 获取 copied stream metadata 形态是否就绪。</summary>
    public bool CopiedMetadataShapeReady => CopiedInterfaceInfoMetadataReady;

    /// <summary>Gets whether the public surface remains pointer-free. 获取 public surface 是否保持无裸指针。</summary>
    public bool PointerFreeSurfaceReady =>
        !StreamReaderPointerExposed &&
        !StreamWriterPointerExposed &&
        !StreamCallbackInvocationEnabled &&
        !StreamBufferExposed;

    /// <summary>Gets whether this design gate is ready as non-proof evidence. 获取该 design gate 作为非 proof 证据是否就绪。</summary>
    public bool DesignGateReady =>
        LineSupportsStreamIo &&
        CopiedMetadataShapeReady &&
        PointerFreeSurfaceReady;

    /// <summary>Gets whether this gate can be promoted without runtime proof. 获取该结果是否可在无 runtime proof 情况下晋级。</summary>
    public bool CanPromoteWithoutRuntimeProof => false;

    /// <summary>Gets whether this result can be promoted as runtime proof. 获取该结果是否可晋级为 runtime proof。</summary>
    public bool CanPromoteRuntimeProof => false;

    /// <summary>Gets whether runtime proof remains blocked. 获取 runtime proof 是否仍被阻塞。</summary>
    public bool RuntimeProofBlocked => !CanPromoteRuntimeProof;

    /// <summary>Gets whether direct deferred rows are still required. 获取 direct deferred 行是否仍必须保留。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets the candidate TensorRT interfaces covered by this design gate. 获取该设计门覆盖的候选 TensorRT 接口。</summary>
    public ReadOnlyCollection<string> CandidateInterfaces =>
        Array.AsReadOnly(new[]
        {
            "IStreamReader",
            "IStreamReaderV2",
            "IStreamWriter"
        });

    /// <summary>Gets the direct deferred methods covered by this design gate. 获取该设计门覆盖的 direct deferred 方法。</summary>
    public ReadOnlyCollection<string> CandidateMethods =>
        Array.AsReadOnly(new[]
        {
            "IStreamReader::getInterfaceInfo",
            "IStreamReaderV2::getInterfaceInfo",
            "IStreamWriter::getInterfaceInfo"
        });

    /// <summary>Gets direct callback methods that remain deferred. 获取仍保持 deferred 的 direct callback 方法。</summary>
    public ReadOnlyCollection<string> DirectCallbackMethods =>
        Array.AsReadOnly(new[]
        {
            "IStreamReader::read",
            "IStreamReaderV2::read",
            "IStreamReaderV2::seek",
            "IStreamWriter::write"
        });

    /// <summary>Gets versioned-interface language metadata candidates for future owner-scoped implementation. 获取未来 owner-scoped 实现可考虑的 versioned-interface language metadata 候选。</summary>
    public ReadOnlyCollection<string> ApiLanguageCandidateMethods =>
        Array.AsReadOnly(new[]
        {
            "IStreamReader::getAPILanguage",
            "IStreamReaderV2::getAPILanguage",
            "IStreamWriter::getAPILanguage",
            "IVersionedInterface::getAPILanguage"
        });

    /// <summary>Gets candidate owner handle shapes for a future managed stream bridge. 获取未来托管 stream bridge 的 owner handle 候选形态。</summary>
    public ReadOnlyCollection<string> OwnerHandleCandidates =>
        Array.AsReadOnly(new[]
        {
            "SafeHandle-derived TensorRtStreamReaderOwnerHandle",
            "SafeHandle-derived TensorRtStreamWriterOwnerHandle",
            "GCHandle-backed managed stream owner state",
            "native noncopyable stream owner storage"
        });

    /// <summary>Gets lifecycle requirements that must pass before metadata or callback promotion. 获取 metadata 或 callback 提升前必须满足的生命周期要求。</summary>
    public ReadOnlyCollection<string> OwnerLifecycleRequirements =>
        Array.AsReadOnly(new[]
        {
            "pin managed owner state before native attach and free it only after detach",
            "keep delegate targets alive for the full native owner lifetime",
            "create and destroy native stream owners symmetrically",
            "detach stream owner from TensorRT before release",
            "never expose borrowed IStreamReader/IStreamReaderV2/IStreamWriter pointers"
        });

    /// <summary>Gets callback safety requirements that must pass before read/seek/write bridge promotion. 获取 read/seek/write bridge 提升前必须满足的 callback 安全要求。</summary>
    public ReadOnlyCollection<string> CallbackSafetyRequirements =>
        Array.AsReadOnly(new[]
        {
            "native vtable callbacks must be noexcept",
            "managed exceptions must be captured and mapped to status diagnostics",
            "read/write buffers must be caller-owned and never retained",
            "seek state must remain owner-scoped and thread-safety documented",
            "callback reentrancy must be blocked or explicitly serialized"
        });

    /// <summary>Gets all currently tracked stream IO methods in this owner ledger. 获取该 owner 台账当前追踪的所有 stream IO 方法。</summary>
    public ReadOnlyCollection<string> OwnerLedgerTrackedMethods
    {
        get
        {
            List<string> methods = new List<string>();
            methods.AddRange(CandidateMethods);
            methods.AddRange(DirectCallbackMethods);
            methods.AddRange(ApiLanguageCandidateMethods);
            return Array.AsReadOnly(methods.ToArray());
        }
    }

    /// <summary>Gets the copied output mode required before implementation. 获取实现前要求的 copied 输出模式。</summary>
    public string RequiredOutputMode => "copied stream reader/writer interface metadata snapshot";

    /// <summary>Gets the next safe implementation step. 获取下一步安全实现动作。</summary>
    public string NextSafeImplementationStep =>
        "Design a stream owner that copies interface metadata without exposing reader/writer handles or invoking read/write callbacks.";

    /// <summary>Gets the direct candidate method count. 获取 direct 候选方法数量。</summary>
    public int CandidateMethodCount => CandidateMethods.Count;

    /// <summary>Gets the direct callback method count. 获取 direct callback 方法数量。</summary>
    public int DirectCallbackMethodCount => DirectCallbackMethods.Count;

    /// <summary>Gets the API language candidate method count. 获取 API language 候选方法数量。</summary>
    public int ApiLanguageCandidateMethodCount => ApiLanguageCandidateMethods.Count;

    /// <summary>Gets the owner ledger tracked method count. 获取 owner 台账追踪方法数量。</summary>
    public int OwnerLedgerTrackedMethodCount => OwnerLedgerTrackedMethods.Count;

    /// <summary>Gets copied blocked prerequisites. 获取已复制的阻塞前置项。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 获取已复制阻塞前置项数量。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the design gate status. 获取设计门状态。</summary>
    public string Status => DesignGateReady ? "design-gate-ready" : "design-gate-blocked";

    /// <summary>Gets a compact diagnostic summary. 获取简短诊断摘要。</summary>
    public string Diagnostic =>
        "stream-io-interface-info-design-gate; RuntimeEvidenceKind=design-gate; " +
        "IsRuntimeExecutionEvidence=False; IsRuntimeExecutionProof=False; " +
        "CopiedMetadataShapeReady=" + CopiedMetadataShapeReady + "; " +
        "PointerFreeSurfaceReady=" + PointerFreeSurfaceReady + "; " +
        "StreamReaderPointerExposed=False; StreamWriterPointerExposed=False; " +
        "StreamCallbackInvocationEnabled=False; StreamBufferExposed=False; " +
        "DirectStreamCallbackRowsDeferred=True; RequiredOutputMode=" + RequiredOutputMode + "; " +
        "CandidateMethodCount=" + CandidateMethodCount + "; DirectCallbackMethodCount=" + DirectCallbackMethodCount + "; " +
        "ApiLanguageCandidateMethodCount=" + ApiLanguageCandidateMethodCount + "; OwnerLedgerTrackedMethodCount=" + OwnerLedgerTrackedMethodCount + "; " +
        "ManagedOwnedStreamOwnerLedgerReady=" + ManagedOwnedStreamOwnerLedgerReady + "; ManagedOwnerLifetimeReady=False; " +
        "NativeOwnerCreateDestroySymmetric=False; NoThrowVTableReady=False; ExceptionToStatusMappingReady=False; " +
        "DetachBeforeReleaseReady=False; StreamCallbackBridgeReady=False; CanImplementStreamMetadataNow=False; " +
        "CanPromoteWithoutRuntimeProof=False; " +
        "RuntimeProofBlocked=True; DeferredRowsStillRequired=True; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 返回简短诊断表示。</summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:pointerFree={PointerFreeSurfaceReady}:proof={IsRuntimeExecutionProof}";
    }
}

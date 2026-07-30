using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports runtime deserialization boundary status without exposing native engine pointers.
/// 报告 runtime deserialization 边界状态，不暴露原生 engine 指针。
/// </summary>
/// <remarks>
/// This result is a runtime precheck. It keeps direct <c>IRuntime::deserializeCudaEngineV2</c>
/// and <c>IRuntime::loadRuntime</c> rows deferred until dependency diagnostics, ownership, and
/// full package consumer runtime proof exist.
/// 该结果是 runtime precheck；在 dependency diagnostics、ownership 和完整 package consumer runtime proof
/// 出现前，direct <c>IRuntime::deserializeCudaEngineV2</c> 与 <c>IRuntime::loadRuntime</c> 行继续 deferred。
/// </remarks>
public readonly struct TensorRtRuntimeDeserializationBoundaryPrecheckResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtRuntimeDeserializationBoundaryPrecheckResult(
        TensorRtApiLine line,
        bool lineSupportsRuntimeDeserialization,
        bool lineSupportsDeserializeCudaEngineV2,
        bool managedByteArrayDeserializeReady,
        bool managedArraySegmentDeserializeReady,
        bool managedReadOnlySpanDeserializeReady,
        bool managedStreamDeserializeReady,
        bool managedFileDeserializeReady,
        bool hostMemoryDeserializeReady,
        bool serializedBufferCopiedBeforeInterop,
        bool pinnedBufferScopedToInteropCall,
        bool hostMemoryHandleOwnedByWrapper,
        bool engineHandleOwnedByWrapper,
        bool pluginLibraryDependencyDiagnosticsReady,
        bool fullPackageConsumerRuntimeEvidenceReady,
        string[] blockedPrerequisites)
    {
        Line = line;
        LineSupportsRuntimeDeserialization = lineSupportsRuntimeDeserialization;
        LineSupportsDeserializeCudaEngineV2 = lineSupportsDeserializeCudaEngineV2;
        ManagedByteArrayDeserializeReady = managedByteArrayDeserializeReady;
        ManagedArraySegmentDeserializeReady = managedArraySegmentDeserializeReady;
        ManagedReadOnlySpanDeserializeReady = managedReadOnlySpanDeserializeReady;
        ManagedStreamDeserializeReady = managedStreamDeserializeReady;
        ManagedFileDeserializeReady = managedFileDeserializeReady;
        HostMemoryDeserializeReady = hostMemoryDeserializeReady;
        SerializedBufferCopiedBeforeInterop = serializedBufferCopiedBeforeInterop;
        PinnedBufferScopedToInteropCall = pinnedBufferScopedToInteropCall;
        HostMemoryHandleOwnedByWrapper = hostMemoryHandleOwnedByWrapper;
        EngineHandleOwnedByWrapper = engineHandleOwnedByWrapper;
        PluginLibraryDependencyDiagnosticsReady = pluginLibraryDependencyDiagnosticsReady;
        FullPackageConsumerRuntimeEvidenceReady = fullPackageConsumerRuntimeEvidenceReady;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this precheck. 获取该预检的证据标记。</summary>
    public string EvidenceKind => "runtime-deserialization-boundary-precheck";

    /// <summary>Gets the diagnostics kind represented by this precheck. 获取该预检代表的诊断类型。</summary>
    public string DiagnosticsKind => "runtime-deserialization-boundary";

    /// <summary>Gets the runtime evidence kind. 获取 runtime evidence 类型。</summary>
    public string RuntimeEvidenceKind => "runtime-precheck";

    /// <summary>Gets whether this precheck is runtime execution evidence. 获取该预检是否为 runtime execution evidence。</summary>
    public bool IsRuntimeExecutionEvidence => false;

    /// <summary>Gets whether this precheck may be promoted as runtime execution proof. 获取该预检是否可晋级为 runtime execution proof。</summary>
    public bool IsRuntimeExecutionProof => false;

    /// <summary>Gets the TensorRT API line. 获取 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets whether the selected TensorRT line supports the runtime deserialization boundary. 获取当前 TensorRT line 是否支持 runtime deserialization 边界。</summary>
    public bool LineSupportsRuntimeDeserialization { get; }

    /// <summary>Gets whether the selected TensorRT line has deserializeCudaEngineV2 rows. 获取当前 TensorRT line 是否有 deserializeCudaEngineV2 行。</summary>
    public bool LineSupportsDeserializeCudaEngineV2 { get; }

    /// <summary>Gets whether managed byte-array deserialization is available. 获取 managed byte[] 反序列化是否可用。</summary>
    public bool ManagedByteArrayDeserializeReady { get; }

    /// <summary>Gets whether managed ArraySegment deserialization copies into an exact array. 获取 managed ArraySegment 反序列化是否复制为精确数组。</summary>
    public bool ManagedArraySegmentDeserializeReady { get; }

    /// <summary>Gets whether managed ReadOnlySpan deserialization copies into managed memory. 获取 managed ReadOnlySpan 反序列化是否复制到托管内存。</summary>
    public bool ManagedReadOnlySpanDeserializeReady { get; }

    /// <summary>Gets whether managed stream deserialization copies into managed memory. 获取 managed stream 反序列化是否复制到托管内存。</summary>
    public bool ManagedStreamDeserializeReady { get; }

    /// <summary>Gets whether managed file deserialization uses managed bytes. 获取 managed file 反序列化是否使用托管字节。</summary>
    public bool ManagedFileDeserializeReady { get; }

    /// <summary>Gets whether TensorRT host-memory deserialization is available. 获取 TensorRT host memory 反序列化是否可用。</summary>
    public bool HostMemoryDeserializeReady { get; }

    /// <summary>Gets whether caller-owned buffers are copied before interop when needed. 获取必要时是否在 interop 前复制调用方 buffer。</summary>
    public bool SerializedBufferCopiedBeforeInterop { get; }

    /// <summary>Gets whether pinned managed buffers are scoped to the native call. 获取 pinned 托管 buffer 是否只在 native 调用期间有效。</summary>
    public bool PinnedBufferScopedToInteropCall { get; }

    /// <summary>Gets whether a caller-owned serialized buffer can escape this public surface. 获取调用方 serialized buffer 是否可能逃逸 public surface。</summary>
    public bool BorrowedSerializedBufferEscaped => false;

    /// <summary>Gets whether host-memory handles are owned by managed wrappers. 获取 host memory handle 是否由 managed wrapper 管理。</summary>
    public bool HostMemoryHandleOwnedByWrapper { get; }

    /// <summary>Gets whether returned engine handles are owned by managed wrappers. 获取返回 engine handle 是否由 managed wrapper 管理。</summary>
    public bool EngineHandleOwnedByWrapper { get; }

    /// <summary>Gets whether this public surface exposes a native engine pointer. 获取 public surface 是否暴露 native engine 指针。</summary>
    public bool EnginePointerExposed => false;

    /// <summary>Gets whether this precheck produces a native engine pointer for callers. 获取该预检是否向调用方产生 native engine 指针。</summary>
    public bool EnginePointerProduced => false;

    /// <summary>Gets whether direct IRuntime::deserializeCudaEngine rows intentionally remain deferred. 获取 direct IRuntime::deserializeCudaEngine 行是否继续 deferred。</summary>
    public bool DirectDeserializeCudaEngineRowsDeferred => false;

    /// <summary>Gets whether direct IRuntime::deserializeCudaEngine is covered by the scoped-buffer bridge. 获取 direct IRuntime::deserializeCudaEngine 是否已由调用期 buffer bridge 覆盖。</summary>
    public bool DirectDeserializeCudaEngineRowsImplemented => SafeDeserializeBridgeReady;

    /// <summary>Gets whether direct IRuntime::deserializeCudaEngineV2 rows intentionally remain deferred where present. 获取存在时 direct IRuntime::deserializeCudaEngineV2 行是否继续 deferred。</summary>
    public bool DirectDeserializeCudaEngineV2RowsDeferred => LineSupportsDeserializeCudaEngineV2;

    /// <summary>Gets whether IRuntime::loadRuntime intentionally remains deferred. 获取 IRuntime::loadRuntime 是否继续 deferred。</summary>
    public bool LoadRuntimeDeferred => true;

    /// <summary>Gets whether plugin/library dependency diagnostics are complete enough for promotion. 获取 plugin/library dependency 诊断是否足以晋级。</summary>
    public bool PluginLibraryDependencyDiagnosticsReady { get; }

    /// <summary>Gets whether the public surface remains pointer-free. 获取 public surface 是否保持无裸指针。</summary>
    public bool PointerFreeSurfaceReady =>
        !BorrowedSerializedBufferEscaped &&
        !EnginePointerExposed &&
        !EnginePointerProduced;

    /// <summary>Gets whether managed deserialization overloads are ready. 获取 managed deserialization overload 是否就绪。</summary>
    public bool ManagedDeserializeSurfaceReady =>
        ManagedByteArrayDeserializeReady &&
        ManagedArraySegmentDeserializeReady &&
        ManagedReadOnlySpanDeserializeReady &&
        ManagedStreamDeserializeReady &&
        ManagedFileDeserializeReady;

    /// <summary>Gets whether the bridge owns returned handles and does not leak caller buffers. 获取 bridge 是否管理返回 handle 且不泄漏调用方 buffer。</summary>
    public bool SafeDeserializeBridgeReady =>
        LineSupportsRuntimeDeserialization &&
        ManagedDeserializeSurfaceReady &&
        HostMemoryDeserializeReady &&
        SerializedBufferCopiedBeforeInterop &&
        PinnedBufferScopedToInteropCall &&
        HostMemoryHandleOwnedByWrapper &&
        EngineHandleOwnedByWrapper &&
        PointerFreeSurfaceReady;

    /// <summary>Gets whether this precheck has enough source evidence to be considered ready. 获取该 precheck 的 source evidence 是否就绪。</summary>
    public bool PrecheckReady => SafeDeserializeBridgeReady;

    /// <summary>Gets whether this precheck can attempt runtime proof. 获取该预检是否可尝试 runtime proof。</summary>
    public bool CanAttemptRuntimeProof => false;

    /// <summary>Gets whether this precheck can be promoted without runtime proof. 获取该预检是否可在无 runtime proof 情况下晋级。</summary>
    public bool CanPromoteWithoutRuntimeProof => false;

    /// <summary>Gets whether a full package consumer runtime proof is ready. 获取完整 package consumer runtime proof 是否就绪。</summary>
    public bool FullPackageConsumerRuntimeEvidenceReady { get; }

    /// <summary>Gets whether this result can be promoted as runtime proof. 获取该结果是否可晋级为 runtime proof。</summary>
    public bool CanPromoteRuntimeProof => false;

    /// <summary>Gets whether runtime proof remains blocked. 获取 runtime proof 是否仍被阻塞。</summary>
    public bool RuntimeProofBlocked => !CanPromoteRuntimeProof;

    /// <summary>Gets whether direct deferred rows are still required. 获取 direct deferred 行是否仍必须保留。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets copied blocked prerequisites. 获取已复制的阻塞前置项。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied blocked prerequisite count. 获取已复制阻塞前置项数量。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the precheck status. 获取预检状态。</summary>
    public string Status => PrecheckReady ? "precheck-ready" : "precheck-blocked";

    /// <summary>Gets a compact diagnostic summary. 获取简短诊断摘要。</summary>
    public string Diagnostic =>
        "runtime-deserialization-boundary-precheck; RuntimeEvidenceKind=runtime-precheck; " +
        "IsRuntimeExecutionEvidence=False; IsRuntimeExecutionProof=False; " +
        "ManagedByteArrayDeserializeReady=" + ManagedByteArrayDeserializeReady + "; " +
        "ManagedStreamDeserializeReady=" + ManagedStreamDeserializeReady + "; " +
        "HostMemoryDeserializeReady=" + HostMemoryDeserializeReady + "; " +
        "SerializedBufferCopiedBeforeInterop=" + SerializedBufferCopiedBeforeInterop + "; " +
        "PinnedBufferScopedToInteropCall=" + PinnedBufferScopedToInteropCall + "; " +
        "BorrowedSerializedBufferEscaped=False; " +
        "EngineHandleOwnedByWrapper=" + EngineHandleOwnedByWrapper + "; " +
        "EnginePointerExposed=False; " +
        "DirectDeserializeCudaEngineRowsDeferred=False; " +
        "DirectDeserializeCudaEngineRowsImplemented=" + DirectDeserializeCudaEngineRowsImplemented + "; " +
        "DirectDeserializeCudaEngineV2RowsDeferred=" + DirectDeserializeCudaEngineV2RowsDeferred + "; " +
        "LoadRuntimeDeferred=True; CanAttemptRuntimeProof=False; " +
        "CanPromoteWithoutRuntimeProof=False; RuntimeProofBlocked=True; " +
        "DeferredRowsStillRequired=True; BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 返回简短诊断表示。</summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:safe={SafeDeserializeBridgeReady}:proof={IsRuntimeExecutionProof}";
    }
}

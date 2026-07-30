using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Evaluates the pointer-free TensorRT runtime deserialization boundary precheck.
/// 评估 TensorRT runtime deserialization 边界的无裸指针预检。
/// </summary>
/// <remarks>
/// This precheck describes the managed safe deserialization surface. It does not expose an
/// <c>ICudaEngine*</c>, does not call <c>IRuntime::loadRuntime</c>, and is not runtime execution proof.
/// 该预检描述托管安全反序列化边界；不会暴露 <c>ICudaEngine*</c>，不会调用
/// <c>IRuntime::loadRuntime</c>，也不是 runtime execution proof。
/// </remarks>
public static class TensorRtRuntimeDeserializationBoundaryPrecheck
{
    /// <summary>
    /// Evaluates the known public runtime deserialization surface for a TensorRT API line.
    /// 基于已知 public runtime deserialization 边界评估指定 TensorRT API line。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <returns>A pointer-free runtime deserialization precheck result. 无裸指针 runtime deserialization 预检结果。</returns>
    public static TensorRtRuntimeDeserializationBoundaryPrecheckResult EvaluateKnownSurface(TensorRtApiLine line)
    {
        return Evaluate(
            line,
            managedByteArrayDeserializeReady: true,
            managedArraySegmentDeserializeReady: true,
            managedReadOnlySpanDeserializeReady: true,
            managedStreamDeserializeReady: true,
            managedFileDeserializeReady: true,
            hostMemoryDeserializeReady: true,
            serializedBufferCopiedBeforeInterop: true,
            pinnedBufferScopedToInteropCall: true,
            hostMemoryHandleOwnedByWrapper: true,
            engineHandleOwnedByWrapper: true,
            pluginLibraryDependencyDiagnosticsReady: false,
            fullPackageConsumerRuntimeEvidenceReady: false);
    }

    /// <summary>
    /// Evaluates the runtime deserialization boundary from explicit capability flags.
    /// 根据显式能力标记评估 runtime deserialization 边界。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <param name="managedByteArrayDeserializeReady">Whether byte-array deserialization is available. byte[] 反序列化是否可用。</param>
    /// <param name="managedArraySegmentDeserializeReady">Whether ArraySegment deserialization copies into an exact array. ArraySegment 反序列化是否复制为精确数组。</param>
    /// <param name="managedReadOnlySpanDeserializeReady">Whether ReadOnlySpan deserialization copies into managed memory. ReadOnlySpan 反序列化是否复制到托管内存。</param>
    /// <param name="managedStreamDeserializeReady">Whether stream deserialization copies into managed memory. Stream 反序列化是否复制到托管内存。</param>
    /// <param name="managedFileDeserializeReady">Whether file deserialization uses managed bytes. 文件反序列化是否使用托管字节。</param>
    /// <param name="hostMemoryDeserializeReady">Whether TensorRT host-memory deserialization is available. TensorRT host memory 反序列化是否可用。</param>
    /// <param name="serializedBufferCopiedBeforeInterop">Whether caller-owned buffers are copied before interop when needed. 必要时是否在 interop 前复制调用方 buffer。</param>
    /// <param name="pinnedBufferScopedToInteropCall">Whether pinned managed buffers are scoped to the native call. pinned 托管 buffer 是否只在 native 调用期间有效。</param>
    /// <param name="hostMemoryHandleOwnedByWrapper">Whether host-memory handles are owned by wrappers. host memory handle 是否由 wrapper 管理。</param>
    /// <param name="engineHandleOwnedByWrapper">Whether returned engine handles are owned by wrappers. 返回 engine handle 是否由 wrapper 管理。</param>
    /// <param name="pluginLibraryDependencyDiagnosticsReady">Whether plugin/library dependency diagnostics are sufficient for promotion. plugin/library dependency 诊断是否足以晋级。</param>
    /// <param name="fullPackageConsumerRuntimeEvidenceReady">Whether full package consumer runtime proof is ready. 完整 package consumer runtime proof 是否就绪。</param>
    /// <returns>A pointer-free runtime deserialization precheck result. 无裸指针 runtime deserialization 预检结果。</returns>
    public static TensorRtRuntimeDeserializationBoundaryPrecheckResult Evaluate(
        TensorRtApiLine line,
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
        bool fullPackageConsumerRuntimeEvidenceReady)
    {
        bool lineSupportsRuntimeDeserialization =
            line == TensorRtApiLine.TensorRt8 ||
            line == TensorRtApiLine.TensorRt10 ||
            line == TensorRtApiLine.TensorRt11;
        bool lineSupportsDeserializeCudaEngineV2 =
            line == TensorRtApiLine.TensorRt10 ||
            line == TensorRtApiLine.TensorRt11;

        List<string> blockers = new List<string>();
        if (!lineSupportsRuntimeDeserialization)
        {
            blockers.Add("TensorRT 8, 10, or 11 runtime deserialization line support has not been selected.");
        }

        if (!managedByteArrayDeserializeReady)
        {
            blockers.Add("managed byte-array deserialization is not ready.");
        }

        if (!managedArraySegmentDeserializeReady)
        {
            blockers.Add("managed ArraySegment deserialization copy policy is not ready.");
        }

        if (!managedReadOnlySpanDeserializeReady)
        {
            blockers.Add("managed ReadOnlySpan deserialization copy policy is not ready.");
        }

        if (!managedStreamDeserializeReady)
        {
            blockers.Add("managed stream deserialization copy policy is not ready.");
        }

        if (!managedFileDeserializeReady)
        {
            blockers.Add("managed file deserialization entry point is not ready.");
        }

        if (!hostMemoryDeserializeReady)
        {
            blockers.Add("TensorRT host-memory deserialization is not ready.");
        }

        if (!serializedBufferCopiedBeforeInterop)
        {
            blockers.Add("caller-owned serialized buffers are not copied before interop.");
        }

        if (!pinnedBufferScopedToInteropCall)
        {
            blockers.Add("pinned managed serialized buffers are not scoped to the native call.");
        }

        if (!hostMemoryHandleOwnedByWrapper)
        {
            blockers.Add("host-memory handle ownership is not modeled by managed wrappers.");
        }

        if (!engineHandleOwnedByWrapper)
        {
            blockers.Add("deserialized engine handle ownership is not modeled by managed wrappers.");
        }

        if (!pluginLibraryDependencyDiagnosticsReady)
        {
            blockers.Add("plugin/library dependency diagnostics are not complete enough to promote runtime proof.");
        }

        if (!fullPackageConsumerRuntimeEvidenceReady)
        {
            blockers.Add("full package consumer runtime execution proof is not ready.");
        }

        if (lineSupportsDeserializeCudaEngineV2)
        {
            blockers.Add("direct IRuntime::deserializeCudaEngineV2 rows remain deferred by design.");
        }

        blockers.Add("IRuntime::loadRuntime plugin-host-code and returned runtime lifetime remain deferred by design.");

        return new TensorRtRuntimeDeserializationBoundaryPrecheckResult(
            line,
            lineSupportsRuntimeDeserialization,
            lineSupportsDeserializeCudaEngineV2,
            managedByteArrayDeserializeReady,
            managedArraySegmentDeserializeReady,
            managedReadOnlySpanDeserializeReady,
            managedStreamDeserializeReady,
            managedFileDeserializeReady,
            hostMemoryDeserializeReady,
            serializedBufferCopiedBeforeInterop,
            pinnedBufferScopedToInteropCall,
            hostMemoryHandleOwnedByWrapper,
            engineHandleOwnedByWrapper,
            pluginLibraryDependencyDiagnosticsReady,
            fullPackageConsumerRuntimeEvidenceReady,
            blockers.ToArray());
    }
}

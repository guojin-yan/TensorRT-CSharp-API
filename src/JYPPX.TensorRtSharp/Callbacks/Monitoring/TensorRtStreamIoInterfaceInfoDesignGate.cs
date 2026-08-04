using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.TensorRtSharp.Shared.Interop;

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

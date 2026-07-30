using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Reports copied OutputAllocator attach/detach design diagnostics without exposing native pointers.
/// 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。
/// </summary>
/// <remarks>
/// This result is a design gate. <see cref="RealCallbackRuntime"/> and
/// <see cref="IsRealCallbackRuntimeProof"/> remain <see langword="false"/> until full package consumer smoke produces
/// complete <c>real-callback-runtime</c> evidence.
/// 中文：这些诊断仍是设计门禁或脚手架证据，不能替代真实 OutputAllocator 回调执行证据。
/// </remarks>
public readonly struct TensorRtOutputAllocatorAttachDetachDesignGateResult
{
    private readonly string[] _blockedPrerequisites;

    internal TensorRtOutputAllocatorAttachDetachDesignGateResult(
        TensorRtApiLine line,
        bool lineSupportsOutputAllocator,
        bool ownerDesignReady,
        bool managedOwnerStateMachineReady,
        bool pointerFreeSurfaceReady,
        bool detachClearControlAvailable,
        bool attachControlAvailable,
        bool stableNativeOwnerAddressReady,
        bool noThrowNativeVTableReady,
        bool outputBufferOwnershipRuntimeReady,
        string[] blockedPrerequisites)
    {
        Line = line;
        LineSupportsOutputAllocator = lineSupportsOutputAllocator;
        OwnerDesignReady = ownerDesignReady;
        ManagedOwnerStateMachineReady = managedOwnerStateMachineReady;
        PointerFreeSurfaceReady = pointerFreeSurfaceReady;
        DetachClearControlAvailable = detachClearControlAvailable;
        AttachControlAvailable = attachControlAvailable;
        StableNativeOwnerAddressReady = stableNativeOwnerAddressReady;
        NoThrowNativeVTableReady = noThrowNativeVTableReady;
        OutputBufferOwnershipRuntimeReady = outputBufferOwnershipRuntimeReady;
        _blockedPrerequisites = blockedPrerequisites == null ? Array.Empty<string>() : (string[])blockedPrerequisites.Clone();
    }

    /// <summary>Gets the marker used by readiness to identify this gate. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string EvidenceKind => "output-allocator-attach-detach-design-gate";

    /// <summary>Gets the callback kind represented by this gate. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string CallbackKind => "output-allocator-attach-detach";

    /// <summary>Gets the runtime evidence kind. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string RuntimeEvidenceKind => "design-gate";

    /// <summary>Gets whether this gate proves a real TensorRT callback runtime. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this gate as real callback runtime proof. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the TensorRT API line represented by the copied owner design snapshot. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets whether the selected TensorRT line has OutputAllocator API support. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool LineSupportsOutputAllocator { get; }

    /// <summary>Gets whether the owner design snapshot was clean owner-design evidence. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool OwnerDesignReady { get; }

    /// <summary>Gets whether the managed owner state machine has clean dispose and in-flight drain evidence. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool ManagedOwnerStateMachineReady { get; }

    /// <summary>Gets whether the public surface remains pointer-free. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool PointerFreeSurfaceReady { get; }

    /// <summary>Gets whether TensorRT 8/10/11 detach/clear control exists through <c>setOutputAllocator(nullptr)</c>. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool DetachClearControlAvailable { get; }

    /// <summary>Gets whether a non-null OutputAllocator attach bridge is available. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool AttachControlAvailable { get; }

    /// <summary>Gets whether the line-specific attach/detach pair is ready. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool LineSpecificAttachDetachReady => AttachControlAvailable && DetachClearControlAvailable;

    /// <summary>Gets whether a stable native owner address is ready. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool StableNativeOwnerAddressReady { get; }

    /// <summary>Gets whether a no-throw native vtable trampoline is ready. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NoThrowNativeVTableReady { get; }

    /// <summary>Gets whether native vtable readiness is complete. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool NativeVTableReady => StableNativeOwnerAddressReady && NoThrowNativeVTableReady;

    /// <summary>Gets whether output buffer ownership and current-memory reuse rules are ready. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool OutputBufferOwnershipRuntimeReady { get; }

    /// <summary>Gets whether this design gate has enough copied evidence to feed the runtime-proof precheck. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool DesignGateReady =>
        LineSupportsOutputAllocator &&
        OwnerDesignReady &&
        ManagedOwnerStateMachineReady &&
        PointerFreeSurfaceReady &&
        DetachClearControlAvailable;

    /// <summary>Gets whether all prerequisites are satisfied to attempt a real runtime proof. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool CanAttemptRuntimeProof =>
        DesignGateReady &&
        LineSpecificAttachDetachReady &&
        NativeVTableReady &&
        OutputBufferOwnershipRuntimeReady;

    /// <summary>Gets whether real runtime proof is still blocked. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool RuntimeProofBlocked => !CanAttemptRuntimeProof;

    /// <summary>Gets whether direct deferred callback rows must remain deferred. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public bool DeferredRowsStillRequired => true;

    /// <summary>Gets the copied list of prerequisites that still block real runtime proof. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public ReadOnlyCollection<string> BlockedPrerequisites =>
        Array.AsReadOnly(_blockedPrerequisites ?? Array.Empty<string>());

    /// <summary>Gets the copied number of blocked prerequisites. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public int BlockedPrerequisiteCount => (_blockedPrerequisites ?? Array.Empty<string>()).Length;

    /// <summary>Gets the design gate status. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string Status => DesignGateReady ? "design-gate-ready" : "design-gate-blocked";

    /// <summary>Gets a copied diagnostic summary. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    public string Diagnostic =>
        "output-allocator-attach-detach-design-gate; RuntimeEvidenceKind=design-gate; RealCallbackRuntime=False; " +
        "IsRealCallbackRuntimeProof=False; AttachControlAvailable=" + AttachControlAvailable + "; " +
        "DetachClearControlAvailable=" + DetachClearControlAvailable + "; " +
        "LineSpecificAttachDetachReady=" + LineSpecificAttachDetachReady + "; " +
        "StableNativeOwnerAddressReady=" + StableNativeOwnerAddressReady + "; " +
        "NoThrowNativeVTableReady=" + NoThrowNativeVTableReady + "; " +
        "OutputBufferOwnershipRuntimeReady=" + OutputBufferOwnershipRuntimeReady + "; " +
        "BlockedPrerequisiteCount=" + BlockedPrerequisiteCount + ".";

    /// <summary>Returns a compact diagnostic representation. 中文：该成员报告 OutputAllocator 无裸指针边界状态，不代表真实 TensorRT 回调运行证明。</summary>
    /// <returns>A diagnostic string. 中文：返回复制后的 OutputAllocator 无裸指针诊断结果。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:line={(int)Line}:status={Status}:attach={AttachControlAvailable}:detach={DetachClearControlAvailable}:proof={IsRealCallbackRuntimeProof}";
    }
}

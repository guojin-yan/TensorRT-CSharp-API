using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

public static partial class TensorRtDebugListenerRuntimeProofPrecheck
{
    /// <summary>
    /// Evaluates the current DebugListener runtime proof precheck from copied owner, gate, and native attach/no-throw preflight evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="attachDetachDesignGate">The copied attach/detach design gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="borrowedTensorSafetyGate">The copied borrowed tensor safety gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="attachVTableSafetyGate">The copied attach/vtable safety gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeAttachNoThrowPreflight">The copied native attach/no-throw preflight result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <returns>A pointer-free precheck result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerRuntimeProofPrecheckResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachDesignGate,
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorSafetyGate,
        TensorRtDebugListenerAttachVTableSafetyGateResult attachVTableSafetyGate,
        TensorRtDebugListenerNativeAttachNoThrowPreflightResult nativeAttachNoThrowPreflight)
    {
        TensorRtDebugListenerNativeOwnerAddressDesignGateResult nativeOwnerAddressDesignGate =
            TensorRtDebugListenerNativeOwnerAddressDesignGate.Evaluate(
                ownerDesignSnapshot,
                attachDetachDesignGate,
                borrowedTensorSafetyGate,
                attachVTableSafetyGate,
                nativeAttachNoThrowPreflight);
        return Evaluate(
            ownerDesignSnapshot,
            attachDetachDesignGate,
            borrowedTensorSafetyGate,
            attachVTableSafetyGate,
            nativeAttachNoThrowPreflight,
            nativeOwnerAddressDesignGate);
    }

    /// <summary>
    /// Evaluates the current DebugListener runtime proof precheck from copied owner, gate, preflight, and native owner address design evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="attachDetachDesignGate">The copied attach/detach design gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="borrowedTensorSafetyGate">The copied borrowed tensor safety gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="attachVTableSafetyGate">The copied attach/vtable safety gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeAttachNoThrowPreflight">The copied native attach/no-throw preflight result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeOwnerAddressDesignGate">The copied native owner address design gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <returns>A pointer-free precheck result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerRuntimeProofPrecheckResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachDesignGate,
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorSafetyGate,
        TensorRtDebugListenerAttachVTableSafetyGateResult attachVTableSafetyGate,
        TensorRtDebugListenerNativeAttachNoThrowPreflightResult nativeAttachNoThrowPreflight,
        TensorRtDebugListenerNativeOwnerAddressDesignGateResult nativeOwnerAddressDesignGate)
    {
        TensorRtDebugListenerNativeNoThrowVTableDesignGateResult nativeNoThrowVTableDesignGate =
            TensorRtDebugListenerNativeNoThrowVTableDesignGate.Evaluate(
                ownerDesignSnapshot,
                attachDetachDesignGate,
                borrowedTensorSafetyGate,
                attachVTableSafetyGate,
                nativeAttachNoThrowPreflight,
                nativeOwnerAddressDesignGate);
        return Evaluate(
            ownerDesignSnapshot,
            attachDetachDesignGate,
            borrowedTensorSafetyGate,
            attachVTableSafetyGate,
            nativeAttachNoThrowPreflight,
            nativeOwnerAddressDesignGate,
            nativeNoThrowVTableDesignGate);
    }

    /// <summary>
    /// Evaluates the current DebugListener runtime proof precheck from copied owner, gate, preflight, owner address, and no-throw vtable design evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="attachDetachDesignGate">The copied attach/detach design gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="borrowedTensorSafetyGate">The copied borrowed tensor safety gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="attachVTableSafetyGate">The copied attach/vtable safety gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeAttachNoThrowPreflight">The copied native attach/no-throw preflight result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeOwnerAddressDesignGate">The copied native owner address design gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeNoThrowVTableDesignGate">The copied native no-throw vtable design gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <returns>A pointer-free precheck result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerRuntimeProofPrecheckResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachDesignGate,
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorSafetyGate,
        TensorRtDebugListenerAttachVTableSafetyGateResult attachVTableSafetyGate,
        TensorRtDebugListenerNativeAttachNoThrowPreflightResult nativeAttachNoThrowPreflight,
        TensorRtDebugListenerNativeOwnerAddressDesignGateResult nativeOwnerAddressDesignGate,
        TensorRtDebugListenerNativeNoThrowVTableDesignGateResult nativeNoThrowVTableDesignGate)
    {
        TensorRtDebugListenerNativeAttachEntryDesignGateResult nativeAttachEntryDesignGate =
            TensorRtDebugListenerNativeAttachEntryDesignGate.Evaluate(
                ownerDesignSnapshot,
                attachDetachDesignGate,
                borrowedTensorSafetyGate,
                attachVTableSafetyGate,
                nativeAttachNoThrowPreflight,
                nativeOwnerAddressDesignGate,
                nativeNoThrowVTableDesignGate);
        return Evaluate(
            ownerDesignSnapshot,
            attachDetachDesignGate,
            borrowedTensorSafetyGate,
            attachVTableSafetyGate,
            nativeAttachNoThrowPreflight,
            nativeOwnerAddressDesignGate,
            nativeNoThrowVTableDesignGate,
            nativeAttachEntryDesignGate);
    }

    /// <summary>
    /// Evaluates the current DebugListener runtime proof precheck from copied owner, gate, preflight, owner address,
    /// no-throw vtable, and attach entry design evidence.
    /// 该成员提供 DebugListener callback 安全边界的只读诊断信息；不能作为真实 TensorRT callback runtime proof。
    /// </summary>
    /// <param name="ownerDesignSnapshot">The copied DebugListener owner design snapshot. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="attachDetachDesignGate">The copied attach/detach design gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="borrowedTensorSafetyGate">The copied borrowed tensor safety gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="attachVTableSafetyGate">The copied attach/vtable safety gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeAttachNoThrowPreflight">The copied native attach/no-throw preflight result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeOwnerAddressDesignGate">The copied native owner address design gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeNoThrowVTableDesignGate">The copied native no-throw vtable design gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <param name="nativeAttachEntryDesignGate">The copied native attach entry design gate result. 该参数传入已复制的 DebugListener 安全门禁或 owner snapshot evidence。</param>
    /// <returns>A pointer-free precheck result. 返回不暴露 native 指针的 pointer-free 诊断结果。</returns>
    public static TensorRtDebugListenerRuntimeProofPrecheckResult Evaluate(
        TensorRtDebugListenerCallbackOwnerSnapshot ownerDesignSnapshot,
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachDesignGate,
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorSafetyGate,
        TensorRtDebugListenerAttachVTableSafetyGateResult attachVTableSafetyGate,
        TensorRtDebugListenerNativeAttachNoThrowPreflightResult nativeAttachNoThrowPreflight,
        TensorRtDebugListenerNativeOwnerAddressDesignGateResult nativeOwnerAddressDesignGate,
        TensorRtDebugListenerNativeNoThrowVTableDesignGateResult nativeNoThrowVTableDesignGate,
        TensorRtDebugListenerNativeAttachEntryDesignGateResult nativeAttachEntryDesignGate)
    {
        TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult nativeDetachBeforeReleaseDesignGate =
            TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate.Evaluate(
                ownerDesignSnapshot,
                attachDetachDesignGate,
                borrowedTensorSafetyGate,
                attachVTableSafetyGate,
                nativeAttachNoThrowPreflight,
                nativeOwnerAddressDesignGate,
                nativeNoThrowVTableDesignGate,
                nativeAttachEntryDesignGate);
        TensorRtDebugListenerNativeOwnerLifecycleDryRunResult nativeOwnerLifecycleDryRun =
            TensorRtDebugListenerNativeOwnerLifecycleDryRun.Evaluate(
                ownerDesignSnapshot,
                attachDetachDesignGate,
                borrowedTensorSafetyGate,
                attachVTableSafetyGate,
                nativeAttachNoThrowPreflight,
                nativeOwnerAddressDesignGate,
                nativeNoThrowVTableDesignGate,
                nativeAttachEntryDesignGate,
                nativeDetachBeforeReleaseDesignGate);
        return Evaluate(
            ownerDesignSnapshot,
            attachDetachDesignGate,
            borrowedTensorSafetyGate,
            attachVTableSafetyGate,
            nativeAttachNoThrowPreflight,
            nativeOwnerAddressDesignGate,
            nativeNoThrowVTableDesignGate,
            nativeAttachEntryDesignGate,
            nativeDetachBeforeReleaseDesignGate,
            nativeOwnerLifecycleDryRun);
    }

}

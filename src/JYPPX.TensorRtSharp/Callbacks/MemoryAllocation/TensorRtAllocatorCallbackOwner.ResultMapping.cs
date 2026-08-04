using System;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtAllocatorCallbackOwner
{
    private static TensorRtAllocatorNativeDryRunResult CreateNativeDryRunResult(NativeTensorRtAllocatorOwnerDiagnosticInfo info)
    {
        return new TensorRtAllocatorNativeDryRunResult(
            line: (TensorRtApiLine)info.Line,
            invocationCount: info.InvocationCount,
            failureCount: info.FailureCount,
            lastStatus: (BridgeStatusCode)info.LastStatus,
            isAttached: info.IsAttached != 0,
            lastSize: info.LastSize,
            lastAlignment: info.LastAlignment,
            diagnostic: BridgeInfoMapper.ReadFixedUtf8(info.LastDiagnostic));
    }

    private static TensorRtAllocatorOwnerStateDryRunResult CreateStateDryRunResult(NativeTensorRtAllocatorOwnerStateInfo info)
    {
        return new TensorRtAllocatorOwnerStateDryRunResult(
            line: (TensorRtApiLine)info.Line,
            ownerId: info.OwnerId,
            stateTransitionCount: info.StateTransitionCount,
            ledgerAllocationCount: info.LedgerAllocationCount,
            ledgerReleaseCount: info.LedgerReleaseCount,
            ledgerFailureCount: info.LedgerFailureCount,
            lastAllocationId: info.LastAllocationId,
            lastReleaseAllocationId: info.LastReleaseAllocationId,
            lastSize: info.LastSize,
            lastAlignment: info.LastAlignment,
            lastStreamValue: info.LastStreamValue,
            attachState: info.AttachState,
            lastStatus: (BridgeStatusCode)info.LastStatus,
            isAttached: info.IsAttached != 0,
            hasLiveAllocation: info.HasLiveAllocation != 0,
            lastOperation: BridgeInfoMapper.ReadFixedUtf8(info.LastOperation),
            diagnostic: BridgeInfoMapper.ReadFixedUtf8(info.LastDiagnostic));
    }

}

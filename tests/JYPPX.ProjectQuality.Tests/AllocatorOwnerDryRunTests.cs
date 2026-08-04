using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class AllocatorOwnerDryRunTests
{
    [Fact]
    public void DryRunOwnerRecordsSuccessFailureAndDisposeWithoutNativeBridge()
    {
        using TensorRtAllocatorCallbackOwner owner = new TensorRtAllocatorCallbackOwner(static request =>
        {
            if (request.Reason == "throw")
            {
                throw new InvalidOperationException("dry-run failure");
            }

            return TensorRtAllocatorDryRunResult.Success($"{request.Reason}:{request.Size}:{request.Alignment}");
        });

        TensorRtAllocatorDryRunResult success = owner.RunDryRunDiagnostic(new TensorRtAllocatorDryRunRequest(1024, 256, "quality"));
        Assert.True(success.Succeeded);
        Assert.Equal("quality:1024:256", success.Diagnostic);
        Assert.False(owner.IsAttached);
        Assert.False(owner.IsDisposed);
        Assert.Equal(1, owner.CallbackInvocationCount);
        Assert.Equal(0, owner.CallbackFailureCount);
        Assert.Null(owner.LastCallbackException);
        Assert.Equal("quality:1024:256", owner.LastDiagnostic);

        TensorRtAllocatorDryRunResult failure = owner.RunDryRunDiagnostic(new TensorRtAllocatorDryRunRequest(2048, 512, "throw"));
        Assert.False(failure.Succeeded);
        Assert.Contains("InvalidOperationException", failure.Diagnostic);
        Assert.Equal(2, owner.CallbackInvocationCount);
        Assert.Equal(1, owner.CallbackFailureCount);
        Assert.IsType<InvalidOperationException>(owner.LastCallbackException);
        Assert.Equal(failure.Diagnostic, owner.LastDiagnostic);

        owner.Dispose();
        Assert.True(owner.IsDisposed);
        Assert.Throws<ObjectDisposedException>(() => owner.RunDryRunDiagnostic(new TensorRtAllocatorDryRunRequest(1, 1, "after-dispose")));
    }

    [Fact]
    public void DryRunRequestRejectsZeroAlignment()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new TensorRtAllocatorDryRunRequest(1, 0, "bad"));
    }

    [Fact]
    public void InternalRuntimePrototypeRecordsPinningInFlightExceptionAndReleaseHook()
    {
        using TensorRtAllocatorCallbackOwner owner = new TensorRtAllocatorCallbackOwner(static request =>
            TensorRtAllocatorDryRunResult.Success($"prototype:{request.Reason}:{request.Size}:{request.Alignment}"));

        TensorRtAllocatorCallbackOwnerSnapshot success =
            owner.RunLifecycleDiagnostic(new TensorRtAllocatorDryRunRequest(4096, 256, "quality-prototype"));
        Assert.Equal("allocator-owner-internal-runtime-prototype", success.EvidenceKind);
        Assert.Equal("not-present", success.RuntimeEvidenceKind);
        Assert.False(success.RealCallbackRuntime);
        Assert.False(success.IsRealCallbackRuntimeProof);
        Assert.Equal("sync-allocator-prototype", success.CallbackKind);
        Assert.Equal(JYPPX.TensorRtSharp.Shared.Interop.BridgeStatusCode.Ok, success.LastStatus);
        Assert.Equal(1, success.InvocationCount);
        Assert.Equal(0, success.FailureCount);
        Assert.Equal(0, success.InFlightCallbackCount);
        Assert.Equal(1, success.MaxInFlightCallbackCount);
        Assert.True(success.CallbackStatePinned);
        Assert.True(success.DelegatePinned);
        Assert.False(success.DisposeRequested);
        Assert.False(success.IsAttached);
        Assert.False(success.DevicePointerExposed);
        Assert.False(success.DevicePointerProduced);
        Assert.False(success.BorrowedPointerEscaped);
        Assert.True(success.ManagedKeepAliveReady);
        Assert.True(success.PointerFreeSurfaceReady);
        Assert.Contains("prototype:quality-prototype:4096:256", success.LastDiagnostic);

        TensorRtAllocatorCallbackOwnerSnapshot beforeDispose = owner.GetSnapshot("pre-dispose");
        Assert.Equal(0, beforeDispose.ReleaseHookCount);
        Assert.True(beforeDispose.CallbackStatePinned);
        Assert.True(beforeDispose.DelegatePinned);

        owner.Dispose();
        TensorRtAllocatorCallbackOwnerSnapshot afterDispose = owner.GetSnapshot("post-dispose");
        Assert.True(afterDispose.DisposeRequested);
        Assert.False(afterDispose.CallbackStatePinned);
        Assert.False(afterDispose.DelegatePinned);
        Assert.Equal(1, afterDispose.ReleaseHookCount);
        Assert.Equal(0, afterDispose.InFlightCallbackCount);
        Assert.True(afterDispose.DisposeReleaseReady);
        Assert.Contains("release hook", afterDispose.ReleaseDiagnostic);

        using TensorRtAllocatorCallbackOwner throwingOwner = new TensorRtAllocatorCallbackOwner(static _ =>
            throw new InvalidOperationException("quality prototype failure"));
        TensorRtAllocatorCallbackOwnerSnapshot failure =
            throwingOwner.RunLifecycleDiagnostic(new TensorRtAllocatorDryRunRequest(8192, 512, "quality-prototype-throw"));
        Assert.False(failure.RealCallbackRuntime);
        Assert.Equal(JYPPX.TensorRtSharp.Shared.Interop.BridgeStatusCode.InvalidState, failure.LastStatus);
        Assert.Equal(1, failure.InvocationCount);
        Assert.Equal(1, failure.FailureCount);
        Assert.Equal(0, failure.InFlightCallbackCount);
        Assert.False(failure.Succeeded);
        Assert.Contains("InvalidOperationException", failure.LastDiagnostic);
    }
}

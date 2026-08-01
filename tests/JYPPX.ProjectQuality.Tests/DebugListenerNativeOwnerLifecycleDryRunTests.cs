using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerNativeOwnerLifecycleDryRunTests
{
    [Fact]
    public void NativeOwnerLifecycleDryRunCopiesOwnerEvidenceWithoutRuntimeProof()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-native-owner-lifecycle-dry-run",
            isInput: true,
            isExecutionTensor: true);

        TensorRtDebugListenerCallbackOwnerSnapshot diagnostic = owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        Assert.True(diagnostic.DebugTensorMetadataCopied);

        owner.Dispose();
        TensorRtDebugListenerCallbackOwnerSnapshot disposed = owner.GetSnapshot("post-dispose");
        TensorRtDebugListenerNativeOwnerLifecycleDryRunResult dryRun =
            TensorRtDebugListenerNativeOwnerLifecycleDryRun.Evaluate(disposed);

        Assert.Equal("debug-listener-native-owner-lifecycle-dry-run", dryRun.EvidenceKind);
        Assert.Equal("debug-listener-process-debug-tensor", dryRun.CallbackKind);
        Assert.Equal("dry-run", dryRun.RuntimeEvidenceKind);
        Assert.False(dryRun.RealCallbackRuntime);
        Assert.False(dryRun.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, dryRun.Line);
        Assert.True(dryRun.OwnerId > 0);
        Assert.Equal(BridgeStatusCode.Ok, dryRun.LastStatus);
        Assert.True(dryRun.ReleaseHookCount > 0);
        Assert.Equal(0, dryRun.InFlightCallbackCount);
        Assert.False(dryRun.CallbackStatePinned);
        Assert.False(dryRun.DelegatePinned);
        Assert.True(dryRun.DisposeRequested);
        Assert.True(dryRun.NativeDetachBeforeReleaseDesignGateReady);
        Assert.True(dryRun.NativeAttachEntryDesignGateReady);
        Assert.True(dryRun.NativeNoThrowVTableDesignGateReady);
        Assert.True(dryRun.NativeOwnerAddressDesignGateReady);
        Assert.True(dryRun.NativeDetachEntryLocated);
        Assert.False(dryRun.NativeAttachEntryLocated);
        Assert.False(dryRun.StableNativeOwnerIdentityReady);
        Assert.False(dryRun.NativeOwnerNonCopyableReady);
        Assert.False(dryRun.NativeOwnerDisposeOrderReady);
        Assert.False(dryRun.NativeOwnerReleaseHookReady);
        Assert.False(dryRun.NativeOwnerInFlightDrainReady);
        Assert.False(dryRun.DetachBeforeReleaseReady);
        Assert.False(dryRun.ReleaseHookOrderingReady);
        Assert.False(dryRun.DisposeIdempotencyReady);
        Assert.False(dryRun.InFlightDrainBeforeReleaseReady);
        Assert.False(dryRun.CallbackStateUnpinAfterDetachReady);
        Assert.False(dryRun.DelegateUnpinAfterDetachReady);
        Assert.False(dryRun.NoThrowNativeDestructorReady);
        Assert.False(dryRun.NativeOwnerLifecycleReady);
        Assert.False(dryRun.NativeVTableDesignReady);
        Assert.True(dryRun.ManagedCallbackKeepAliveDesignReady);
        Assert.True(dryRun.BorrowedDebugTensorMetadataCopyDesignReady);
        Assert.True(dryRun.BorrowedDebugTensorPointerEscapeBlocked);
        Assert.True(dryRun.DryRunReady);
        Assert.False(dryRun.BorrowedDebugTensorLifetimeRuntimeReady);
        Assert.False(dryRun.BorrowedDebugTensorDataLifetimeRuntimeReady);
        Assert.False(dryRun.ProcessDebugTensorRuntimeReady);
        Assert.False(dryRun.FullPackageConsumerRuntimeEvidenceReady);
        Assert.False(dryRun.CanImplementNativeAttach);
        Assert.False(dryRun.CanAttemptRuntimeProof);
        Assert.True(dryRun.RuntimeProofBlocked);
        Assert.True(dryRun.DeferredRowsStillRequired);
        Assert.Equal("dry-run-ready", dryRun.Status);
        Assert.True(dryRun.BlockedPrerequisiteCount >= 10);
        Assert.Contains(dryRun.BlockedPrerequisites, item => item.Contains("stable owner identity", StringComparison.Ordinal));
        Assert.Contains(dryRun.BlockedPrerequisites, item => item.Contains("non-copyable storage", StringComparison.Ordinal));
        Assert.Contains(dryRun.BlockedPrerequisites, item => item.Contains("release hook ordering", StringComparison.Ordinal));
        Assert.Contains(dryRun.BlockedPrerequisites, item => item.Contains("idempotency", StringComparison.Ordinal));
        Assert.Contains(dryRun.BlockedPrerequisites, item => item.Contains("in-flight callback drain", StringComparison.Ordinal));
        Assert.Contains(dryRun.BlockedPrerequisites, item => item.Contains("callback state post-detach unpin", StringComparison.Ordinal));
        Assert.Contains(dryRun.BlockedPrerequisites, item => item.Contains("delegate post-detach unpin", StringComparison.Ordinal));
        Assert.Contains(dryRun.BlockedPrerequisites, item => item.Contains("no-throw destructor", StringComparison.Ordinal));
        Assert.Contains("RuntimeEvidenceKind=dry-run", dryRun.Diagnostic);
        Assert.Contains("DryRunReady=True", dryRun.Diagnostic);
        Assert.Contains("NativeDetachBeforeReleaseDesignGateReady=True", dryRun.Diagnostic);
        Assert.Contains("NativeDetachEntryLocated=True", dryRun.Diagnostic);
        Assert.Contains("NativeAttachEntryLocated=False", dryRun.Diagnostic);
        Assert.Contains("StableNativeOwnerIdentityReady=False", dryRun.Diagnostic);
        Assert.Contains("NativeOwnerNonCopyableReady=False", dryRun.Diagnostic);
        Assert.Contains("ReleaseHookOrderingReady=False", dryRun.Diagnostic);
        Assert.Contains("DisposeIdempotencyReady=False", dryRun.Diagnostic);
        Assert.Contains("InFlightDrainBeforeReleaseReady=False", dryRun.Diagnostic);
        Assert.Contains("CallbackStateUnpinAfterDetachReady=False", dryRun.Diagnostic);
        Assert.Contains("DelegateUnpinAfterDetachReady=False", dryRun.Diagnostic);
        Assert.Contains("NoThrowNativeDestructorReady=False", dryRun.Diagnostic);
        Assert.Contains("NativeOwnerLifecycleReady=False", dryRun.Diagnostic);
        Assert.Contains("CanImplementNativeAttach=False", dryRun.Diagnostic);
        Assert.Contains("RuntimeProofBlocked=True", dryRun.Diagnostic);
    }

    [Fact]
    public void RuntimeProofPrecheckConsumesNativeOwnerLifecycleDryRunAndStaysBlocked()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-native-owner-lifecycle-dry-run-precheck",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        TensorRtDebugListenerRuntimeProofPrecheckResult precheck =
            TensorRtDebugListenerRuntimeProofPrecheck.Evaluate(owner.GetSnapshot("post-dispose"));

        Assert.True(precheck.NativeDetachBeforeReleaseDesignGateReady);
        Assert.True(precheck.NativeOwnerLifecycleDryRunReady);
        Assert.False(precheck.NativeAttachEntryLocated);
        Assert.True(precheck.NativeDetachEntryLocated);
        Assert.False(precheck.DetachBeforeReleaseReady);
        Assert.False(precheck.ReleaseHookOrderingReady);
        Assert.False(precheck.DisposeIdempotencyReady);
        Assert.False(precheck.InFlightDrainBeforeReleaseReady);
        Assert.False(precheck.CallbackStateUnpinAfterDetachReady);
        Assert.False(precheck.DelegateUnpinAfterDetachReady);
        Assert.True(precheck.NativeOwnerNonCopyableStorageReady);
        Assert.True(precheck.NativeOwnerNonCopyableReady);
        Assert.True(precheck.NativeOwnerCopyBlocked);
        Assert.True(precheck.NativeOwnerMoveBlocked);
        Assert.False(precheck.NativeOwnerAddressExposed);
        Assert.False(precheck.NativeOwnerPointerProduced);
        Assert.True(precheck.NativeNoThrowDestructorGateReady);
        Assert.True(precheck.DestructorNoThrowScaffoldReady);
        Assert.True(precheck.DestructorExceptionEscapeBlocked);
        Assert.False(precheck.DestructorAddressExposed);
        Assert.False(precheck.DestructorPointerProduced);
        Assert.True(precheck.NoThrowNativeDestructorReady);
        Assert.False(precheck.NativeOwnerLifecycleReady);
        Assert.False(precheck.CanImplementNativeAttach);
        Assert.False(precheck.CanAttemptRuntimeProof);
        Assert.True(precheck.RuntimeProofBlocked);
        Assert.Contains("NativeOwnerLifecycleDryRunReady=True", precheck.Diagnostic);
        Assert.Contains("NativeOwnerNonCopyableStorageReady=True", precheck.Diagnostic);
        Assert.Contains("NativeOwnerNonCopyableReady=True", precheck.Diagnostic);
        Assert.Contains("NativeOwnerCopyBlocked=True", precheck.Diagnostic);
        Assert.Contains("NativeOwnerMoveBlocked=True", precheck.Diagnostic);
        Assert.Contains("NativeOwnerAddressExposed=False", precheck.Diagnostic);
        Assert.Contains("NativeOwnerPointerProduced=False", precheck.Diagnostic);
        Assert.Contains("NativeNoThrowDestructorGateReady=True", precheck.Diagnostic);
        Assert.Contains("DestructorNoThrowScaffoldReady=True", precheck.Diagnostic);
        Assert.Contains("DestructorExceptionEscapeBlocked=True", precheck.Diagnostic);
        Assert.Contains("DestructorAddressExposed=False", precheck.Diagnostic);
        Assert.Contains("DestructorPointerProduced=False", precheck.Diagnostic);
        Assert.Contains("ReleaseHookOrderingReady=False", precheck.Diagnostic);
        Assert.Contains("DisposeIdempotencyReady=False", precheck.Diagnostic);
        Assert.Contains("InFlightDrainBeforeReleaseReady=False", precheck.Diagnostic);
        Assert.Contains("CallbackStateUnpinAfterDetachReady=False", precheck.Diagnostic);
        Assert.Contains("DelegateUnpinAfterDetachReady=False", precheck.Diagnostic);
        Assert.Contains("NoThrowNativeDestructorReady=True", precheck.Diagnostic);
    }

    [Fact]
    public void PublicNativeOwnerLifecycleDryRunSurfaceDoesNotExposeRawPointers()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtDebugListenerNativeOwnerLifecycleDryRun),
            typeof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult)
        };

        foreach (Type type in publicTypes)
        {
            foreach (ConstructorInfo constructor in type.GetConstructors(BindingFlags.Instance | BindingFlags.Public))
            {
                foreach (ParameterInfo parameter in constructor.GetParameters())
                {
                    Assert.NotEqual(typeof(IntPtr), parameter.ParameterType);
                    Assert.NotEqual(typeof(UIntPtr), parameter.ParameterType);
                }
            }

            foreach (PropertyInfo property in type.GetProperties(BindingFlags.Instance | BindingFlags.Public | BindingFlags.Static))
            {
                Assert.NotEqual(typeof(IntPtr), property.PropertyType);
                Assert.NotEqual(typeof(UIntPtr), property.PropertyType);
            }

            foreach (MethodInfo method in type.GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.DeclaredOnly))
            {
                Assert.NotEqual(typeof(IntPtr), method.ReturnType);
                Assert.NotEqual(typeof(UIntPtr), method.ReturnType);
                foreach (ParameterInfo parameter in method.GetParameters())
                {
                    Assert.NotEqual(typeof(IntPtr), parameter.ParameterType);
                    Assert.NotEqual(typeof(UIntPtr), parameter.ParameterType);
                }
            }
        }
    }

    [Fact]
    public void ReadinessSmokeAndDocsKeepNativeOwnerLifecycleDryRunSeparateFromRuntimeProof()
    {
        string dryRunSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerNativeOwnerLifecycleDryRun.cs");
        string precheckSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerRuntimeProofPrecheck.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string dryRunDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-native-owner-lifecycle-dry-run.md");
        string detachGateDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-native-detach-before-release-design-gate.md");
        string precheckDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-runtime-proof-precheck.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string index = ReadSource("docs", "index.md");
        string toc = ReadSource("docs", "toc.yml");
        string smokeReadme = ReadSource("smoke", "README.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("public static class TensorRtDebugListenerNativeOwnerLifecycleDryRun", dryRunSource);
        Assert.Contains("public readonly struct TensorRtDebugListenerNativeOwnerLifecycleDryRunResult", dryRunSource);
        Assert.Contains("EvidenceKind => \"debug-listener-native-owner-lifecycle-dry-run\"", dryRunSource);
        Assert.Contains("RuntimeEvidenceKind => \"dry-run\"", dryRunSource);
        Assert.Contains("RealCallbackRuntime => false", dryRunSource);
        Assert.Contains("IsRealCallbackRuntimeProof => false", dryRunSource);
        Assert.Contains("DryRunReady", dryRunSource);
        Assert.Contains("StableNativeOwnerIdentityReady", dryRunSource);
        Assert.Contains("NativeOwnerNonCopyableReady", dryRunSource);
        Assert.Contains("ReleaseHookOrderingReady", dryRunSource);
        Assert.Contains("DisposeIdempotencyReady", dryRunSource);
        Assert.Contains("InFlightDrainBeforeReleaseReady", dryRunSource);
        Assert.Contains("CallbackStateUnpinAfterDetachReady", dryRunSource);
        Assert.Contains("DelegateUnpinAfterDetachReady", dryRunSource);
        Assert.Contains("NoThrowNativeDestructorReady", dryRunSource);
        Assert.DoesNotContain("public IntPtr", dryRunSource);
        Assert.DoesNotContain("public nint", dryRunSource);

        Assert.Contains("TensorRtDebugListenerNativeOwnerLifecycleDryRun", precheckSource);
        Assert.Contains("NativeOwnerLifecycleDryRunReady", precheckSource);
        Assert.Contains("NativeOwnerLifecycleDryRunReady &&", precheckSource);

        Assert.Contains("debug-listener-native-owner-lifecycle-dry-run", smokeProgram);
        Assert.Contains("DebugListenerNativeOwnerLifecycleDryRun=", smokeProgram);
        Assert.Contains("NativeOwnerLifecycleDryRunReady", smokeProgram);

        Assert.Contains("debugListenerNativeOwnerLifecycleDryRun", readiness);
        Assert.Contains("New-DebugListenerNativeOwnerLifecycleDryRunEvidence", readiness);
        Assert.Contains("debug-listener-native-owner-lifecycle-dry-run", readiness);
        Assert.Contains("dry-run-ready", readiness);
        Assert.Contains("NativeOwnerLifecycleDryRunReady", readiness);
        Assert.Contains("canImplementNativeAttach = $false", readiness);
        Assert.Contains("canAttemptRuntimeProof = $false", readiness);
        Assert.Contains("runtimeProofBlocked = $true", readiness);

        Assert.Contains("debug-listener-native-owner-lifecycle-dry-run", packageConsumer);
        Assert.Contains("debug-listener-native-owner-lifecycle-dry-run", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeOwnerLifecycleDryRun", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeOwnerLifecycleDryRunResult", bridgeConsumer);

        Assert.Contains("debug-listener-native-owner-lifecycle-dry-run", dryRunDoc);
        Assert.Contains("RuntimeEvidenceKind=dry-run", dryRunDoc);
        Assert.Contains("DryRunReady=True", dryRunDoc);
        Assert.Contains("NativeDetachBeforeReleaseDesignGateReady=True", dryRunDoc);
        Assert.Contains("NativeDetachEntryLocated=True", dryRunDoc);
        Assert.Contains("NativeAttachEntryLocated=False", dryRunDoc);
        Assert.Contains("StableNativeOwnerIdentityReady=False", dryRunDoc);
        Assert.Contains("NativeOwnerNonCopyableReady=False", dryRunDoc);
        Assert.Contains("ReleaseHookOrderingReady=False", dryRunDoc);
        Assert.Contains("DisposeIdempotencyReady=False", dryRunDoc);
        Assert.Contains("InFlightDrainBeforeReleaseReady=False", dryRunDoc);
        Assert.Contains("CallbackStateUnpinAfterDetachReady=False", dryRunDoc);
        Assert.Contains("DelegateUnpinAfterDetachReady=False", dryRunDoc);
        Assert.Contains("NoThrowNativeDestructorReady=False", dryRunDoc);
        Assert.Contains("NativeOwnerLifecycleReady=False", dryRunDoc);
        Assert.Contains("not proof", dryRunDoc);
        Assert.Contains("debug-listener-native-owner-lifecycle-dry-run", detachGateDoc);
        Assert.Contains("debug-listener-native-owner-lifecycle-dry-run", precheckDoc);
        Assert.Contains("debug-listener-native-owner-lifecycle-dry-run", trampolineGate);
        Assert.Contains("debug-listener-native-owner-lifecycle-dry-run", schema);
        Assert.Contains("debug-listener-native-owner-lifecycle-dry-run", latest);
        Assert.Contains("debug-listener-native-owner-lifecycle-dry-run", index);
        Assert.Contains("debug-listener-native-owner-lifecycle-dry-run.md", toc);
        Assert.Contains("debug-listener-native-owner-lifecycle-dry-run", smokeReadme);

        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IOutputAllocator\",\"notifyShape\",\"IOutputAllocator::notifyShape\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IOutputAllocator\",\"reallocateOutput\",\"IOutputAllocator::reallocateOutput\",\"other\",\"deferred-only\"", comparison);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return RepositorySourceReader.Read(path);
    }
}

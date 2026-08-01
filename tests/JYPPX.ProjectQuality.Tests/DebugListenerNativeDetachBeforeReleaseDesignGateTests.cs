using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerNativeDetachBeforeReleaseDesignGateTests
{
    [Fact]
    public void NativeDetachBeforeReleaseDesignGateCopiesAttachEntryStateWithoutRuntimeProof()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-native-detach-before-release-design-gate",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        TensorRtDebugListenerCallbackOwnerSnapshot disposed = owner.GetSnapshot("post-dispose");
        TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult gate =
            TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate.Evaluate(disposed);

        Assert.Equal("debug-listener-native-detach-before-release-design-gate", gate.EvidenceKind);
        Assert.Equal("debug-listener-process-debug-tensor", gate.CallbackKind);
        Assert.Equal("design-gate", gate.RuntimeEvidenceKind);
        Assert.False(gate.RealCallbackRuntime);
        Assert.False(gate.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, gate.Line);
        Assert.True(gate.NativeAttachEntryDesignGateReady);
        Assert.True(gate.NativeNoThrowVTableDesignGateReady);
        Assert.True(gate.NativeOwnerAddressDesignGateReady);
        Assert.True(gate.NativeDetachEntryLocated);
        Assert.False(gate.NativeAttachEntryLocated);
        Assert.False(gate.LineSpecificAttachEntryDesignReady);
        Assert.False(gate.AttachEntryNoThrowReady);
        Assert.False(gate.AttachEntryVersionGuardReady);
        Assert.False(gate.AttachEntryOwnershipReady);
        Assert.False(gate.DetachBeforeReleaseReady);
        Assert.False(gate.ReleaseHookOrderingReady);
        Assert.False(gate.DisposeIdempotencyReady);
        Assert.False(gate.InFlightDrainBeforeReleaseReady);
        Assert.False(gate.CallbackStateUnpinAfterDetachReady);
        Assert.False(gate.DelegateUnpinAfterDetachReady);
        Assert.False(gate.NativeOwnerLifecycleReady);
        Assert.False(gate.NativeVTableDesignReady);
        Assert.True(gate.ManagedCallbackKeepAliveDesignReady);
        Assert.True(gate.BorrowedDebugTensorMetadataCopyDesignReady);
        Assert.True(gate.BorrowedDebugTensorPointerEscapeBlocked);
        Assert.False(gate.BorrowedDebugTensorLifetimeRuntimeReady);
        Assert.False(gate.BorrowedDebugTensorDataLifetimeRuntimeReady);
        Assert.False(gate.ProcessDebugTensorRuntimeReady);
        Assert.False(gate.FullPackageConsumerRuntimeEvidenceReady);
        Assert.True(gate.DesignGateReady);
        Assert.False(gate.CanImplementNativeAttach);
        Assert.False(gate.CanAttemptRuntimeProof);
        Assert.True(gate.RuntimeProofBlocked);
        Assert.True(gate.DeferredRowsStillRequired);
        Assert.Equal("design-gate-ready", gate.Status);
        Assert.True(gate.BlockedPrerequisiteCount >= 10);
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("detach-before-release", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("release hook ordering", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("idempotency", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("in-flight callback drain", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("callback state unpin", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("delegate unpin", StringComparison.Ordinal));
        Assert.Contains("RuntimeEvidenceKind=design-gate", gate.Diagnostic);
        Assert.Contains("NativeAttachEntryDesignGateReady=True", gate.Diagnostic);
        Assert.Contains("NativeDetachEntryLocated=True", gate.Diagnostic);
        Assert.Contains("NativeAttachEntryLocated=False", gate.Diagnostic);
        Assert.Contains("DetachBeforeReleaseReady=False", gate.Diagnostic);
        Assert.Contains("ReleaseHookOrderingReady=False", gate.Diagnostic);
        Assert.Contains("DisposeIdempotencyReady=False", gate.Diagnostic);
        Assert.Contains("InFlightDrainBeforeReleaseReady=False", gate.Diagnostic);
        Assert.Contains("CallbackStateUnpinAfterDetachReady=False", gate.Diagnostic);
        Assert.Contains("DelegateUnpinAfterDetachReady=False", gate.Diagnostic);
        Assert.Contains("CanImplementNativeAttach=False", gate.Diagnostic);
        Assert.Contains("RuntimeProofBlocked=True", gate.Diagnostic);
    }

    [Fact]
    public void RuntimeProofPrecheckConsumesNativeDetachBeforeReleaseDesignGateAndStaysBlocked()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-native-detach-before-release-design-gate-precheck",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        TensorRtDebugListenerRuntimeProofPrecheckResult precheck =
            TensorRtDebugListenerRuntimeProofPrecheck.Evaluate(owner.GetSnapshot("post-dispose"));

        Assert.True(precheck.NativeDetachBeforeReleaseDesignGateReady);
        Assert.False(precheck.DetachBeforeReleaseReady);
        Assert.False(precheck.ReleaseHookOrderingReady);
        Assert.False(precheck.DisposeIdempotencyReady);
        Assert.False(precheck.InFlightDrainBeforeReleaseReady);
        Assert.False(precheck.CallbackStateUnpinAfterDetachReady);
        Assert.False(precheck.DelegateUnpinAfterDetachReady);
        Assert.False(precheck.CanImplementNativeAttach);
        Assert.False(precheck.CanAttemptRuntimeProof);
        Assert.True(precheck.RuntimeProofBlocked);
        Assert.Contains("NativeDetachBeforeReleaseDesignGateReady=True", precheck.Diagnostic);
        Assert.Contains("ReleaseHookOrderingReady=False", precheck.Diagnostic);
        Assert.Contains("DisposeIdempotencyReady=False", precheck.Diagnostic);
        Assert.Contains("InFlightDrainBeforeReleaseReady=False", precheck.Diagnostic);
        Assert.Contains("CallbackStateUnpinAfterDetachReady=False", precheck.Diagnostic);
        Assert.Contains("DelegateUnpinAfterDetachReady=False", precheck.Diagnostic);
    }

    [Fact]
    public void PublicNativeDetachBeforeReleaseGateSurfaceDoesNotExposeRawPointers()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate),
            typeof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult)
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
    public void ReadinessSmokeAndDocsKeepNativeDetachBeforeReleaseGateSeparateFromRuntimeProof()
    {
        string gateSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate.cs");
        string precheckSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerRuntimeProofPrecheck.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string gateDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-native-detach-before-release-design-gate.md");
        string attachEntryDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-native-attach-entry-design-gate.md");
        string precheckDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-runtime-proof-precheck.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string index = ReadSource("docs", "index.md");
        string toc = ReadSource("docs", "toc.yml");
        string smokeReadme = ReadSource("smoke", "README.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("public static class TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate", gateSource);
        Assert.Contains("public readonly struct TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult", gateSource);
        Assert.Contains("EvidenceKind => \"debug-listener-native-detach-before-release-design-gate\"", gateSource);
        Assert.Contains("RuntimeEvidenceKind => \"design-gate\"", gateSource);
        Assert.Contains("RealCallbackRuntime => false", gateSource);
        Assert.Contains("IsRealCallbackRuntimeProof => false", gateSource);
        Assert.Contains("ReleaseHookOrderingReady", gateSource);
        Assert.Contains("DisposeIdempotencyReady", gateSource);
        Assert.Contains("InFlightDrainBeforeReleaseReady", gateSource);
        Assert.Contains("CallbackStateUnpinAfterDetachReady", gateSource);
        Assert.Contains("DelegateUnpinAfterDetachReady", gateSource);
        Assert.DoesNotContain("public IntPtr", gateSource);
        Assert.DoesNotContain("public nint", gateSource);

        Assert.Contains("TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate", precheckSource);
        Assert.Contains("NativeDetachBeforeReleaseDesignGateReady", precheckSource);
        Assert.Contains("ReleaseHookOrderingReady", precheckSource);

        Assert.Contains("debug-listener-native-detach-before-release-design-gate", smokeProgram);
        Assert.Contains("DebugListenerNativeDetachBeforeReleaseDesignGate=", smokeProgram);
        Assert.Contains("NativeDetachBeforeReleaseDesignGateReady", smokeProgram);

        Assert.Contains("debugListenerNativeDetachBeforeReleaseDesignGate", readiness);
        Assert.Contains("New-DebugListenerNativeDetachBeforeReleaseDesignGateEvidence", readiness);
        Assert.Contains("debug-listener-native-detach-before-release-design-gate", readiness);
        Assert.Contains("NativeDetachBeforeReleaseDesignGateReady", readiness);
        Assert.Contains("ReleaseHookOrderingReady=False", readiness);
        Assert.Contains("DisposeIdempotencyReady=False", readiness);
        Assert.Contains("InFlightDrainBeforeReleaseReady=False", readiness);
        Assert.Contains("CallbackStateUnpinAfterDetachReady=False", readiness);
        Assert.Contains("DelegateUnpinAfterDetachReady=False", readiness);
        Assert.Contains("canImplementNativeAttach = $false", readiness);
        Assert.Contains("canAttemptRuntimeProof = $false", readiness);
        Assert.Contains("runtimeProofBlocked = $true", readiness);

        Assert.Contains("debug-listener-native-detach-before-release-design-gate", packageConsumer);
        Assert.Contains("debug-listener-native-detach-before-release-design-gate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult", bridgeConsumer);

        Assert.Contains("debug-listener-native-detach-before-release-design-gate", gateDoc);
        Assert.Contains("RuntimeEvidenceKind=design-gate", gateDoc);
        Assert.Contains("NativeAttachEntryDesignGateReady=True", gateDoc);
        Assert.Contains("NativeDetachEntryLocated=True", gateDoc);
        Assert.Contains("NativeAttachEntryLocated=False", gateDoc);
        Assert.Contains("DetachBeforeReleaseReady=False", gateDoc);
        Assert.Contains("ReleaseHookOrderingReady=False", gateDoc);
        Assert.Contains("DisposeIdempotencyReady=False", gateDoc);
        Assert.Contains("InFlightDrainBeforeReleaseReady=False", gateDoc);
        Assert.Contains("CallbackStateUnpinAfterDetachReady=False", gateDoc);
        Assert.Contains("DelegateUnpinAfterDetachReady=False", gateDoc);
        Assert.Contains("CanImplementNativeAttach=False", gateDoc);
        Assert.Contains("not proof", gateDoc);
        Assert.Contains("debug-listener-native-detach-before-release-design-gate", attachEntryDoc);
        Assert.Contains("debug-listener-native-detach-before-release-design-gate", precheckDoc);
        Assert.Contains("debug-listener-native-detach-before-release-design-gate", trampolineGate);
        Assert.Contains("debug-listener-native-detach-before-release-design-gate", schema);
        Assert.Contains("debug-listener-native-detach-before-release-design-gate", latest);
        Assert.Contains("debug-listener-native-detach-before-release-design-gate", index);
        Assert.Contains("debug-listener-native-detach-before-release-design-gate.md", toc);
        Assert.Contains("debug-listener-native-detach-before-release-design-gate", smokeReadme);

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

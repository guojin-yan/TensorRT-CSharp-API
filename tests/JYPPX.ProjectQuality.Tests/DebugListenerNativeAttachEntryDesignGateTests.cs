using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerNativeAttachEntryDesignGateTests
{
    [Fact]
    public void NativeAttachEntryDesignGateCopiesNoThrowVTableStateWithoutRuntimeProof()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-native-attach-entry-design-gate",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        TensorRtDebugListenerCallbackOwnerSnapshot disposed = owner.GetSnapshot("post-dispose");
        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachGate =
            TensorRtDebugListenerAttachDetachDesignGate.Evaluate(disposed);
        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorGate =
            TensorRtDebugListenerBorrowedTensorSafetyGate.Evaluate(disposed, attachDetachGate);
        TensorRtDebugListenerAttachVTableSafetyGateResult attachVTableGate =
            TensorRtDebugListenerAttachVTableSafetyGate.Evaluate(disposed, attachDetachGate, borrowedTensorGate);
        TensorRtDebugListenerNativeAttachNoThrowPreflightResult preflight =
            TensorRtDebugListenerNativeAttachNoThrowPreflight.Evaluate(disposed, attachDetachGate, borrowedTensorGate, attachVTableGate);
        TensorRtDebugListenerNativeOwnerAddressDesignGateResult ownerAddressGate =
            TensorRtDebugListenerNativeOwnerAddressDesignGate.Evaluate(disposed, attachDetachGate, borrowedTensorGate, attachVTableGate, preflight);
        TensorRtDebugListenerNativeNoThrowVTableDesignGateResult noThrowVTableGate =
            TensorRtDebugListenerNativeNoThrowVTableDesignGate.Evaluate(disposed, attachDetachGate, borrowedTensorGate, attachVTableGate, preflight, ownerAddressGate);
        TensorRtDebugListenerNativeAttachEntryDesignGateResult gate =
            TensorRtDebugListenerNativeAttachEntryDesignGate.Evaluate(disposed, attachDetachGate, borrowedTensorGate, attachVTableGate, preflight, ownerAddressGate, noThrowVTableGate);

        Assert.Equal("debug-listener-native-attach-entry-design-gate", gate.EvidenceKind);
        Assert.Equal("debug-listener-process-debug-tensor", gate.CallbackKind);
        Assert.Equal("design-gate", gate.RuntimeEvidenceKind);
        Assert.False(gate.RealCallbackRuntime);
        Assert.False(gate.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, gate.Line);
        Assert.True(gate.NativeNoThrowVTableDesignGateReady);
        Assert.True(gate.NativeOwnerAddressDesignGateReady);
        Assert.True(gate.NativeAttachNoThrowPreflightReady);
        Assert.True(gate.NativeDetachEntryLocated);
        Assert.False(gate.NativeAttachEntryLocated);
        Assert.False(gate.LineSpecificAttachEntryDesignReady);
        Assert.False(gate.AttachEntryNoThrowReady);
        Assert.False(gate.AttachEntryVersionGuardReady);
        Assert.False(gate.AttachEntryOwnershipReady);
        Assert.False(gate.DetachBeforeReleaseReady);
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
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("setDebugListener(non-null)", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("line-specific", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("version guard", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("ownership", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("detach-before-release", StringComparison.Ordinal));
        Assert.Contains("RuntimeEvidenceKind=design-gate", gate.Diagnostic);
        Assert.Contains("NativeNoThrowVTableDesignGateReady=True", gate.Diagnostic);
        Assert.Contains("NativeDetachEntryLocated=True", gate.Diagnostic);
        Assert.Contains("NativeAttachEntryLocated=False", gate.Diagnostic);
        Assert.Contains("LineSpecificAttachEntryDesignReady=False", gate.Diagnostic);
        Assert.Contains("AttachEntryNoThrowReady=False", gate.Diagnostic);
        Assert.Contains("AttachEntryVersionGuardReady=False", gate.Diagnostic);
        Assert.Contains("AttachEntryOwnershipReady=False", gate.Diagnostic);
        Assert.Contains("DetachBeforeReleaseReady=False", gate.Diagnostic);
        Assert.Contains("CanImplementNativeAttach=False", gate.Diagnostic);
        Assert.Contains("RuntimeProofBlocked=True", gate.Diagnostic);
    }

    [Fact]
    public void RuntimeProofPrecheckConsumesNativeAttachEntryDesignGateAndStaysBlocked()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-native-attach-entry-design-gate-precheck",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        TensorRtDebugListenerRuntimeProofPrecheckResult precheck =
            TensorRtDebugListenerRuntimeProofPrecheck.Evaluate(owner.GetSnapshot("post-dispose"));

        Assert.True(precheck.NativeAttachEntryDesignGateReady);
        Assert.False(precheck.LineSpecificAttachEntryDesignReady);
        Assert.False(precheck.AttachEntryNoThrowReady);
        Assert.False(precheck.AttachEntryVersionGuardReady);
        Assert.False(precheck.AttachEntryOwnershipReady);
        Assert.False(precheck.DetachBeforeReleaseReady);
        Assert.False(precheck.CanImplementNativeAttach);
        Assert.False(precheck.CanAttemptRuntimeProof);
        Assert.True(precheck.RuntimeProofBlocked);
        Assert.Contains("NativeAttachEntryDesignGateReady=True", precheck.Diagnostic);
        Assert.Contains("LineSpecificAttachEntryDesignReady=False", precheck.Diagnostic);
        Assert.Contains("AttachEntryNoThrowReady=False", precheck.Diagnostic);
        Assert.Contains("AttachEntryVersionGuardReady=False", precheck.Diagnostic);
        Assert.Contains("AttachEntryOwnershipReady=False", precheck.Diagnostic);
        Assert.Contains("DetachBeforeReleaseReady=False", precheck.Diagnostic);
    }

    [Fact]
    public void PublicNativeAttachEntryGateSurfaceDoesNotExposeRawPointers()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtDebugListenerNativeAttachEntryDesignGate),
            typeof(TensorRtDebugListenerNativeAttachEntryDesignGateResult)
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
    public void ReadinessSmokeAndDocsKeepNativeAttachEntryGateSeparateFromRuntimeProof()
    {
        string gateSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerNativeAttachEntryDesignGate.cs");
        string precheckSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerRuntimeProofPrecheck.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string gateDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-native-attach-entry-design-gate.md");
        string noThrowDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-native-nothrow-vtable-design-gate.md");
        string precheckDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-runtime-proof-precheck.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string index = ReadSource("docs", "index.md");
        string toc = ReadSource("docs", "toc.yml");
        string runtimeSplitReadme = ReadSource("pack", "runtime-split", "README.md");
        string smokeReadme = ReadSource("smoke", "README.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("public static class TensorRtDebugListenerNativeAttachEntryDesignGate", gateSource);
        Assert.Contains("public readonly struct TensorRtDebugListenerNativeAttachEntryDesignGateResult", gateSource);
        Assert.Contains("EvidenceKind => \"debug-listener-native-attach-entry-design-gate\"", gateSource);
        Assert.Contains("RuntimeEvidenceKind => \"design-gate\"", gateSource);
        Assert.Contains("RealCallbackRuntime => false", gateSource);
        Assert.Contains("IsRealCallbackRuntimeProof => false", gateSource);
        Assert.Contains("LineSpecificAttachEntryDesignReady", gateSource);
        Assert.Contains("AttachEntryNoThrowReady", gateSource);
        Assert.Contains("AttachEntryVersionGuardReady", gateSource);
        Assert.Contains("AttachEntryOwnershipReady", gateSource);
        Assert.Contains("DetachBeforeReleaseReady", gateSource);
        Assert.DoesNotContain("public IntPtr", gateSource);
        Assert.DoesNotContain("public nint", gateSource);

        Assert.Contains("TensorRtDebugListenerNativeAttachEntryDesignGate", precheckSource);
        Assert.Contains("NativeAttachEntryDesignGateReady", precheckSource);
        Assert.Contains("LineSpecificAttachEntryDesignReady", precheckSource);

        Assert.Contains("debug-listener-native-attach-entry-design-gate", smokeProgram);
        Assert.Contains("DebugListenerNativeAttachEntryDesignGate=", smokeProgram);
        Assert.Contains("NativeAttachEntryDesignGateReady", smokeProgram);

        Assert.Contains("debugListenerNativeAttachEntryDesignGate", readiness);
        Assert.Contains("New-DebugListenerNativeAttachEntryDesignGateEvidence", readiness);
        Assert.Contains("debug-listener-native-attach-entry-design-gate", readiness);
        Assert.Contains("NativeAttachEntryDesignGateReady", readiness);
        Assert.Contains("LineSpecificAttachEntryDesignReady=False", readiness);
        Assert.Contains("AttachEntryNoThrowReady=False", readiness);
        Assert.Contains("AttachEntryVersionGuardReady=False", readiness);
        Assert.Contains("AttachEntryOwnershipReady=False", readiness);
        Assert.Contains("DetachBeforeReleaseReady=False", readiness);
        Assert.Contains("canImplementNativeAttach = $false", readiness);
        Assert.Contains("canAttemptRuntimeProof = $false", readiness);
        Assert.Contains("runtimeProofBlocked = $true", readiness);

        Assert.Contains("debug-listener-native-attach-entry-design-gate", packageConsumer);
        Assert.Contains("debug-listener-native-attach-entry-design-gate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeAttachEntryDesignGate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeAttachEntryDesignGateResult", bridgeConsumer);

        Assert.Contains("debug-listener-native-attach-entry-design-gate", gateDoc);
        Assert.Contains("RuntimeEvidenceKind=design-gate", gateDoc);
        Assert.Contains("NativeNoThrowVTableDesignGateReady=True", gateDoc);
        Assert.Contains("NativeDetachEntryLocated=True", gateDoc);
        Assert.Contains("NativeAttachEntryLocated=False", gateDoc);
        Assert.Contains("LineSpecificAttachEntryDesignReady=False", gateDoc);
        Assert.Contains("AttachEntryNoThrowReady=False", gateDoc);
        Assert.Contains("AttachEntryVersionGuardReady=False", gateDoc);
        Assert.Contains("AttachEntryOwnershipReady=False", gateDoc);
        Assert.Contains("DetachBeforeReleaseReady=False", gateDoc);
        Assert.Contains("CanImplementNativeAttach=False", gateDoc);
        Assert.Contains("not proof", gateDoc);
        Assert.Contains("debug-listener-native-attach-entry-design-gate", noThrowDoc);
        Assert.Contains("debug-listener-native-attach-entry-design-gate", precheckDoc);
        Assert.Contains("debug-listener-native-attach-entry-design-gate", trampolineGate);
        Assert.Contains("debug-listener-native-attach-entry-design-gate", schema);
        Assert.Contains("debug-listener-native-attach-entry-design-gate", latest);
        Assert.Contains("debug-listener-native-attach-entry-design-gate", index);
        Assert.Contains("debug-listener-native-attach-entry-design-gate.md", toc);
        Assert.Contains("debugListenerNativeAttachEntryDesignGate", runtimeSplitReadme);
        Assert.Contains("debug-listener-native-attach-entry-design-gate", smokeReadme);

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

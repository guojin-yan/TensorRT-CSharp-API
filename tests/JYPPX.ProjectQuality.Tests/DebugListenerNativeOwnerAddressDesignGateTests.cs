using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerNativeOwnerAddressDesignGateTests
{
    [Fact]
    public void NativeOwnerAddressDesignGateCopiesPreflightStateWithoutPointerExposureOrRuntimeProof()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-native-owner-address-design-gate",
            isInput: true,
            isExecutionTensor: true);

        TensorRtDebugListenerCallbackOwnerSnapshot diagnostic = owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        Assert.True(diagnostic.DebugTensorMetadataCopied);

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
        TensorRtDebugListenerNativeOwnerAddressDesignGateResult gate =
            TensorRtDebugListenerNativeOwnerAddressDesignGate.Evaluate(disposed, attachDetachGate, borrowedTensorGate, attachVTableGate, preflight);

        Assert.Equal("debug-listener-native-owner-address-design-gate", gate.EvidenceKind);
        Assert.Equal("debug-listener-native-owner-address", gate.CallbackKind);
        Assert.Equal("design-gate", gate.RuntimeEvidenceKind);
        Assert.False(gate.RealCallbackRuntime);
        Assert.False(gate.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, gate.Line);
        Assert.True(gate.NativeAttachNoThrowPreflightReady);
        Assert.False(gate.NativeAttachEntryLocated);
        Assert.True(gate.NativeDetachEntryLocated);
        Assert.False(gate.StableNativeOwnerAddressReady);
        Assert.False(gate.StableNativeOwnerAddressDesignReady);
        Assert.True(gate.ManagedCallbackKeepAliveDesignReady);
        Assert.False(gate.NativeOwnerNonCopyableReady);
        Assert.False(gate.NativeOwnerDisposeOrderReady);
        Assert.False(gate.NativeOwnerReleaseHookReady);
        Assert.False(gate.NativeOwnerInFlightDrainReady);
        Assert.False(gate.NoThrowNativeDestructorReady);
        Assert.False(gate.NoThrowVTableDesignReady);
        Assert.False(gate.ExceptionToStatusMappingDesignReady);
        Assert.True(gate.BorrowedDebugTensorMetadataCopyDesignReady);
        Assert.True(gate.BorrowedDebugTensorPointerEscapeBlocked);
        Assert.False(gate.NativeOwnerLifecycleReady);
        Assert.False(gate.NativeVTableDesignReady);
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
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("stable native DebugListener owner address", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("non-copyable", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("no-throw destructor", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("IDebugListener::processDebugTensor", StringComparison.Ordinal));
        Assert.Contains("RuntimeEvidenceKind=design-gate", gate.Diagnostic);
        Assert.Contains("DesignGateReady=True", gate.Diagnostic);
        Assert.Contains("NativeAttachNoThrowPreflightReady=True", gate.Diagnostic);
        Assert.Contains("StableNativeOwnerAddressDesignReady=False", gate.Diagnostic);
        Assert.Contains("NativeOwnerNonCopyableReady=False", gate.Diagnostic);
        Assert.Contains("NoThrowNativeDestructorReady=False", gate.Diagnostic);
        Assert.Contains("NativeOwnerLifecycleReady=False", gate.Diagnostic);
        Assert.Contains("CanImplementNativeAttach=False", gate.Diagnostic);
        Assert.Contains("RuntimeProofBlocked=True", gate.Diagnostic);
    }

    [Fact]
    public void RuntimeProofPrecheckConsumesNativeOwnerAddressGateAndStaysBlocked()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-native-owner-address-design-gate-precheck",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        TensorRtDebugListenerCallbackOwnerSnapshot disposed = owner.GetSnapshot("post-dispose");
        TensorRtDebugListenerRuntimeProofPrecheckResult precheck =
            TensorRtDebugListenerRuntimeProofPrecheck.Evaluate(disposed);

        Assert.True(precheck.NativeAttachNoThrowPreflightReady);
        Assert.True(precheck.NativeOwnerAddressDesignGateReady);
        Assert.False(precheck.NativeAttachEntryLocated);
        Assert.True(precheck.NativeDetachEntryLocated);
        Assert.False(precheck.StableNativeOwnerAddressDesignReady);
        Assert.True(precheck.ManagedCallbackKeepAliveDesignReady);
        Assert.True(precheck.NativeOwnerNonCopyableStorageReady);
        Assert.True(precheck.NativeOwnerNonCopyableReady);
        Assert.True(precheck.NativeOwnerCopyBlocked);
        Assert.True(precheck.NativeOwnerMoveBlocked);
        Assert.False(precheck.NativeOwnerAddressExposed);
        Assert.False(precheck.NativeOwnerPointerProduced);
        Assert.False(precheck.NativeOwnerDisposeOrderReady);
        Assert.False(precheck.NativeOwnerReleaseHookReady);
        Assert.False(precheck.NativeOwnerInFlightDrainReady);
        Assert.True(precheck.NativeNoThrowDestructorGateReady);
        Assert.True(precheck.DestructorNoThrowScaffoldReady);
        Assert.True(precheck.DestructorExceptionEscapeBlocked);
        Assert.False(precheck.DestructorAddressExposed);
        Assert.False(precheck.DestructorPointerProduced);
        Assert.True(precheck.NoThrowNativeDestructorReady);
        Assert.False(precheck.NativeOwnerLifecycleReady);
        Assert.False(precheck.NoThrowVTableDesignReady);
        Assert.False(precheck.ExceptionToStatusMappingDesignReady);
        Assert.False(precheck.CanImplementNativeAttach);
        Assert.False(precheck.CanAttemptRuntimeProof);
        Assert.True(precheck.RuntimeProofBlocked);
        Assert.Contains("NativeOwnerAddressDesignGateReady=True", precheck.Diagnostic);
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
        Assert.Contains("NativeOwnerDisposeOrderReady=False", precheck.Diagnostic);
        Assert.Contains("NativeOwnerReleaseHookReady=False", precheck.Diagnostic);
        Assert.Contains("NativeOwnerInFlightDrainReady=False", precheck.Diagnostic);
        Assert.Contains("NoThrowNativeDestructorReady=True", precheck.Diagnostic);
        Assert.Contains("NativeOwnerLifecycleReady=False", precheck.Diagnostic);
    }

    [Fact]
    public void PublicNativeOwnerAddressGateSurfaceDoesNotExposeRawPointers()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtDebugListenerNativeOwnerAddressDesignGate),
            typeof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult)
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
    public void ReadinessSmokeAndDocsKeepNativeOwnerAddressGateSeparateFromRuntimeProof()
    {
        string gateSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerNativeOwnerAddressDesignGate.cs");
        string precheckSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerRuntimeProofPrecheck.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string gateDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-native-owner-address-design-gate.md");
        string preflightDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-native-attach-nothrow-preflight.md");
        string precheckDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-runtime-proof-precheck.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string index = ReadSource("docs", "index.md");
        string toc = ReadSource("docs", "toc.yml");
        string smokeReadme = ReadSource("smoke", "README.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("public static class TensorRtDebugListenerNativeOwnerAddressDesignGate", gateSource);
        Assert.Contains("public readonly struct TensorRtDebugListenerNativeOwnerAddressDesignGateResult", gateSource);
        Assert.Contains("EvidenceKind => \"debug-listener-native-owner-address-design-gate\"", gateSource);
        Assert.Contains("RuntimeEvidenceKind => \"design-gate\"", gateSource);
        Assert.Contains("RealCallbackRuntime => false", gateSource);
        Assert.Contains("IsRealCallbackRuntimeProof => false", gateSource);
        Assert.Contains("StableNativeOwnerAddressDesignReady", gateSource);
        Assert.Contains("NativeOwnerNonCopyableReady", gateSource);
        Assert.Contains("NativeOwnerDisposeOrderReady", gateSource);
        Assert.Contains("NativeOwnerReleaseHookReady", gateSource);
        Assert.Contains("NativeOwnerInFlightDrainReady", gateSource);
        Assert.Contains("NoThrowNativeDestructorReady", gateSource);
        Assert.Contains("NativeOwnerLifecycleReady", gateSource);
        Assert.Contains("CanImplementNativeAttach", gateSource);
        Assert.DoesNotContain("public IntPtr", gateSource);
        Assert.DoesNotContain("public nint", gateSource);

        Assert.Contains("TensorRtDebugListenerNativeOwnerAddressDesignGate", precheckSource);
        Assert.Contains("NativeOwnerAddressDesignGateReady", precheckSource);
        Assert.Contains("NativeOwnerLifecycleReady", precheckSource);

        Assert.Contains("debug-listener-native-owner-address-design-gate", smokeProgram);
        Assert.Contains("DebugListenerNativeOwnerAddressDesignGate=", smokeProgram);
        Assert.Contains("NativeOwnerLifecycleReady", smokeProgram);

        Assert.Contains("debugListenerNativeOwnerAddressDesignGate", readiness);
        Assert.Contains("New-DebugListenerNativeOwnerAddressDesignGateEvidence", readiness);
        Assert.Contains("debug-listener-native-owner-address-design-gate", readiness);
        Assert.Contains("design-gate-ready", readiness);
        Assert.Contains("NativeOwnerAddressDesignGateReady", readiness);
        Assert.Contains("canImplementNativeAttach = $false", readiness);
        Assert.Contains("canAttemptRuntimeProof = $false", readiness);
        Assert.Contains("runtimeProofBlocked = $true", readiness);

        Assert.Contains("debug-listener-native-owner-address-design-gate", packageConsumer);
        Assert.Contains("debug-listener-native-owner-address-design-gate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeOwnerAddressDesignGate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeOwnerAddressDesignGateResult", bridgeConsumer);

        Assert.Contains("debug-listener-native-owner-address-design-gate", gateDoc);
        Assert.Contains("RuntimeEvidenceKind=design-gate", gateDoc);
        Assert.Contains("DesignGateReady=True", gateDoc);
        Assert.Contains("NativeAttachNoThrowPreflightReady=True", gateDoc);
        Assert.Contains("StableNativeOwnerAddressDesignReady=False", gateDoc);
        Assert.Contains("NativeOwnerNonCopyableReady=False", gateDoc);
        Assert.Contains("NativeOwnerDisposeOrderReady=False", gateDoc);
        Assert.Contains("NativeOwnerReleaseHookReady=False", gateDoc);
        Assert.Contains("NativeOwnerInFlightDrainReady=False", gateDoc);
        Assert.Contains("NoThrowNativeDestructorReady=False", gateDoc);
        Assert.Contains("NativeOwnerLifecycleReady=False", gateDoc);
        Assert.Contains("CanImplementNativeAttach=False", gateDoc);
        Assert.Contains("not proof", gateDoc);
        Assert.Contains("debug-listener-native-owner-address-design-gate", preflightDoc);
        Assert.Contains("debug-listener-native-owner-address-design-gate", precheckDoc);
        Assert.Contains("debug-listener-native-owner-address-design-gate", trampolineGate);
        Assert.Contains("debug-listener-native-owner-address-design-gate", schema);
        Assert.Contains("debug-listener-native-owner-address-design-gate", latest);
        Assert.Contains("debug-listener-native-owner-address-design-gate", index);
        Assert.Contains("debug-listener-native-owner-address-design-gate.md", toc);
        Assert.Contains("debug-listener-native-owner-address-design-gate", smokeReadme);

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

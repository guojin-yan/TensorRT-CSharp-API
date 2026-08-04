using System.Reflection;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerNativeNoThrowVTableDesignGateTests
{
    [Fact]
    public void NativeNoThrowVTableDesignGateCopiesOwnerAddressStateWithoutRuntimeProof()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-native-nothrow-vtable-design-gate",
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
        TensorRtDebugListenerNativeNoThrowVTableDesignGateResult gate =
            TensorRtDebugListenerNativeNoThrowVTableDesignGate.Evaluate(disposed, attachDetachGate, borrowedTensorGate, attachVTableGate, preflight, ownerAddressGate);

        Assert.Equal("debug-listener-native-nothrow-vtable-design-gate", gate.EvidenceKind);
        Assert.Equal("debug-listener-process-debug-tensor", gate.CallbackKind);
        Assert.Equal("design-gate", gate.RuntimeEvidenceKind);
        Assert.False(gate.RealCallbackRuntime);
        Assert.False(gate.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, gate.Line);
        Assert.True(gate.NativeOwnerAddressDesignGateReady);
        Assert.True(gate.NativeAttachNoThrowPreflightReady);
        Assert.False(gate.NativeAttachEntryLocated);
        Assert.False(gate.NativeOwnerLifecycleReady);
        Assert.True(gate.ManagedCallbackKeepAliveDesignReady);
        Assert.False(gate.NoThrowNativeDestructorReady);
        Assert.False(gate.NoThrowVTableDesignReady);
        Assert.False(gate.ExceptionToStatusMappingDesignReady);
        Assert.True(gate.BorrowedDebugTensorMetadataCopyDesignReady);
        Assert.True(gate.BorrowedDebugTensorPointerEscapeBlocked);
        Assert.False(gate.NativeVTableTrampolineReady);
        Assert.False(gate.CallbackExceptionCaptureReady);
        Assert.False(gate.CallbackStatusMappingReady);
        Assert.False(gate.CallbackInFlightAccountingReady);
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
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("no-throw vtable", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("exception-to-status", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("in-flight accounting", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("IDebugListener::processDebugTensor", StringComparison.Ordinal));
        Assert.Contains("RuntimeEvidenceKind=design-gate", gate.Diagnostic);
        Assert.Contains("DesignGateReady=True", gate.Diagnostic);
        Assert.Contains("NativeOwnerAddressDesignGateReady=True", gate.Diagnostic);
        Assert.Contains("NoThrowVTableDesignReady=False", gate.Diagnostic);
        Assert.Contains("ExceptionToStatusMappingDesignReady=False", gate.Diagnostic);
        Assert.Contains("NativeVTableTrampolineReady=False", gate.Diagnostic);
        Assert.Contains("CallbackExceptionCaptureReady=False", gate.Diagnostic);
        Assert.Contains("CallbackStatusMappingReady=False", gate.Diagnostic);
        Assert.Contains("CallbackInFlightAccountingReady=False", gate.Diagnostic);
        Assert.Contains("CanImplementNativeAttach=False", gate.Diagnostic);
        Assert.Contains("RuntimeProofBlocked=True", gate.Diagnostic);
    }

    [Fact]
    public void RuntimeProofPrecheckConsumesNativeNoThrowVTableGateAndStaysBlocked()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-native-nothrow-vtable-design-gate-precheck",
            isInput: true,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        TensorRtDebugListenerRuntimeProofPrecheckResult precheck =
            TensorRtDebugListenerRuntimeProofPrecheck.Evaluate(owner.GetSnapshot("post-dispose"));

        Assert.True(precheck.NativeNoThrowVTableDesignGateReady);
        Assert.False(precheck.NoThrowVTableDesignReady);
        Assert.False(precheck.ExceptionToStatusMappingDesignReady);
        Assert.False(precheck.NativeVTableTrampolineReady);
        Assert.False(precheck.CallbackExceptionCaptureReady);
        Assert.False(precheck.CallbackStatusMappingReady);
        Assert.False(precheck.CallbackInFlightAccountingReady);
        Assert.False(precheck.CanImplementNativeAttach);
        Assert.False(precheck.CanAttemptRuntimeProof);
        Assert.True(precheck.RuntimeProofBlocked);
        Assert.Contains("NativeNoThrowVTableDesignGateReady=True", precheck.Diagnostic);
        Assert.Contains("NativeVTableTrampolineReady=False", precheck.Diagnostic);
        Assert.Contains("CallbackExceptionCaptureReady=False", precheck.Diagnostic);
        Assert.Contains("CallbackStatusMappingReady=False", precheck.Diagnostic);
        Assert.Contains("CallbackInFlightAccountingReady=False", precheck.Diagnostic);
    }

    [Fact]
    public void PublicNativeNoThrowVTableGateSurfaceDoesNotExposeRawPointers()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtDebugListenerNativeNoThrowVTableDesignGate),
            typeof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult)
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
    public void ReadinessSmokeAndDocsKeepNativeNoThrowVTableGateSeparateFromRuntimeProof()
    {
        string gateSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerNativeNoThrowVTableDesignGate.cs");
        string precheckSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerRuntimeProofPrecheck.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string gateDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-native-nothrow-vtable-design-gate.md");
        string ownerAddressDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-native-owner-address-design-gate.md");
        string precheckDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-runtime-proof-precheck.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string index = ReadSource("docs", "index.md");
        string toc = ReadSource("docs", "toc.yml");
        string smokeReadme = ReadSource("smoke", "README.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("public static class TensorRtDebugListenerNativeNoThrowVTableDesignGate", gateSource);
        Assert.Contains("public readonly struct TensorRtDebugListenerNativeNoThrowVTableDesignGateResult", gateSource);
        Assert.Contains("EvidenceKind => \"debug-listener-native-nothrow-vtable-design-gate\"", gateSource);
        Assert.Contains("RuntimeEvidenceKind => \"design-gate\"", gateSource);
        Assert.Contains("RealCallbackRuntime => false", gateSource);
        Assert.Contains("IsRealCallbackRuntimeProof => false", gateSource);
        Assert.Contains("NativeVTableTrampolineReady", gateSource);
        Assert.Contains("CallbackExceptionCaptureReady", gateSource);
        Assert.Contains("CallbackStatusMappingReady", gateSource);
        Assert.Contains("CallbackInFlightAccountingReady", gateSource);
        Assert.DoesNotContain("public IntPtr", gateSource);
        Assert.DoesNotContain("public nint", gateSource);

        Assert.Contains("TensorRtDebugListenerNativeNoThrowVTableDesignGate", precheckSource);
        Assert.Contains("NativeNoThrowVTableDesignGateReady", precheckSource);
        Assert.Contains("CallbackExceptionCaptureReady", precheckSource);

        Assert.Contains("debug-listener-native-nothrow-vtable-design-gate", smokeProgram);
        Assert.Contains("DebugListenerNativeNoThrowVTableDesignGate=", smokeProgram);
        Assert.Contains("NativeNoThrowVTableDesignGateReady", smokeProgram);

        Assert.Contains("debugListenerNativeNoThrowVTableDesignGate", readiness);
        Assert.Contains("New-DebugListenerNativeNoThrowVTableDesignGateEvidence", readiness);
        Assert.Contains("debug-listener-native-nothrow-vtable-design-gate", readiness);
        Assert.Contains("NativeNoThrowVTableDesignGateReady", readiness);
        Assert.Contains("CallbackExceptionCaptureReady=False", readiness);
        Assert.Contains("CallbackStatusMappingReady=False", readiness);
        Assert.Contains("CallbackInFlightAccountingReady=False", readiness);
        Assert.Contains("canImplementNativeAttach = $false", readiness);
        Assert.Contains("canAttemptRuntimeProof = $false", readiness);
        Assert.Contains("runtimeProofBlocked = $true", readiness);

        Assert.Contains("debug-listener-native-nothrow-vtable-design-gate", packageConsumer);
        Assert.Contains("debug-listener-native-nothrow-vtable-design-gate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeNoThrowVTableDesignGate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerNativeNoThrowVTableDesignGateResult", bridgeConsumer);

        Assert.Contains("debug-listener-native-nothrow-vtable-design-gate", gateDoc);
        Assert.Contains("RuntimeEvidenceKind=design-gate", gateDoc);
        Assert.Contains("NativeOwnerAddressDesignGateReady=True", gateDoc);
        Assert.Contains("NoThrowNativeDestructorReady=False", gateDoc);
        Assert.Contains("NoThrowVTableDesignReady=False", gateDoc);
        Assert.Contains("ExceptionToStatusMappingDesignReady=False", gateDoc);
        Assert.Contains("NativeVTableTrampolineReady=False", gateDoc);
        Assert.Contains("CallbackExceptionCaptureReady=False", gateDoc);
        Assert.Contains("CallbackStatusMappingReady=False", gateDoc);
        Assert.Contains("CallbackInFlightAccountingReady=False", gateDoc);
        Assert.Contains("CanImplementNativeAttach=False", gateDoc);
        Assert.Contains("not proof", gateDoc);
        Assert.Contains("debug-listener-native-nothrow-vtable-design-gate", ownerAddressDoc);
        Assert.Contains("debug-listener-native-nothrow-vtable-design-gate", precheckDoc);
        Assert.Contains("debug-listener-native-nothrow-vtable-design-gate", trampolineGate);
        Assert.Contains("debug-listener-native-nothrow-vtable-design-gate", schema);
        Assert.Contains("debug-listener-native-nothrow-vtable-design-gate", latest);
        Assert.Contains("debug-listener-native-nothrow-vtable-design-gate", index);
        Assert.Contains("debug-listener-native-nothrow-vtable-design-gate.md", toc);
        Assert.Contains("debug-listener-native-nothrow-vtable-design-gate", smokeReadme);

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

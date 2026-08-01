using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerAttachVTableSafetyGateTests
{
    [Fact]
    public void GateCopiesAttachVTableStateWithoutPointerExposureOrRuntimeProof()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-attach-vtable-safety-gate",
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
        TensorRtDebugListenerAttachVTableSafetyGateResult gate =
            TensorRtDebugListenerAttachVTableSafetyGate.Evaluate(disposed, attachDetachGate, borrowedTensorGate);

        Assert.Equal("debug-listener-attach-vtable-safety-gate", gate.EvidenceKind);
        Assert.Equal("debug-listener-attach-vtable", gate.CallbackKind);
        Assert.Equal("design-gate", gate.RuntimeEvidenceKind);
        Assert.False(gate.RealCallbackRuntime);
        Assert.False(gate.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, gate.Line);
        Assert.True(gate.LineSupportsDebugListener);
        Assert.True(gate.OwnerDesignReady);
        Assert.True(gate.AttachDetachDesignGateReady);
        Assert.True(gate.BorrowedTensorSafetyGateReady);
        Assert.True(gate.ManagedOwnerStateMachineReady);
        Assert.True(gate.DebugTensorMetadataCopied);
        Assert.True(gate.PointerFreeSurfaceReady);
        Assert.True(gate.BorrowedDebugTensorPointerEscapeBlocked);
        Assert.True(gate.DetachClearControlAvailable);
        Assert.False(gate.AttachControlAvailable);
        Assert.False(gate.LineSpecificAttachDetachReady);
        Assert.False(gate.StableNativeOwnerAddressReady);
        Assert.False(gate.NoThrowNativeVTableReady);
        Assert.False(gate.NativeVTableReady);
        Assert.False(gate.ExceptionToStatusMappingReady);
        Assert.False(gate.BorrowedDebugTensorLifetimeReady);
        Assert.False(gate.BorrowedDebugTensorDataLifetimeReady);
        Assert.False(gate.ProcessDebugTensorRuntimeReady);
        Assert.False(gate.FullPackageConsumerRuntimeEvidenceReady);
        Assert.True(gate.SafetyGateReady);
        Assert.True(gate.DeferredRowsStillRequired);
        Assert.False(gate.CanAttemptRuntimeProof);
        Assert.True(gate.RuntimeProofBlocked);
        Assert.Equal("safety-gate-ready", gate.Status);
        Assert.True(gate.BlockedPrerequisiteCount >= 6);
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("setDebugListener(non-null)", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("stable native DebugListener owner address", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("no-throw vtable", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("exception-to-status mapping", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("IDebugListener::processDebugTensor runtime callback", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("real-callback-runtime debug listener attach/vtable", StringComparison.Ordinal));
        Assert.Contains("SafetyGateReady=True", gate.Diagnostic);
        Assert.Contains("AttachControlAvailable=False", gate.Diagnostic);
        Assert.Contains("StableNativeOwnerAddressReady=False", gate.Diagnostic);
        Assert.Contains("NoThrowNativeVTableReady=False", gate.Diagnostic);
        Assert.Contains("ExceptionToStatusMappingReady=False", gate.Diagnostic);
        Assert.Contains("ProcessDebugTensorRuntimeReady=False", gate.Diagnostic);
        Assert.Contains("RuntimeProofBlocked=True", gate.Diagnostic);
    }

    [Fact]
    public void PublicGateSurfaceDoesNotExposeRawPointers()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtDebugListenerAttachVTableSafetyGate),
            typeof(TensorRtDebugListenerAttachVTableSafetyGateResult)
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
    public void ReadinessSmokeAndDocsKeepGateSeparateFromRuntimeProof()
    {
        string gateSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerAttachVTableSafetyGate.cs");
        string precheckSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerRuntimeProofPrecheck.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string gateDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-attach-vtable-safety-gate.md");
        string precheckDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-runtime-proof-precheck.md");
        string attachDetachDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-attach-detach-design-gate.md");
        string borrowedTensorDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-borrowed-tensor-safety-gate.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string index = ReadSource("docs", "index.md");
        string toc = ReadSource("docs", "toc.yml");
        string smokeReadme = ReadSource("smoke", "README.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("public static class TensorRtDebugListenerAttachVTableSafetyGate", gateSource);
        Assert.Contains("public readonly struct TensorRtDebugListenerAttachVTableSafetyGateResult", gateSource);
        Assert.Contains("EvidenceKind => \"debug-listener-attach-vtable-safety-gate\"", gateSource);
        Assert.Contains("CallbackKind => \"debug-listener-attach-vtable\"", gateSource);
        Assert.Contains("RuntimeEvidenceKind => \"design-gate\"", gateSource);
        Assert.Contains("RealCallbackRuntime => false", gateSource);
        Assert.Contains("IsRealCallbackRuntimeProof => false", gateSource);
        Assert.Contains("AttachControlAvailable", gateSource);
        Assert.Contains("StableNativeOwnerAddressReady", gateSource);
        Assert.Contains("NoThrowNativeVTableReady", gateSource);
        Assert.Contains("ExceptionToStatusMappingReady", gateSource);
        Assert.Contains("ProcessDebugTensorRuntimeReady", gateSource);
        Assert.Contains("FullPackageConsumerRuntimeEvidenceReady", gateSource);
        Assert.DoesNotContain("public IntPtr", gateSource);
        Assert.DoesNotContain("public nint", gateSource);

        Assert.Contains("TensorRtDebugListenerAttachVTableSafetyGate", precheckSource);
        Assert.Contains("AttachVTableSafetyGateReady", precheckSource);
        Assert.Contains("ExceptionToStatusMappingReady", precheckSource);

        Assert.Contains("debug-listener-attach-vtable-safety-gate", smokeProgram);
        Assert.Contains("DebugListenerAttachVTableSafetyGate=", smokeProgram);
        Assert.Contains("ExceptionToStatusMappingReady", smokeProgram);

        Assert.Contains("debugListenerAttachVTableSafetyGate", readiness);
        Assert.Contains("New-DebugListenerAttachVTableSafetyGateEvidence", readiness);
        Assert.Contains("debug-listener-attach-vtable-safety-gate", readiness);
        Assert.Contains("safety-gate-ready", readiness);
        Assert.Contains("canAttemptRuntimeProof = $false", readiness);
        Assert.Contains("runtimeProofBlocked = $true", readiness);
        Assert.Contains("isRealCallbackRuntimeProof = $false", readiness);

        Assert.Contains("debug-listener-attach-vtable-safety-gate", packageConsumer);
        Assert.Contains("debug-listener-attach-vtable-safety-gate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerAttachVTableSafetyGate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerAttachVTableSafetyGateResult", bridgeConsumer);

        Assert.Contains("debug-listener-attach-vtable-safety-gate", gateDoc);
        Assert.Contains("RuntimeEvidenceKind=design-gate", gateDoc);
        Assert.Contains("AttachControlAvailable=False", gateDoc);
        Assert.Contains("StableNativeOwnerAddressReady=False", gateDoc);
        Assert.Contains("NoThrowNativeVTableReady=False", gateDoc);
        Assert.Contains("ExceptionToStatusMappingReady=False", gateDoc);
        Assert.Contains("ProcessDebugTensorRuntimeReady=False", gateDoc);
        Assert.Contains("not proof", gateDoc);
        Assert.Contains("debug-listener-attach-vtable-safety-gate", precheckDoc);
        Assert.Contains("debug-listener-attach-vtable-safety-gate", attachDetachDoc);
        Assert.Contains("debug-listener-attach-vtable-safety-gate", borrowedTensorDoc);
        Assert.Contains("debug-listener-attach-vtable-safety-gate", trampolineGate);
        Assert.Contains("debug-listener-attach-vtable-safety-gate", schema);
        Assert.Contains("debug-listener-attach-vtable-safety-gate", latest);
        Assert.Contains("debug-listener-attach-vtable-safety-gate", index);
        Assert.Contains("debug-listener-attach-vtable-safety-gate.md", toc);
        Assert.Contains("debug-listener-attach-vtable-safety-gate", smokeReadme);

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

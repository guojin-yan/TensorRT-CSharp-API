using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerAttachDetachDesignGateTests
{
    [Fact]
    public void GateCopiesAttachDetachReadinessWithoutPointerExposureOrRuntimeProof()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "quality-debug-listener-attach-detach-design-gate",
            isInput: true,
            isExecutionTensor: true);

        TensorRtDebugListenerCallbackOwnerSnapshot diagnostic = owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        Assert.True(diagnostic.DebugTensorMetadataCopied);

        owner.Dispose();
        TensorRtDebugListenerCallbackOwnerSnapshot disposed = owner.GetSnapshot("post-dispose");
        TensorRtDebugListenerAttachDetachDesignGateResult gate =
            TensorRtDebugListenerAttachDetachDesignGate.Evaluate(disposed);

        Assert.Equal("debug-listener-attach-detach-design-gate", gate.EvidenceKind);
        Assert.Equal("debug-listener-attach-detach", gate.CallbackKind);
        Assert.Equal("design-gate", gate.RuntimeEvidenceKind);
        Assert.False(gate.RealCallbackRuntime);
        Assert.False(gate.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, gate.Line);
        Assert.True(gate.LineSupportsDebugListener);
        Assert.True(gate.OwnerDesignReady);
        Assert.True(gate.ManagedOwnerStateMachineReady);
        Assert.True(gate.DebugTensorMetadataCopied);
        Assert.True(gate.PointerFreeSurfaceReady);
        Assert.False(gate.AttachControlAvailable);
        Assert.True(gate.DetachClearControlAvailable);
        Assert.False(gate.LineSpecificAttachDetachReady);
        Assert.False(gate.StableNativeOwnerAddressReady);
        Assert.False(gate.NoThrowNativeVTableReady);
        Assert.False(gate.NativeVTableReady);
        Assert.False(gate.BorrowedDebugTensorLifetimeReady);
        Assert.True(gate.DesignGateReady);
        Assert.False(gate.CanAttemptRuntimeProof);
        Assert.True(gate.RuntimeProofBlocked);
        Assert.True(gate.DeferredRowsStillRequired);
        Assert.Equal("design-gate-ready", gate.Status);
        Assert.True(gate.BlockedPrerequisiteCount >= 4);
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("setDebugListener(non-null)", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("stable address", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("no-throw vtable", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("lifetime rules", StringComparison.Ordinal));
        Assert.Contains("AttachControlAvailable=False", gate.Diagnostic);
        Assert.Contains("DetachClearControlAvailable=True", gate.Diagnostic);
        Assert.Contains("LineSpecificAttachDetachReady=False", gate.Diagnostic);
    }

    [Fact]
    public void PublicGateSurfaceDoesNotExposeRawPointers()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtDebugListenerAttachDetachDesignGate),
            typeof(TensorRtDebugListenerAttachDetachDesignGateResult)
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
        string gateSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerAttachDetachDesignGate.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string gateDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-attach-detach-design-gate.md");
        string precheckDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-runtime-proof-precheck.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string runtimeSplitReadme = ReadSource("pack", "runtime-split", "README.md");
        string smokeReadme = ReadSource("smoke", "README.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("public static class TensorRtDebugListenerAttachDetachDesignGate", gateSource);
        Assert.Contains("public readonly struct TensorRtDebugListenerAttachDetachDesignGateResult", gateSource);
        Assert.Contains("RuntimeEvidenceKind => \"design-gate\"", gateSource);
        Assert.Contains("RealCallbackRuntime => false", gateSource);
        Assert.Contains("IsRealCallbackRuntimeProof => false", gateSource);
        Assert.Contains("AttachControlAvailable", gateSource);
        Assert.Contains("DetachClearControlAvailable", gateSource);
        Assert.Contains("LineSpecificAttachDetachReady", gateSource);
        Assert.Contains("NativeVTableReady", gateSource);
        Assert.Contains("RuntimeProofBlocked", gateSource);
        Assert.DoesNotContain("public IntPtr", gateSource);
        Assert.DoesNotContain("public nint", gateSource);

        Assert.Contains("debug-listener-attach-detach-design-gate", smokeProgram);
        Assert.Contains("DebugListenerAttachDetachDesignGate=", smokeProgram);
        Assert.Contains("AttachControlAvailable", smokeProgram);
        Assert.Contains("DetachClearControlAvailable", smokeProgram);

        Assert.Contains("debugListenerAttachDetachDesignGate", readiness);
        Assert.Contains("New-DebugListenerAttachDetachDesignGateEvidence", readiness);
        Assert.Contains("design-gate-ready", readiness);
        Assert.Contains("isRealCallbackRuntimeProof = $false", readiness);

        Assert.Contains("debug-listener-attach-detach-design-gate", packageConsumer);
        Assert.Contains("debug-listener-attach-detach-design-gate", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerAttachDetachDesignGate", bridgeConsumer);

        Assert.Contains("debug-listener-attach-detach-design-gate", gateDoc);
        Assert.Contains("RuntimeEvidenceKind=design-gate", gateDoc);
        Assert.Contains("AttachControlAvailable=False", gateDoc);
        Assert.Contains("DetachClearControlAvailable=True", gateDoc);
        Assert.Contains("LineSpecificAttachDetachReady=False", gateDoc);
        Assert.Contains("RuntimeProofBlocked=True", gateDoc);
        Assert.Contains("not proof", gateDoc);
        Assert.Contains("debug-listener-attach-detach-design-gate", precheckDoc);
        Assert.Contains("debug-listener-attach-detach-design-gate", trampolineGate);
        Assert.Contains("debug-listener-attach-detach-design-gate", schema);
        Assert.Contains("debug-listener-attach-detach-design-gate", latest);
        Assert.Contains("debugListenerAttachDetachDesignGate", runtimeSplitReadme);
        Assert.Contains("debug-listener-attach-detach-design-gate", smokeReadme);

        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IOutputAllocator\",\"notifyShape\",\"IOutputAllocator::notifyShape\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IOutputAllocator\",\"reallocateOutput\",\"IOutputAllocator::reallocateOutput\",\"other\",\"deferred-only\"", comparison);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}

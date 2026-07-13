using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OutputAllocatorAttachDetachDesignGateTests
{
    [Fact]
    public void GateCopiesAttachDetachReadinessWithoutPointerExposureOrRuntimeProof()
    {
        using TensorRtOutputAllocatorCallbackOwner owner = new TensorRtOutputAllocatorCallbackOwner();
        TensorRtOutputAllocatorCallbackRequest request = new TensorRtOutputAllocatorCallbackRequest(
            "quality_output_allocator",
            8192UL,
            256UL,
            new long[] { 1, 16, 32 },
            "quality-output-allocator-attach-detach-design-gate",
            hasCurrentMemory: true);

        TensorRtOutputAllocatorCallbackOwnerSnapshot diagnostic =
            owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request, 0UL);
        Assert.Equal(1, diagnostic.NotifyShapeCount);
        Assert.Equal(1, diagnostic.ReallocateOutputCount);

        owner.Dispose();
        TensorRtOutputAllocatorCallbackOwnerSnapshot disposed = owner.GetSnapshot("post-dispose");
        TensorRtOutputAllocatorAttachDetachDesignGateResult gate =
            TensorRtOutputAllocatorAttachDetachDesignGate.Evaluate(disposed);

        Assert.Equal("output-allocator-attach-detach-design-gate", gate.EvidenceKind);
        Assert.Equal("output-allocator-attach-detach", gate.CallbackKind);
        Assert.Equal("design-gate", gate.RuntimeEvidenceKind);
        Assert.False(gate.RealCallbackRuntime);
        Assert.False(gate.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, gate.Line);
        Assert.True(gate.LineSupportsOutputAllocator);
        Assert.True(gate.OwnerDesignReady);
        Assert.True(gate.ManagedOwnerStateMachineReady);
        Assert.True(gate.PointerFreeSurfaceReady);
        Assert.False(gate.AttachControlAvailable);
        Assert.True(gate.DetachClearControlAvailable);
        Assert.False(gate.LineSpecificAttachDetachReady);
        Assert.False(gate.StableNativeOwnerAddressReady);
        Assert.False(gate.NoThrowNativeVTableReady);
        Assert.False(gate.NativeVTableReady);
        Assert.False(gate.OutputBufferOwnershipRuntimeReady);
        Assert.True(gate.DesignGateReady);
        Assert.False(gate.CanAttemptRuntimeProof);
        Assert.True(gate.RuntimeProofBlocked);
        Assert.True(gate.DeferredRowsStillRequired);
        Assert.Equal("design-gate-ready", gate.Status);
        Assert.True(gate.BlockedPrerequisiteCount >= 4);
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("setOutputAllocator(non-null)", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("stable address", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("no-throw vtable", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("output buffer ownership", StringComparison.Ordinal));
        Assert.Contains("AttachControlAvailable=False", gate.Diagnostic);
        Assert.Contains("DetachClearControlAvailable=True", gate.Diagnostic);
        Assert.Contains("LineSpecificAttachDetachReady=False", gate.Diagnostic);
        Assert.Contains("OutputBufferOwnershipRuntimeReady=False", gate.Diagnostic);
    }

    [Fact]
    public void PublicGateSurfaceDoesNotExposeRawPointers()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtOutputAllocatorAttachDetachDesignGate),
            typeof(TensorRtOutputAllocatorAttachDetachDesignGateResult)
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
        string gateSource = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtOutputAllocatorAttachDetachDesignGate.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string gateDoc = ReadSource("docs", "articles", "zh-cn", "output-allocator-attach-detach-design-gate.md");
        string precheckDoc = ReadSource("docs", "articles", "zh-cn", "output-allocator-runtime-proof-precheck.md");
        string designDoc = ReadSource("docs", "articles", "zh-cn", "output-allocator-callback-owner-design.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string runtimeSplitReadme = ReadSource("pack", "runtime-split", "README.md");
        string smokeReadme = ReadSource("smoke", "README.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("public static class TensorRtOutputAllocatorAttachDetachDesignGate", gateSource);
        Assert.Contains("public readonly struct TensorRtOutputAllocatorAttachDetachDesignGateResult", gateSource);
        Assert.Contains("RuntimeEvidenceKind => \"design-gate\"", gateSource);
        Assert.Contains("RealCallbackRuntime => false", gateSource);
        Assert.Contains("IsRealCallbackRuntimeProof => false", gateSource);
        Assert.Contains("AttachControlAvailable", gateSource);
        Assert.Contains("DetachClearControlAvailable", gateSource);
        Assert.Contains("LineSpecificAttachDetachReady", gateSource);
        Assert.Contains("NativeVTableReady", gateSource);
        Assert.Contains("OutputBufferOwnershipRuntimeReady", gateSource);
        Assert.Contains("RuntimeProofBlocked", gateSource);
        Assert.DoesNotContain("public IntPtr", gateSource);
        Assert.DoesNotContain("public nint", gateSource);

        Assert.Contains("output-allocator-attach-detach-design-gate", smokeProgram);
        Assert.Contains("OutputAllocatorAttachDetachDesignGate=", smokeProgram);
        Assert.Contains("AttachControlAvailable", smokeProgram);
        Assert.Contains("DetachClearControlAvailable", smokeProgram);

        Assert.Contains("outputAllocatorAttachDetachDesignGate", readiness);
        Assert.Contains("New-OutputAllocatorAttachDetachDesignGateEvidence", readiness);
        Assert.Contains("output-allocator-attach-detach-design-gate", readiness);
        Assert.Contains("design-gate-ready", readiness);
        Assert.Contains("isRealCallbackRuntimeProof = $false", readiness);

        Assert.Contains("output-allocator-attach-detach-design-gate", packageConsumer);
        Assert.Contains("output-allocator-attach-detach-design-gate", bridgeConsumer);
        Assert.Contains("TensorRtOutputAllocatorAttachDetachDesignGate", bridgeConsumer);
        Assert.Contains("TensorRtOutputAllocatorAttachDetachDesignGateResult", bridgeConsumer);

        Assert.Contains("output-allocator-attach-detach-design-gate", gateDoc);
        Assert.Contains("RuntimeEvidenceKind=design-gate", gateDoc);
        Assert.Contains("AttachControlAvailable=False", gateDoc);
        Assert.Contains("DetachClearControlAvailable=True", gateDoc);
        Assert.Contains("LineSpecificAttachDetachReady=False", gateDoc);
        Assert.Contains("OutputBufferOwnershipRuntimeReady=False", gateDoc);
        Assert.Contains("RuntimeProofBlocked=True", gateDoc);
        Assert.Contains("not proof", gateDoc);
        Assert.Contains("output-allocator-attach-detach-design-gate", precheckDoc);
        Assert.Contains("output-allocator-attach-detach-design-gate", designDoc);
        Assert.Contains("output-allocator-attach-detach-design-gate", trampolineGate);
        Assert.Contains("output-allocator-attach-detach-design-gate", schema);
        Assert.Contains("output-allocator-attach-detach-design-gate", latest);
        Assert.Contains("outputAllocatorAttachDetachDesignGate", runtimeSplitReadme);
        Assert.Contains("output-allocator-attach-detach-design-gate", smokeReadme);

        Assert.Contains("\"IOutputAllocator\",\"notifyShape\",\"IOutputAllocator::notifyShape\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IOutputAllocator\",\"reallocateOutput\",\"IOutputAllocator::reallocateOutput\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}

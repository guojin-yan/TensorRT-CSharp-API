using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OutputBufferOwnershipSafetyGateTests
{
    [Fact]
    public void GateCopiesOwnershipReadinessWithoutPointerExposureOrRuntimeProof()
    {
        using TensorRtOutputAllocatorCallbackOwner owner = new TensorRtOutputAllocatorCallbackOwner();
        TensorRtOutputAllocatorCallbackRequest request = new TensorRtOutputAllocatorCallbackRequest(
            "quality_output_buffer",
            8192UL,
            256UL,
            new long[] { 1, 16, 32 },
            "quality-output-buffer-ownership-safety-gate",
            hasCurrentMemory: true);

        TensorRtOutputAllocatorCallbackOwnerSnapshot diagnostic =
            owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request, 0UL);
        Assert.Equal(1, diagnostic.NotifyShapeCount);
        Assert.Equal(1, diagnostic.ReallocateOutputCount);

        owner.Dispose();
        TensorRtOutputAllocatorCallbackOwnerSnapshot disposed = owner.GetSnapshot("post-dispose");
        TensorRtOutputAllocatorAttachDetachDesignGateResult attachDetachGate =
            TensorRtOutputAllocatorAttachDetachDesignGate.Evaluate(disposed);
        TensorRtOutputBufferOwnershipSafetyGateResult gate =
            TensorRtOutputBufferOwnershipSafetyGate.Evaluate(disposed, attachDetachGate);

        Assert.Equal("output-buffer-ownership-safety-gate", gate.EvidenceKind);
        Assert.Equal("output-allocator-output-buffer-ownership", gate.CallbackKind);
        Assert.Equal("design-gate", gate.RuntimeEvidenceKind);
        Assert.False(gate.RealCallbackRuntime);
        Assert.False(gate.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, gate.Line);
        Assert.Equal("quality_output_buffer", gate.TensorName);
        Assert.Equal(8192UL, gate.RequestedSize);
        Assert.Equal(256UL, gate.Alignment);
        Assert.Equal(3, gate.ShapeRank);
        Assert.True(gate.HasCurrentMemory);
        Assert.Equal(1, gate.NotifyShapeCount);
        Assert.Equal(1, gate.ReallocateOutputCount);
        Assert.True(gate.AttachDetachDesignGateReady);
        Assert.True(gate.OwnerDesignReady);
        Assert.True(gate.PointerFreeSurfaceReady);
        Assert.True(gate.CopiedCurrentMemoryMetadataReady);
        Assert.True(gate.CopiedShapeMetadataReady);
        Assert.True(gate.CopiedRequestMetadataReady);
        Assert.True(gate.SafetyGateReady);
        Assert.False(gate.OutputBufferOwnershipRuntimeReady);
        Assert.False(gate.CurrentMemoryReusePolicyReady);
        Assert.True(gate.BorrowedPointerEscapeBlocked);
        Assert.False(gate.OwnedDevicePointerReleasePolicyReady);
        Assert.False(gate.ShapeNotificationOrderingReady);
        Assert.False(gate.ReallocateOutputRuntimeReady);
        Assert.False(gate.FullPackageConsumerRuntimeEvidenceReady);
        Assert.True(gate.DeferredRowsStillRequired);
        Assert.False(gate.CanAttemptRuntimeProof);
        Assert.True(gate.RuntimeProofBlocked);
        Assert.Equal("safety-gate-ready", gate.Status);
        Assert.True(gate.BlockedPrerequisiteCount >= 5);
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("currentMemory reuse policy", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("owned device pointer release policy", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("notifyShape before reallocateOutput", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("reallocateOutput runtime callback", StringComparison.Ordinal));
        Assert.Contains(gate.BlockedPrerequisites, item => item.Contains("real-callback-runtime", StringComparison.Ordinal));
        Assert.Contains("SafetyGateReady=True", gate.Diagnostic);
        Assert.Contains("OutputBufferOwnershipRuntimeReady=False", gate.Diagnostic);
        Assert.Contains("CurrentMemoryReusePolicyReady=False", gate.Diagnostic);
        Assert.Contains("BorrowedPointerEscapeBlocked=True", gate.Diagnostic);
        Assert.Contains("ReallocateOutputRuntimeReady=False", gate.Diagnostic);
    }

    [Fact]
    public void PublicGateSurfaceDoesNotExposeRawPointers()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtOutputBufferOwnershipSafetyGate),
            typeof(TensorRtOutputBufferOwnershipSafetyGateResult)
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
        string gateSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "MemoryAllocation", "TensorRtOutputBufferOwnershipSafetyGate.cs");
        string precheckSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "MemoryAllocation", "TensorRtOutputAllocatorRuntimeProofPrecheck.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string gateDoc = ReadSource("docs", "articles", "zh-cn", "output-buffer-ownership-safety-gate.md");
        string precheckDoc = ReadSource("docs", "articles", "zh-cn", "output-allocator-runtime-proof-precheck.md");
        string attachDetachDoc = ReadSource("docs", "articles", "zh-cn", "output-allocator-attach-detach-design-gate.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string runtimeSplitReadme = ReadSource("pack", "runtime-split", "README.md");
        string smokeReadme = ReadSource("smoke", "README.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("public static class TensorRtOutputBufferOwnershipSafetyGate", gateSource);
        Assert.Contains("public readonly struct TensorRtOutputBufferOwnershipSafetyGateResult", gateSource);
        Assert.Contains("EvidenceKind => \"output-buffer-ownership-safety-gate\"", gateSource);
        Assert.Contains("RuntimeEvidenceKind => \"design-gate\"", gateSource);
        Assert.Contains("RealCallbackRuntime => false", gateSource);
        Assert.Contains("IsRealCallbackRuntimeProof => false", gateSource);
        Assert.Contains("CurrentMemoryReusePolicyReady", gateSource);
        Assert.Contains("BorrowedPointerEscapeBlocked", gateSource);
        Assert.Contains("OwnedDevicePointerReleasePolicyReady", gateSource);
        Assert.Contains("ShapeNotificationOrderingReady", gateSource);
        Assert.Contains("ReallocateOutputRuntimeReady", gateSource);
        Assert.DoesNotContain("public IntPtr", gateSource);
        Assert.DoesNotContain("public nint", gateSource);

        Assert.Contains("TensorRtOutputBufferOwnershipSafetyGate", precheckSource);
        Assert.Contains("OutputBufferOwnershipSafetyGateReady", precheckSource);
        Assert.Contains("CurrentMemoryReusePolicyReady", precheckSource);
        Assert.Contains("ReallocateOutputRuntimeReady", precheckSource);

        Assert.Contains("output-buffer-ownership-safety-gate", smokeProgram);
        Assert.Contains("OutputBufferOwnershipSafetyGate=", smokeProgram);
        Assert.Contains("CurrentMemoryReusePolicyReady", smokeProgram);
        Assert.Contains("BorrowedPointerEscapeBlocked", smokeProgram);

        Assert.Contains("outputBufferOwnershipSafetyGate", readiness);
        Assert.Contains("New-OutputBufferOwnershipSafetyGateEvidence", readiness);
        Assert.Contains("output-buffer-ownership-safety-gate", readiness);
        Assert.Contains("safety-gate-ready", readiness);
        Assert.Contains("isRealCallbackRuntimeProof = $false", readiness);

        Assert.Contains("output-buffer-ownership-safety-gate", packageConsumer);
        Assert.Contains("output-buffer-ownership-safety-gate", bridgeConsumer);
        Assert.Contains("TensorRtOutputBufferOwnershipSafetyGate", bridgeConsumer);
        Assert.Contains("TensorRtOutputBufferOwnershipSafetyGateResult", bridgeConsumer);

        Assert.Contains("output-buffer-ownership-safety-gate", gateDoc);
        Assert.Contains("RuntimeEvidenceKind=design-gate", gateDoc);
        Assert.Contains("OutputBufferOwnershipRuntimeReady=False", gateDoc);
        Assert.Contains("CurrentMemoryReusePolicyReady=False", gateDoc);
        Assert.Contains("BorrowedPointerEscapeBlocked=True", gateDoc);
        Assert.Contains("ReallocateOutputRuntimeReady=False", gateDoc);
        Assert.Contains("not proof", gateDoc);
        Assert.Contains("output-buffer-ownership-safety-gate", precheckDoc);
        Assert.Contains("output-buffer-ownership-safety-gate", attachDetachDoc);
        Assert.Contains("output-buffer-ownership-safety-gate", trampolineGate);
        Assert.Contains("output-buffer-ownership-safety-gate", schema);
        Assert.Contains("output-buffer-ownership-safety-gate", latest);
        Assert.Contains("outputBufferOwnershipSafetyGate", runtimeSplitReadme);
        Assert.Contains("output-buffer-ownership-safety-gate", smokeReadme);

        Assert.Contains("\"IOutputAllocator\",\"notifyShape\",\"IOutputAllocator::notifyShape\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IOutputAllocator\",\"reallocateOutput\",\"IOutputAllocator::reallocateOutput\",\"other\",\"deferred-only\"", comparison);
        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return RepositorySourceReader.Read(path);
    }
}

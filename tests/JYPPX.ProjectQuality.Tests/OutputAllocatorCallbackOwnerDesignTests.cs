using System.Reflection;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OutputAllocatorCallbackOwnerDesignTests
{
    [Fact]
    public void PublicDesignOwnerCopiesDiagnosticsWithoutPointerOwnershipOrRuntimeProof()
    {
        using TensorRtOutputAllocatorCallbackOwner owner = new TensorRtOutputAllocatorCallbackOwner();
        long[] shape = { 1, 1000 };
        TensorRtOutputAllocatorCallbackRequest request = new TensorRtOutputAllocatorCallbackRequest(
            "quality_output",
            4096UL,
            256UL,
            shape,
            "quality-output-owner-design",
            hasCurrentMemory: true);
        shape[0] = 99;

        TensorRtOutputAllocatorCallbackOwnerSnapshot snapshot = owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request, 0UL);

        Assert.Equal("output-allocator-callback-owner-design", snapshot.EvidenceKind);
        Assert.Equal("output-allocator-prototype", snapshot.CallbackKind);
        Assert.Equal("not-present", snapshot.RuntimeEvidenceKind);
        Assert.False(snapshot.RealCallbackRuntime);
        Assert.False(snapshot.IsRealCallbackRuntimeProof);
        Assert.False(snapshot.IsAttached);
        Assert.False(snapshot.OutputBufferPointerExposed);
        Assert.False(snapshot.OutputBufferPointerProduced);
        Assert.Equal("quality_output", snapshot.TensorName);
        Assert.Equal("[1x1000]", snapshot.ShapeSummary);
        Assert.Equal(4096UL, snapshot.RequestedSize);
        Assert.Equal(256UL, snapshot.Alignment);
        Assert.Equal(2, snapshot.InvocationCount);
        Assert.Equal(1, snapshot.NotifyShapeCount);
        Assert.Equal(1, snapshot.ReallocateOutputCount);
        Assert.Equal(0, snapshot.FailureCount);
        Assert.Equal(0, snapshot.InFlightCallbackCount);
        Assert.True(snapshot.CallbackStatePinned);
        Assert.True(snapshot.DelegatePinned);
        Assert.Contains("RealCallbackRuntime=False", snapshot.LastDiagnostic);
        Assert.Contains("IsRealCallbackRuntimeProof=False", snapshot.LastDiagnostic);

        if (snapshot.NativeLedgerAvailable)
        {
            Assert.Equal(BridgeStatusCode.Ok, snapshot.NativeLedgerStatus);
            Assert.Equal(2UL, snapshot.StateTransitionCount);
            Assert.Equal(1UL, snapshot.LedgerAllocationCount);
            Assert.Equal(1UL, snapshot.LedgerReleaseCount);
            Assert.Equal(0UL, snapshot.LedgerFailureCount);
            Assert.False(snapshot.HasLiveAllocation);
        }
        else
        {
            Assert.Equal(BridgeStatusCode.RuntimeError, snapshot.NativeLedgerStatus);
            Assert.Contains("native ledger diagnostic unavailable", snapshot.NativeLedgerDiagnostic);
        }

        owner.Dispose();
        TensorRtOutputAllocatorCallbackOwnerSnapshot disposed = owner.GetSnapshot("post-dispose");
        Assert.True(disposed.DisposeRequested);
        Assert.False(disposed.CallbackStatePinned);
        Assert.False(disposed.DelegatePinned);
        Assert.True(disposed.ReleaseHookCount >= 1);
        Assert.Contains("release hook", disposed.ReleaseDiagnostic);
    }

    [Fact]
    public void PublicDesignOwnerRejectsUnsafeRequestShape()
    {
        Assert.Throws<ArgumentException>(() => new TensorRtOutputAllocatorCallbackRequest("", 1UL, 1UL, Array.Empty<long>()));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TensorRtOutputAllocatorCallbackRequest("x", 1UL, 0UL, Array.Empty<long>()));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TensorRtOutputAllocatorCallbackRequest("x", 1UL, 1UL, new long[9]));
    }

    [Fact]
    public void PublicDesignOwnerSurfaceDoesNotExposeRawPointers()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtOutputAllocatorCallbackOwner),
            typeof(TensorRtOutputAllocatorCallbackRequest),
            typeof(TensorRtOutputAllocatorCallbackOwnerSnapshot)
        };

        foreach (Type type in publicTypes)
        {
            foreach (PropertyInfo property in type.GetProperties(BindingFlags.Instance | BindingFlags.Public))
            {
                Assert.NotEqual(typeof(IntPtr), property.PropertyType);
                Assert.NotEqual(typeof(UIntPtr), property.PropertyType);
            }

            foreach (MethodInfo method in type.GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.DeclaredOnly))
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
    public void ReadinessSmokeAndDocsKeepDesignGateSeparateFromRuntimeProof()
    {
        string ownerSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "MemoryAllocation", "TensorRtOutputAllocatorCallbackOwner.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string designDoc = ReadSource("docs", "articles", "zh-cn", "output-allocator-callback-owner-design.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string smokeReadme = ReadSource("smoke", "README.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("public sealed partial class TensorRtOutputAllocatorCallbackOwner", ownerSource);
        Assert.Contains("public readonly struct TensorRtOutputAllocatorCallbackOwnerSnapshot", ownerSource);
        Assert.Contains("public TensorRtOutputAllocatorCallbackOwnerSnapshot RunDesignDiagnostic", ownerSource);
        Assert.Contains("RuntimeEvidenceKind => \"not-present\"", ownerSource);
        Assert.Contains("RealCallbackRuntime => false", ownerSource);
        Assert.Contains("IsRealCallbackRuntimeProof => false", ownerSource);
        Assert.Contains("OutputBufferPointerExposed => false", ownerSource);
        Assert.Contains("OutputBufferPointerProduced => false", ownerSource);
        Assert.DoesNotContain("public IntPtr", ownerSource);
        Assert.DoesNotContain("public nint", ownerSource);

        Assert.Contains("output-allocator-callback-owner-design", smokeProgram);
        Assert.Contains("OutputAllocatorCallbackOwnerDesign=", smokeProgram);
        Assert.Contains("OutputAllocatorCallbackOwnerDesignDispose=", smokeProgram);
        Assert.Contains("RuntimeEvidenceKind", smokeProgram);
        Assert.Contains("IsRealCallbackRuntimeProof", smokeProgram);
        Assert.Contains("NativeLedgerAvailable", smokeProgram);
        Assert.Contains("StateTransitionCount", smokeProgram);
        Assert.Contains("LedgerAllocationCount", smokeProgram);

        Assert.Contains("outputAllocatorCallbackOwnerDesign", readiness);
        Assert.Contains("New-OutputAllocatorCallbackOwnerDesignEvidence", readiness);
        Assert.Contains("owner-design-gate", readiness);
        Assert.Contains("design-ready", readiness);
        Assert.Contains("isRealCallbackRuntimeProof = $false", readiness);

        Assert.Contains("output-allocator-callback-owner-design", designDoc);
        Assert.Contains("RuntimeEvidenceKind=not-present", designDoc);
        Assert.Contains("RealCallbackRuntime=False", designDoc);
        Assert.Contains("IsRealCallbackRuntimeProof=False", designDoc);
        Assert.Contains("not proof", designDoc);
        Assert.Contains("output-allocator-callback-owner-design", trampolineGate);
        Assert.Contains("output-allocator-callback-owner-design", schema);
        Assert.Contains("output-allocator-callback-owner-design", smokeReadme);

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

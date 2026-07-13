using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CallbackOwnerClosureMatrixTests
{
    [Fact]
    public void ClosureMatrixAggregatesCallbackFamiliesWithoutPromotingRuntimeProof()
    {
        TensorRtCallbackOwnerClosureMatrixResult matrix = CreateMatrix();

        Assert.Equal("callback-owner-closure-matrix", matrix.EvidenceKind);
        Assert.Equal("closure-matrix", matrix.RuntimeEvidenceKind);
        Assert.False(matrix.RealCallbackRuntime);
        Assert.False(matrix.IsRealCallbackRuntimeProof);
        Assert.Equal(5, matrix.FamilyCount);
        Assert.Equal(0, matrix.ClosureReadyFamilyCount);
        Assert.Equal(0, matrix.RuntimeProofAttemptReadyFamilyCount);
        Assert.Equal(0, matrix.PackageConsumerRuntimeProofReadyFamilyCount);
        Assert.True(matrix.PointerFreeSurfaceReady);
        Assert.False(matrix.AllFamiliesClosureReady);
        Assert.False(matrix.CanAttemptRuntimeProof);
        Assert.True(matrix.RuntimeProofBlocked);
        Assert.True(matrix.DeferredRowsStillRequired);
        Assert.True(matrix.BlockedPrerequisiteCount > 0);
        Assert.Contains("RealCallbackRuntime=False", matrix.Summary);
        Assert.Contains("IsRealCallbackRuntimeProof=False", matrix.Summary);

        TensorRtCallbackOwnerClosureMatrixRow debug = matrix.Rows.Single(row => row.OwnerFamily == "DebugListener");
        Assert.Equal("debug-listener-process-debug-tensor", debug.CallbackKind);
        Assert.Contains("IDebugListener::processDebugTensor", debug.CallbackMethods);
        Assert.True(debug.DesignGateReady);
        Assert.True(debug.NativeNonCopyableOwnerStorageReady);
        Assert.True(debug.NoThrowDestructorReady);
        Assert.True(debug.BorrowedPointerEscapeBlocked);
        Assert.False(debug.AttachDetachClearControlReady);
        Assert.False(debug.OptInRuntimeSmokeReady);
        Assert.False(debug.PackageConsumerRuntimeProofReady);
        Assert.True(debug.RuntimeProofBlocked);

        TensorRtCallbackOwnerClosureMatrixRow output = matrix.Rows.Single(row => row.OwnerFamily == "OutputAllocator");
        Assert.Contains("IOutputAllocator::notifyShape", output.CallbackMethods);
        Assert.True(output.DesignGateReady);
        Assert.True(output.BorrowedPointerEscapeBlocked);
        Assert.False(output.AttachDetachClearControlReady);
        Assert.False(output.NoThrowVTableReady);

        TensorRtCallbackOwnerClosureMatrixRow asyncAllocator = matrix.Rows.Single(row => row.OwnerFamily == "GpuAsyncAllocator");
        Assert.Contains("IGpuAsyncAllocator::allocateAsync", asyncAllocator.CallbackMethods);
        Assert.False(asyncAllocator.OptInRuntimeSmokeReady);
        Assert.Contains("CUDA stream lifetime", asyncAllocator.NextWorkItem);

        TensorRtCallbackOwnerClosureMatrixRow stream = matrix.Rows.Single(row => row.OwnerFamily == "StreamReaderWriter");
        Assert.Contains("IStreamReaderV2::seek", stream.CallbackMethods);
        Assert.True(stream.DesignGateReady);
        Assert.True(stream.BorrowedPointerEscapeBlocked);
        Assert.False(stream.ManagedOwnerStateReady);
        Assert.False(stream.NoThrowVTableReady);
    }

    [Fact]
    public void ClosureMatrixPublicSurfaceDoesNotExposeRawPointers()
    {
        AssertNoRawPointerTypes(typeof(TensorRtCallbackOwnerClosureMatrix));
        AssertNoRawPointerTypes(typeof(TensorRtCallbackOwnerClosureMatrixResult));
        AssertNoRawPointerTypes(typeof(TensorRtCallbackOwnerClosureMatrixRow));

        string source = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtCallbackOwnerClosureMatrix.cs");
        Assert.DoesNotContain("public IntPtr", source);
        Assert.DoesNotContain("public UIntPtr", source);
        Assert.DoesNotContain("public nint", source);
        Assert.Contains("PackageConsumerRuntimeProofRequired", source);
        Assert.Contains("PackageConsumerRuntimeProofReady", source);
        Assert.Contains("DeferredRowsStillRequired", source);
        Assert.Contains("BorrowedPointerEscapeBlocked", source);
    }

    [Fact]
    public void SmokeAndDocsExposeClosureMatrixBoundary()
    {
        string source = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtCallbackOwnerClosureMatrix.cs");
        string smoke = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string smokeReadme = ReadSource("smoke", "README.md");
        string doc = ReadSource("docs", "articles", "zh-cn", "callback-owner-closure-matrix.md");
        string boundaryGuide = ReadSource("docs", "articles", "zh-cn", "callback-allocator-boundary-guide.md");
        string roadmap = ReadSource("docs", "articles", "zh-cn", "callback-allocator-safety-bridge-roadmap.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string toc = ReadSource("docs", "toc.yml");
        string index = ReadSource("docs", "index.md");

        Assert.Contains("TensorRtCallbackOwnerClosureMatrix", source);
        Assert.Contains("TensorRtCallbackOwnerClosureMatrixResult", source);
        Assert.Contains("TensorRtCallbackOwnerClosureMatrixRow", source);
        Assert.Contains("RealCallbackRuntime", smoke);
        Assert.Contains("IsRealCallbackRuntimeProof", smoke);
        Assert.Contains("PackageConsumerRuntimeProofReady", smoke);
        AssertClosureMatrixOutputMarkers(smokeReadme);
        AssertClosureMatrixOutputMarkers(doc);
        Assert.Contains("TensorRtCallbackOwnerClosureMatrix.Evaluate", smoke);
        Assert.Contains("CallbackOwnerClosureMatrix=", smoke);
        Assert.Contains("GpuAllocator", doc);
        Assert.Contains("GpuAsyncAllocator", doc);
        Assert.Contains("OutputAllocator", doc);
        Assert.Contains("DebugListener", doc);
        Assert.Contains("StreamReaderWriter", doc);
        Assert.Contains("not proof", doc);
        Assert.Contains("不能作为真实 TensorRT callback runtime proof", doc);
        Assert.Contains("TensorRtCallbackOwnerClosureMatrix", boundaryGuide);
        Assert.Contains("TensorRtCallbackOwnerClosureMatrixResult", roadmap);
        Assert.Contains("callback-owner-closure-matrix", schema);
        Assert.Contains("不能作为真实 TensorRT callback runtime proof", schema);
        AssertBridgePackageConsumerMarkers(bridgeConsumer);
        AssertRuntimeReadinessMarkers(readiness);
        Assert.Contains("callback-owner-closure-matrix.md", toc);
        Assert.Contains("callback-owner-closure-matrix.md", index);
    }

    private static TensorRtCallbackOwnerClosureMatrixResult CreateMatrix()
    {
        using TensorRtAllocatorCallbackOwner allocatorOwner = new TensorRtAllocatorCallbackOwner(static request =>
            TensorRtAllocatorDryRunResult.Success("quality-closure-matrix:" + request.Reason));
        TensorRtAllocatorLedgerSafetyGateResult allocatorGate = TensorRtAllocatorLedgerSafetyGate.Evaluate(
            allocatorOwner,
            TensorRtApiLine.TensorRt11,
            new TensorRtAllocatorDryRunRequest(32768UL, 256UL, "quality-closure-matrix-allocator"),
            "IGpuAllocator",
            0UL);

        using TensorRtOutputAllocatorCallbackOwner outputOwner = new TensorRtOutputAllocatorCallbackOwner();
        _ = outputOwner.RunDesignDiagnostic(
            TensorRtApiLine.TensorRt11,
            new TensorRtOutputAllocatorCallbackRequest(
                "quality_closure_matrix_output",
                4096UL,
                256UL,
                new long[] { 1, 1000 },
                "quality-closure-matrix-output",
                hasCurrentMemory: true),
            0UL);
        outputOwner.Dispose();
        TensorRtOutputAllocatorRuntimeProofPrecheckResult outputPrecheck =
            TensorRtOutputAllocatorRuntimeProofPrecheck.Evaluate(outputOwner.GetSnapshot("post-dispose-closure-matrix"));

        using TensorRtDebugListenerCallbackOwner debugOwner = new TensorRtDebugListenerCallbackOwner();
        _ = debugOwner.RunDesignDiagnostic(
            TensorRtApiLine.TensorRt11,
            new TensorRtDebugListenerCallbackRequest(
                "quality_closure_matrix_debug",
                TensorRtDataType.Float,
                TensorRtTensorLocation.Device,
                new long[] { 1, 3, 224, 224 },
                "quality-closure-matrix-debug",
                isInput: true,
                isExecutionTensor: true));
        debugOwner.Dispose();
        TensorRtDebugListenerRuntimeProofPrecheckResult debugPrecheck =
            TensorRtDebugListenerRuntimeProofPrecheck.Evaluate(debugOwner.GetSnapshot("post-dispose-closure-matrix"));

        TensorRtStreamIoInterfaceInfoDesignGateResult streamGate =
            TensorRtStreamIoInterfaceInfoDesignGate.EvaluateKnownSurface(TensorRtApiLine.TensorRt11);

        return TensorRtCallbackOwnerClosureMatrix.Evaluate(
            allocatorGate,
            outputPrecheck,
            debugPrecheck,
            streamGate);
    }

    private static void AssertClosureMatrixOutputMarkers(string text)
    {
        Assert.Contains("callback-owner-closure-matrix", text);
        Assert.Contains("TensorRtCallbackOwnerClosureMatrix", text);
        Assert.Contains("TensorRtCallbackOwnerClosureMatrixResult", text);
        Assert.Contains("TensorRtCallbackOwnerClosureMatrixRow", text);
        Assert.Contains("RuntimeEvidenceKind", text);
        Assert.Contains("RealCallbackRuntime=False", text);
        Assert.Contains("IsRealCallbackRuntimeProof=False", text);
        Assert.Contains("PackageConsumerRuntimeProofRequired", text);
        Assert.Contains("PackageConsumerRuntimeProofReady", text);
        Assert.Contains("RuntimeProofBlocked", text);
        Assert.Contains("DeferredRowsStillRequired", text);
    }

    private static void AssertBridgePackageConsumerMarkers(string text)
    {
        Assert.Contains("TensorRtCallbackOwnerClosureMatrix.Evaluate", text);
        Assert.Contains("callbackOwnerClosureMatrix", text);
        Assert.Contains("callbackOwnerClosureMatrixSummary", text);
        Assert.Contains("TensorRtStreamIoInterfaceInfoDesignGate.EvaluateKnownSurface", text);
        Assert.Contains("callback-owner-closure-matrix", text);
        Assert.Contains("TensorRtCallbackOwnerClosureMatrixResult.FamilyCount", text);
        Assert.Contains("TensorRtCallbackOwnerClosureMatrixResult.DesignGateReadyFamilyCount", text);
        Assert.Contains("TensorRtCallbackOwnerClosureMatrixResult.ClosureReadyFamilyCount", text);
        Assert.Contains("TensorRtCallbackOwnerClosureMatrixResult.RuntimeProofAttemptReadyFamilyCount", text);
        Assert.Contains("TensorRtCallbackOwnerClosureMatrixResult.PackageConsumerRuntimeProofReadyFamilyCount", text);
        Assert.Contains("TensorRtCallbackOwnerClosureMatrixResult.RuntimeProofBlocked", text);
        Assert.Contains("TensorRtCallbackOwnerClosureMatrixResult.DeferredRowsStillRequired", text);
    }

    private static void AssertRuntimeReadinessMarkers(string text)
    {
        Assert.Contains("New-CallbackOwnerClosureMatrixEvidence", text);
        Assert.Contains("callbackOwnerClosureMatrixEvidence", text);
        Assert.Contains("callbackOwnerClosureMatrix = $callbackOwnerClosureMatrixEvidence", text);
        Assert.Contains("hasCallbackOwnerClosureMatrix", text);
        Assert.Contains("closure-matrix-ready", text);
        Assert.Contains("callbackOwnerClosureMatrix", text);
        Assert.Contains("callback owner closure matrix:", text);
        Assert.Contains("Callback owner closure matrix missing evidence", text);
        AssertClosureMatrixOutputMarkers(text);
        Assert.Contains("FamilyCount=5", text);
        Assert.Contains("DesignGateReadyFamilyCount", text);
        Assert.Contains("ClosureReadyFamilyCount", text);
        Assert.Contains("RuntimeProofAttemptReadyFamilyCount", text);
        Assert.Contains("PackageConsumerRuntimeProofReadyFamilyCount", text);
        Assert.Contains("IStreamReaderV2::seek", text);
        Assert.Contains("IStreamWriter::write", text);
    }

    private static void AssertNoRawPointerTypes(Type type)
    {
        foreach (ConstructorInfo constructor in type.GetConstructors(BindingFlags.Instance | BindingFlags.Public))
        {
            foreach (ParameterInfo parameter in constructor.GetParameters())
            {
                AssertNoRawPointerType(parameter.ParameterType);
            }
        }

        foreach (PropertyInfo property in type.GetProperties(BindingFlags.Instance | BindingFlags.Public | BindingFlags.Static))
        {
            AssertNoRawPointerType(property.PropertyType);
        }

        foreach (MethodInfo method in type.GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.DeclaredOnly))
        {
            AssertNoRawPointerType(method.ReturnType);
            foreach (ParameterInfo parameter in method.GetParameters())
            {
                AssertNoRawPointerType(parameter.ParameterType);
            }
        }
    }

    private static void AssertNoRawPointerType(Type type)
    {
        Assert.NotEqual(typeof(IntPtr), type);
        Assert.NotEqual(typeof(UIntPtr), type);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}

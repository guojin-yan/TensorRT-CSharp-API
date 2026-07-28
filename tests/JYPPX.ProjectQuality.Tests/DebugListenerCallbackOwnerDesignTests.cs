using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerCallbackOwnerDesignTests
{
    [Fact]
    public void PublicDesignOwnerCopiesDebugTensorMetadataWithoutPointerOwnershipOrRuntimeProof()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        long[] shape = { 1, 3, 224, 224 };
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            shape,
            "quality-debug-listener-owner-design",
            isInput: true,
            isOutput: false,
            isShapeTensor: false,
            isExecutionTensor: true);
        shape[0] = 99;

        TensorRtDebugListenerCallbackOwnerSnapshot snapshot = owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);

        Assert.Equal("debug-listener-callback-owner-design", snapshot.EvidenceKind);
        Assert.Equal("debug-listener-prototype", snapshot.CallbackKind);
        Assert.Equal("not-present", snapshot.RuntimeEvidenceKind);
        Assert.False(snapshot.RealCallbackRuntime);
        Assert.False(snapshot.IsRealCallbackRuntimeProof);
        Assert.False(snapshot.IsAttached);
        Assert.False(snapshot.DebugTensorPointerExposed);
        Assert.False(snapshot.DebugTensorPointerProduced);
        Assert.False(snapshot.BorrowedDebugTensorPointerEscaped);
        Assert.Equal("quality_debug_tensor", snapshot.TensorName);
        Assert.Equal(TensorRtDataType.Float, snapshot.DataType);
        Assert.Equal(TensorRtTensorLocation.Device, snapshot.Location);
        Assert.Equal(4, snapshot.ShapeRank);
        Assert.Equal("[1x3x224x224]", snapshot.ShapeSummary);
        Assert.True(snapshot.IsInput);
        Assert.False(snapshot.IsOutput);
        Assert.False(snapshot.IsShapeTensor);
        Assert.True(snapshot.IsExecutionTensor);
        Assert.True(snapshot.DebugTensorMetadataCopied);
        Assert.Equal(1, snapshot.InvocationCount);
        Assert.Equal(1, snapshot.ProcessDebugTensorCount);
        Assert.Equal(0, snapshot.FailureCount);
        Assert.Equal(0, snapshot.InFlightCallbackCount);
        Assert.True(snapshot.CallbackStatePinned);
        Assert.True(snapshot.DelegatePinned);
        Assert.Equal(BridgeStatusCode.Ok, snapshot.LastStatus);
        Assert.True(snapshot.Succeeded);

        owner.Dispose();
        TensorRtDebugListenerCallbackOwnerSnapshot disposed = owner.GetSnapshot("post-dispose");
        Assert.True(disposed.DisposeRequested);
        Assert.False(disposed.CallbackStatePinned);
        Assert.False(disposed.DelegatePinned);
        Assert.True(disposed.ReleaseHookCount >= 1);
        Assert.Contains("release hook", disposed.ReleaseDiagnostic);
    }

    [Fact]
    public void PublicDesignOwnerMapsSyntheticHandlerExceptionToInvalidState()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 64 },
            "throw");

        TensorRtDebugListenerCallbackOwnerSnapshot snapshot = owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);

        Assert.Equal(BridgeStatusCode.InvalidState, snapshot.LastStatus);
        Assert.False(snapshot.Succeeded);
        Assert.Equal(1, snapshot.InvocationCount);
        Assert.Equal(1, snapshot.ProcessDebugTensorCount);
        Assert.Equal(1, snapshot.FailureCount);
        Assert.Equal(0, snapshot.InFlightCallbackCount);
        Assert.False(snapshot.RealCallbackRuntime);
        Assert.False(snapshot.IsRealCallbackRuntimeProof);
        Assert.Contains("InvalidOperationException", snapshot.LastDiagnostic);
    }

    [Fact]
    public void PublicDesignOwnerRejectsUnsafeRequestShapeAndEnums()
    {
        Assert.Throws<ArgumentException>(() => new TensorRtDebugListenerCallbackRequest("", TensorRtDataType.Float, TensorRtTensorLocation.Device, Array.Empty<long>()));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TensorRtDebugListenerCallbackRequest("x", (TensorRtDataType)12345, TensorRtTensorLocation.Device, Array.Empty<long>()));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TensorRtDebugListenerCallbackRequest("x", TensorRtDataType.Float, (TensorRtTensorLocation)99, Array.Empty<long>()));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TensorRtDebugListenerCallbackRequest("x", TensorRtDataType.Float, TensorRtTensorLocation.Device, new long[9]));
    }

    [Fact]
    public void PublicDesignOwnerSurfaceDoesNotExposeRawPointers()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtDebugListenerCallbackOwner),
            typeof(TensorRtDebugListenerCallbackRequest),
            typeof(TensorRtDebugListenerCallbackOwnerSnapshot)
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
        string ownerSource = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "Debugging", "TensorRtDebugListenerCallbackOwner.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string designDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-callback-owner-design.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string runtimeSplitReadme = ReadSource("pack", "runtime-split", "README.md");
        string smokeReadme = ReadSource("smoke", "README.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        Assert.Contains("public sealed class TensorRtDebugListenerCallbackOwner", ownerSource);
        Assert.Contains("public readonly struct TensorRtDebugListenerCallbackOwnerSnapshot", ownerSource);
        Assert.Contains("public TensorRtDebugListenerCallbackOwnerSnapshot RunDesignDiagnostic", ownerSource);
        Assert.Contains("RuntimeEvidenceKind => \"not-present\"", ownerSource);
        Assert.Contains("RealCallbackRuntime => false", ownerSource);
        Assert.Contains("IsRealCallbackRuntimeProof => false", ownerSource);
        Assert.Contains("DebugTensorPointerExposed => false", ownerSource);
        Assert.Contains("DebugTensorPointerProduced => false", ownerSource);
        Assert.Contains("BorrowedDebugTensorPointerEscaped => false", ownerSource);
        Assert.DoesNotContain("public IntPtr", ownerSource);
        Assert.DoesNotContain("public nint", ownerSource);

        Assert.Contains("debug-listener-callback-owner-design", smokeProgram);
        Assert.Contains("DebugListenerCallbackOwnerDesign=", smokeProgram);
        Assert.Contains("DebugListenerCallbackOwnerDesignDispose=", smokeProgram);
        Assert.Contains("RuntimeEvidenceKind", smokeProgram);
        Assert.Contains("IsRealCallbackRuntimeProof", smokeProgram);
        Assert.Contains("ProcessDebugTensorCount", smokeProgram);
        Assert.Contains("DebugTensorPointerExposed", smokeProgram);
        Assert.Contains("DebugTensorPointerProduced", smokeProgram);
        Assert.Contains("BorrowedDebugTensorPointerEscaped", smokeProgram);

        Assert.Contains("debugListenerCallbackOwnerDesign", readiness);
        Assert.Contains("New-DebugListenerCallbackOwnerDesignEvidence", readiness);
        Assert.Contains("owner-design-gate", readiness);
        Assert.Contains("design-ready", readiness);
        Assert.Contains("isRealCallbackRuntimeProof = $false", readiness);

        Assert.Contains("debug-listener-callback-owner-design", packageConsumer);
        Assert.Contains("output-allocator-callback-owner-design", packageConsumer);
        Assert.Contains("debug-listener-callback-owner-design", bridgeConsumer);
        Assert.Contains("TensorRtDebugListenerCallbackOwner", bridgeConsumer);

        Assert.Contains("debug-listener-callback-owner-design", designDoc);
        Assert.Contains("RuntimeEvidenceKind=not-present", designDoc);
        Assert.Contains("RealCallbackRuntime=False", designDoc);
        Assert.Contains("IsRealCallbackRuntimeProof=False", designDoc);
        Assert.Contains("setDebugListener", designDoc);
        Assert.Contains("not proof", designDoc);
        Assert.Contains("debug-listener-callback-owner-design", trampolineGate);
        Assert.Contains("debug-listener-callback-owner-design", schema);
        Assert.Contains("debug-listener-callback-owner-design", latest);
        Assert.Contains("debugListenerCallbackOwnerDesign", runtimeSplitReadme);
        Assert.Contains("debug-listener-callback-owner-design", smokeReadme);

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

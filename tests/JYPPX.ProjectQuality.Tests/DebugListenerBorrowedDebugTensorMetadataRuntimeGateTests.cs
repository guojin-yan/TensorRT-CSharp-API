using System.Reflection;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerBorrowedDebugTensorMetadataRuntimeGateTests
{
    [Fact]
    public void MetadataGateCopiesDebugTensorMetadataAndStaysNonProof()
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "quality_metadata_tensor",
            TensorRtDataType.Half,
            TensorRtTensorLocation.Device,
            new long[] { 2, 4, 16 },
            "quality-debug-listener-borrowed-debug-tensor-metadata-runtime-gate",
            isInput: false,
            isOutput: true,
            isShapeTensor: false,
            isExecutionTensor: true);

        owner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, request);
        owner.Dispose();
        TensorRtDebugListenerCallbackOwnerSnapshot snapshot = owner.GetSnapshot("post-dispose");
        TensorRtDebugListenerBorrowedTensorSafetyGateResult safety =
            TensorRtDebugListenerBorrowedTensorSafetyGate.Evaluate(snapshot);
        TensorRtDebugListenerNoThrowVTableCallbackStubResult callbackStub =
            TensorRtDebugListenerNoThrowVTableCallbackStub.Evaluate(snapshot);
        TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult gate =
            TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate.Evaluate(snapshot, safety, callbackStub);
        TensorRtDebugListenerRuntimeProofPrecheckResult precheck =
            TensorRtDebugListenerRuntimeProofPrecheck.Evaluate(snapshot);

        Assert.Equal("debug-listener-borrowed-debug-tensor-metadata-runtime-gate", gate.EvidenceKind);
        Assert.Equal("borrowed-debug-tensor-metadata-gate", gate.RuntimeEvidenceKind);
        Assert.Equal("debug-listener-process-debug-tensor", gate.CallbackKind);
        Assert.False(gate.RealCallbackRuntime);
        Assert.False(gate.IsRealCallbackRuntimeProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, gate.Line);
        Assert.True(gate.OwnerId > 0);
        Assert.Equal(BridgeStatusCode.Ok, gate.LastStatus);
        Assert.Equal("quality_metadata_tensor", gate.TensorName);
        Assert.Equal("quality_metadata_tensor".Length, gate.TensorNameLength);
        Assert.Equal(TensorRtDataType.Half, gate.DataType);
        Assert.Equal(TensorRtTensorLocation.Device, gate.Location);
        Assert.Equal(3, gate.TensorShapeRank);
        Assert.Contains("16", gate.ShapeSummary);
        Assert.False(gate.IsInput);
        Assert.True(gate.IsOutput);
        Assert.False(gate.IsShapeTensor);
        Assert.True(gate.IsExecutionTensor);
        Assert.True(gate.BorrowedTensorSafetyGateReady);
        Assert.True(gate.CallbackStubGateReady);
        Assert.True(gate.MetadataGateReady);
        Assert.True(gate.TensorNameCopied);
        Assert.True(gate.TensorTypeCopied);
        Assert.True(gate.TensorLocationCopied);
        Assert.True(gate.TensorShapeCopied);
        Assert.True(gate.TensorFlagsCopied);
        Assert.True(gate.BorrowedDebugTensorMetadataCopyReady);
        Assert.True(gate.BorrowedDebugTensorPointerEscapeBlocked);
        Assert.True(gate.BorrowedDebugTensorDataPointerEscapeBlocked);
        Assert.False(gate.DebugTensorPointerExposed);
        Assert.False(gate.DebugTensorDataPointerExposed);
        Assert.False(gate.BorrowedDebugTensorLifetimeReady);
        Assert.False(gate.BorrowedDebugTensorDataLifetimeReady);
        Assert.False(gate.SetDebugListenerNonNullEnabled);
        Assert.False(gate.NativeVTableInstalled);
        Assert.False(gate.ProcessDebugTensorRuntimeReady);
        Assert.False(gate.CanCallProcessDebugTensorRuntime);
        Assert.False(gate.CanAttemptRuntimeProof);
        Assert.True(gate.RuntimeProofBlocked);
        Assert.True(gate.DeferredRowsStillRequired);
        Assert.Contains("borrowed-debug-tensor-metadata-gate", gate.ReasonMetadataRuntimeStillBlocked, StringComparison.Ordinal);
        Assert.Equal("borrowed-debug-tensor-metadata-gate-ready", gate.Status);
        Assert.Contains("RuntimeEvidenceKind=borrowed-debug-tensor-metadata-gate", gate.Diagnostic);
        Assert.Contains("MetadataGateReady=True", gate.Diagnostic);
        Assert.Contains("BorrowedDebugTensorDataPointerEscapeBlocked=True", gate.Diagnostic);
        Assert.Contains("BorrowedDebugTensorLifetimeReady=False", gate.Diagnostic);
        Assert.Contains("ProcessDebugTensorRuntimeReady=False", gate.Diagnostic);

        Assert.False(precheck.NativeVTableReady);
        Assert.False(precheck.ProcessDebugTensorRuntimeReady);
        Assert.False(precheck.CanAttemptRuntimeProof);
        Assert.True(precheck.RuntimeProofBlocked);
    }

    [Fact]
    public void PublicMetadataGateSurfaceDoesNotExposeRawPointers()
    {
        AssertNoRawPointerTypes(typeof(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate));
        AssertNoRawPointerTypes(typeof(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult));

        string source = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate.cs");
        Assert.DoesNotContain("public IntPtr", source);
        Assert.DoesNotContain("public UIntPtr", source);
        Assert.DoesNotContain("public nint", source);
        Assert.Contains("DebugTensorPointerExposed", source);
        Assert.Contains("DebugTensorDataPointerExposed", source);
        Assert.Contains("BorrowedDebugTensorDataPointerEscapeBlocked", source);
    }

    [Fact]
    public void SourceNativeSmokeReadinessPackageAndDocsContainMetadataGateEvidenceButNotRuntimeProof()
    {
        string source = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate.cs");
        string nativeSource = ReadSource("native", "src", "tensorrt", "common", "debug_listener_borrowed_debug_tensor_metadata_runtime_gate.inc");
        string trt8Api = ReadSource("native", "src", "tensorrt", "v8", "api.cpp");
        string trt10Api = ReadSource("native", "src", "tensorrt", "v10", "api.cpp");
        string trt11Api = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");
        string smoke = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string packageConsumer = ReadSource("eng", "Test-PackageConsumer.ps1");
        string doc = ReadSource("docs", "articles", "zh-cn", "debug-listener-borrowed-debug-tensor-metadata-runtime-gate.md");
        string stubDoc = ReadSource("docs", "articles", "zh-cn", "debug-listener-nothrow-vtable-callback-stub.md");
        string schema = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string trampolineGate = ReadSource("docs", "articles", "zh-cn", "real-callback-trampoline-gate.md");
        string smokeReadme = ReadSource("smoke", "README.md");
        string runtimeSplitReadme = ReadSource("pack", "runtime-split", "README.md");
        string toc = ReadSource("docs", "toc.yml");
        string index = ReadSource("docs", "index.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");

        AssertMetadataGateSurfaceMarkers(source);
        AssertMetadataGateSurfaceMarkers(smoke);
        AssertMetadataGateSurfaceMarkers(readiness);
        AssertMetadataGateSurfaceMarkers(bridgeConsumer);
        AssertMetadataGateSurfaceMarkers(doc);
        AssertMetadataGateSurfaceMarkers(packageConsumer);
        AssertMetadataGateEvidenceKindMarkers(readiness);
        AssertMetadataGateEvidenceKindMarkers(doc);
        AssertMetadataGateEvidenceKindMarkers(latest);
        AssertMetadataGateEvidenceKindMarkers(smokeReadme);
        AssertMetadataGateEvidenceKindMarkers(runtimeSplitReadme);
        AssertMetadataGateOverviewMarkers(schema);
        AssertMetadataGateOverviewMarkers(trampolineGate);
        AssertMetadataGateOverviewMarkers(stubDoc);
        Assert.Contains("debug-listener-borrowed-debug-tensor-metadata-runtime-gate.md", toc);
        Assert.Contains("debug-listener-borrowed-debug-tensor-metadata-runtime-gate.md", index);
        Assert.Contains("DebugListenerBorrowedDebugTensorMetadataRuntimeGate final", nativeSource);
        Assert.Contains("copy_metadata", nativeSource);
        Assert.Contains("metadata_copy_ready", nativeSource);
        Assert.Contains("borrowed_tensor_pointer_escape_blocked", nativeSource);
        Assert.Contains("borrowed_tensor_data_pointer_escape_blocked", nativeSource);
        Assert.Contains("borrowed_tensor_lifetime_runtime_ready", nativeSource);
        Assert.Contains("borrowed_tensor_data_lifetime_runtime_ready", nativeSource);
        Assert.Contains("process_debug_tensor_runtime_ready", nativeSource);
        Assert.Contains("debug_listener_borrowed_debug_tensor_metadata_runtime_gate.inc", trt8Api);
        Assert.Contains("debug_listener_borrowed_debug_tensor_metadata_runtime_gate.inc", trt10Api);
        Assert.Contains("debug_listener_borrowed_debug_tensor_metadata_runtime_gate.inc", trt11Api);
        Assert.Contains("DebugListenerBorrowedDebugTensorMetadataRuntimeGate=", smoke);
        Assert.Contains("hasDebugListenerBorrowedDebugTensorMetadataRuntimeGate", readiness);
        Assert.Contains("New-DebugListenerBorrowedDebugTensorMetadataRuntimeGateEvidence", readiness);
        Assert.Contains("borrowed-debug-tensor-metadata-gate", packageConsumer);
        Assert.Contains("borrowed-debug-tensor-metadata-gate", bridgeConsumer);
        Assert.Contains("not proof", doc);
        Assert.Contains("not proof", schema);
        Assert.Contains("not proof", latest);
        Assert.Contains("not proof", trampolineGate);
        Assert.Contains("\"IDebugListener\",\"processDebugTensor\",\"IDebugListener::processDebugTensor\",\"other\",\"deferred-only\"", comparison);
    }

    private static void AssertMetadataGateSurfaceMarkers(string text)
    {
        Assert.Contains("debug-listener-borrowed-debug-tensor-metadata-runtime-gate", text);
        Assert.Contains("TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate", text);
        Assert.Contains("TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult", text);
        Assert.Contains("MetadataGateReady", text);
        Assert.Contains("TensorNameCopied", text);
        Assert.Contains("TensorNameLength", text);
        Assert.Contains("TensorTypeCopied", text);
        Assert.Contains("TensorLocationCopied", text);
        Assert.Contains("TensorShapeCopied", text);
        Assert.Contains("TensorFlagsCopied", text);
        Assert.Contains("BorrowedDebugTensorMetadataCopyReady", text);
        Assert.Contains("BorrowedDebugTensorPointerEscapeBlocked", text);
        Assert.Contains("BorrowedDebugTensorDataPointerEscapeBlocked", text);
        Assert.Contains("DebugTensorPointerExposed", text);
        Assert.Contains("DebugTensorDataPointerExposed", text);
        Assert.Contains("BorrowedDebugTensorLifetimeReady", text);
        Assert.Contains("BorrowedDebugTensorDataLifetimeReady", text);
        Assert.Contains("CanCallProcessDebugTensorRuntime", text);
        Assert.Contains("RuntimeProofBlocked", text);
    }

    private static void AssertMetadataGateEvidenceKindMarkers(string text)
    {
        Assert.Contains("borrowed-debug-tensor-metadata-gate", text);
        Assert.Contains("RuntimeEvidenceKind", text);
        Assert.Contains("RealCallbackRuntime=False", text);
        Assert.Contains("IsRealCallbackRuntimeProof=False", text);
    }

    private static void AssertMetadataGateOverviewMarkers(string text)
    {
        Assert.Contains("debug-listener-borrowed-debug-tensor-metadata-runtime-gate", text);
        Assert.Contains("borrowed-debug-tensor-metadata-gate", text);
        Assert.Contains("not proof", text);
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

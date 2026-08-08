using System.Reflection;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class RuntimeDeserializationBoundaryPrecheckTests
{
    [Fact]
    public void KnownSurfacePrecheckReportsSafeManagedDeserializeWithoutRuntimeProof()
    {
        TensorRtRuntimeDeserializationBoundaryPrecheckResult precheck =
            TensorRtRuntimeDeserializationBoundaryPrecheck.EvaluateKnownSurface(TensorRtApiLine.TensorRt11);

        Assert.Equal("runtime-deserialization-boundary-precheck", precheck.EvidenceKind);
        Assert.Equal("runtime-deserialization-boundary", precheck.DiagnosticsKind);
        Assert.Equal("runtime-precheck", precheck.RuntimeEvidenceKind);
        Assert.False(precheck.IsRuntimeExecutionEvidence);
        Assert.False(precheck.IsRuntimeExecutionProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, precheck.Line);
        Assert.True(precheck.LineSupportsRuntimeDeserialization);
        Assert.True(precheck.LineSupportsDeserializeCudaEngineV2);
        Assert.True(precheck.ManagedByteArrayDeserializeReady);
        Assert.True(precheck.ManagedArraySegmentDeserializeReady);
        Assert.True(precheck.ManagedReadOnlySpanDeserializeReady);
        Assert.True(precheck.ManagedStreamDeserializeReady);
        Assert.True(precheck.ManagedFileDeserializeReady);
        Assert.True(precheck.HostMemoryDeserializeReady);
        Assert.True(precheck.SerializedBufferCopiedBeforeInterop);
        Assert.True(precheck.PinnedBufferScopedToInteropCall);
        Assert.False(precheck.BorrowedSerializedBufferEscaped);
        Assert.True(precheck.HostMemoryHandleOwnedByWrapper);
        Assert.True(precheck.EngineHandleOwnedByWrapper);
        Assert.False(precheck.EnginePointerExposed);
        Assert.False(precheck.EnginePointerProduced);
        Assert.False(precheck.DirectDeserializeCudaEngineRowsDeferred);
        Assert.True(precheck.DirectDeserializeCudaEngineRowsImplemented);
        Assert.True(precheck.DirectDeserializeCudaEngineV2RowsDeferred);
        Assert.True(precheck.LoadRuntimeDeferred);
        Assert.False(precheck.PluginLibraryDependencyDiagnosticsReady);
        Assert.True(precheck.PointerFreeSurfaceReady);
        Assert.True(precheck.ManagedDeserializeSurfaceReady);
        Assert.True(precheck.SafeDeserializeBridgeReady);
        Assert.True(precheck.PrecheckReady);
        Assert.False(precheck.CanAttemptRuntimeProof);
        Assert.False(precheck.CanPromoteWithoutRuntimeProof);
        Assert.False(precheck.FullPackageConsumerRuntimeEvidenceReady);
        Assert.False(precheck.CanPromoteRuntimeProof);
        Assert.True(precheck.RuntimeProofBlocked);
        Assert.True(precheck.DeferredRowsStillRequired);
        Assert.Equal("precheck-ready", precheck.Status);
        Assert.True(precheck.BlockedPrerequisiteCount >= 4);
        Assert.Contains(precheck.BlockedPrerequisites, item => item.Contains("plugin/library dependency diagnostics", StringComparison.Ordinal));
        Assert.Contains(precheck.BlockedPrerequisites, item => item.Contains("runtime execution proof", StringComparison.Ordinal));
        Assert.Contains(precheck.BlockedPrerequisites, item => item.Contains("deserializeCudaEngineV2", StringComparison.Ordinal));
        Assert.Contains(precheck.BlockedPrerequisites, item => item.Contains("loadRuntime", StringComparison.Ordinal));
        Assert.Contains("ManagedByteArrayDeserializeReady=True", precheck.Diagnostic);
        Assert.Contains("SerializedBufferCopiedBeforeInterop=True", precheck.Diagnostic);
        Assert.Contains("EngineHandleOwnedByWrapper=True", precheck.Diagnostic);
        Assert.Contains("LoadRuntimeDeferred=True", precheck.Diagnostic);
        Assert.Contains("RuntimeProofBlocked=True", precheck.Diagnostic);
    }

    [Fact]
    public void TensorRt8KnownSurfaceHasNoV2RowButKeepsLoadRuntimeDeferred()
    {
        TensorRtRuntimeDeserializationBoundaryPrecheckResult precheck =
            TensorRtRuntimeDeserializationBoundaryPrecheck.EvaluateKnownSurface(TensorRtApiLine.TensorRt8);

        Assert.True(precheck.LineSupportsRuntimeDeserialization);
        Assert.False(precheck.LineSupportsDeserializeCudaEngineV2);
        Assert.False(precheck.DirectDeserializeCudaEngineV2RowsDeferred);
        Assert.False(precheck.DirectDeserializeCudaEngineRowsDeferred);
        Assert.True(precheck.DirectDeserializeCudaEngineRowsImplemented);
        Assert.True(precheck.LoadRuntimeDeferred);
        Assert.True(precheck.PrecheckReady);
        Assert.True(precheck.RuntimeProofBlocked);
        Assert.True(precheck.DeferredRowsStillRequired);
    }

    [Fact]
    public void DependencyDiagnosticsReportsDependencyProbeAndDriverBlockedWithoutRuntimeProof()
    {
        TensorRtRuntimeDeserializationBoundaryPrecheckResult precheck =
            TensorRtRuntimeDeserializationBoundaryPrecheck.EvaluateKnownSurface(TensorRtApiLine.TensorRt11);

        TensorRtRuntimeDeserializationDependencyDiagnosticsResult diagnostics =
            TensorRtRuntimeDeserializationDependencyDiagnostics.Evaluate(
                precheck,
                fullPackageConsumerReportPresent: true,
                fullPackageConsumerSmokeRequested: true,
                fullPackageConsumerSmokeResult: "blocked-by-cuda-driver",
                dependencyProbeOnly: true,
                blockedByCudaDriver: true);

        Assert.Equal("runtime-deserialization-dependency-diagnostics", diagnostics.EvidenceKind);
        Assert.Equal("runtime-deserialization-dependency-diagnostics", diagnostics.DiagnosticsKind);
        Assert.Equal("dependency-diagnostics", diagnostics.RuntimeEvidenceKind);
        Assert.False(diagnostics.IsRuntimeExecutionEvidence);
        Assert.False(diagnostics.IsRuntimeExecutionProof);
        Assert.Equal(TensorRtApiLine.TensorRt11, diagnostics.Line);
        Assert.True(diagnostics.PrecheckReady);
        Assert.True(diagnostics.ManagedDeserializeSurfaceReady);
        Assert.True(diagnostics.SafeDeserializeBridgeReady);
        Assert.True(diagnostics.FullPackageConsumerReportPresent);
        Assert.True(diagnostics.FullPackageConsumerSmokeRequested);
        Assert.Equal("blocked-by-cuda-driver", diagnostics.FullPackageConsumerSmokeResult);
        Assert.True(diagnostics.DependencyProbeOnly);
        Assert.True(diagnostics.BlockedByCudaDriver);
        Assert.True(diagnostics.DriverRuntimeMismatchClassified);
        Assert.False(diagnostics.PackageConsumerRuntimeProofPresent);
        Assert.True(diagnostics.ExternalRuntimeProofRequired);
        Assert.True(diagnostics.RuntimeProofOwnerActionRequired);
        Assert.Equal("cuda-driver-runtime-compatibility", diagnostics.RuntimeProofBlockerCategory);
        Assert.Equal("runtime-smoke-driver-blocked", diagnostics.PackageConsumerEvidenceClassification);
        Assert.False(diagnostics.PluginLibraryDependencyDiagnosticsComplete);
        Assert.False(diagnostics.LoadRuntimeOwnershipModeled);
        Assert.False(diagnostics.DirectDeserializeCudaEngineRowsDeferred);
        Assert.True(diagnostics.DirectDeserializeCudaEngineV2RowsDeferred);
        Assert.True(diagnostics.LoadRuntimeDeferred);
        Assert.True(diagnostics.PointerFreeSurfaceReady);
        Assert.False(diagnostics.CanAttemptRuntimeProof);
        Assert.False(diagnostics.CanPromoteRuntimeProof);
        Assert.True(diagnostics.RuntimeProofBlocked);
        Assert.True(diagnostics.DeferredRowsStillRequired);
        Assert.Equal("dependency-diagnostics-ready", diagnostics.Status);
        Assert.Contains("not runtime execution proof", diagnostics.WhyNotRuntimeProof, StringComparison.Ordinal);
        Assert.Contains("compatible NVIDIA driver", diagnostics.NextOwnerAction, StringComparison.Ordinal);
        Assert.Contains(diagnostics.BlockedPrerequisites, item => item.Contains("dependency-probe-only", StringComparison.Ordinal));
        Assert.Contains(diagnostics.BlockedPrerequisites, item => item.Contains("CUDA driver/runtime", StringComparison.Ordinal));
        Assert.Contains(diagnostics.BlockedPrerequisites, item => item.Contains("plugin library dependency", StringComparison.Ordinal));
        Assert.Contains(diagnostics.BlockedPrerequisites, item => item.Contains("loadRuntime", StringComparison.Ordinal));
        Assert.Contains("RuntimeEvidenceKind=dependency-diagnostics", diagnostics.Diagnostic);
        Assert.Contains("BlockedByCudaDriver=True", diagnostics.Diagnostic);
        Assert.Contains("RuntimeProofBlockerCategory=cuda-driver-runtime-compatibility", diagnostics.Diagnostic);
        Assert.Contains("ExternalRuntimeProofRequired=True", diagnostics.Diagnostic);
        Assert.Contains("CanPromoteRuntimeProof=False", diagnostics.Diagnostic);
    }

    [Fact]
    public void PublicPrecheckSurfaceDoesNotExposeRawPointers()
    {
        Type[] publicTypes =
        {
            typeof(TensorRtRuntimeDeserializationBoundaryPrecheck),
            typeof(TensorRtRuntimeDeserializationBoundaryPrecheckResult),
            typeof(TensorRtRuntimeDeserializationDependencyDiagnostics),
            typeof(TensorRtRuntimeDeserializationDependencyDiagnosticsResult)
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
    public void ManagedDeserializeSurfaceCopiesCallerBuffersAndKeepsDirectRowsDeferred()
    {
        string precheckSource = ReadSource("src", "JYPPX.TensorRtSharp", "Runtime", "TensorRtRuntimeDeserializationBoundaryPrecheck.cs");
        string diagnosticsSource = ReadSource("src", "JYPPX.TensorRtSharp", "Runtime", "TensorRtRuntimeDeserializationDependencyDiagnostics.cs");
        string runtimeSource = ReadSource("src", "JYPPX.TensorRtSharp", "Runtime", "TensorRtRuntime.cs");
        string nativeBridgeApi = ReadSource("src", "JYPPX.TensorRtSharp", "Internal", "Interop", "Serialization", "NativeBridgeApi.EngineDeserialization.cs");
        string smokeProgram = ReadSource("smoke", "CallbackAllocatorSafeControlsSmokeRunner", "Program.cs");
        string designDoc = ReadSource("docs", "articles", "zh-cn", "runtime-deserialization-boundary-precheck.md");
        string diagnosticsDoc = ReadSource("docs", "articles", "zh-cn", "runtime-deserialization-dependency-diagnostics.md");
        string manualGroups = ReadSource("docs", "articles", "zh-cn", "deferred-manual-design-groups.md");
        string docsIndex = ReadSource("docs", "index.md");
        string docsToc = ReadSource("docs", "toc.yml");
        string latest = ReadSource("docs", "articles", "zh-cn", "windows-api-completion-latest.md");
        string smokeReadme = ReadSource("smoke", "README.md");
        string splitReadme = ReadSource("pack", "runtime-split", "README.md");
        string readiness = ReadSource("eng", "Test-RuntimePackageReadiness.ps1");
        string releaseEvidence = ReadSource("eng", "Export-ReleaseEvidenceBundle.ps1");
        string bridgeConsumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");
        string trt8Deferred = ReadTensorRtManifest("v8", "trt8-cross-version-seventh-batch-deferred-boundaries.manifest.json");
        string trt10Deferred = ReadTensorRtManifest("v10", "trt10-cross-version-fifth-batch-runtime-serialization-deferred.manifest.json");
        string trt11Deferred = ReadTensorRtManifest("v11", "trt11-twenty-third-batch-deferred-coverage.manifest.json");

        Assert.Contains("public static class TensorRtRuntimeDeserializationBoundaryPrecheck", precheckSource);
        Assert.Contains("public readonly struct TensorRtRuntimeDeserializationBoundaryPrecheckResult", precheckSource);
        Assert.Contains("RuntimeEvidenceKind => \"runtime-precheck\"", precheckSource);
        Assert.Contains("IsRuntimeExecutionEvidence => false", precheckSource);
        Assert.Contains("IsRuntimeExecutionProof => false", precheckSource);
        Assert.Contains("ManagedByteArrayDeserializeReady", precheckSource);
        Assert.Contains("ManagedStreamDeserializeReady", precheckSource);
        Assert.Contains("HostMemoryDeserializeReady", precheckSource);
        Assert.Contains("SerializedBufferCopiedBeforeInterop", precheckSource);
        Assert.Contains("PinnedBufferScopedToInteropCall", precheckSource);
        Assert.Contains("BorrowedSerializedBufferEscaped => false", precheckSource);
        Assert.Contains("EngineHandleOwnedByWrapper", precheckSource);
        Assert.Contains("EnginePointerExposed => false", precheckSource);
        Assert.Contains("DirectDeserializeCudaEngineRowsDeferred => false", precheckSource);
        Assert.Contains("DirectDeserializeCudaEngineRowsImplemented => SafeDeserializeBridgeReady", precheckSource);
        Assert.Contains("LoadRuntimeDeferred => true", precheckSource);
        Assert.Contains("CanAttemptRuntimeProof => false", precheckSource);
        Assert.Contains("DeferredRowsStillRequired => true", precheckSource);
        Assert.DoesNotContain("public IntPtr", precheckSource);
        Assert.DoesNotContain("public nint", precheckSource);
        Assert.Contains("public static class TensorRtRuntimeDeserializationDependencyDiagnostics", diagnosticsSource);
        Assert.Contains("public readonly struct TensorRtRuntimeDeserializationDependencyDiagnosticsResult", diagnosticsSource);
        Assert.Contains("RuntimeEvidenceKind => \"dependency-diagnostics\"", diagnosticsSource);
        Assert.Contains("IsRuntimeExecutionEvidence => false", diagnosticsSource);
        Assert.Contains("IsRuntimeExecutionProof => false", diagnosticsSource);
        Assert.Contains("FullPackageConsumerReportPresent", diagnosticsSource);
        Assert.Contains("FullPackageConsumerSmokeRequested", diagnosticsSource);
        Assert.Contains("FullPackageConsumerSmokeResult", diagnosticsSource);
        Assert.Contains("DependencyProbeOnly", diagnosticsSource);
        Assert.Contains("BlockedByCudaDriver", diagnosticsSource);
        Assert.Contains("DriverRuntimeMismatchClassified", diagnosticsSource);
        Assert.Contains("PackageConsumerEvidenceClassification", diagnosticsSource);
        Assert.Contains("RuntimeProofBlockerCategory", diagnosticsSource);
        Assert.Contains("RuntimeProofOwnerActionRequired", diagnosticsSource);
        Assert.Contains("ExternalRuntimeProofRequired", diagnosticsSource);
        Assert.Contains("PackageConsumerRuntimeProofPresent", diagnosticsSource);
        Assert.Contains("WhyNotRuntimeProof", diagnosticsSource);
        Assert.Contains("NextOwnerAction", diagnosticsSource);
        Assert.Contains("PluginLibraryDependencyDiagnosticsComplete => false", diagnosticsSource);
        Assert.Contains("LoadRuntimeOwnershipModeled => false", diagnosticsSource);
        Assert.Contains("CanPromoteRuntimeProof => false", diagnosticsSource);
        Assert.Contains("DeferredRowsStillRequired => true", diagnosticsSource);
        Assert.DoesNotContain("public IntPtr", diagnosticsSource);
        Assert.DoesNotContain("public nint", diagnosticsSource);

        Assert.Contains("public TensorRtEngine Deserialize(TensorRtHostMemory hostMemory)", runtimeSource);
        Assert.Contains("public TensorRtEngine Deserialize(byte[] serializedEngine)", runtimeSource);
        Assert.Contains("public TensorRtEngine Deserialize(ArraySegment<byte> serializedEngine)", runtimeSource);
        Assert.Contains("public TensorRtEngine Deserialize(ReadOnlySpan<byte> serializedEngine)", runtimeSource);
        Assert.Contains("public TensorRtEngine Deserialize(Stream serializedEngineStream)", runtimeSource);
        Assert.Contains("public TensorRtEngine DeserializeFromFile(string filePath)", runtimeSource);
        Assert.Contains("Buffer.BlockCopy(serializedEngine.Array, serializedEngine.Offset, buffer, 0, serializedEngine.Count)", runtimeSource);
        Assert.Contains("return Deserialize(serializedEngine.ToArray())", runtimeSource);
        Assert.Contains("serializedEngineStream.CopyTo(copy)", runtimeSource);
        Assert.Contains("File.ReadAllBytes(filePath)", runtimeSource);
        Assert.Contains("GCHandle.Alloc(engineData, GCHandleType.Pinned)", nativeBridgeApi);
        Assert.Contains("pinned.Free()", nativeBridgeApi);
        Assert.Contains("jyppx_trt10_runtime_deserialize_engine", nativeBridgeApi);
        Assert.Contains("jyppx_trt11_runtime_deserialize_engine", nativeBridgeApi);
        Assert.Contains("jyppx_trt8_runtime_deserialize_engine", nativeBridgeApi);

        Assert.Contains("runtime-deserialization-boundary-precheck", smokeProgram);
        Assert.Contains("RuntimeDeserializationBoundaryPrecheck=", smokeProgram);
        Assert.Contains("ManagedByteArrayDeserializeReady", smokeProgram);
        Assert.Contains("SerializedBufferCopiedBeforeInterop", smokeProgram);
        Assert.Contains("EngineHandleOwnedByWrapper", smokeProgram);
        Assert.Contains("LoadRuntimeDeferred", smokeProgram);
        Assert.Contains("RuntimeProofBlocked", smokeProgram);
        Assert.Contains("runtime-deserialization-dependency-diagnostics", smokeProgram);
        Assert.Contains("RuntimeDeserializationDependencyDiagnostics=", smokeProgram);
        Assert.Contains("DependencyProbeOnly", smokeProgram);
        Assert.Contains("BlockedByCudaDriver", smokeProgram);
        Assert.Contains("DriverRuntimeMismatchClassified", smokeProgram);
        Assert.Contains("PackageConsumerEvidenceClassification", smokeProgram);
        Assert.Contains("RuntimeProofBlockerCategory", smokeProgram);
        Assert.Contains("RuntimeProofOwnerActionRequired", smokeProgram);
        Assert.Contains("ExternalRuntimeProofRequired", smokeProgram);
        Assert.Contains("WhyNotRuntimeProof", smokeProgram);
        Assert.Contains("NextOwnerAction", smokeProgram);
        Assert.Contains("CanPromoteRuntimeProof", smokeProgram);

        Assert.Contains("Runtime Deserialization Boundary Precheck", designDoc);
        Assert.Contains("RuntimeEvidenceKind=runtime-precheck", designDoc);
        Assert.Contains("ManagedByteArrayDeserializeReady=True", designDoc);
        Assert.Contains("ManagedStreamDeserializeReady=True", designDoc);
        Assert.Contains("HostMemoryDeserializeReady=True", designDoc);
        Assert.Contains("SerializedBufferCopiedBeforeInterop=True", designDoc);
        Assert.Contains("EngineHandleOwnedByWrapper=True", designDoc);
        Assert.Contains("DirectDeserializeCudaEngineRowsDeferred=False", designDoc);
        Assert.Contains("DirectDeserializeCudaEngineRowsImplemented=True", designDoc);
        Assert.Contains("DirectDeserializeCudaEngineV2RowsDeferred=True", designDoc);
        Assert.Contains("LoadRuntimeDeferred=True", designDoc);
        Assert.Contains("not proof", designDoc);
        Assert.Contains("Runtime Deserialization Dependency Diagnostics", diagnosticsDoc);
        Assert.Contains("RuntimeEvidenceKind=dependency-diagnostics", diagnosticsDoc);
        Assert.Contains("DependencyProbeOnly", diagnosticsDoc);
        Assert.Contains("BlockedByCudaDriver", diagnosticsDoc);
        Assert.Contains("RuntimeProofBlockerCategory", diagnosticsDoc);
        Assert.Contains("PackageConsumerEvidenceClassification", diagnosticsDoc);
        Assert.Contains("ExternalRuntimeProofRequired", diagnosticsDoc);
        Assert.Contains("NextOwnerAction", diagnosticsDoc);
        Assert.Contains("PluginLibraryDependencyDiagnosticsComplete=False", diagnosticsDoc);
        Assert.Contains("LoadRuntimeOwnershipModeled=False", diagnosticsDoc);
        Assert.Contains("CanPromoteRuntimeProof=False", diagnosticsDoc);
        Assert.Contains("blocked-by-cuda-driver", diagnosticsDoc);
        Assert.Contains("runtime-deserialization-boundary-precheck.md", docsIndex);
        Assert.Contains("runtime-deserialization-dependency-diagnostics.md", docsIndex);
        Assert.Contains("runtime-deserialization-boundary-precheck.md", docsToc);
        Assert.Contains("runtime-deserialization-dependency-diagnostics.md", docsToc);
        Assert.Contains("runtime-deserialization-boundary-precheck", manualGroups);
        Assert.Contains("runtime-deserialization-dependency-diagnostics", manualGroups);
        Assert.Contains("runtime-deserialization-boundary-precheck", latest);
        Assert.Contains("runtime-deserialization-dependency-diagnostics", latest);
        Assert.Contains("runtime-deserialization-boundary-precheck", smokeReadme);
        Assert.Contains("runtime-deserialization-dependency-diagnostics", smokeReadme);
        Assert.Contains("New-RuntimeDeserializationBoundaryPrecheckEvidence", readiness);
        Assert.Contains("New-RuntimeDeserializationDependencyDiagnosticsEvidence", readiness);
        Assert.Contains("runtimeDeserializationDependencyDiagnostics", readiness);
        Assert.Contains("runtimeProofBlockerCategory", readiness);
        Assert.Contains("packageConsumerEvidenceClassification", readiness);
        Assert.Contains("externalRuntimeProofRequired", readiness);
        Assert.Contains("nextOwnerAction", readiness);
        Assert.Contains("runtimeProofBlockerOwnerAction", readiness);
        Assert.Contains("New-RuntimeProofBlockerOwnerAction", readiness);
        Assert.Contains("runtimeDeserializationBoundaryPrecheck", readiness);
        Assert.Contains("hasRuntimeDeserializationBoundaryPrecheck", readiness);
        Assert.Contains("runtime-precheck", readiness);
        Assert.Contains("dependency-diagnostics", readiness);
        Assert.Contains("Runtime deserialization boundary precheck missing evidence", readiness);
        Assert.Contains("Runtime deserialization dependency diagnostics missing evidence", readiness);
        Assert.Contains("runtime-deserialization-boundary-precheck", releaseEvidence);
        Assert.Contains("runtime-deserialization-dependency-diagnostics", releaseEvidence);
        Assert.Contains("runtimeDeserializationDependencyDiagnosticsRuntimeProofBlockerCategory", releaseEvidence);
        Assert.Contains("runtimeDeserializationDependencyDiagnosticsPackageConsumerEvidenceClassification", releaseEvidence);
        Assert.Contains("runtimeDeserializationDependencyDiagnosticsExternalRuntimeProofRequired", releaseEvidence);
        Assert.Contains("runtimeDeserializationDependencyDiagnosticsNextOwnerAction", releaseEvidence);
        Assert.Contains("runtimeProofBlockerOwnerActionStatus", releaseEvidence);
        Assert.Contains("direct deserializeCudaEngineV2/loadRuntime rows deferred", releaseEvidence);
        Assert.Contains("TensorRtRuntimeDeserializationBoundaryPrecheck", bridgeConsumer);
        Assert.Contains("runtime-deserialization-boundary-precheck", bridgeConsumer);
        Assert.Contains("LoadRuntimeDeferred", bridgeConsumer);

        Assert.Contains("\"IRuntime\",\"deserializeCudaEngineV2\",\"IRuntime::deserializeCudaEngineV2\",\"runtime-serialization\",\"deferred-only\"", comparison);
        Assert.Contains("\"IRuntime\",\"loadRuntime\",\"IRuntime::loadRuntime\",\"runtime-serialization\",\"deferred-only\"", comparison);
        Assert.Contains("trt10-runtime-deserialize-cuda-engine-v2-deferred", comparison);
        Assert.Contains("trt11-runtime-deserialize-cuda-engine-v2-deferred", comparison);
        Assert.Contains("trt8-runtime-load-runtime-deferred", comparison);
        Assert.Contains("trt10-runtime-load-runtime-deferred", comparison);
        Assert.Contains("trt11-runtime-load-runtime-deferred", comparison);
        Assert.Contains("trt8-runtime-load-runtime-deferred", trt8Deferred);
        Assert.Contains("trt10-runtime-deserialize-cuda-engine-v2-deferred", trt10Deferred);
        Assert.Contains("trt10-runtime-load-runtime-deferred", trt10Deferred);
        Assert.Contains("trt11-runtime-deserialize-cuda-engine-v2-deferred", trt11Deferred);
        Assert.Contains("jyppx_trt11_runtime_load_runtime_deferred", trt11Deferred);
    }

    private static string ReadTensorRtManifest(string lineDirectory, string manifestName)
    {
        return ReadSource("native", "manifests", "tensorrt", lineDirectory, manifestName);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return RepositorySourceReader.Read(path);
    }
}

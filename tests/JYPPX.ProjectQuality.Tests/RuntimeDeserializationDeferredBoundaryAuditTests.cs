using System.Text.Json;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class RuntimeDeserializationDeferredBoundaryAuditTests
{
    [Fact]
    public void AuditRecordsFiveRuntimeBoundaryRowsAsDeferredOnly()
    {
        using JsonDocument document = JsonDocument.Parse(ReadSource("artifacts", "interface-coverage", "runtime-deserialization-deferred-boundary-audit.json"));
        JsonElement root = document.RootElement;
        JsonElement scope = root.GetProperty("scope");

        Assert.Equal("runtime-deserialization-deferred-boundary-audit.v1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("runtime-deserialization-boundary", scope.GetProperty("designGroup").GetString());
        Assert.Equal(5, scope.GetProperty("candidateCount").GetInt32());
        Assert.False(scope.GetProperty("canPromoteWithoutDesignGate").GetBoolean());
        Assert.False(scope.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.True(scope.GetProperty("deferredHistoryRetained").GetBoolean());
        Assert.False(scope.GetProperty("nativePromotionAttempted").GetBoolean());
        Assert.True(scope.GetProperty("publicApiPointerFree").GetBoolean());
        Assert.False(scope.GetProperty("githubActionsUsed").GetBoolean());
        Assert.False(scope.GetProperty("publicPublishSideEffects").GetBoolean());

        JsonElement rows = root.GetProperty("blockedRows");
        Assert.Equal(5, rows.GetArrayLength());
        AssertRow(rows, "IRuntime::deserializeCudaEngineV2", "10", "trt10-runtime-deserialize-cuda-engine-v2-deferred");
        AssertRow(rows, "IRuntime::deserializeCudaEngineV2", "11", "trt11-runtime-deserialize-cuda-engine-v2-deferred");
        AssertRow(rows, "IRuntime::loadRuntime", "8", "trt8-runtime-load-runtime-deferred");
        AssertRow(rows, "IRuntime::loadRuntime", "10", "trt10-runtime-load-runtime-deferred");
        AssertRow(rows, "IRuntime::loadRuntime", "11", "trt11-runtime-load-runtime-deferred");

        foreach (JsonElement row in rows.EnumerateArray())
        {
            Assert.Equal("deferred-only", row.GetProperty("implementationStatus").GetString());
            Assert.Equal("medium", row.GetProperty("candidatePlanRisk").GetString());
            Assert.Equal("keep-deferred", row.GetProperty("decision").GetString());
            Assert.Contains("deferred", row.GetProperty("manifestId").GetString() ?? string.Empty, StringComparison.Ordinal);
            Assert.Contains("JYPPX_TENSORRT_VERSION_MAJOR_NUM ==", row.GetProperty("versionGuard").GetString() ?? string.Empty, StringComparison.Ordinal);
            Assert.False(string.IsNullOrWhiteSpace(row.GetProperty("ownershipBlocker").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(row.GetProperty("safeAlternative").GetString()));
        }
    }

    [Fact]
    public void SafeAlternativePrecheckAndDiagnosticsRemainNonProof()
    {
        TensorRtRuntimeDeserializationBoundaryPrecheckResult precheck =
            TensorRtRuntimeDeserializationBoundaryPrecheck.EvaluateKnownSurface(TensorRtApiLine.TensorRt11);
        TensorRtRuntimeDeserializationDependencyDiagnosticsResult diagnostics =
            TensorRtRuntimeDeserializationDependencyDiagnostics.EvaluateKnownSurface(TensorRtApiLine.TensorRt11);
        TensorRtStreamIoInterfaceInfoDesignGateResult streamIo =
            TensorRtStreamIoInterfaceInfoDesignGate.EvaluateKnownSurface(TensorRtApiLine.TensorRt11);

        Assert.True(precheck.PrecheckReady);
        Assert.True(precheck.SafeDeserializeBridgeReady);
        Assert.False(precheck.DirectDeserializeCudaEngineRowsDeferred);
        Assert.True(precheck.DirectDeserializeCudaEngineRowsImplemented);
        Assert.True(precheck.DirectDeserializeCudaEngineV2RowsDeferred);
        Assert.True(precheck.LoadRuntimeDeferred);
        Assert.False(precheck.CanPromoteRuntimeProof);
        Assert.True(precheck.RuntimeProofBlocked);

        Assert.True(diagnostics.PrecheckReady);
        Assert.True(diagnostics.DependencyProbeOnly);
        Assert.False(diagnostics.PluginLibraryDependencyDiagnosticsComplete);
        Assert.False(diagnostics.LoadRuntimeOwnershipModeled);
        Assert.False(diagnostics.CanPromoteRuntimeProof);
        Assert.True(diagnostics.RuntimeProofBlocked);
        Assert.Equal("full-package-consumer-report-missing", diagnostics.RuntimeProofBlockerCategory);

        Assert.True(streamIo.DesignGateReady);
        Assert.True(streamIo.DirectStreamCallbackRowsDeferred);
        Assert.True(streamIo.ManagedOwnedStreamOwnerLedgerReady);
        Assert.False(streamIo.ManagedOwnerLifetimeReady);
        Assert.False(streamIo.NativeOwnerCreateDestroySymmetric);
        Assert.False(streamIo.CanImplementStreamMetadataNow);
        Assert.False(streamIo.CanPromoteRuntimeProof);
        Assert.True(streamIo.RuntimeProofBlocked);
    }

    [Fact]
    public void AuditIsConsistentWithCoverageManifestsAndDocs()
    {
        string audit = ReadSource("artifacts", "interface-coverage", "runtime-deserialization-deferred-boundary-audit.md");
        string json = ReadSource("artifacts", "interface-coverage", "runtime-deserialization-deferred-boundary-audit.json");
        string candidatePlan = ReadSource("artifacts", "interface-coverage", "deferred-readonly-api-candidate-plan.json");
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");
        string precheckDoc = ReadSource("docs", "articles", "zh-cn", "runtime-deserialization-boundary-precheck.md");
        string dependencyDoc = ReadSource("docs", "articles", "zh-cn", "runtime-deserialization-dependency-diagnostics.md");
        string streamIoDoc = ReadSource("docs", "articles", "zh-cn", "stream-io-interface-info-design-gate.md");
        string trt8Deferred = ReadSource("native", "manifests", "tensorrt", "v8", "trt8-cross-version-seventh-batch-deferred-boundaries.manifest.json");
        string trt10Deferred = ReadSource("native", "manifests", "tensorrt", "v10", "trt10-cross-version-fifth-batch-runtime-serialization-deferred.manifest.json");
        string trt11Deferred = ReadSource("native", "manifests", "tensorrt", "v11", "trt11-twenty-third-batch-deferred-coverage.manifest.json");

        Assert.Contains("runtime-deserialization-deferred-boundary-audit", audit);
        Assert.Contains("TensorRtRuntimeDeserializationBoundaryPrecheck.EvaluateKnownSurface", audit);
        Assert.Contains("TensorRtRuntimeDeserializationDependencyDiagnostics.EvaluateKnownSurface", audit);
        Assert.Contains("TensorRtStreamIoInterfaceInfoDesignGate.EvaluateKnownSurface", audit);
        Assert.Contains("不修改 native ABI surface", audit);
        Assert.Contains("不触发 GitHub Actions", audit);
        Assert.Contains("publicPublishSideEffects", json);
        Assert.Contains("stream reader/writer callback handles", json);

        Assert.Contains("\"designGroup\": \"runtime-deserialization-boundary\"", candidatePlan);
        Assert.Contains("\"IRuntime::deserializeCudaEngineV2 [TRT10]\"", candidatePlan);
        Assert.Contains("\"IRuntime::deserializeCudaEngineV2 [TRT11]\"", candidatePlan);
        Assert.Contains("\"IRuntime::loadRuntime [TRT8]\"", candidatePlan);
        Assert.Contains("\"IRuntime::loadRuntime [TRT10]\"", candidatePlan);
        Assert.Contains("\"IRuntime::loadRuntime [TRT11]\"", candidatePlan);

        Assert.Contains("\"IRuntime\",\"deserializeCudaEngineV2\",\"IRuntime::deserializeCudaEngineV2\",\"runtime-serialization\",\"deferred-only\"", comparison);
        Assert.Contains("\"IRuntime\",\"loadRuntime\",\"IRuntime::loadRuntime\",\"runtime-serialization\",\"deferred-only\"", comparison);
        Assert.Contains("trt8-runtime-load-runtime-deferred", trt8Deferred);
        Assert.Contains("trt10-runtime-deserialize-cuda-engine-v2-deferred", trt10Deferred);
        Assert.Contains("trt10-runtime-load-runtime-deferred", trt10Deferred);
        Assert.Contains("trt11-runtime-deserialize-cuda-engine-v2-deferred", trt11Deferred);
        Assert.Contains("trt11-runtime-load-runtime-deferred", trt11Deferred);

        Assert.Contains("DirectDeserializeCudaEngineV2RowsDeferred=True", precheckDoc);
        Assert.Contains("LoadRuntimeDeferred=True", precheckDoc);
        Assert.Contains("PluginLibraryDependencyDiagnosticsComplete=False", dependencyDoc);
        Assert.Contains("LoadRuntimeOwnershipModeled=False", dependencyDoc);
        Assert.Contains("IStreamReaderV2::seek", streamIoDoc);
        Assert.Contains("IStreamWriter::write", streamIoDoc);
    }

    private static void AssertRow(JsonElement rows, string interfaceName, string line, string manifestId)
    {
        foreach (JsonElement row in rows.EnumerateArray())
        {
            if (row.GetProperty("interface").GetString() == interfaceName &&
                row.GetProperty("tensorRtLine").GetString() == line)
            {
                Assert.Equal(manifestId, row.GetProperty("manifestId").GetString());
                return;
            }
        }

        throw new InvalidOperationException($"Missing runtime boundary audit row: {interfaceName} [{line}].");
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}

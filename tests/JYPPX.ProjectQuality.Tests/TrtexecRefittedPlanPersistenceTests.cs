using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TrtexecRefittedPlanPersistenceTests
{
    [Fact]
    public void PersistenceImplementationDisposesOriginalBeforeIndependentReload()
    {
        string service = ReadSource("src", "JYPPX.TensorRtSharp.Tools", "OnnxEngineBuildService.cs");
        string snapshot = ReadSource("src", "JYPPX.TensorRtSharp.Tools", "OnnxEngineRefitPersistenceSnapshot.cs");
        string diagnostics = ReadSource("src", "JYPPX.TensorRtSharp.Tools", "OnnxEngineBuildDiagnostics.cs");
        string schema = ReadSource("applications", "TensorRtExec", "tensor-rt-exec-report.schema.json");
        string form = ReadSource("applications", "TensorRtExec", "WinForms", "MainForm.cs");

        int runtimeSelection = service.IndexOf("ExternalOnnxRefitReload", StringComparison.Ordinal);
        int helper = service.IndexOf("private static (TensorRtEngine Engine, OnnxEngineRefitPersistenceSnapshot Snapshot) PersistAndReloadRefittedEngine", StringComparison.Ordinal);
        int createConfig = service.IndexOf("refittedEngine.CreateSerializationConfig()", helper, StringComparison.Ordinal);
        int clearExcludeWeights = service.IndexOf("ClearFlag(TensorRtSerializationFlag.ExcludeWeights)", createConfig, StringComparison.Ordinal);
        int serialize = service.IndexOf("refittedEngine.Serialize(serializationConfig)", clearExcludeWeights, StringComparison.Ordinal);
        int dispose = service.IndexOf("refittedEngine.Dispose()", serialize, StringComparison.Ordinal);
        int reload = service.IndexOf("runtime.DeserializeFromFile(persistedPlanPath)", dispose, StringComparison.Ordinal);

        Assert.True(runtimeSelection >= 0 && helper > runtimeSelection);
        Assert.True(createConfig > helper && clearExcludeWeights > createConfig && serialize > clearExcludeWeights && dispose > serialize && reload > dispose);
        Assert.Contains("ArtifactDiffersFromStrippedPlan", snapshot, StringComparison.Ordinal);
        Assert.Contains("OriginalRefittedEngineDisposedBeforeReload", snapshot, StringComparison.Ordinal);
        Assert.Contains("InferenceRanFromReloadedEngine", snapshot, StringComparison.Ordinal);
        Assert.Contains("RefittableWeightsIncludedInSerialization", snapshot, StringComparison.Ordinal);
        Assert.Contains("bool reloadGate = ioTensorCount > 0 && layerCount > 0 && profileCount > 0", service, StringComparison.Ordinal);
        Assert.DoesNotContain("bool reloadGate = reloadRefittable", service, StringComparison.Ordinal);
        Assert.Contains("validateRefittableState: !refitPersistenceSnapshot.Succeeded", service, StringComparison.Ordinal);
        Assert.Contains("full-weight-reload-does-not-require-refittable-state", service, StringComparison.Ordinal);
        Assert.Contains("result.RefitPersistenceSnapshot", diagnostics, StringComparison.Ordinal);
        Assert.Contains("\"RefitPersistenceSnapshot\"", schema, StringComparison.Ordinal);
        Assert.Contains("_saveRefittedEnginePath", form, StringComparison.Ordinal);
        Assert.DoesNotContain("IntPtr", snapshot, StringComparison.Ordinal);
        Assert.DoesNotContain("SafeHandle", snapshot, StringComparison.Ordinal);
    }

    [Fact]
    public void CompactEvidenceProvesPersistDisposeReloadAndSecondProcessMatch()
    {
        string path = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "trtexec-refitted-plan-persistence-evidence.json");
        Assert.True(File.Exists(path), "The compact refitted-plan persistence evidence must be checked in.");

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        JsonElement root = document.RootElement;
        JsonElement trt10 = root.GetProperty("tensorRt10");

        Assert.Equal("trtexec-refitted-plan-persistence-evidence.v1", root.GetProperty("schemaVersion").GetString());
        Assert.True(trt10.GetProperty("persisted").GetBoolean());
        Assert.True(trt10.GetProperty("artifactDiffersFromStrippedPlan").GetBoolean());
        Assert.NotEqual(0, trt10.GetProperty("serializationFlagsBefore").GetInt32() & trt10.GetProperty("excludeWeightsFlag").GetInt32());
        Assert.Equal(0, trt10.GetProperty("serializationFlagsAfter").GetInt32() & trt10.GetProperty("excludeWeightsFlag").GetInt32());
        Assert.True(trt10.GetProperty("refittableWeightsIncludedInSerialization").GetBoolean());
        Assert.True(trt10.GetProperty("originalEngineDisposedBeforeReload").GetBoolean());
        Assert.True(trt10.GetProperty("sameProcessReloadSucceeded").GetBoolean());
        Assert.True(trt10.GetProperty("reloadIoTensorCount").GetInt32() > 0);
        Assert.True(trt10.GetProperty("reloadLayerCount").GetInt32() > 0);
        Assert.True(trt10.GetProperty("reloadOptimizationProfileCount").GetInt32() > 0);
        Assert.True(trt10.GetProperty("reloadContextCreationAllowed").GetBoolean());
        Assert.True(trt10.GetProperty("secondProcessReloadSucceeded").GetBoolean());
        Assert.True(trt10.GetProperty("sameProcessOutputExactMatch").GetBoolean());
        Assert.True(trt10.GetProperty("secondProcessOutputExactMatch").GetBoolean());
        Assert.True(root.GetProperty("tensorRt8").GetProperty("nonDryGuardedBeforeNativeExecution").GetBoolean());
        Assert.Equal("dependency-probe-only", root.GetProperty("tensorRt11").GetProperty("state").GetString());
        Assert.False(root.GetProperty("tensorRt11").GetProperty("saveRefittedEngineApplied").GetBoolean());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("canPublishPublicly").GetBoolean());
    }

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray()));
    }
}

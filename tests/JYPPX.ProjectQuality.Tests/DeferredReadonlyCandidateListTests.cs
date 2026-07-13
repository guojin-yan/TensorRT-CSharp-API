using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DeferredReadonlyCandidateListTests
{
    [Fact]
    public void DeferredReadonlyCandidateListIsMachineReadableAndExcludesUnsafeAreas()
    {
        string path = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-readonly-candidate-list.json");
        Assert.True(File.Exists(path), path);

        string text = File.ReadAllText(path);
        using JsonDocument document = JsonDocument.Parse(text);
        JsonElement root = document.RootElement;

        Assert.Equal("deferred-readonly-candidate-list.v1", root.GetProperty("schemaVersion").GetString());
        Assert.Contains("not runtime proof", root.GetProperty("description").GetString(), StringComparison.OrdinalIgnoreCase);

        JsonElement groups = root.GetProperty("groups");
        foreach (string group in new[]
                 {
                     "pluginFieldMetadata",
                     "engineLayerMetadata",
                     "builderConfigReadback",
                     "errorRecorderSnapshot",
                     "readonlyDiagnostics",
                 })
        {
            Assert.True(groups.TryGetProperty(group, out JsonElement candidates), group);
            Assert.True(candidates.GetArrayLength() >= 1, group);
        }

        foreach (string term in new[]
                 {
                     "candidateId",
                     "apiArea",
                     "riskLevel",
                     "outputMode",
                     "nativeLayerRequired",
                     "managedWrapperRequired",
                     "smokeRequired",
                     "deferReason",
                     "callback trampoline",
                     "allocator callback",
                     "plugin instance create",
                     "plugin instance clone",
                     "plugin enqueue",
                     "plugin resource acquire",
                     "plugin resource release",
                     "borrowed pointer public exposure",
                     "runtime proof",
                     "dimension-expression-snapshot-design-002",
                     "error-recorder-interface-info-design-002",
                     "plugin-creator-v3-metadata-design-003",
                     "algorithm-snapshot-design-003",
                     "stream-io-interface-info-design-004",
                     "calibrator-interface-info-design-004",
                     "logger-finder-metadata-design-004",
                     "debug-listener-interface-info-design-004",
                     "allocator-interface-info-design-004",
                     "candidateMethods",
                     "IPluginCreatorV3One::getPluginName",
                     "IVersionedInterface::getInterfaceInfo",
                     "IAlgorithm::getTimingMSec",
                     "IAlgorithmContext::getName",
                     "IAlgorithmIOInfo::getStrides",
                     "IAlgorithmVariant::getTactic",
                     "IStreamReader::getInterfaceInfo",
                     "IStreamReaderV2::getInterfaceInfo",
                     "IStreamWriter::getInterfaceInfo",
                     "IInt8EntropyCalibrator::getInterfaceInfo",
                     "ILoggerFinder::getInterfaceInfo",
                     "IDebugListener::processDebugTensor",
                     "IGpuAsyncAllocator::getInterfaceInfo",
                     "IOutputAllocator::getInterfaceInfo",
                     "IDimensionExpr::getConstantValue",
                     "IErrorRecorder::getInterfaceInfo",
                     "build-only",
                     "dry-run",
                     "template",
                     "local feed",
                     "ProjectReference",
                     "direct `.nupkg`",
                     "TensorRtExec report",
                     "YoloVision matrix",
                     "OnnxToEngine report",
                     "readonly diagnostics",
                 })
        {
            Assert.Contains(term, text, StringComparison.Ordinal);
        }

        Assert.DoesNotContain("YoloDet", text, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("canPublishPublicly=true", text, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("canCloseReleaseIssue=true", text, StringComparison.OrdinalIgnoreCase);
    }
}

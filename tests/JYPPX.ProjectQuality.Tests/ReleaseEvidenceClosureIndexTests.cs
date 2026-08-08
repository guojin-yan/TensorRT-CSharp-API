using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseEvidenceClosureIndexTests
{
    [Fact]
    public void ClosureIndexKeepsEveryLaneNonProofUntilOwnerEvidenceArrives()
    {
        using JsonDocument document = ReadFinalReleaseJson("release-evidence-closure-index.json");
        JsonElement[] lanes = document.RootElement.GetProperty("lanes").EnumerateArray().ToArray();

        Assert.True(lanes.Length >= 6);
        Assert.All(lanes, static lane =>
        {
            Assert.False(lane.GetProperty("isRuntimeProof").GetBoolean());
            Assert.True(lane.GetProperty("requiredBeforePromotion").GetArrayLength() >= 3);
            Assert.True(lane.GetProperty("nonSubstitutes").GetArrayLength() >= 3);
        });

        string[] laneIds = lanes.Select(static item => item.GetProperty("id").GetString()!).ToArray();
        foreach (string laneId in new[]
        {
            "yolovision-real-model",
            "onnx-to-engine-build-report",
            "tensorrtexec-tool",
            "article-system",
            "package-consumer-runtime",
            "post-publish-verification"
        })
        {
            Assert.Contains(laneId, laneIds);
        }
    }

    [Fact]
    public void ClosureIndexLocksForbiddenSubstitutesAndCloseGates()
    {
        using JsonDocument document = ReadFinalReleaseJson("release-evidence-closure-index.json");
        JsonElement root = document.RootElement;
        string[] forbiddenSubstitutes = root.GetProperty("forbiddenSubstitutes")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        foreach (string forbidden in new[]
        {
            "TensorRtExec report",
            "YoloVision matrix",
            "OnnxToEngine report",
            "build-only",
            "dry-run",
            "ProjectReference",
            "direct .nupkg",
            "article",
            "roadmap"
        })
        {
            Assert.Contains(forbidden, forbiddenSubstitutes);
        }

        JsonElement[] gates = root.GetProperty("closureGates").EnumerateArray().ToArray();
        Assert.All(gates, static gate => Assert.True(gate.GetProperty("requiredForClose").GetBoolean()));
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        string path = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName);
        return JsonDocument.Parse(File.ReadAllText(path));
    }
}

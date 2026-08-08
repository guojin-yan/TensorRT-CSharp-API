using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OwnerRealProofImportAuditBundleTests
{
    [Fact]
    public void OwnerRealProofImportAuditBundleKeepsLanesBlockedAndSeparate()
    {
        using JsonDocument document = ReadFinalReleaseJson("owner-real-proof-import-audit-bundle.json");
        JsonElement root = document.RootElement;

        Assert.Equal("owner-real-proof-import-audit-bundle.v1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("blocked-owner-real-proof-required", root.GetProperty("auditState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(root.GetProperty("importReadiness").GetBoolean());

        string[] laneIds = root.GetProperty("proofLanes")
            .EnumerateArray()
            .Select(static lane => lane.GetProperty("laneId").GetString()!)
            .ToArray();

        Assert.Equal(new[]
        {
            "sample-run-evidence",
            "package-consumer-runtime",
            "post-publish-verification",
            "release-close-owner-approval"
        }, laneIds);

        foreach (JsonElement lane in root.GetProperty("proofLanes").EnumerateArray())
        {
            Assert.False(lane.GetProperty("isProof").GetBoolean());
            Assert.True(lane.GetProperty("requiredOwnerInputs").GetArrayLength() >= 9);
            Assert.Contains("pwsh", lane.GetProperty("strictValidator").GetString(), StringComparison.Ordinal);
            Assert.Contains("-FailOn", lane.GetProperty("strictValidator").GetString(), StringComparison.Ordinal);
        }

        string raw = root.GetRawText();
        Assert.Contains("sample-run-evidence cannot replace package-consumer-runtime", raw, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime cannot replace post-publish verification", raw, StringComparison.Ordinal);
        AssertForbiddenSubstitutes(root.GetProperty("forbiddenProofSubstitutes").EnumerateArray().Select(static item => item.GetString()!).ToArray());
    }

    internal static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    internal static void AssertForbiddenSubstitutes(string[] substitutes)
    {
        foreach (string expected in new[]
        {
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
            "blocked-by-cuda-driver"
        })
        {
            Assert.Contains(expected, substitutes);
        }
    }

}

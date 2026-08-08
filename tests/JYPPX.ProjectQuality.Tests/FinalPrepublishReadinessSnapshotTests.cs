using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class FinalPrepublishReadinessSnapshotTests
{
    [Fact]
    public void SnapshotKeepsReleaseBlockedByOwnerProof()
    {
        using JsonDocument document = ReadFinalReleaseJson("final-prepublish-readiness-snapshot.json");
        JsonElement root = document.RootElement;

        Assert.Equal("final-prepublish-readiness-snapshot", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-proof-missing", root.GetProperty("snapshotState").GetString());
        Assert.Equal("owner-real-proof-missing", root.GetProperty("publishBlockedReason").GetString());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("approvesPublicRelease").GetBoolean());
    }

    [Fact]
    public void SnapshotIncludesFinalGateInputs()
    {
        using JsonDocument document = ReadFinalReleaseJson("final-prepublish-readiness-snapshot.json");
        JsonElement[] gates = document.RootElement.GetProperty("finalGates").EnumerateArray().ToArray();
        Assert.Equal(3, gates.Length);

        foreach (string gateId in new[] { "release-proof-owner-backfill-summary-validation", "release-final-blocker-convergence", "owner-real-proof-final-action-worklist" })
        {
            JsonElement gate = Assert.Single(gates, item => item.GetProperty("id").GetString() == gateId);
            Assert.True(gate.GetProperty("exists").GetBoolean());
            Assert.False(gate.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(gate.GetProperty("canCloseReleaseIssue").GetBoolean());
        }
    }

    [Fact]
    public void SnapshotChecksPackageDocsAndSampleRenameReadiness()
    {
        using JsonDocument document = ReadFinalReleaseJson("final-prepublish-readiness-snapshot.json");
        JsonElement root = document.RootElement;

        Assert.Equal("ready-for-owner-proof", root.GetProperty("packageReadiness").GetProperty("state").GetString());
        Assert.Equal("ready-for-owner-proof", root.GetProperty("docsReadiness").GetProperty("state").GetString());
        Assert.Equal("ready-for-owner-proof", root.GetProperty("sampleRenameReadiness").GetProperty("state").GetString());
        Assert.True(root.GetProperty("sampleRenameReadiness").GetProperty("legacySampleIdentityRemoved").GetBoolean());
        Assert.Equal("YoloVision", root.GetProperty("sampleRenameReadiness").GetProperty("currentSampleName").GetString());

        string solutionText = ReadText("TensorRtSharp.sln");
        Assert.Contains("applications\\YoloVision\\YoloVision.csproj", solutionText, StringComparison.Ordinal);
        Assert.DoesNotContain("samples\\YoloDet\\YoloDet.csproj", solutionText, StringComparison.Ordinal);
    }

    [Fact]
    public void SnapshotKeepsOwnerEvidenceBlocked()
    {
        using JsonDocument document = ReadFinalReleaseJson("final-prepublish-readiness-snapshot.json");
        JsonElement ownerEvidence = document.RootElement.GetProperty("ownerEvidenceReadiness");

        Assert.Equal("blocked-owner-action-required", ownerEvidence.GetProperty("state").GetString());
        Assert.Equal(4, ownerEvidence.GetProperty("missingOwnerEvidenceDirectoryCount").GetInt32());

        foreach (JsonElement check in ownerEvidence.GetProperty("checks").EnumerateArray())
        {
            Assert.True(check.GetProperty("passed").GetBoolean());
            Assert.Equal("blocked-owner-action-required", check.GetProperty("state").GetString());
        }
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(ReadFinalReleaseText(fileName));
    }

    private static string ReadFinalReleaseText(string fileName)
    {
        return ReadText("artifacts", "final-release", fileName);
    }

    private static string ReadText(params string[] pathParts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
    }
}

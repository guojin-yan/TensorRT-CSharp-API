using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class RealExternalProofBackfillExecutionBundleTests
{
    [Fact]
    public void BundleAggregatesExecutionTracksWithoutPromotingProof()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseProofWorklist.ps1"));

        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealExternalProofBackfillExecutionBundle.ps1"));
        Assert.Contains("Real external proof backfill execution bundle written", output, StringComparison.Ordinal);

        using JsonDocument bundleDocument = ReadFinalReleaseJson("real-external-proof-backfill-execution-bundle.json");
        JsonElement bundle = bundleDocument.RootElement;

        Assert.Equal("real-external-proof-backfill-execution-bundle", bundle.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-external-proof-execution-required", bundle.GetProperty("bundleState").GetString());
        Assert.Equal(6, bundle.GetProperty("trackCount").GetInt32());
        Assert.Equal(6, bundle.GetProperty("blockedTrackCount").GetInt32());
        Assert.False(bundle.GetProperty("performsPublish").GetBoolean());
        Assert.False(bundle.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(bundle.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(bundle.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(bundle.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(bundle.GetProperty("isReleaseCloseProof").GetBoolean());
        Assert.Contains("not proof", bundle.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] trackIds = bundle.GetProperty("proofTracks").EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();
        Assert.Contains("package-consumer-runtime-proof-execution", trackIds);
        Assert.Contains("post-publish-verification-execution", trackIds);
        Assert.Contains("linux-runner-proof-execution", trackIds);
        Assert.Contains("real-model-runtime-proof-execution", trackIds);
        Assert.Contains("release-close-owner-input-execution", trackIds);
        Assert.Contains("strict-close-validation-execution", trackIds);

        JsonElement packageConsumerTrack = bundle.GetProperty("proofTracks")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "package-consumer-runtime-proof-execution");
        string[] forbiddenSubstitutes = packageConsumerTrack.GetProperty("forbiddenSubstitutes")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("local feed", forbiddenSubstitutes);
        Assert.Contains("ProjectReference", forbiddenSubstitutes);
        Assert.Contains("direct .nupkg", forbiddenSubstitutes);
        Assert.Contains("worklist", forbiddenSubstitutes);
        Assert.Contains("hash-only audit", forbiddenSubstitutes);

        string markdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "real-external-proof-backfill-execution-bundle.md"));
        Assert.Contains("## Proof Tracks", markdown, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime-proof-execution", markdown, StringComparison.Ordinal);
        Assert.Contains("Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseEvidenceAndDocsIncludeExecutionBundle()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseProofWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealExternalProofBackfillExecutionBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-real-external-proof-execution-required", evidence.GetProperty("realExternalProofBackfillExecutionBundleState").GetString());
        Assert.Equal(6, evidence.GetProperty("realExternalProofBackfillExecutionBundleTrackCount").GetInt32());
        Assert.Equal(6, evidence.GetProperty("realExternalProofBackfillExecutionBundleBlockedTrackCount").GetInt32());
        Assert.False(evidence.GetProperty("realExternalProofBackfillExecutionBundleCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("realExternalProofBackfillExecutionBundleCanPromoteRuntimeProof").GetBoolean());
        Assert.False(evidence.GetProperty("realExternalProofBackfillExecutionBundleIsRuntimeExecutionProof").GetBoolean());
        Assert.False(evidence.GetProperty("realExternalProofBackfillExecutionBundleIsReleaseCloseProof").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "real-external-proof-backfill-execution-bundle");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("guidance", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("artifacts/final-release/real-external-proof-backfill-execution-bundle.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-external-proof-backfill-execution-bundle.md", sourceArtifacts);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string releaseEvidenceDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));
        string bundleDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "real-external-proof-backfill-execution-bundle.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("real-external-proof-backfill-execution-bundle.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("real-external-proof-backfill-execution-bundle.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("Export-RealExternalProofBackfillExecutionBundle.ps1", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("real-external-proof-backfill-execution-bundle", bundleDoc, StringComparison.Ordinal);
        Assert.Contains("real external proof backfill execution bundle", evidenceMarkdown, StringComparison.OrdinalIgnoreCase);
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
    {
        using Process process = new();
        process.StartInfo.FileName = "pwsh";
        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(scriptPath);
        foreach (string argument in arguments)
        {
            process.StartInfo.ArgumentList.Add(argument);
        }

        process.StartInfo.WorkingDirectory = RepositoryPaths.Root;
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.RedirectStandardError = true;
        process.StartInfo.UseShellExecute = false;

        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptPath}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        return stdout;
    }
}

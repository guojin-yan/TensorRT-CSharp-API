using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleaseCloseProofWorklistTests
{
    [Fact]
    public void ReleaseCloseProofWorklistAggregatesFinalCloseBlockersWithoutPromotingProof()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofWorklist.ps1"));

        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseProofWorklist.ps1"));
        Assert.Contains("Release close proof worklist written", output, StringComparison.Ordinal);

        using JsonDocument worklistDocument = ReadFinalReleaseJson("release-close-proof-worklist.json");
        JsonElement worklist = worklistDocument.RootElement;

        Assert.Equal("release-close-proof-worklist", worklist.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-release-close-real-proof-required", worklist.GetProperty("worklistState").GetString());
        Assert.Equal(8, worklist.GetProperty("workItemCount").GetInt32());
        Assert.True(worklist.GetProperty("blockedWorkItemCount").GetInt32() >= 1);
        Assert.True(worklist.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        Assert.True(worklist.GetProperty("failedValidationItemCount").GetInt32() >= 1);
        Assert.False(worklist.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(worklist.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(worklist.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(worklist.GetProperty("performsPublish").GetBoolean());
        Assert.False(worklist.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(worklist.GetProperty("isReleaseCloseProof").GetBoolean());
        Assert.Contains("not runtime proof", worklist.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] itemIds = worklist.GetProperty("workItems").EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();
        Assert.Contains("package-consumer-runtime-proof", itemIds);
        Assert.Contains("post-publish-verification-proof", itemIds);
        Assert.Contains("linux-runner-proof", itemIds);
        Assert.Contains("real-model-runtime-proof", itemIds);
        Assert.Contains("release-issue-close-owner-input", itemIds);
        Assert.Contains("release-issue-close-candidate", itemIds);
        Assert.Contains("release-issue-final-close-decision", itemIds);
        Assert.Contains("strict-close-validator", itemIds);

        JsonElement strictCloseItem = worklist.GetProperty("workItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "strict-close-validator");
        Assert.True(strictCloseItem.GetProperty("blocked").GetBoolean());
        Assert.Contains("FailOnNotCloseReady", strictCloseItem.GetProperty("validatorCommand").GetString(), StringComparison.Ordinal);

        string[] sourceArtifacts = worklist.GetProperty("sourceArtifacts").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-worklist.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/post-publish-verification-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/user-acceptance/sample-run-evidence-record-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/user-acceptance/real-model-owner-handoff.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-issue-close-record-validation.json", sourceArtifacts);

        string markdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-close-proof-worklist.md"));
        Assert.Contains("## Work Items", markdown, StringComparison.Ordinal);
        Assert.Contains("strict-close-validator", markdown, StringComparison.Ordinal);
        Assert.Contains("Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseEvidenceAndDocsIncludeReleaseCloseProofWorklist()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseProofWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-release-close-real-proof-required", evidence.GetProperty("releaseCloseProofWorklistState").GetString());
        Assert.Equal(8, evidence.GetProperty("releaseCloseProofWorklistItemCount").GetInt32());
        Assert.True(evidence.GetProperty("releaseCloseProofWorklistBlockedItemCount").GetInt32() >= 1);
        Assert.True(evidence.GetProperty("releaseCloseProofWorklistFailedActionRequiredCount").GetInt32() >= 1);
        Assert.False(evidence.GetProperty("releaseCloseProofWorklistCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("releaseCloseProofWorklistIsReleaseCloseProof").GetBoolean());
        Assert.False(evidence.GetProperty("releaseCloseProofWorklistIsRuntimeExecutionProof").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "release-close-proof-worklist");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("not proof", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("artifacts/final-release/release-close-proof-worklist.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-close-proof-worklist.md", sourceArtifacts);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string releaseEvidenceDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));
        string worklistDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-close-proof-worklist.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("release-close-proof-worklist.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("release-close-proof-worklist.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("Export-ReleaseCloseProofWorklist.ps1", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("release-close-proof-worklist", worklistDoc, StringComparison.Ordinal);
        Assert.Contains("release close proof worklist", evidenceMarkdown, StringComparison.OrdinalIgnoreCase);
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

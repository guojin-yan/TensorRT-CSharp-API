using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleaseIssueCloseRecordCandidateTests
{
    [Fact]
    public void ReleaseIssueCloseRecordCandidateExportsBlockedCloseInputSurface()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofBackfillExecutionPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofExecutionHandoff.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerExternalProofInputPreflight.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofInputRepairPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofInputDraftPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerExternalProofBackfillOrchestrator.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-StaleReleaseClaims.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseIssueCloseRecordCandidate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseIssueCloseRecordCandidate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument candidateDocument = ReadFinalReleaseJson("release-issue-close-record-candidate.json");
        JsonElement candidate = candidateDocument.RootElement;

        Assert.Equal("release-issue-close-record-candidate", candidate.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-release-close-real-proof-required", candidate.GetProperty("candidateState").GetString());
        Assert.Equal("release-issue-close-record", candidate.GetProperty("proofLineId").GetString());
        Assert.False(candidate.GetProperty("performsPublish").GetBoolean());
        Assert.False(candidate.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(candidate.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains("Test-ReleaseIssueCloseRecord.ps1", candidate.GetProperty("strictCloseValidatorCommand").GetString(), StringComparison.Ordinal);
        Assert.Contains("-FailOnNotCloseReady", candidate.GetProperty("strictCloseValidatorCommand").GetString(), StringComparison.Ordinal);

        string[] rules = candidate.GetProperty("requiredRealInputRules").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string rule in new[]
        {
            "releaseEvidenceBundleSha256",
            "releaseClosePreflightPathAndHash",
            "staleClaimsAuditPathAndHash",
            "postPublishProofValidationPathAndHash",
            "rollbackPlan",
            "ownerFinalCloseDecision",
            "strictCloseValidatorCommand",
        })
        {
            Assert.Contains(rule, rules);
        }

        Assert.StartsWith("<owner-fill-", candidate.GetProperty("rollbackPlan").GetString(), StringComparison.Ordinal);
        Assert.StartsWith("<owner-fill-", candidate.GetProperty("ownerFinalCloseDecision").GetString(), StringComparison.Ordinal);
        Assert.Equal("missing-real-post-publish-proof-validation", candidate.GetProperty("postPublishProofValidationState").GetString());

        using JsonDocument validationDocument = ReadFinalReleaseJson("release-issue-close-record-candidate-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("release-issue-close-record-candidate-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-release-close-real-proof-required", validation.GetProperty("validationState").GetString());
        Assert.True(validation.GetProperty("isValidCandidate").GetBoolean());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("blocked-release-close-real-proof-required", evidence.GetProperty("releaseIssueCloseRecordCandidateState").GetString());
        Assert.Equal("blocked-release-close-real-proof-required", evidence.GetProperty("releaseIssueCloseRecordCandidateValidationState").GetString());
        Assert.False(evidence.GetProperty("releaseIssueCloseRecordCandidateCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("releaseIssueCloseRecordCandidateRollbackPlanReady").GetBoolean());
        Assert.False(evidence.GetProperty("releaseIssueCloseRecordCandidateOwnerDecisionReady").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "release-issue-close-record-candidate");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("owner input surface only", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/release-issue-close-record-candidate.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-issue-close-record-candidate-validation.json", sourceArtifacts);

        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-issue-close-record-candidate.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("release-issue-close-record-candidate", readme, StringComparison.Ordinal);
        Assert.Contains("release-issue-close-record-candidate", readmeZh, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-issue-close-record-candidate.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-issue-close-record-candidate.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("candidateState=blocked-release-close-real-proof-required", article, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("release issue close record candidate: `blocked-release-close-real-proof-required`", evidenceMarkdown, StringComparison.Ordinal);
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

using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleaseCloseStrictRecordCandidateTests
{
    [Fact]
    public void ReleaseCloseStrictRecordCandidateStaysBlockedAndAuditable()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerReleaseExecutionPackage.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerReleaseExecutionPackage.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseIssueFinalCloseDecisionTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseIssueFinalCloseDecision.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalEvidenceFreeze.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalEvidenceFreeze.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealExternalProofOverlayPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-RealExternalProofOverlayPack.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseIssueCloseRecordOverlayCandidate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseIssueCloseRecordOverlayCandidate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerExternalExecutionResultBackfillKit.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerExternalExecutionResultBackfillKit.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerInputCrossHashAudit.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerInputCrossHashAudit.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseStrictRecordCandidate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseCloseStrictRecordCandidate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument candidateDocument = ReadFinalReleaseJson("release-close-strict-record-candidate.json");
        JsonElement candidate = candidateDocument.RootElement;
        Assert.Equal("release-close-strict-record-candidate", candidate.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-release-close-strict-record-owner-input-required", candidate.GetProperty("candidateState").GetString());
        Assert.Equal(0, candidate.GetProperty("mismatchedHashCount").GetInt32());
        Assert.True(candidate.GetProperty("missingOwnerInputCount").GetInt32() >= 1);
        Assert.True(candidate.GetProperty("missingRealProofCount").GetInt32() >= 1);
        Assert.True(candidate.GetProperty("hashLines").GetArrayLength() >= 9);
        Assert.False(candidate.GetProperty("performsPublish").GetBoolean());
        Assert.False(candidate.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(candidate.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains("Test-ReleaseIssueCloseRecord.ps1", candidate.GetProperty("strictValidatorCommand").GetString(), StringComparison.Ordinal);
        Assert.Contains("-FailOnNotCloseReady", candidate.GetProperty("strictValidatorCommand").GetString(), StringComparison.Ordinal);

        using JsonDocument validationDocument = ReadFinalReleaseJson("release-close-strict-record-candidate-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("release-close-strict-record-candidate-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-release-close-strict-record-owner-input-required", validation.GetProperty("validationState").GetString());
        Assert.True(validation.GetProperty("isValidStrictCandidateShape").GetBoolean());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        Assert.Equal(0, validation.GetProperty("mismatchedHashCount").GetInt32());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("blocked-release-close-strict-record-owner-input-required", evidence.GetProperty("releaseCloseStrictRecordCandidateState").GetString());
        Assert.Equal("blocked-release-close-strict-record-owner-input-required", evidence.GetProperty("releaseCloseStrictRecordCandidateValidationState").GetString());
        Assert.Equal(0, evidence.GetProperty("releaseCloseStrictRecordCandidateFailedBlockerCount").GetInt32());
        Assert.True(evidence.GetProperty("releaseCloseStrictRecordCandidateFailedActionRequiredCount").GetInt32() >= 1);
        Assert.Equal(0, evidence.GetProperty("releaseCloseStrictRecordCandidateMismatchedHashCount").GetInt32());
        Assert.False(evidence.GetProperty("releaseCloseStrictRecordCandidateCanCloseReleaseIssue").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "release-close-strict-record-candidate");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("hash consistency and blocked-shape validity cannot substitute", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/release-close-strict-record-candidate.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-close-strict-record-candidate-validation.json", sourceArtifacts);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-close-strict-record-candidate.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("articles/zh-cn/release-close-strict-record-candidate.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-close-strict-record-candidate.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("release-close-strict-record-candidate", readme, StringComparison.Ordinal);
        Assert.Contains("release-close-strict-record-candidate", readmeZh, StringComparison.Ordinal);
        Assert.Contains("candidateState=blocked-release-close-strict-record-owner-input-required", article, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("release close strict record candidate validation: `blocked-release-close-strict-record-owner-input-required`", evidenceMarkdown, StringComparison.Ordinal);
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

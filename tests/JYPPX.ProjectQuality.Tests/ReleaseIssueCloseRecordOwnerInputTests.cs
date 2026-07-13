using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleaseIssueCloseRecordOwnerInputTests
{
    [Fact]
    public void ReleaseIssueCloseRecordOwnerInputExportsBlockedOverlaySurface()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-StaleReleaseClaims.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseIssueCloseRecordOwnerInputTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseIssueCloseRecordOwnerInput.ps1"), "-Strict");
        RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseIssueCloseRecordCandidate.ps1"),
            "-OwnerInputPath",
            "artifacts/final-release/release-issue-close-record-owner-input.template.json");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseIssueCloseRecordCandidate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument templateDocument = ReadFinalReleaseJson("release-issue-close-record-owner-input.template.json");
        JsonElement template = templateDocument.RootElement;
        Assert.Equal("release-issue-close-record-owner-input", template.GetProperty("recordKind").GetString());
        Assert.Equal("template-owner-input-required", template.GetProperty("ownerInputState").GetString());
        Assert.Equal("release-issue-close-record", template.GetProperty("proofLineId").GetString());
        Assert.Contains("Test-ReleaseIssueCloseRecord.ps1", template.GetProperty("strictCloseValidatorCommand").GetString(), StringComparison.Ordinal);
        Assert.False(template.GetProperty("performsPublish").GetBoolean());
        Assert.False(template.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(template.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument validationDocument = ReadFinalReleaseJson("release-issue-close-record-owner-input-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("release-issue-close-record-owner-input-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-input-required", validation.GetProperty("validationState").GetString());
        Assert.True(validation.GetProperty("isValidOwnerInputShape").GetBoolean());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());

        string[] validationItemIds = validation.GetProperty("validationItems")
            .EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();
        Assert.Contains("release-evidence-bundle-hash", validationItemIds);
        Assert.Contains("post-publish-validation-real-proof", validationItemIds);
        Assert.Contains("field-rollbackPlan", validationItemIds);
        Assert.Contains("field-ownerFinalCloseDecision", validationItemIds);

        using JsonDocument candidateDocument = ReadFinalReleaseJson("release-issue-close-record-candidate.json");
        JsonElement candidate = candidateDocument.RootElement;
        Assert.True(candidate.GetProperty("ownerInputOverlayApplied").GetBoolean());
        Assert.Equal("artifacts/final-release/release-issue-close-record-owner-input.template.json", candidate.GetProperty("ownerInputPath").GetString());
        Assert.False(candidate.GetProperty("performsPublish").GetBoolean());
        Assert.False(candidate.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(candidate.GetProperty("canCloseReleaseIssue").GetBoolean());

        string[] candidateSourceArtifacts = candidate.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/release-issue-close-record-owner-input.template.json", candidateSourceArtifacts);
        Assert.Contains("artifacts/final-release/release-issue-close-record-owner-input-validation.json", candidateSourceArtifacts);

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("blocked-owner-input-required", evidence.GetProperty("releaseIssueCloseRecordOwnerInputValidationState").GetString());
        Assert.True(evidence.GetProperty("releaseIssueCloseRecordOwnerInputFailedActionRequiredCount").GetInt32() >= 1);
        Assert.Equal(0, evidence.GetProperty("releaseIssueCloseRecordOwnerInputFailedBlockerCount").GetInt32());
        Assert.False(evidence.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("performsPublish").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "release-issue-close-record-owner-input");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("cannot close", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/release-issue-close-record-owner-input.template.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-issue-close-record-owner-input-validation.json", sourceArtifacts);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-issue-close-record-owner-input.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("articles/zh-cn/release-issue-close-record-owner-input.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-issue-close-record-owner-input.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("release-issue-close-record-owner-input", article, StringComparison.Ordinal);
        Assert.Contains("release issue close record owner input validation: `blocked-owner-input-required`", evidenceMarkdown, StringComparison.Ordinal);
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

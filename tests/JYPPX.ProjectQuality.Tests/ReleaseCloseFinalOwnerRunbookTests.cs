using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleaseCloseFinalOwnerRunbookTests
{
    [Fact]
    public void ReleaseCloseFinalOwnerRunbookStaysBlockedAndAuditable()
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
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofRealBackfillExecutionPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerProofRealBackfillExecutionPack.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseIssueCloseRecordRealInputMap.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseIssueCloseRecordRealInputMap.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofRealInputConvergence.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerProofRealInputConvergence.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseFinalOwnerRunbook.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseCloseFinalOwnerRunbook.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument runbookDocument = ReadFinalReleaseJson("release-close-final-owner-runbook.json");
        JsonElement runbook = runbookDocument.RootElement;
        Assert.Equal("release-close-final-owner-runbook", runbook.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-release-close-final-owner-action-required", runbook.GetProperty("runbookState").GetString());
        Assert.True(runbook.GetProperty("runbookStepCount").GetInt32() >= 12);
        Assert.True(runbook.GetProperty("blockedStepCount").GetInt32() >= 1);
        Assert.True(runbook.GetProperty("ownerActionStepCount").GetInt32() >= 1);
        Assert.True(runbook.GetProperty("strictValidatorCount").GetInt32() >= 1);
        Assert.False(runbook.GetProperty("performsPublish").GetBoolean());
        Assert.False(runbook.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(runbook.GetProperty("canCloseReleaseIssue").GetBoolean());

        string runbookSteps = string.Join('\n', runbook.GetProperty("runbookSteps").EnumerateArray().Select(static item => item.GetRawText()));
        Assert.Contains("public package", runbookSteps, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("clean consumer", runbookSteps, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("final close decision", runbookSteps, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("FailOnNotCloseReady", runbookSteps, StringComparison.Ordinal);

        string[] nonSubstitutes = runbook.GetProperty("nonSubstituteProofKinds").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string requiredKind in new[]
                 {
                     "local feed",
                     "ProjectReference",
                     "direct .nupkg",
                     "template",
                     "draft",
                     "candidate",
                     "schema-only",
                     "preflight-only",
                     "dependency-probe-only",
                     "blocked-by-cuda-driver",
                 })
        {
            Assert.Contains(requiredKind, nonSubstitutes);
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("release-close-final-owner-runbook-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("release-close-final-owner-runbook-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-release-close-final-owner-action-required", validation.GetProperty("validationState").GetString());
        Assert.True(validation.GetProperty("isValidRunbookShape").GetBoolean());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        Assert.True(validation.GetProperty("blockedStepCount").GetInt32() >= 1);
        Assert.True(validation.GetProperty("ownerActionStepCount").GetInt32() >= 1);
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("blocked-release-close-final-owner-action-required", evidence.GetProperty("releaseCloseFinalOwnerRunbookState").GetString());
        Assert.Equal("blocked-release-close-final-owner-action-required", evidence.GetProperty("releaseCloseFinalOwnerRunbookValidationState").GetString());
        Assert.Equal(0, evidence.GetProperty("releaseCloseFinalOwnerRunbookFailedBlockerCount").GetInt32());
        Assert.True(evidence.GetProperty("releaseCloseFinalOwnerRunbookFailedActionRequiredCount").GetInt32() >= 1);
        Assert.True(evidence.GetProperty("releaseCloseFinalOwnerRunbookBlockedStepCount").GetInt32() >= 1);
        Assert.True(evidence.GetProperty("releaseCloseFinalOwnerRunbookOwnerActionStepCount").GetInt32() >= 1);
        Assert.False(evidence.GetProperty("releaseCloseFinalOwnerRunbookCanCloseReleaseIssue").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "release-close-final-owner-runbook");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("final owner execution manual", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/release-close-final-owner-runbook.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-close-final-owner-runbook-validation.json", sourceArtifacts);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-close-final-owner-runbook.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("articles/zh-cn/release-close-final-owner-runbook.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-close-final-owner-runbook.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("release-close-final-owner-runbook", readme, StringComparison.Ordinal);
        Assert.Contains("release-close-final-owner-runbook", readmeZh, StringComparison.Ordinal);
        Assert.Contains("runbookState=blocked-release-close-final-owner-action-required", article, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("release close final owner runbook validation: `blocked-release-close-final-owner-action-required`", evidenceMarkdown, StringComparison.Ordinal);
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

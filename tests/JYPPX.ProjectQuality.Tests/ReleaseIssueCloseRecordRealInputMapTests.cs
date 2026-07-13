using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleaseIssueCloseRecordRealInputMapTests
{
    [Fact]
    public void ReleaseIssueCloseRecordRealInputMapStaysBlockedAndAuditable()
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
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument mapDocument = ReadFinalReleaseJson("release-issue-close-record-real-input-map.json");
        JsonElement map = mapDocument.RootElement;
        Assert.Equal("release-issue-close-record-real-input-map", map.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-release-close-real-input-required", map.GetProperty("mapState").GetString());
        Assert.True(map.GetProperty("mappedInputCount").GetInt32() >= 8);
        Assert.True(map.GetProperty("missingRealInputCount").GetInt32() >= 1);
        Assert.True(map.GetProperty("blockedMappingCount").GetInt32() >= 1);
        Assert.False(map.GetProperty("performsPublish").GetBoolean());
        Assert.False(map.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(map.GetProperty("canCloseReleaseIssue").GetBoolean());

        string[] releaseCloseFields = map.GetProperty("realInputMappings")
            .EnumerateArray()
            .Select(static item => item.GetProperty("releaseCloseField").GetString()!)
            .ToArray();
        Assert.Contains("rollbackPlan", releaseCloseFields);
        Assert.Contains("rollbackOwner", releaseCloseFields);
        Assert.Contains("rollbackTrigger", releaseCloseFields);
        Assert.Contains("ownerFinalCloseDecision", releaseCloseFields);
        Assert.Contains("releaseIssueId", releaseCloseFields);
        Assert.Contains("releaseIssueUrl", releaseCloseFields);
        Assert.Contains("publicChannelPackageSource", releaseCloseFields);
        Assert.Contains("cleanConsumerRuntimeSmokeLog", releaseCloseFields);

        using JsonDocument validationDocument = ReadFinalReleaseJson("release-issue-close-record-real-input-map-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("release-issue-close-record-real-input-map-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-release-close-real-input-required", validation.GetProperty("validationState").GetString());
        Assert.True(validation.GetProperty("isValidRealInputMapShape").GetBoolean());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        Assert.True(validation.GetProperty("missingRealInputCount").GetInt32() >= 1);
        Assert.True(validation.GetProperty("blockedMappingCount").GetInt32() >= 1);
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("blocked-release-close-real-input-required", evidence.GetProperty("releaseIssueCloseRecordRealInputMapState").GetString());
        Assert.Equal("blocked-release-close-real-input-required", evidence.GetProperty("releaseIssueCloseRecordRealInputMapValidationState").GetString());
        Assert.Equal(0, evidence.GetProperty("releaseIssueCloseRecordRealInputMapFailedBlockerCount").GetInt32());
        Assert.True(evidence.GetProperty("releaseIssueCloseRecordRealInputMapFailedActionRequiredCount").GetInt32() >= 1);
        Assert.True(evidence.GetProperty("releaseIssueCloseRecordRealInputMapMissingRealInputCount").GetInt32() >= 1);
        Assert.True(evidence.GetProperty("releaseIssueCloseRecordRealInputMapBlockedMappingCount").GetInt32() >= 1);
        Assert.False(evidence.GetProperty("releaseIssueCloseRecordRealInputMapCanCloseReleaseIssue").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "release-issue-close-record-real-input-map");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("owner input mapping only", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/release-issue-close-record-real-input-map.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-issue-close-record-real-input-map-validation.json", sourceArtifacts);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-issue-close-record-real-input-map.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("articles/zh-cn/release-issue-close-record-real-input-map.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-issue-close-record-real-input-map.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("release-issue-close-record-real-input-map", readme, StringComparison.Ordinal);
        Assert.Contains("release-issue-close-record-real-input-map", readmeZh, StringComparison.Ordinal);
        Assert.Contains("mapState=blocked-release-close-real-input-required", article, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("release issue close record real input map validation: `blocked-release-close-real-input-required`", evidenceMarkdown, StringComparison.Ordinal);
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

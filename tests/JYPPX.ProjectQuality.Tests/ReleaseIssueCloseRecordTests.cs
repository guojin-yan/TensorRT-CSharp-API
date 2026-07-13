using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleaseIssueCloseRecordTests
{
    [Fact]
    public void ReleaseIssueCloseRecordTemplateAndValidatorStayBlockedUntilRealProofExists()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseIssueCloseRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseIssueCloseRecord.ps1"));

        using JsonDocument template = ReadFinalReleaseJson("release-issue-close-record-template.json");
        JsonElement templateRoot = template.RootElement;

        Assert.Equal("release-issue-close-record-template", templateRoot.GetProperty("recordKind").GetString());
        Assert.Equal("template-only", templateRoot.GetProperty("recordState").GetString());
        Assert.Equal("template-only", templateRoot.GetProperty("proofClassification").GetString());
        Assert.Equal("blocked-real-proof-required", templateRoot.GetProperty("closeReadinessState").GetString());
        Assert.False(templateRoot.GetProperty("performsPublish").GetBoolean());
        Assert.False(templateRoot.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(templateRoot.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(templateRoot.GetProperty("canPromoteReleaseIssueCloseRecord").GetBoolean());
        Assert.True(templateRoot.GetProperty("requiredOwnerInputFieldCount").GetInt32() >= 40);

        string[] requiredFields = templateRoot.GetProperty("requiredOwnerInputFields").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("releaseIssue.url", requiredFields);
        Assert.Contains("ownerDecision.finalCloseDecision", requiredFields);
        Assert.Contains("selectedChannel.sourceUri", requiredFields);
        Assert.Contains("packages.managed.nupkgSha256", requiredFields);
        Assert.Contains("packages.runtime.nupkgSha256", requiredFields);
        Assert.Contains("postPublishVerification.isPostPublishVerificationProof", requiredFields);
        Assert.Contains("releaseClosePreflight.canCloseReleaseIssue", requiredFields);
        Assert.Contains("releaseEvidenceBundle.sha256", requiredFields);
        Assert.Contains("rollbackPlan.packageYankOrDeprecatePlan", requiredFields);

        using JsonDocument validation = ReadFinalReleaseJson("release-issue-close-record-validation.json");
        JsonElement validationRoot = validation.RootElement;

        Assert.Equal("release-issue-close-record-validation", validationRoot.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-template-only", validationRoot.GetProperty("validationState").GetString());
        Assert.Equal("template-only", validationRoot.GetProperty("proofClassification").GetString());
        Assert.True(validationRoot.GetProperty("isTemplate").GetBoolean());
        Assert.False(validationRoot.GetProperty("canPromoteReleaseIssueCloseRecord").GetBoolean());
        Assert.False(validationRoot.GetProperty("performsPublish").GetBoolean());
        Assert.False(validationRoot.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validationRoot.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.True(validationRoot.GetProperty("failedValidationItemCount").GetInt32() >= 8);

        JsonElement[] validationItems = validationRoot.GetProperty("validationItems").EnumerateArray().ToArray();
        Assert.Contains(validationItems, static item => item.GetProperty("id").GetString() == "not-template-record" && !item.GetProperty("passed").GetBoolean());
        Assert.Contains(validationItems, static item => item.GetProperty("id").GetString() == "owner-final-close-decision" && !item.GetProperty("passed").GetBoolean());
        Assert.Contains(validationItems, static item => item.GetProperty("id").GetString() == "post-publish-verification-proof" && !item.GetProperty("passed").GetBoolean());
        Assert.Contains(validationItems, static item => item.GetProperty("id").GetString() == "release-close-preflight-passed" && !item.GetProperty("passed").GetBoolean());
        Assert.Contains(validationItems, static item => item.GetProperty("id").GetString() == "evidence-bundle-sha256-matches" && !item.GetProperty("passed").GetBoolean());
    }

    [Fact]
    public void ReleaseIssueCloseRecordPropagatesThroughCloseArtifactsButCannotCloseByDefault()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseIssueCloseRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseIssueCloseRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseClosePreflight.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseRcOwnerHandoff.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerReleaseExecutionPackage.ps1"));

        using JsonDocument preflight = ReadFinalReleaseJson("release-close-preflight.json");
        JsonElement preflightRoot = preflight.RootElement;
        Assert.Equal("blocked-template-only", preflightRoot.GetProperty("releaseIssueCloseRecordValidationState").GetString());
        Assert.False(preflightRoot.GetProperty("releaseIssueCloseRecordCanPromote").GetBoolean());
        Assert.False(preflightRoot.GetProperty("releaseIssueCloseRecordCanCloseReleaseIssue").GetBoolean());
        JsonElement closeItem = preflightRoot.GetProperty("preflightItems").EnumerateArray().Single(static item => item.GetProperty("id").GetString() == "release-issue-close-record");
        Assert.False(closeItem.GetProperty("passed").GetBoolean());
        Assert.Contains("Test-ReleaseIssueCloseRecord.ps1", closeItem.GetProperty("ownerAction").GetString(), StringComparison.Ordinal);

        using JsonDocument evidence = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidenceRoot = evidence.RootElement;
        Assert.Equal("blocked-template-only", evidenceRoot.GetProperty("releaseIssueCloseRecordValidationState").GetString());
        Assert.Equal("template-only", evidenceRoot.GetProperty("releaseIssueCloseRecordProofClassification").GetString());
        Assert.False(evidenceRoot.GetProperty("releaseIssueCloseRecordCanPromote").GetBoolean());
        Assert.False(evidenceRoot.GetProperty("releaseIssueCloseRecordCanPublishPublicly").GetBoolean());
        Assert.False(evidenceRoot.GetProperty("releaseIssueCloseRecordCanCloseReleaseIssue").GetBoolean());
        Assert.Contains(evidenceRoot.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/release-issue-close-record-validation.json");
        Assert.Contains(evidenceRoot.GetProperty("sourceArtifacts").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/release-issue-close-record-template.json");
        Assert.Contains(evidenceRoot.GetProperty("evidenceItems").EnumerateArray(), static item => item.GetProperty("id").GetString() == "release-issue-close-record-validation");

        using JsonDocument handoff = ReadFinalReleaseJson("release-rc-owner-handoff.json");
        JsonElement handoffRoot = handoff.RootElement;
        Assert.Contains(handoffRoot.GetProperty("sourceArtifacts").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/release-issue-close-record-template.json");
        Assert.Contains(handoffRoot.GetProperty("sourceArtifacts").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/release-issue-close-record-validation.json");
        Assert.Equal("release-issue-close-record", handoffRoot.GetProperty("finalCloseOwnerAction").GetProperty("id").GetString());
        Assert.False(handoffRoot.GetProperty("finalCloseOwnerAction").GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument ownerPackage = ReadFinalReleaseJson("owner-release-execution-package.json");
        JsonElement ownerPackageRoot = ownerPackage.RootElement;
        JsonElement closeStep = ownerPackageRoot.GetProperty("executionSteps").EnumerateArray().Single(static item => item.GetProperty("id").GetString() == "validate-release-issue-close-record");
        Assert.Equal("release-close-review", closeStep.GetProperty("phase").GetString());
        Assert.Equal("Test-ReleaseIssueCloseRecord.ps1", closeStep.GetProperty("validator").GetString());
        Assert.Contains(ownerPackageRoot.GetProperty("sourceArtifacts").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/release-issue-close-record-validation.json");
        Assert.Contains(ownerPackageRoot.GetProperty("proofBackfillOrder").EnumerateArray(), static item => item.GetString() == "fill and validate release-issue-close-record.json");
        Assert.False(ownerPackageRoot.GetProperty("performsPublish").GetBoolean());
        Assert.False(ownerPackageRoot.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(ownerPackageRoot.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static string RunPowerShell(string scriptPath)
    {
        using Process process = new();
        process.StartInfo.FileName = "pwsh";
        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(scriptPath);
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

using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleaseOwnerProofInputRecordTests
{
    private static readonly string[] RequiredNonSubstitutes =
    {
        "managed-readiness",
        "CallbackAllocatorReadinessSnapshot",
        "precheck-only",
        "dry-run-only",
        "schema-only",
    };

    [Fact]
    public void ReleaseOwnerProofInputRecordTemplateAndValidatorStayBlockedUntilOwnerEvidenceExists()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseOwnerProofInputRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseOwnerProofInputRecord.ps1"));

        using JsonDocument template = ReadFinalReleaseJson("release-owner-proof-input-record-template.json");
        JsonElement templateRoot = template.RootElement;

        Assert.Equal("release-owner-proof-input-record-template", templateRoot.GetProperty("recordKind").GetString());
        Assert.Equal("template-only", templateRoot.GetProperty("recordState").GetString());
        Assert.Equal("template-only", templateRoot.GetProperty("proofClassification").GetString());
        Assert.False(templateRoot.GetProperty("performsPublish").GetBoolean());
        Assert.False(templateRoot.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(templateRoot.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(templateRoot.GetProperty("canPromoteOwnerProofInput").GetBoolean());
        Assert.True(templateRoot.GetProperty("requiredOwnerInputFieldCount").GetInt32() >= 40);

        string[] requiredFields = templateRoot.GetProperty("requiredOwnerInputFields").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("ownerAuthorization.ownerName", requiredFields);
        Assert.Contains("selectedChannel.channelSourceUri", requiredFields);
        Assert.Contains("packages.managed.nupkgSha256", requiredFields);
        Assert.Contains("packages.runtime.nupkgSha256", requiredFields);
        Assert.Contains("runtimeEvidence.runtimeSmokeLogSha256", requiredFields);
        Assert.Contains("hostMetadata.tensorRtRuntimeVersion", requiredFields);
        Assert.Contains("acknowledgements.hashesComputedFromReferencedFiles", requiredFields);

        string[] templateNonSubstitutes = templateRoot.GetProperty("nonSubstituteProofKinds").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string marker in RequiredNonSubstitutes)
        {
            Assert.Contains(marker, templateNonSubstitutes);
        }

        using JsonDocument validation = ReadFinalReleaseJson("release-owner-proof-input-record-validation.json");
        JsonElement validationRoot = validation.RootElement;

        Assert.Equal("release-owner-proof-input-record-validation", validationRoot.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-template-only", validationRoot.GetProperty("validationState").GetString());
        Assert.Equal("template-only", validationRoot.GetProperty("proofClassification").GetString());
        Assert.True(validationRoot.GetProperty("isTemplate").GetBoolean());
        Assert.False(validationRoot.GetProperty("isRealOwnerProofInput").GetBoolean());
        Assert.False(validationRoot.GetProperty("canPromoteOwnerProofInput").GetBoolean());
        Assert.False(validationRoot.GetProperty("performsPublish").GetBoolean());
        Assert.False(validationRoot.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validationRoot.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.True(validationRoot.GetProperty("failedValidationItemCount").GetInt32() >= 8);

        JsonElement[] validationItems = validationRoot.GetProperty("validationItems").EnumerateArray().ToArray();
        Assert.Contains(validationItems, static item => item.GetProperty("id").GetString() == "not-template-record" && !item.GetProperty("passed").GetBoolean());
        Assert.Contains(validationItems, static item => item.GetProperty("id").GetString() == "clean-consumer-boundary" && !item.GetProperty("passed").GetBoolean());
        Assert.Contains(validationItems, static item => item.GetProperty("id").GetString() == "runtime-evidence-fields" && !item.GetProperty("passed").GetBoolean());
        Assert.Contains(validationItems, static item => item.GetProperty("id").GetString() == "host-metadata-fields" && !item.GetProperty("passed").GetBoolean());
    }

    [Fact]
    public void ReleaseOwnerProofInputRecordPropagatesThroughRcAndReleaseCloseArtifacts()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseOwnerProofInputRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseOwnerProofInputRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseRcProofDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseRcProofDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseRcOwnerHandoff.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseClosePreflight.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerReleaseExecutionPackage.ps1"));

        using JsonDocument dashboard = ReadFinalReleaseJson("release-rc-proof-dashboard.json");
        JsonElement ownerAuthorization = dashboard.RootElement.GetProperty("proofBlockers").EnumerateArray().Single(static item => item.GetProperty("id").GetString() == "owner-authorization");
        Assert.Contains("Test-ReleaseOwnerProofInputRecord.ps1", ownerAuthorization.GetProperty("requiredValidator").GetString(), StringComparison.Ordinal);
        Assert.Contains(ownerAuthorization.GetProperty("sourceArtifacts").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/release-owner-proof-input-record-template.json");
        Assert.Contains(ownerAuthorization.GetProperty("sourceArtifacts").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/release-owner-proof-input-record-validation.json");
        Assert.Contains(ownerAuthorization.GetProperty("requiredOwnerInputs").EnumerateArray(), static item => item.GetString() == "release-owner-proof-input-record.json");

        using JsonDocument dashboardValidation = ReadFinalReleaseJson("release-rc-proof-dashboard-validation.json");
        Assert.True(dashboardValidation.RootElement.GetProperty("ownerProofInputArtifactsPresent").GetBoolean());

        using JsonDocument ownerHandoff = ReadFinalReleaseJson("release-rc-owner-handoff.json");
        Assert.Contains(ownerHandoff.RootElement.GetProperty("sourceArtifacts").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/release-owner-proof-input-record-template.json");
        Assert.Contains(ownerHandoff.RootElement.GetProperty("sourceArtifacts").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/release-owner-proof-input-record-validation.json");

        using JsonDocument evidence = ReadFinalReleaseJson("release-evidence-bundle.json");
        Assert.Equal("blocked-template-only", evidence.RootElement.GetProperty("releaseOwnerProofInputRecordValidationState").GetString());
        Assert.Equal("template-only", evidence.RootElement.GetProperty("releaseOwnerProofInputRecordProofClassification").GetString());
        Assert.False(evidence.RootElement.GetProperty("releaseOwnerProofInputRecordCanPromote").GetBoolean());
        Assert.False(evidence.RootElement.GetProperty("releaseOwnerProofInputRecordCanPublishPublicly").GetBoolean());
        Assert.False(evidence.RootElement.GetProperty("releaseOwnerProofInputRecordCanCloseReleaseIssue").GetBoolean());
        Assert.Contains(evidence.RootElement.GetProperty("sourceEvidence").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/release-owner-proof-input-record-validation.json");
        Assert.Contains(evidence.RootElement.GetProperty("sourceArtifacts").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/release-owner-proof-input-record-template.json");

        using JsonDocument preflight = ReadFinalReleaseJson("release-close-preflight.json");
        Assert.Equal("blocked-template-only", preflight.RootElement.GetProperty("releaseOwnerProofInputValidationState").GetString());
        Assert.False(preflight.RootElement.GetProperty("releaseOwnerProofInputCanPromote").GetBoolean());
        JsonElement preflightItem = preflight.RootElement.GetProperty("preflightItems").EnumerateArray().Single(static item => item.GetProperty("id").GetString() == "release-owner-proof-input-record");
        Assert.False(preflightItem.GetProperty("passed").GetBoolean());
        Assert.Contains("Test-ReleaseOwnerProofInputRecord.ps1", preflightItem.GetProperty("ownerAction").GetString(), StringComparison.Ordinal);

        using JsonDocument ownerPackage = ReadFinalReleaseJson("owner-release-execution-package.json");
        JsonElement ownerProofStep = ownerPackage.RootElement.GetProperty("executionSteps").EnumerateArray().Single(static item => item.GetProperty("id").GetString() == "validate-owner-proof-input-record");
        Assert.Equal("owner-proof-input", ownerProofStep.GetProperty("phase").GetString());
        Assert.Equal("Test-ReleaseOwnerProofInputRecord.ps1", ownerProofStep.GetProperty("validator").GetString());
        Assert.False(ownerPackage.RootElement.GetProperty("performsPublish").GetBoolean());
        Assert.False(ownerPackage.RootElement.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(ownerPackage.RootElement.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains(ownerPackage.RootElement.GetProperty("sourceArtifacts").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/release-owner-proof-input-record-validation.json");
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

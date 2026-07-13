using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleaseProofOwnerInputSchemaAndForbiddenScanTests
{
    [Fact]
    public void ReleaseProofSurfacesProjectOwnerInputSchemaAndForbiddenSubstituteScan()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Import-PackageConsumerRuntimeProofOwnerInput.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalReleaseDryRun.ps1"), "-AllowRuntimeSmokeBlocked", "-WarnOnly");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalReleaseCloseBlockerDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalReleaseCloseBlockerDashboard.ps1"), "-Strict");

        using JsonDocument releaseEvidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement releaseEvidence = releaseEvidenceDocument.RootElement;
        Assert.True(releaseEvidence.GetProperty("ownerInputSchemaReady").GetBoolean());
        Assert.True(releaseEvidence.GetProperty("packageConsumerRuntimeProofOwnerInputSchemaReady").GetBoolean());
        Assert.True(releaseEvidence.GetProperty("packageConsumerRuntimeProofOwnerInputSchemaFieldCount").GetInt32() >= 40);
        Assert.Equal(0, releaseEvidence.GetProperty("packageConsumerRuntimeProofOwnerInputSchemaPlaceholderAllowedFieldCount").GetInt32());
        Assert.False(releaseEvidence.GetProperty("packageConsumerRuntimeProofOwnerInputSchemaCanPromoteRuntimeProof").GetBoolean());
        Assert.False(releaseEvidence.GetProperty("packageConsumerRuntimeProofOwnerInputSchemaCanPublishPublicly").GetBoolean());
        Assert.False(releaseEvidence.GetProperty("packageConsumerRuntimeProofOwnerInputSchemaCanCloseReleaseIssue").GetBoolean());
        Assert.Equal("blocked-forbidden-substitute-detected", releaseEvidence.GetProperty("forbiddenSubstituteScanState").GetString());
        Assert.True(releaseEvidence.GetProperty("detectedForbiddenSubstituteCount").GetInt32() >= 1);
        Assert.False(releaseEvidence.GetProperty("packageConsumerRuntimeProofForbiddenSubstituteScanCanPromoteRuntimeProof").GetBoolean());
        Assert.False(releaseEvidence.GetProperty("packageConsumerRuntimeProofForbiddenSubstituteScanCanPublishPublicly").GetBoolean());
        Assert.False(releaseEvidence.GetProperty("packageConsumerRuntimeProofForbiddenSubstituteScanCanCloseReleaseIssue").GetBoolean());
        Assert.False(releaseEvidence.GetProperty("cleanOwnerInputReady").GetBoolean());
        Assert.False(releaseEvidence.GetProperty("ownerInputForbiddenSubstituteFree").GetBoolean());
        Assert.False(releaseEvidence.GetProperty("ownerInputHashFieldsReady").GetBoolean());
        Assert.False(releaseEvidence.GetProperty("ownerInputSmokeLogReady").GetBoolean());
        Assert.False(releaseEvidence.GetProperty("ownerInputCanPromoteRuntimeProof").GetBoolean());

        JsonElement[] evidenceItems = releaseEvidence.GetProperty("evidenceItems").EnumerateArray().ToArray();
        Assert.Contains(evidenceItems, item => item.GetProperty("id").GetString() == "package-consumer-runtime-proof-owner-input-schema");
        Assert.Contains(evidenceItems, item => item.GetProperty("id").GetString() == "package-consumer-runtime-proof-forbidden-substitute-scan");

        string[] releaseEvidenceSources = releaseEvidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.json", releaseEvidenceSources);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.md", releaseEvidenceSources);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.json", releaseEvidenceSources);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.md", releaseEvidenceSources);

        using JsonDocument dryRunDocument = ReadFinalReleaseJson("final-release-dry-run-summary.json");
        JsonElement dryRun = dryRunDocument.RootElement;
        Assert.True(dryRun.GetProperty("ownerInputSchemaReady").GetBoolean());
        Assert.Equal("blocked-forbidden-substitute-detected", dryRun.GetProperty("forbiddenSubstituteScanState").GetString());
        Assert.True(dryRun.GetProperty("detectedForbiddenSubstituteCount").GetInt32() >= 1);
        Assert.Equal("blocked-forbidden-substitute-detected", dryRun.GetProperty("ownerProofSchemaScanStatus").GetString());
        Assert.False(dryRun.GetProperty("ownerInputSchemaCanPromoteRuntimeProof").GetBoolean());
        Assert.False(dryRun.GetProperty("forbiddenSubstituteScanCanPromoteRuntimeProof").GetBoolean());
        Assert.False(dryRun.GetProperty("forbiddenSubstituteScanCanCloseReleaseIssue").GetBoolean());
        Assert.False(dryRun.GetProperty("cleanOwnerInputReady").GetBoolean());
        Assert.Equal("blocked-owner-input-required", dryRun.GetProperty("ownerInputReadinessStatus").GetString());
        Assert.False(dryRun.GetProperty("ownerInputCanPromoteRuntimeProof").GetBoolean());

        JsonElement[] gates = dryRun.GetProperty("gates").EnumerateArray().ToArray();
        Assert.Contains(gates, item =>
            item.GetProperty("name").GetString() == "OwnerProof schema and forbidden substitute scan" &&
            item.GetProperty("status").GetString() == "blocked-forbidden-substitute-detected");

        using JsonDocument dashboardDocument = ReadFinalReleaseJson("final-release-close-blocker-dashboard.json");
        JsonElement dashboard = dashboardDocument.RootElement;
        Assert.True(dashboard.GetProperty("ownerInputSchemaReady").GetBoolean());
        Assert.Equal("blocked-forbidden-substitute-detected", dashboard.GetProperty("forbiddenSubstituteScanState").GetString());
        Assert.True(dashboard.GetProperty("detectedForbiddenSubstituteCount").GetInt32() >= 1);
        Assert.False(dashboard.GetProperty("performsPublish").GetBoolean());
        Assert.False(dashboard.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(dashboard.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(dashboard.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(dashboard.GetProperty("cleanOwnerInputReady").GetBoolean());
        Assert.False(dashboard.GetProperty("ownerInputCanPromoteRuntimeProof").GetBoolean());
        Assert.Equal("blocked-owner-compatible-host-runtime-smoke-field-alignment", dashboard.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentState").GetString());
        Assert.Equal("blocked-owner-compatible-host-runtime-smoke-field-alignment-valid", dashboard.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState").GetString());
        Assert.Equal("Smoke=not-requested", dashboard.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus").GetString());
        Assert.True(dashboard.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount").GetInt32() >= 30);
        Assert.Equal(0, dashboard.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount").GetInt32());
        Assert.Equal(0, dashboard.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount").GetInt32());

        JsonElement[] blockers = dashboard.GetProperty("blockers").EnumerateArray().ToArray();
        JsonElement alignmentBlocker = Assert.Single(blockers, item => item.GetProperty("blockerId").GetString() == "package-consumer-owner-runtime-smoke-field-alignment");
        Assert.False(alignmentBlocker.GetProperty("ready").GetBoolean());
        Assert.False(alignmentBlocker.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(alignmentBlocker.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(alignmentBlocker.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains("Field alignment cannot prove runtime execution", alignmentBlocker.GetProperty("whyNonSubstitute").GetString(), StringComparison.Ordinal);
        JsonElement schemaScanBlocker = Assert.Single(blockers, item => item.GetProperty("blockerId").GetString() == "package-consumer-runtime-ownerproof-schema-scan");
        Assert.False(schemaScanBlocker.GetProperty("ready").GetBoolean());
        Assert.False(schemaScanBlocker.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(schemaScanBlocker.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(schemaScanBlocker.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains("cannot prove runtime execution", schemaScanBlocker.GetProperty("whyNonSubstitute").GetString(), StringComparison.Ordinal);

        string[] dashboardSources = dashboard.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.json", dashboardSources);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.json", dashboardSources);
        Assert.Contains("artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment.json", dashboardSources);
        Assert.Contains("artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment-validation.json", dashboardSources);

        using JsonDocument validationDocument = ReadFinalReleaseJson("final-release-close-blocker-dashboard-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-release-close-blocker-dashboard-ready", validation.GetProperty("validationState").GetString());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains(
            validation.GetProperty("validationItems").EnumerateArray(),
            item => item.GetProperty("id").GetString() == "ownerproof-schema-scan-projected" && item.GetProperty("passed").GetBoolean());
        Assert.Contains(
            validation.GetProperty("validationItems").EnumerateArray(),
            item => item.GetProperty("id").GetString() == "clean-owner-input-readiness-projected" && item.GetProperty("passed").GetBoolean());
        Assert.Contains(
            validation.GetProperty("validationItems").EnumerateArray(),
            item => item.GetProperty("id").GetString() == "owner-runtime-smoke-field-alignment-projected" && item.GetProperty("passed").GetBoolean());
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

using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerRealEvidenceInputAcceptanceTests
{
    [Fact]
    public void MissingOwnerInputIsBlockedAndDoesNotFakeProof()
    {
        RunBaseExports();
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Import-OwnerRealEvidenceInput.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerRealEvidenceInput.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerRealEvidenceAcceptanceDashboard.ps1"));

        using JsonDocument importDocument = ReadFinalReleaseJson("owner-real-evidence-input-import.json");
        JsonElement import = importDocument.RootElement;

        Assert.Equal("owner-real-evidence-input-import", import.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-input-file-required", import.GetProperty("importState").GetString());
        Assert.False(import.GetProperty("inputExists").GetBoolean());
        Assert.Equal(0, import.GetProperty("acceptedLaneCount").GetInt32());
        Assert.Equal(6, import.GetProperty("failedActionRequiredCount").GetInt32());
        AssertFalsePublishAndCloseFlags(import);
        Assert.False(import.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.False(import.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(import.GetProperty("executesDotnetNugetPush").GetBoolean());
        Assert.False(import.GetProperty("uploadsGitHubReleaseAssets").GetBoolean());

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-real-evidence-input-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("owner-real-evidence-input-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-evidence-input-required-validation-valid", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("failedActionRequiredCount").GetInt32());
        AssertFalsePublishAndCloseFlags(validation);

        using JsonDocument dashboardDocument = ReadFinalReleaseJson("owner-real-evidence-acceptance-dashboard.json");
        JsonElement dashboard = dashboardDocument.RootElement;

        Assert.Equal("owner-real-evidence-acceptance-dashboard", dashboard.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-evidence-required", dashboard.GetProperty("dashboardState").GetString());
        Assert.False(dashboard.GetProperty("canPrepareOwnerOnlyPublishCandidate").GetBoolean());
        AssertFalsePublishAndCloseFlags(dashboard);
    }

    [Fact]
    public void OwnerInputSchemaAndTemplateCoverSixLanesRequiredFieldsAndForbiddenSubstitutes()
    {
        string schemaPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "owner-input", "owner-real-evidence-input.schema.json");
        string templatePath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "owner-input", "owner-real-evidence-input.template.json");
        Assert.True(File.Exists(schemaPath), schemaPath);
        Assert.True(File.Exists(templatePath), templatePath);

        string schema = File.ReadAllText(schemaPath);
        string template = File.ReadAllText(templatePath);
        using JsonDocument schemaDocument = JsonDocument.Parse(schema);
        using JsonDocument templateDocument = JsonDocument.Parse(template);

        Assert.Equal("https://json-schema.org/draft/2020-12/schema", schemaDocument.RootElement.GetProperty("$schema").GetString());
        Assert.Equal("owner-real-evidence-input", templateDocument.RootElement.GetProperty("recordKind").GetString());
        Assert.Equal(6, templateDocument.RootElement.GetProperty("lanes").GetArrayLength());

        foreach (string laneId in RequiredFinalActionIds)
        {
            Assert.Contains(laneId, schema, StringComparison.Ordinal);
            Assert.Contains(laneId, template, StringComparison.Ordinal);
        }

        foreach (string required in new[] { "artifactPath", "artifactSha256", "logPath", "logSha256", "recordPath", "recordSha256", "hostMetadata", "commandLine", "exitCode", "startedAtUtc", "finishedAtUtc", "ownerReview", "decision", "publicPackageIdentity", "publicChannel", "rollbackPlan" })
        {
            Assert.Contains(required, schema, StringComparison.Ordinal);
        }

        foreach (string forbidden in ForbiddenSubstitutes)
        {
            Assert.Contains(forbidden, schema + template, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void OwnerOnlyPublishCandidateRemainsManualOnlyAndBlockedWithoutRealInput()
    {
        RunBaseExports();
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Import-OwnerRealEvidenceInput.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerRealEvidenceInput.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerRealEvidenceAcceptanceDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerOnlyPublishExecutionCandidate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerOnlyPublishExecutionCandidate.ps1"), "-Strict");

        using JsonDocument candidateDocument = ReadFinalReleaseJson("owner-only-publish-execution-candidate.json");
        JsonElement candidate = candidateDocument.RootElement;

        Assert.Equal("owner-only-publish-execution-candidate", candidate.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-evidence-required", candidate.GetProperty("candidateState").GetString());
        Assert.True(candidate.GetProperty("ownerManualOnly").GetBoolean());
        Assert.Equal(0, candidate.GetProperty("acceptedLaneCount").GetInt32());
        Assert.Equal(6, candidate.GetProperty("blockedLaneCount").GetInt32());
        AssertFalsePublishAndCloseFlags(candidate);
        Assert.False(candidate.GetProperty("executesDotnetNugetPush").GetBoolean());
        Assert.False(candidate.GetProperty("uploadsGitHubReleaseAssets").GetBoolean());
        Assert.False(candidate.GetProperty("closesReleaseIssue").GetBoolean());

        foreach (JsonElement command in candidate.GetProperty("manualCommands").EnumerateArray())
        {
            Assert.Equal("owner-manual-only", command.GetProperty("mode").GetString());
            Assert.False(command.GetProperty("executesDotnetNugetPush").GetBoolean());
            Assert.False(command.GetProperty("uploadsGitHubReleaseAssets").GetBoolean());
        }

        string raw = candidate.GetRawText();
        Assert.Contains("dotnet nuget push", raw, StringComparison.Ordinal);
        Assert.Contains("GitHub Release assets", raw, StringComparison.Ordinal);
        Assert.Contains("public-channel install", raw, StringComparison.Ordinal);

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-only-publish-execution-candidate-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("owner-only-publish-execution-candidate-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-only-publish-execution-candidate-valid", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("failedActionRequiredCount").GetInt32());
        AssertFalsePublishAndCloseFlags(validation);
    }

    [Fact]
    public void FinalActionMapLinksOwnerInputAcceptanceSurfacesAndKeepsActionRequired()
    {
        RunBaseExports();
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Import-OwnerRealEvidenceInput.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerRealEvidenceInput.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerRealEvidenceAcceptanceDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerOnlyPublishExecutionCandidate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerOnlyPublishExecutionCandidate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalPublishActionRequiredEvidenceMap.ps1"));

        using JsonDocument document = ReadFinalReleaseJson("final-publish-action-required-evidence-map.json");
        JsonElement root = document.RootElement;

        Assert.Equal("final-publish-action-required-evidence-map", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-owner-evidence-required", root.GetProperty("mapState").GetString());
        Assert.Equal(6, root.GetProperty("actionRequiredCount").GetInt32());
        Assert.Equal("artifacts/final-release/owner-real-evidence-input-import.json", root.GetProperty("sourceOwnerRealEvidenceInputImport").GetString());
        Assert.Equal("artifacts/final-release/owner-real-evidence-input-validation.json", root.GetProperty("sourceOwnerRealEvidenceInputValidation").GetString());
        Assert.Equal("artifacts/final-release/owner-real-evidence-acceptance-dashboard.json", root.GetProperty("sourceOwnerRealEvidenceAcceptanceDashboard").GetString());
        Assert.Equal("artifacts/final-release/owner-only-publish-execution-candidate.json", root.GetProperty("sourceOwnerOnlyPublishExecutionCandidate").GetString());
        Assert.Equal("blocked-owner-input-file-required", root.GetProperty("ownerRealEvidenceInputImportState").GetString());
        Assert.Equal("blocked-owner-real-evidence-input-required-validation-valid", root.GetProperty("ownerRealEvidenceInputValidationState").GetString());
        Assert.Equal("blocked-owner-real-evidence-required", root.GetProperty("ownerRealEvidenceAcceptanceDashboardState").GetString());
        Assert.Equal("blocked-owner-real-evidence-required", root.GetProperty("ownerOnlyPublishExecutionCandidateState").GetString());
        Assert.Equal("blocked-owner-only-publish-execution-candidate-valid", root.GetProperty("ownerOnlyPublishExecutionCandidateValidationState").GetString());
        Assert.Equal(0, root.GetProperty("ownerRealEvidenceInputAcceptedLaneCount").GetInt32());
        Assert.Equal(6, root.GetProperty("ownerRealEvidenceInputBlockedLaneCount").GetInt32());
        AssertFalsePublishAndCloseFlags(root);
    }

    [Fact]
    public void FinalEvidenceFreezeCarriesOwnerInputAcceptanceStatesWithoutChangingPublishFlags()
    {
        RunBaseExports();
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Import-OwnerRealEvidenceInput.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerRealEvidenceInput.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerRealEvidenceAcceptanceDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerOnlyPublishExecutionCandidate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerOnlyPublishExecutionCandidate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalPublishActionRequiredEvidenceMap.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCandidateFinalEvidenceFreeze.ps1"));

        using JsonDocument document = ReadFinalReleaseJson("release-candidate-final-evidence-freeze.json");
        JsonElement root = document.RootElement;

        Assert.Equal("release-candidate-final-evidence-freeze", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-proof-required", root.GetProperty("freezeState").GetString());
        Assert.Equal("blocked-owner-input-file-required", root.GetProperty("ownerRealEvidenceInputImportState").GetString());
        Assert.Equal("blocked-owner-real-evidence-input-required-validation-valid", root.GetProperty("ownerRealEvidenceInputValidationState").GetString());
        Assert.Equal(0, root.GetProperty("ownerRealEvidenceInputAcceptedLaneCount").GetInt32());
        Assert.Equal(6, root.GetProperty("ownerRealEvidenceInputBlockedLaneCount").GetInt32());
        Assert.Equal(0, root.GetProperty("ownerRealEvidenceInputValidationFailedBlockerCount").GetInt32());
        Assert.Equal(6, root.GetProperty("ownerRealEvidenceInputValidationFailedActionRequiredCount").GetInt32());
        Assert.Equal("blocked-owner-real-evidence-required", root.GetProperty("ownerRealEvidenceAcceptanceDashboardState").GetString());
        Assert.False(root.GetProperty("ownerRealEvidenceAcceptanceCanPrepareOwnerOnlyPublishCandidate").GetBoolean());
        Assert.Equal("blocked-owner-real-evidence-required", root.GetProperty("ownerOnlyPublishExecutionCandidateState").GetString());
        Assert.Equal("blocked-owner-only-publish-execution-candidate-valid", root.GetProperty("ownerOnlyPublishExecutionCandidateValidationState").GetString());
        Assert.Equal("blocked-real-owner-evidence-required", root.GetProperty("finalPublishActionRequiredEvidenceMapState").GetString());
        Assert.Equal(6, root.GetProperty("finalPublishActionRequiredEvidenceMapActionRequiredCount").GetInt32());
        AssertFalsePublishAndCloseFlags(root);
    }

    [Fact]
    public void PublicDocsGateStillHasNoBlockedProofOrPublicationClaims()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PublicDocsAndPackageMetadataGate.ps1"));

        using JsonDocument document = ReadFinalReleaseJson("public-docs-package-metadata-gate.json");
        JsonElement root = document.RootElement;

        Assert.Equal("public-docs-package-metadata-gate", root.GetProperty("recordKind").GetString());
        Assert.Equal(0, root.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(0, root.GetProperty("blockedMatchCount").GetInt32());
        AssertFalsePublishAndCloseFlags(root);
    }

    private static readonly string[] RequiredFinalActionIds =
    {
        "real-model-runtime-owner-proof-required",
        "package-consumer-runtime-owner-proof-required",
        "post-publish-verification-owner-proof-required",
        "final-owner-real-input-template-pack-owner-input-required",
        "owner-external-proof-result-import-owner-proof-required",
        "owner-result-candidate-bridge-real-proof-required",
    };

    private static readonly string[] ForbiddenSubstitutes =
    {
        "local feed",
        "ProjectReference",
        "direct nupkg",
        "dashboard",
        "template",
        "draft",
        "candidate",
        "build report",
        "article",
        "screenshot",
        "OnnxToEngine report",
        "TensorRtExec report",
        "YoloVision matrix",
    };

    private static void RunBaseExports()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerDualRouteProofPlan.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CleanExternalConsumerExecutionKit.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationIntakeMap.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalPublishActionRequiredEvidenceMap.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseStrictProofExecutionOrder.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerRealEvidenceImportPacket.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerRealEvidenceImportPacket.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PublicPublishAuthorizationPreflight.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PublicPublishAuthorizationPreflight.ps1"), "-Strict");
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static void AssertFalsePublishAndCloseFlags(JsonElement root)
    {
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
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

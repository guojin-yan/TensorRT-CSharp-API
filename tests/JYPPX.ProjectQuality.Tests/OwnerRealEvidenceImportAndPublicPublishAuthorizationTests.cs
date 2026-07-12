using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerRealEvidenceImportAndPublicPublishAuthorizationTests
{
    [Fact]
    public void OwnerRealEvidenceImportPacketCoversAllFinalActionRequiredLanes()
    {
        RunStageExports();
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerRealEvidenceImportPacket.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerRealEvidenceImportPacket.ps1"), "-Strict");

        using JsonDocument packetDocument = ReadFinalReleaseJson("owner-real-evidence-import-packet.json");
        JsonElement packet = packetDocument.RootElement;

        Assert.Equal("owner-real-evidence-import-packet", packet.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-evidence-required", packet.GetProperty("packetState").GetString());
        Assert.Equal(6, packet.GetProperty("laneCount").GetInt32());
        Assert.Equal(6, packet.GetProperty("actionRequiredCount").GetInt32());
        AssertFalsePublishAndCloseFlags(packet);
        Assert.False(packet.GetProperty("performsRuntimeExecution").GetBoolean());
        Assert.False(packet.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.False(packet.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(packet.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(packet.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(packet.GetProperty("isReleaseCloseProof").GetBoolean());

        JsonElement[] lanes = packet.GetProperty("lanes").EnumerateArray().ToArray();
        string[] laneIds = lanes.Select(static lane => lane.GetProperty("laneId").GetString()!).ToArray();
        foreach (string expected in RequiredFinalActionIds)
        {
            Assert.Contains(expected, laneIds);
        }

        foreach (JsonElement lane in lanes)
        {
            Assert.Equal("blocked-owner-real-evidence-required", lane.GetProperty("laneState").GetString());
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("ownerInputArtifact").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("expectedRecord").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("validator").GetString()));
            Assert.True(lane.GetProperty("validators").GetArrayLength() >= 1);
            Assert.True(lane.GetProperty("requiredFields").GetArrayLength() >= 3);
            Assert.True(lane.GetProperty("requiredFiles").GetArrayLength() >= 3);
            Assert.True(lane.GetProperty("requiredHashes").GetArrayLength() >= 3);
            Assert.True(lane.GetProperty("requiredOwnerReview").GetArrayLength() >= 3);
            Assert.False(lane.GetProperty("performsPublish").GetBoolean());
            Assert.False(lane.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(lane.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(lane.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        }

        string raw = packet.GetRawText();
        foreach (string forbidden in new[] { "template-only record", "dashboard-only record", "local feed", "ProjectReference", "direct nupkg" })
        {
            Assert.Contains(forbidden, raw, StringComparison.Ordinal);
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-real-evidence-import-packet-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("owner-real-evidence-import-packet-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-evidence-required-packet-shape-valid", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("failedActionRequiredCount").GetInt32());
        AssertFalsePublishAndCloseFlags(validation);
    }

    [Fact]
    public void PublicPublishAuthorizationPreflightBlocksPublishAndCloseUntilRealOwnerEvidencePasses()
    {
        RunStageExports();
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerRealEvidenceImportPacket.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerRealEvidenceImportPacket.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PublicPublishAuthorizationPreflight.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PublicPublishAuthorizationPreflight.ps1"), "-Strict");

        using JsonDocument preflightDocument = ReadFinalReleaseJson("public-publish-authorization-preflight.json");
        JsonElement preflight = preflightDocument.RootElement;

        Assert.Equal("public-publish-authorization-preflight", preflight.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-public-publish-authorization-required", preflight.GetProperty("preflightState").GetString());
        Assert.Equal(6, preflight.GetProperty("ownerImportLaneCount").GetInt32());
        Assert.Equal(6, preflight.GetProperty("finalActionRequiredCount").GetInt32());
        Assert.True(preflight.GetProperty("releaseCloseStrictExecutionStepCount").GetInt32() >= 6);
        Assert.True(preflight.GetProperty("requirementCount").GetInt32() >= 6);
        Assert.Equal(preflight.GetProperty("requirementCount").GetInt32(), preflight.GetProperty("blockedRequirementCount").GetInt32());
        AssertFalsePublishAndCloseFlags(preflight);
        Assert.False(preflight.GetProperty("performsRuntimeExecution").GetBoolean());
        Assert.False(preflight.GetProperty("uploadsGitHubReleaseAssets").GetBoolean());
        Assert.False(preflight.GetProperty("executesDotnetNugetPush").GetBoolean());
        Assert.False(preflight.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.False(preflight.GetProperty("canPromoteRuntimeProof").GetBoolean());

        JsonElement commandPolicy = preflight.GetProperty("commandPolicy");
        Assert.False(commandPolicy.GetProperty("executesDotnetNugetPush").GetBoolean());
        Assert.False(commandPolicy.GetProperty("uploadsGitHubReleaseAssets").GetBoolean());
        Assert.False(commandPolicy.GetProperty("storesTokens").GetBoolean());
        Assert.False(commandPolicy.GetProperty("closesReleaseIssue").GetBoolean());

        string[] categories = preflight.GetProperty("requirements").EnumerateArray()
            .Select(static requirement => requirement.GetProperty("category").GetString()!)
            .ToArray();
        foreach (string expected in new[] { "public-package-channel", "package-hash-chain", "package-consumer-runtime", "post-publish-verification", "owner-release-decision", "release-close-authorization" })
        {
            Assert.Contains(expected, categories);
        }

        string raw = preflight.GetRawText();
        foreach (string forbidden in new[] { "template-only record", "dashboard-only record", "local feed", "ProjectReference", "direct nupkg", "article readiness map" })
        {
            Assert.Contains(forbidden, raw, StringComparison.Ordinal);
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("public-publish-authorization-preflight-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("public-publish-authorization-preflight-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-public-publish-authorization-required-preflight-valid", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() >= 6);
        AssertFalsePublishAndCloseFlags(validation);
        Assert.False(validation.GetProperty("executesDotnetNugetPush").GetBoolean());
        Assert.False(validation.GetProperty("uploadsGitHubReleaseAssets").GetBoolean());
    }

    [Fact]
    public void FinalActionMapLinksOwnerEvidenceImportAndPublishAuthorizationSurfaces()
    {
        RunStageExports();
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerRealEvidenceImportPacket.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerRealEvidenceImportPacket.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PublicPublishAuthorizationPreflight.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PublicPublishAuthorizationPreflight.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalPublishActionRequiredEvidenceMap.ps1"));

        using JsonDocument document = ReadFinalReleaseJson("final-publish-action-required-evidence-map.json");
        JsonElement root = document.RootElement;

        Assert.Equal("final-publish-action-required-evidence-map", root.GetProperty("recordKind").GetString());
        Assert.Equal("artifacts/final-release/owner-real-evidence-import-packet.json", root.GetProperty("sourceOwnerRealEvidenceImportPacket").GetString());
        Assert.Equal("artifacts/final-release/owner-real-evidence-import-packet-validation.json", root.GetProperty("sourceOwnerRealEvidenceImportPacketValidation").GetString());
        Assert.Equal("artifacts/final-release/public-publish-authorization-preflight.json", root.GetProperty("sourcePublicPublishAuthorizationPreflight").GetString());
        Assert.Equal("artifacts/final-release/public-publish-authorization-preflight-validation.json", root.GetProperty("sourcePublicPublishAuthorizationPreflightValidation").GetString());
        Assert.Equal(6, root.GetProperty("ownerRealEvidenceImportLaneCount").GetInt32());
        Assert.Equal("blocked-owner-real-evidence-required-packet-shape-valid", root.GetProperty("ownerRealEvidenceImportValidationState").GetString());
        Assert.True(root.GetProperty("publicPublishAuthorizationRequirementCount").GetInt32() >= 6);
        Assert.Equal("blocked-owner-public-publish-authorization-required-preflight-valid", root.GetProperty("publicPublishAuthorizationValidationState").GetString());
        AssertFalsePublishAndCloseFlags(root);
        Assert.False(root.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
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
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
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

    private static void RunStageExports()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerDualRouteProofPlan.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CleanExternalConsumerExecutionKit.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PostPublishVerificationIntakeMap.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalPublishActionRequiredEvidenceMap.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseStrictProofExecutionOrder.ps1"));
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

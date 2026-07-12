using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PublicPackageProofAndPostPublishOwnerConfirmationTests
{
    [Fact]
    public void PublicPackageAndPostPublishProofBridgesStayBlockedUntilOwnerProofExists()
    {
        RealExternalProofExecutionAndCloseOwnerInputPipeline.Run();
        RunPowerShell("Export-PublicPackageProofOwnerInputTemplate.ps1");
        RunPowerShell("Test-PublicPackageProofOwnerInput.ps1", "-Strict");
        RunPowerShell("Export-PostPublishProofOwnerConfirmation.ps1");
        RunPowerShell("Test-PostPublishProofOwnerConfirmation.ps1", "-Strict");
        RunPowerShell("Export-ReleaseClosePublicProofBridge.ps1");
        RunPowerShell("Test-ReleaseClosePublicProofBridge.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument publicValidationDocument = ReadFinalReleaseJson("public-package-proof-owner-input-validation.json");
        JsonElement publicValidation = publicValidationDocument.RootElement;
        Assert.Equal("blocked-public-package-proof-owner-input-required", publicValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, publicValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(publicValidation.GetProperty("failedActionRequiredCount").GetInt32() >= 60);
        AssertFalseProofPublishCloseFlags(publicValidation);
        AssertPublicPackageOwnerInputFields(publicValidation);

        using JsonDocument confirmationDocument = ReadFinalReleaseJson("post-publish-proof-owner-confirmation.json");
        JsonElement confirmation = confirmationDocument.RootElement;
        Assert.Equal("blocked-post-publish-proof-owner-confirmation-required", confirmation.GetProperty("confirmationState").GetString());
        Assert.Equal(5, confirmation.GetProperty("confirmationGateCount").GetInt32());
        Assert.Equal(5, confirmation.GetProperty("blockedConfirmationGateCount").GetInt32());
        Assert.Equal(0, confirmation.GetProperty("readyConfirmationGateCount").GetInt32());
        AssertFalseProofPublishCloseFlags(confirmation);

        using JsonDocument confirmationValidationDocument = ReadFinalReleaseJson("post-publish-proof-owner-confirmation-validation.json");
        JsonElement confirmationValidation = confirmationValidationDocument.RootElement;
        Assert.Equal("blocked-post-publish-proof-owner-confirmation-required", confirmationValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, confirmationValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(5, confirmationValidation.GetProperty("failedActionRequiredCount").GetInt32());
        AssertFalseProofPublishCloseFlags(confirmationValidation);

        using JsonDocument bridgeDocument = ReadFinalReleaseJson("release-close-public-proof-bridge.json");
        JsonElement bridge = bridgeDocument.RootElement;
        Assert.Equal("blocked-release-close-public-proof-required", bridge.GetProperty("bridgeState").GetString());
        Assert.Equal(6, bridge.GetProperty("publicProofGateCount").GetInt32());
        Assert.Equal(5, bridge.GetProperty("blockedPublicProofGateCount").GetInt32());
        Assert.Equal(1, bridge.GetProperty("readyPublicProofGateCount").GetInt32());
        AssertFalseProofPublishCloseFlags(bridge);

        using JsonDocument bridgeValidationDocument = ReadFinalReleaseJson("release-close-public-proof-bridge-validation.json");
        JsonElement bridgeValidation = bridgeValidationDocument.RootElement;
        Assert.Equal("blocked-release-close-public-proof-required", bridgeValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, bridgeValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(5, bridgeValidation.GetProperty("failedActionRequiredCount").GetInt32());
        AssertFalseProofPublishCloseFlags(bridgeValidation);

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("blocked-public-package-proof-owner-input-required", evidence.GetProperty("publicPackageProofOwnerInputValidationState").GetString());
        Assert.Equal("blocked-post-publish-proof-owner-confirmation-required", evidence.GetProperty("postPublishProofOwnerConfirmationValidationState").GetString());
        Assert.Equal("blocked-release-close-public-proof-required", evidence.GetProperty("releaseClosePublicProofBridgeValidationState").GetString());
        Assert.False(evidence.GetProperty("publicPackageProofOwnerInputCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("postPublishProofOwnerConfirmationCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("releaseClosePublicProofBridgeIsPostPublishProof").GetBoolean());

        AssertBlockedEvidenceItem(evidence, "public-package-proof-owner-input", "not package push");
        AssertBlockedEvidenceItem(evidence, "post-publish-proof-owner-confirmation", "not post-publish proof");
        AssertBlockedEvidenceItem(evidence, "release-close-public-proof-bridge", "not release close approval");

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/public-package-proof-owner-input.template.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/public-package-proof-owner-input-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/post-publish-proof-owner-confirmation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/post-publish-proof-owner-confirmation-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-close-public-proof-bridge.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-close-public-proof-bridge-validation.json", sourceArtifacts);

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        Assert.Contains(audit.GetProperty("auditedItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "release-close-public-proof-bridge" &&
            item.GetProperty("passed").GetBoolean() == false &&
            item.GetProperty("hasNonProofBoundary").GetBoolean());

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string releaseEvidenceDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));

        Assert.Contains("articles/zh-cn/public-package-proof-owner-input.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/post-publish-proof-owner-confirmation.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("release-close-public-proof-bridge", readme, StringComparison.Ordinal);
        Assert.Contains("public-package-proof-owner-input", readmeZh, StringComparison.Ordinal);
        Assert.Contains("post-publish-proof-owner-confirmation", releaseEvidenceDoc, StringComparison.Ordinal);
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static void AssertFalseProofPublishCloseFlags(JsonElement element)
    {
        RealExternalProofExecutionAndCloseOwnerInputPipeline.AssertFalseProofPublishCloseFlags(element);
        Assert.False(element.GetProperty("isPostPublishProof").GetBoolean());
    }

    private static void AssertBlockedEvidenceItem(JsonElement evidence, string id, string boundaryText)
    {
        JsonElement item = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(item => item.GetProperty("id").GetString() == id);

        Assert.False(item.GetProperty("passed").GetBoolean());
        Assert.Contains(boundaryText, item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    private static void AssertPublicPackageOwnerInputFields(JsonElement publicValidation)
    {
        string[] validationIds = publicValidation.GetProperty("validationItems")
            .EnumerateArray()
            .Select(item => item.GetProperty("id").GetString()!)
            .ToArray();

        string[] requiredIds =
        [
            "field-nugetPackageSource",
            "github-release-releaseUrl",
            "github-release-tagName",
            "github-release-managedAssetPath",
            "github-release-managedAssetSha256",
            "github-release-runtimeAssetPath",
            "github-release-runtimeAssetSha256",
            "managed-publicDownloadUrl",
            "managed-publicDownloadSha256",
            "runtime-publicDownloadUrl",
            "runtime-publicDownloadSha256",
            "clean-consumer-root",
            "clean-consumer-projectPath",
            "clean-consumer-restoreCommand",
            "clean-consumer-buildCommand",
            "clean-consumer-smokeCommand",
            "clean-consumer-restoreLogPath",
            "clean-consumer-restoreLogSha256",
            "clean-consumer-buildLogPath",
            "clean-consumer-buildLogSha256",
            "clean-consumer-smokeLogPath",
            "clean-consumer-smokeLogSha256",
            "clean-consumer-stdoutLogPath",
            "clean-consumer-stdoutLogSha256",
            "clean-consumer-stderrLogPath",
            "clean-consumer-stderrLogSha256",
            "clean-consumer-root-outside-repository",
            "host-metadata-ownerName",
            "host-metadata-machineName",
            "host-metadata-osDescription",
            "host-metadata-architecture",
            "host-metadata-gpuName",
            "host-metadata-cudaDriverVersion",
            "host-metadata-cudaRuntimeVersion",
            "host-metadata-cudnnVersion",
            "host-metadata-tensorRtVersion",
            "host-metadata-tensorRtLine",
            "owner-review-reviewer",
            "owner-review-reviewedAtUtc",
            "owner-review-approvalState",
            "confirmation-confirmsGithubReleaseAssetsReviewed",
            "confirmation-confirmsCleanExternalConsumerRestoreBuildSmoke",
            "confirmation-confirmsStdoutStderrSha256Reviewed",
            "confirmation-confirmsHostMetadataReviewed",
        ];

        foreach (string id in requiredIds)
        {
            Assert.Contains(id, validationIds);
        }
    }

    private static void RunPowerShell(string scriptName, params string[] arguments)
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", scriptName);
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(scriptPath, arguments);
    }
}

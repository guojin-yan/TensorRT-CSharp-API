using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerPostPublishProofValidatorTests
{
    private static readonly (string Id, string BlockedState, int MinimumFields)[] Validators =
    {
        ("public-package-url-hash-proof-validator", "blocked-public-package-url-hash-real-owner-proof-required", 10),
        ("external-clean-consumer-post-publish-proof-validator", "blocked-external-clean-consumer-real-owner-proof-required", 9),
        ("yolovision-real-model-post-publish-proof-validator", "blocked-yolovision-real-model-real-owner-proof-required", 14),
        ("article-publication-proof-validator", "blocked-article-publication-real-owner-proof-required", 10),
        ("release-close-final-bridge-proof-validator", "blocked-release-close-final-bridge-real-owner-proof-required", 9),
    };

    [Fact]
    public void ProofValidatorsStayBlockedWithoutRealOwnerInput()
    {
        RunProofValidatorPipeline();

        foreach ((string id, string blockedState, int minimumFields) in Validators)
        {
            using JsonDocument validatorDocument = ReadFinalReleaseJson($"{id}.json");
            JsonElement validator = validatorDocument.RootElement;
            Assert.Equal(id, validator.GetProperty("recordKind").GetString());
            Assert.Equal(blockedState, validator.GetProperty("validatorState").GetString());
            Assert.False(validator.GetProperty("realOwnerInputPresent").GetBoolean());
            Assert.False(validator.GetProperty("ownerEvidenceAccepted").GetBoolean());
            Assert.False(validator.GetProperty("proofReady").GetBoolean());
            Assert.False(validator.GetProperty("proofPromotionReady").GetBoolean());
            Assert.True(validator.GetProperty("requiredFieldCount").GetInt32() >= minimumFields);
            Assert.Equal(0, validator.GetProperty("readyFieldCount").GetInt32());
            Assert.True(validator.GetProperty("blockedFieldCount").GetInt32() >= minimumFields);
            Assert.True(validator.GetProperty("blockedReasonCount").GetInt32() >= 1);
            Assert.Contains("not package push", validator.GetProperty("boundary").GetString()!, StringComparison.OrdinalIgnoreCase);
            AssertFalseProofPublishCloseFlags(validator);

            using JsonDocument validationDocument = ReadFinalReleaseJson($"{id}-validation.json");
            JsonElement validation = validationDocument.RootElement;
            Assert.Equal($"{id}-validation", validation.GetProperty("recordKind").GetString());
            Assert.Equal($"{id}-ready-non-proof", validation.GetProperty("validationState").GetString());
            Assert.False(validation.GetProperty("ownerEvidenceAccepted").GetBoolean());
            Assert.True(validation.GetProperty("blockedReasonCount").GetInt32() >= 1);
            Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
            AssertFalseProofPublishCloseFlags(validation);
        }

        using JsonDocument closeDocument = ReadFinalReleaseJson("release-close-final-bridge-proof-validator.json");
        JsonElement closeValidator = closeDocument.RootElement;
        Assert.Equal(4, closeValidator.GetProperty("dependencyRequiredCount").GetInt32());
        Assert.Equal(0, closeValidator.GetProperty("dependencyAcceptedCount").GetInt32());

        using JsonDocument externalDocument = ReadFinalReleaseJson("external-clean-consumer-post-publish-proof-validator.json");
        JsonElement external = externalDocument.RootElement;
        Assert.True(external.GetProperty("rejectsLocalFeedProjectReferenceAndDirectNupkg").GetBoolean());
        Assert.False(external.GetProperty("externalWorkspacePathReady").GetBoolean());

        using JsonDocument yoloDocument = ReadFinalReleaseJson("yolovision-real-model-post-publish-proof-validator.json");
        JsonElement yolo = yoloDocument.RootElement;
        Assert.True(yolo.GetProperty("rejectsReadinessTutorialMatrixArtifacts").GetBoolean());
        Assert.True(yolo.GetProperty("yoloVisionHashFieldCount").GetInt32() >= 9);
        Assert.False(yolo.GetProperty("realModelExecutionConfirmationReady").GetBoolean());

        using JsonDocument articleDocument = ReadFinalReleaseJson("article-publication-proof-validator.json");
        JsonElement article = articleDocument.RootElement;
        Assert.True(article.GetProperty("supportsMultipleArticleProofRecords").GetBoolean());
        Assert.Equal(0, article.GetProperty("articleProofReadyRecordCount").GetInt32());
        Assert.False(article.GetProperty("articleProofRecordsReady").GetBoolean());

        Assert.True(closeValidator.GetProperty("manualCloseReviewOnly").GetBoolean());
        Assert.False(closeValidator.GetProperty("ownerCloseDecisionReady").GetBoolean());
    }

    [Fact]
    public void ReleaseEvidenceBundleAndClassificationAuditTrackProofValidatorsAsNonProof()
    {
        RunProofValidatorPipeline();
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("public-package-url-hash-proof-validator-ready-non-proof", evidence.GetProperty("publicPackageUrlHashProofValidatorValidationState").GetString());
        Assert.False(evidence.GetProperty("publicPackageUrlHashProofValidatorOwnerEvidenceAccepted").GetBoolean());
        Assert.Equal("external-clean-consumer-post-publish-proof-validator-ready-non-proof", evidence.GetProperty("externalCleanConsumerPostPublishProofValidatorValidationState").GetString());
        Assert.Equal("yolovision-real-model-post-publish-proof-validator-ready-non-proof", evidence.GetProperty("yoloVisionRealModelPostPublishProofValidatorValidationState").GetString());
        Assert.Equal("article-publication-proof-validator-ready-non-proof", evidence.GetProperty("articlePublicationProofValidatorValidationState").GetString());
        Assert.Equal("release-close-final-bridge-proof-validator-ready-non-proof", evidence.GetProperty("releaseCloseFinalBridgeProofValidatorValidationState").GetString());
        Assert.Equal(0, evidence.GetProperty("releaseCloseFinalBridgeProofValidatorDependencyAcceptedCount").GetInt32());
        Assert.Equal(4, evidence.GetProperty("releaseCloseFinalBridgeProofValidatorDependencyRequiredCount").GetInt32());

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach ((string id, _, _) in Validators)
        {
            Assert.Contains($"artifacts/final-release/{id}-validation.json", sourceArtifacts);
            JsonElement item = evidence.GetProperty("evidenceItems").EnumerateArray().Single(evidenceItem => evidenceItem.GetProperty("id").GetString() == id);
            Assert.False(item.GetProperty("passed").GetBoolean());
            Assert.Contains("ownerEvidenceAccepted=False", item.GetProperty("state").GetString()!, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("not package push", item.GetProperty("boundary").GetString()!, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument classificationDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement classification = classificationDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", classification.GetProperty("auditState").GetString());
        Assert.Equal(0, classification.GetProperty("findingCount").GetInt32());
        foreach ((string id, _, _) in Validators)
        {
            Assert.Contains(
                classification.GetProperty("auditedItems").EnumerateArray(),
                item => item.GetProperty("id").GetString() == id
                    && item.GetProperty("passed").GetBoolean() == false
                    && item.GetProperty("hasNonProofBoundary").GetBoolean());
        }
    }

    private static void RunProofValidatorPipeline()
    {
        RunPowerShell("Export-PublicPackageUrlHashProofValidator.ps1");
        RunPowerShell("Test-PublicPackageUrlHashProofValidator.ps1", "-Strict");
        RunPowerShell("Export-ExternalCleanConsumerPostPublishProofValidator.ps1");
        RunPowerShell("Test-ExternalCleanConsumerPostPublishProofValidator.ps1", "-Strict");
        RunPowerShell("Export-YoloVisionRealModelPostPublishProofValidator.ps1");
        RunPowerShell("Test-YoloVisionRealModelPostPublishProofValidator.ps1", "-Strict");
        RunPowerShell("Export-ArticlePublicationProofValidator.ps1");
        RunPowerShell("Test-ArticlePublicationProofValidator.ps1", "-Strict");
        RunPowerShell("Export-ReleaseCloseFinalBridgeProofValidator.ps1");
        RunPowerShell("Test-ReleaseCloseFinalBridgeProofValidator.ps1", "-Strict");
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    private static void AssertFalseProofPublishCloseFlags(JsonElement root)
    {
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("usesPublishToken").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(root.GetProperty("isReleaseCloseProof").GetBoolean());
        if (root.TryGetProperty("canPromoteRuntimeProof", out JsonElement canPromoteRuntimeProof))
        {
            Assert.False(canPromoteRuntimeProof.GetBoolean());
        }
    }

    private static string RunPowerShell(string scriptName, params string[] arguments)
    {
        using Process process = new()
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "pwsh",
                WorkingDirectory = RepositoryPaths.Root,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            },
        };

        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(Path.Combine(RepositoryPaths.Root, "eng", scriptName));
        foreach (string argument in arguments)
        {
            process.StartInfo.ArgumentList.Add(argument);
        }

        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptName}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        return stdout;
    }
}

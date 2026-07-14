using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerPostPublishRealInputCandidateTests
{
    private static readonly string[] CandidateIds =
    {
        "owner-post-publish-docs-article-sample-real-input-template",
        "owner-post-publish-docs-article-sample-real-input-import",
        "public-package-url-hash-verification-candidate",
        "external-clean-consumer-post-publish-candidate",
        "yolovision-real-model-post-publish-candidate",
        "article-publication-proof-candidate",
        "release-issue-close-material-candidate",
    };

    [Fact]
    public void OwnerRealInputTemplateImportAndCandidatesStayBlockedNonProof()
    {
        RunOwnerPostPublishRealInputPipeline();

        using JsonDocument templateDocument = ReadFinalReleaseJson("owner-post-publish-docs-article-sample-real-input.template.json");
        JsonElement template = templateDocument.RootElement;
        Assert.Equal("owner-post-publish-docs-article-sample-real-input-template", template.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-post-publish-real-input-template", template.GetProperty("templateState").GetString());
        Assert.Equal(5, template.GetProperty("laneCount").GetInt32());
        Assert.True(template.GetProperty("requiredFieldCount").GetInt32() >= 39);
        Assert.Equal(template.GetProperty("requiredFieldCount").GetInt32(), template.GetProperty("placeholderFieldCount").GetInt32());
        AssertFalseProofPublishCloseFlags(template);

        using JsonDocument templateValidationDocument = ReadFinalReleaseJson("owner-post-publish-docs-article-sample-real-input-template-validation.json");
        JsonElement templateValidation = templateValidationDocument.RootElement;
        Assert.Equal("owner-post-publish-docs-article-sample-real-input-template-ready-non-proof", templateValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, templateValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(templateValidation.GetProperty("requiredFieldCount").GetInt32(), templateValidation.GetProperty("placeholderFieldCount").GetInt32());
        AssertFalseProofPublishCloseFlags(templateValidation);

        using JsonDocument importDocument = ReadFinalReleaseJson("owner-post-publish-docs-article-sample-real-input-import.json");
        JsonElement import = importDocument.RootElement;
        Assert.Equal("owner-post-publish-docs-article-sample-real-input-import", import.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-post-publish-docs-article-sample-real-input-required", import.GetProperty("importState").GetString());
        Assert.False(import.GetProperty("realOwnerInputPresent").GetBoolean());
        Assert.False(import.GetProperty("candidateReady").GetBoolean());
        Assert.Equal(0, import.GetProperty("readyFieldCount").GetInt32());
        Assert.True(import.GetProperty("blockedFieldCount").GetInt32() >= 39);
        Assert.True(import.GetProperty("placeholderFieldCount").GetInt32() >= 39);
        AssertFalseProofPublishCloseFlags(import);

        using JsonDocument importValidationDocument = ReadFinalReleaseJson("owner-post-publish-docs-article-sample-real-input-import-validation.json");
        JsonElement importValidation = importValidationDocument.RootElement;
        Assert.Equal("owner-post-publish-docs-article-sample-real-input-import-ready-non-proof", importValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, importValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(importValidation.GetProperty("blockedFieldCount").GetInt32() >= 39);
        AssertFalseProofPublishCloseFlags(importValidation);

        AssertCandidate("public-package-url-hash-verification-candidate", "blocked-public-package-url-hash-owner-proof-required", 10);
        AssertCandidate("external-clean-consumer-post-publish-candidate", "blocked-external-clean-consumer-post-publish-owner-proof-required", 6);
        AssertCandidate("yolovision-real-model-post-publish-candidate", "blocked-yolovision-real-model-owner-proof-required", 9);
        AssertCandidate("article-publication-proof-candidate", "blocked-article-publication-owner-proof-required", 7);
        AssertCandidate("release-issue-close-material-candidate", "blocked-release-issue-close-material-owner-proof-required", 7);
    }

    [Fact]
    public void ReleaseEvidenceBundleAndClassificationAuditCarryOwnerRealInputCandidateBoundaries()
    {
        RunOwnerPostPublishRealInputPipeline();
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("owner-post-publish-docs-article-sample-real-input-template-ready-non-proof", evidence.GetProperty("ownerPostPublishRealInputTemplateValidationState").GetString());
        Assert.True(evidence.GetProperty("ownerPostPublishRealInputTemplateRequiredFieldCount").GetInt32() >= 39);
        Assert.Equal("owner-post-publish-docs-article-sample-real-input-import-ready-non-proof", evidence.GetProperty("ownerPostPublishRealInputImportValidationState").GetString());
        Assert.True(evidence.GetProperty("ownerPostPublishRealInputImportBlockedFieldCount").GetInt32() >= 39);
        Assert.Equal("public-package-url-hash-verification-candidate-ready-non-proof", evidence.GetProperty("publicPackageUrlHashVerificationCandidateValidationState").GetString());
        Assert.Equal("external-clean-consumer-post-publish-candidate-ready-non-proof", evidence.GetProperty("externalCleanConsumerPostPublishCandidateValidationState").GetString());
        Assert.Equal("yolovision-real-model-post-publish-candidate-ready-non-proof", evidence.GetProperty("yoloVisionRealModelPostPublishCandidateValidationState").GetString());
        Assert.Equal("article-publication-proof-candidate-ready-non-proof", evidence.GetProperty("articlePublicationProofCandidateValidationState").GetString());
        Assert.Equal("release-issue-close-material-candidate-ready-non-proof", evidence.GetProperty("releaseIssueCloseMaterialCandidateValidationState").GetString());
        Assert.False(evidence.GetProperty("releaseIssueCloseMaterialCandidateFinalBridgePassed").GetBoolean());

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string fileName in new[]
        {
            "owner-post-publish-docs-article-sample-real-input-template-validation.json",
            "owner-post-publish-docs-article-sample-real-input-import-validation.json",
            "public-package-url-hash-verification-candidate-validation.json",
            "external-clean-consumer-post-publish-candidate-validation.json",
            "yolovision-real-model-post-publish-candidate-validation.json",
            "article-publication-proof-candidate-validation.json",
            "release-issue-close-material-candidate-validation.json",
        })
        {
            Assert.Contains($"artifacts/final-release/{fileName}", sourceArtifacts);
        }

        foreach (string id in CandidateIds)
        {
            JsonElement item = evidence.GetProperty("evidenceItems").EnumerateArray().Single(evidenceItem => evidenceItem.GetProperty("id").GetString() == id);
            Assert.False(item.GetProperty("passed").GetBoolean());
            string boundary = item.GetProperty("boundary").GetString()!;
            Assert.Contains("not", boundary, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("not package push", boundary, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument classificationDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement classification = classificationDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", classification.GetProperty("auditState").GetString());
        Assert.Equal(0, classification.GetProperty("findingCount").GetInt32());
        foreach (string id in CandidateIds)
        {
            Assert.Contains(
                classification.GetProperty("auditedItems").EnumerateArray(),
                item => item.GetProperty("id").GetString() == id
                    && item.GetProperty("passed").GetBoolean() == false
                    && item.GetProperty("hasNonProofBoundary").GetBoolean());
        }
    }

    private static void AssertCandidate(string stem, string blockedState, int minimumFields)
    {
        using JsonDocument candidateDocument = ReadFinalReleaseJson($"{stem}.json");
        JsonElement candidate = candidateDocument.RootElement;
        Assert.Equal(stem, candidate.GetProperty("recordKind").GetString());
        Assert.Equal(blockedState, candidate.GetProperty("candidateState").GetString());
        Assert.False(candidate.GetProperty("candidateReady").GetBoolean());
        Assert.False(candidate.GetProperty("proofReady").GetBoolean());
        Assert.True(candidate.GetProperty("requiredFieldCount").GetInt32() >= minimumFields);
        Assert.Equal(0, candidate.GetProperty("readyFieldCount").GetInt32());
        Assert.True(candidate.GetProperty("blockedFieldCount").GetInt32() >= minimumFields);
        AssertFalseProofPublishCloseFlags(candidate);

        using JsonDocument validationDocument = ReadFinalReleaseJson($"{stem}-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.EndsWith("ready-non-proof", validation.GetProperty("validationState").GetString(), StringComparison.Ordinal);
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("blockedFieldCount").GetInt32() >= minimumFields);
        AssertFalseProofPublishCloseFlags(validation);
    }

    private static void RunOwnerPostPublishRealInputPipeline()
    {
        RunPowerShell("Export-OwnerPostPublishDocsArticleSampleRealInputTemplate.ps1");
        RunPowerShell("Test-OwnerPostPublishDocsArticleSampleRealInputTemplate.ps1", "-Strict");
        RunPowerShell("Import-OwnerPostPublishDocsArticleSampleRealInput.ps1");
        RunPowerShell("Test-OwnerPostPublishDocsArticleSampleRealInput.ps1", "-Strict");
        RunPowerShell("Export-PublicPackageUrlHashVerificationCandidate.ps1");
        RunPowerShell("Test-PublicPackageUrlHashVerificationCandidate.ps1", "-Strict");
        RunPowerShell("Export-ExternalCleanConsumerPostPublishCandidate.ps1");
        RunPowerShell("Test-ExternalCleanConsumerPostPublishCandidate.ps1", "-Strict");
        RunPowerShell("Export-YoloVisionRealModelPostPublishCandidate.ps1");
        RunPowerShell("Test-YoloVisionRealModelPostPublishCandidate.ps1", "-Strict");
        RunPowerShell("Export-ArticlePublicationProofCandidate.ps1");
        RunPowerShell("Test-ArticlePublicationProofCandidate.ps1", "-Strict");
        RunPowerShell("Export-ReleaseIssueCloseMaterialCandidate.ps1");
        RunPowerShell("Test-ReleaseIssueCloseMaterialCandidate.ps1", "-Strict");
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

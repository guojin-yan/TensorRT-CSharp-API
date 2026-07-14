using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerPostPublishAcceptanceAndDownloadVerificationTests
{
    [Fact]
    public void AcceptanceManifestAndDownloadVerificationStayBlockedWithoutOwnerEvidence()
    {
        RunAcceptanceAndDownloadPipeline();

        using JsonDocument manifestDocument = ReadFinalReleaseJson("owner-post-publish-proof-acceptance-manifest.json");
        JsonElement manifest = manifestDocument.RootElement;
        Assert.Equal("owner-post-publish-proof-acceptance-manifest", manifest.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-post-publish-proof-acceptance-real-evidence-required", manifest.GetProperty("manifestState").GetString());
        Assert.Equal(5, manifest.GetProperty("validatorCount").GetInt32());
        Assert.Equal(0, manifest.GetProperty("acceptedValidatorCount").GetInt32());
        Assert.False(manifest.GetProperty("allValidatorsAccepted").GetBoolean());
        Assert.False(manifest.GetProperty("releaseCloseReady").GetBoolean());
        Assert.False(manifest.GetProperty("closeIssueCommandReady").GetBoolean());
        Assert.True(manifest.GetProperty("blockedReasonCount").GetInt32() >= 1);
        AssertFalseProofPublishCloseFlags(manifest);

        using JsonDocument manifestValidationDocument = ReadFinalReleaseJson("owner-post-publish-proof-acceptance-manifest-validation.json");
        JsonElement manifestValidation = manifestValidationDocument.RootElement;
        Assert.Equal("owner-post-publish-proof-acceptance-manifest-ready-non-proof", manifestValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, manifestValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(manifestValidation.GetProperty("releaseCloseReady").GetBoolean());
        AssertFalseProofPublishCloseFlags(manifestValidation);

        using JsonDocument downloadDocument = ReadFinalReleaseJson("public-package-url-hash-download-verification.json");
        JsonElement download = downloadDocument.RootElement;
        Assert.Equal("public-package-url-hash-download-verification", download.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-public-package-url-hash-download-verification-real-owner-input-required", download.GetProperty("verificationState").GetString());
        Assert.False(download.GetProperty("realOwnerInputPresent").GetBoolean());
        Assert.False(download.GetProperty("downloadAllowed").GetBoolean());
        Assert.Equal(2, download.GetProperty("packageCheckCount").GetInt32());
        Assert.Equal(0, download.GetProperty("downloadAttemptedCount").GetInt32());
        Assert.Equal(0, download.GetProperty("hashMatchedCount").GetInt32());
        Assert.False(download.GetProperty("downloadVerificationReady").GetBoolean());
        AssertFalseProofPublishCloseFlags(download);

        using JsonDocument downloadValidationDocument = ReadFinalReleaseJson("public-package-url-hash-download-verification-validation.json");
        JsonElement downloadValidation = downloadValidationDocument.RootElement;
        Assert.Equal("public-package-url-hash-download-verification-ready-non-proof", downloadValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, downloadValidation.GetProperty("downloadAttemptedCount").GetInt32());
        Assert.Equal(0, downloadValidation.GetProperty("failedBlockerCount").GetInt32());
        AssertFalseProofPublishCloseFlags(downloadValidation);
    }

    [Fact]
    public void ReleaseEvidenceBundleTracksAcceptanceAndDownloadVerificationAsNonProof()
    {
        RunAcceptanceAndDownloadPipeline();
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("owner-post-publish-proof-acceptance-manifest-ready-non-proof", evidence.GetProperty("ownerPostPublishProofAcceptanceManifestValidationState").GetString());
        Assert.Equal(0, evidence.GetProperty("ownerPostPublishProofAcceptanceManifestAcceptedValidatorCount").GetInt32());
        Assert.Equal(5, evidence.GetProperty("ownerPostPublishProofAcceptanceManifestValidatorCount").GetInt32());
        Assert.False(evidence.GetProperty("ownerPostPublishProofAcceptanceManifestReleaseCloseReady").GetBoolean());
        Assert.Equal("public-package-url-hash-download-verification-ready-non-proof", evidence.GetProperty("publicPackageUrlHashDownloadVerificationValidationState").GetString());
        Assert.False(evidence.GetProperty("publicPackageUrlHashDownloadVerificationDownloadAllowed").GetBoolean());
        Assert.Equal(0, evidence.GetProperty("publicPackageUrlHashDownloadVerificationDownloadAttemptedCount").GetInt32());

        string[] ids =
        {
            "owner-post-publish-proof-acceptance-manifest",
            "public-package-url-hash-download-verification",
        };
        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string id in ids)
        {
            Assert.Contains($"artifacts/final-release/{id}-validation.json", sourceArtifacts);
            JsonElement item = evidence.GetProperty("evidenceItems").EnumerateArray().Single(evidenceItem => evidenceItem.GetProperty("id").GetString() == id);
            Assert.False(item.GetProperty("passed").GetBoolean());
            Assert.Contains("not package push", item.GetProperty("boundary").GetString()!, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument classificationDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement classification = classificationDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", classification.GetProperty("auditState").GetString());
        Assert.Equal(0, classification.GetProperty("findingCount").GetInt32());
        foreach (string id in ids)
        {
            Assert.Contains(
                classification.GetProperty("auditedItems").EnumerateArray(),
                item => item.GetProperty("id").GetString() == id
                    && item.GetProperty("passed").GetBoolean() == false
                    && item.GetProperty("hasNonProofBoundary").GetBoolean());
        }
    }

    private static void RunAcceptanceAndDownloadPipeline()
    {
        RunPowerShell("Export-OwnerPostPublishProofAcceptanceManifest.ps1");
        RunPowerShell("Test-OwnerPostPublishProofAcceptanceManifest.ps1", "-Strict");
        RunPowerShell("Export-PublicPackageUrlHashDownloadVerification.ps1");
        RunPowerShell("Test-PublicPackageUrlHashDownloadVerification.ps1", "-Strict");
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

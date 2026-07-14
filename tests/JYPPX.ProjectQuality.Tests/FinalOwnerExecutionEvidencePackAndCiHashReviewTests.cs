using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalOwnerExecutionEvidencePackAndCiHashReviewTests
{
    [Fact]
    public void FinalOwnerExecutionEvidencePackDefaultsToBlockedNonProof()
    {
        RunPowerShell("Export-FinalOwnerExecutionEvidencePack.ps1");
        RunPowerShell("Test-FinalOwnerExecutionEvidencePack.ps1", "-Strict");

        using JsonDocument skeletonDocument = ReadFinalReleaseJson("final-owner-execution-input-skeleton-validation.json");
        JsonElement skeleton = skeletonDocument.RootElement;
        Assert.Equal("final-owner-execution-input-skeleton-validation-ready-non-proof", skeleton.GetProperty("validationState").GetString());
        Assert.True(skeleton.GetProperty("laneCount").GetInt32() >= 9);
        Assert.True(skeleton.GetProperty("requiredFieldCount").GetInt32() >= 40);
        AssertFalseProofPublishCloseFlags(skeleton);

        using JsonDocument ciDocument = ReadFinalReleaseJson("github-ci-evidence-from-owner-input-validation.json");
        JsonElement ci = ciDocument.RootElement;
        Assert.Equal("github-ci-evidence-from-owner-input-validation-ready-non-proof", ci.GetProperty("validationState").GetString());
        Assert.False(ci.GetProperty("ciEvidenceAccepted").GetBoolean());
        Assert.True(ci.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        AssertFalseProofPublishCloseFlags(ci);

        using JsonDocument bundleHashDocument = ReadFinalReleaseJson("release-evidence-bundle-hash-review-validation.json");
        JsonElement bundleHash = bundleHashDocument.RootElement;
        Assert.Equal("release-evidence-bundle-hash-review-validation-ready-non-proof", bundleHash.GetProperty("validationState").GetString());
        Assert.False(bundleHash.GetProperty("reviewAccepted").GetBoolean());
        Assert.True(bundleHash.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        Assert.Contains("not package push", bundleHash.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        AssertFalseProofPublishCloseFlags(bundleHash);

        using JsonDocument auditHashDocument = ReadFinalReleaseJson("classification-audit-hash-review-validation.json");
        JsonElement auditHash = auditHashDocument.RootElement;
        Assert.Equal("classification-audit-hash-review-validation-ready-non-proof", auditHash.GetProperty("validationState").GetString());
        Assert.True(auditHash.GetProperty("stateMatchesRequired").GetBoolean());
        Assert.False(auditHash.GetProperty("reviewAccepted").GetBoolean());
        Assert.True(auditHash.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        Assert.Contains("not package push", auditHash.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        AssertFalseProofPublishCloseFlags(auditHash);

        using JsonDocument packDocument = ReadFinalReleaseJson("final-owner-execution-evidence-pack.json");
        JsonElement pack = packDocument.RootElement;
        Assert.Equal("final-owner-execution-evidence-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-execution-evidence-pack-real-owner-proof-required", pack.GetProperty("packState").GetString());
        Assert.True(pack.GetProperty("gateCount").GetInt32() >= 6);
        Assert.True(pack.GetProperty("blockedGateCount").GetInt32() > 0);
        Assert.False(pack.GetProperty("allOwnerExecutionInputsReady").GetBoolean());
        Assert.Contains(pack.GetProperty("rejectedNonProofStates").EnumerateArray(), static state => state.GetString() == "queued-workflow");
        Assert.Contains(pack.GetProperty("rejectedNonProofStates").EnumerateArray(), static state => state.GetString() == "hash-only");
        Assert.Contains("does not use tokens", pack.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        AssertFalseProofPublishCloseFlags(pack);

        using JsonDocument packValidationDocument = ReadFinalReleaseJson("final-owner-execution-evidence-pack-validation.json");
        JsonElement packValidation = packValidationDocument.RootElement;
        Assert.Equal("final-owner-execution-evidence-pack-validation-ready-non-proof", packValidation.GetProperty("validationState").GetString());
        Assert.True(packValidation.GetProperty("gateCount").GetInt32() >= 6);
        Assert.True(packValidation.GetProperty("blockedGateCount").GetInt32() > 0);
        Assert.Equal(0, packValidation.GetProperty("failedBlockerCount").GetInt32());
        AssertFalseProofPublishCloseFlags(packValidation);
    }

    [Fact]
    public void ReleaseBundleAndClassificationAuditTrackFinalOwnerExecutionEvidencePack()
    {
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument bundleDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement bundle = bundleDocument.RootElement;
        Assert.Equal("final-owner-execution-evidence-pack-validation-ready-non-proof", bundle.GetProperty("finalOwnerExecutionEvidencePackValidationState").GetString());
        Assert.True(bundle.GetProperty("finalOwnerExecutionEvidencePackGateCount").GetInt32() >= 6);
        Assert.True(bundle.GetProperty("finalOwnerExecutionEvidencePackBlockedGateCount").GetInt32() > 0);
        Assert.False(bundle.GetProperty("githubCiEvidenceFromOwnerInputAccepted").GetBoolean());
        Assert.False(bundle.GetProperty("releaseEvidenceBundleHashReviewAccepted").GetBoolean());
        Assert.True(bundle.GetProperty("classificationAuditHashReviewStateMatchesRequired").GetBoolean());
        Assert.False(bundle.GetProperty("classificationAuditHashReviewAccepted").GetBoolean());

        string[] ids =
        {
            "final-owner-execution-input-skeleton",
            "github-ci-evidence-from-owner-input",
            "release-evidence-bundle-hash-review",
            "classification-audit-hash-review",
            "final-owner-execution-evidence-pack",
        };
        string[] sourceArtifacts = bundle.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string id in ids)
        {
            Assert.Contains($"artifacts/final-release/{id}.json", sourceArtifacts);
            Assert.Contains($"artifacts/final-release/{id}-validation.json", sourceArtifacts);
            JsonElement[] matchingItems = bundle.GetProperty("evidenceItems")
                .EnumerateArray()
                .Where(evidenceItem => evidenceItem.GetProperty("id").GetString() == id)
                .ToArray();
            Assert.NotEmpty(matchingItems);
            foreach (JsonElement item in matchingItems)
            {
                Assert.False(item.GetProperty("passed").GetBoolean());
                Assert.Contains("not package push", item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
            }
        }

        using JsonDocument classificationDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement classification = classificationDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", classification.GetProperty("auditState").GetString());
        Assert.Equal(0, classification.GetProperty("findingCount").GetInt32());
        foreach (string marker in new[]
        {
            "github ci evidence from owner input",
            "release evidence bundle hash review",
            "classification audit hash review",
            "final owner execution evidence pack",
        })
        {
            Assert.Contains(classification.GetProperty("requiredNonSubstituteMarkers").EnumerateArray(), item => item.GetString() == marker);
        }

        foreach (string id in ids)
        {
            Assert.Contains(
                classification.GetProperty("auditedItems").EnumerateArray(),
                item => item.GetProperty("id").GetString() == id
                    && item.GetProperty("passed").GetBoolean() == false
                    && item.GetProperty("hasNonProofBoundary").GetBoolean());
        }
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


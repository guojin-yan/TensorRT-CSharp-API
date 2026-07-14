using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PostPublishProofValidatorBridgeAndReleaseCloseFinalBridgeTests
{
    [Fact]
    public void BridgesDefaultToBlockedNonProofWithExplicitSubstitutionBoundaries()
    {
        RunPowerShell("Export-PostPublishProofValidatorBridge.ps1");
        RunPowerShell("Test-PostPublishProofValidatorBridge.ps1", "-Strict");
        RunPowerShell("Export-ReleaseCloseFinalBridge.ps1");
        RunPowerShell("Test-ReleaseCloseFinalBridge.ps1", "-Strict");

        using JsonDocument postBridgeDocument = ReadFinalReleaseJson("post-publish-proof-validator-bridge.json");
        JsonElement postBridge = postBridgeDocument.RootElement;
        Assert.Equal("post-publish-proof-validator-bridge", postBridge.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-post-publish-proof-validator-bridge-real-owner-proof-required", postBridge.GetProperty("bridgeState").GetString());
        Assert.Equal(4, postBridge.GetProperty("laneCount").GetInt32());
        Assert.Equal(0, postBridge.GetProperty("proofReadyLaneCount").GetInt32());
        Assert.False(postBridge.GetProperty("allPostPublishInputsAccepted").GetBoolean());
        Assert.True(postBridge.GetProperty("publicPackageHashCannotSubstitutePostPublishProof").GetBoolean());
        Assert.True(postBridge.GetProperty("shapeValidCannotSubstitutePostPublishProof").GetBoolean());
        Assert.Contains("cannot substitute post-publish CleanConsumer runtime proof", postBridge.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        AssertFalseProofPublishCloseFlags(postBridge);
        Assert.Contains(postBridge.GetProperty("lanes").EnumerateArray(), static lane =>
            lane.GetProperty("id").GetString() == "public-package-url-hash" &&
            lane.GetProperty("proofReady").GetBoolean() == false &&
            lane.GetProperty("isPostPublishProof").GetBoolean() == false);

        using JsonDocument postValidationDocument = ReadFinalReleaseJson("post-publish-proof-validator-bridge-validation.json");
        JsonElement postValidation = postValidationDocument.RootElement;
        Assert.Equal("post-publish-proof-validator-bridge-validation-ready-non-proof", postValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, postValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(postValidation.GetProperty("allPostPublishInputsAccepted").GetBoolean());
        Assert.True(postValidation.GetProperty("publicPackageHashCannotSubstitutePostPublishProof").GetBoolean());
        Assert.True(postValidation.GetProperty("shapeValidCannotSubstitutePostPublishProof").GetBoolean());
        AssertFalseProofPublishCloseFlags(postValidation);

        using JsonDocument closeBridgeDocument = ReadFinalReleaseJson("release-close-final-bridge.json");
        JsonElement closeBridge = closeBridgeDocument.RootElement;
        Assert.Equal("release-close-final-bridge", closeBridge.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-release-close-final-bridge-real-owner-proof-required", closeBridge.GetProperty("bridgeState").GetString());
        Assert.True(closeBridge.GetProperty("gateCount").GetInt32() >= 6);
        Assert.True(closeBridge.GetProperty("blockedGateCount").GetInt32() > 0);
        Assert.False(closeBridge.GetProperty("allCloseInputsReady").GetBoolean());
        Assert.True(closeBridge.GetProperty("publicPackageHashCannotSubstitutePostPublishProof").GetBoolean());
        Assert.True(closeBridge.GetProperty("shapeValidCannotSubstituteReleaseCloseProof").GetBoolean());
        Assert.Contains(closeBridge.GetProperty("rejectedNonProofStates").EnumerateArray(), static state => state.GetString() == "public-package-hash-only");
        Assert.Contains(closeBridge.GetProperty("rejectedNonProofStates").EnumerateArray(), static state => state.GetString() == "staging-shape-valid-only");
        Assert.Contains(closeBridge.GetProperty("closeGates").EnumerateArray(), static gate =>
            gate.GetProperty("id").GetString() == "final-owner-close-decision" &&
            gate.GetProperty("ready").GetBoolean() == false);
        AssertFalseProofPublishCloseFlags(closeBridge);

        using JsonDocument closeValidationDocument = ReadFinalReleaseJson("release-close-final-bridge-validation.json");
        JsonElement closeValidation = closeValidationDocument.RootElement;
        Assert.Equal("release-close-final-bridge-validation-ready-non-proof", closeValidation.GetProperty("validationState").GetString());
        Assert.True(closeValidation.GetProperty("gateCount").GetInt32() >= 6);
        Assert.True(closeValidation.GetProperty("blockedGateCount").GetInt32() > 0);
        Assert.Equal(0, closeValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(closeValidation.GetProperty("allCloseInputsReady").GetBoolean());
        AssertFalseProofPublishCloseFlags(closeValidation);
    }

    [Fact]
    public void ReleaseBundleAndClassificationAuditIncludeBothFinalBridgesAsNonProof()
    {
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument bundleDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement bundle = bundleDocument.RootElement;
        Assert.Equal("post-publish-proof-validator-bridge-validation-ready-non-proof", bundle.GetProperty("postPublishProofValidatorBridgeValidationState").GetString());
        Assert.Equal(4, bundle.GetProperty("postPublishProofValidatorBridgeLaneCount").GetInt32());
        Assert.Equal(0, bundle.GetProperty("postPublishProofValidatorBridgeProofReadyLaneCount").GetInt32());
        Assert.True(bundle.GetProperty("postPublishProofValidatorBridgeBlockedLaneCount").GetInt32() > 0);
        Assert.True(bundle.GetProperty("postPublishProofValidatorBridgePublicPackageHashCannotSubstitutePostPublishProof").GetBoolean());
        Assert.True(bundle.GetProperty("postPublishProofValidatorBridgeShapeValidCannotSubstitutePostPublishProof").GetBoolean());
        Assert.Equal("release-close-final-bridge-validation-ready-non-proof", bundle.GetProperty("releaseCloseFinalBridgeValidationState").GetString());
        Assert.True(bundle.GetProperty("releaseCloseFinalBridgeGateCount").GetInt32() >= 6);
        Assert.True(bundle.GetProperty("releaseCloseFinalBridgeBlockedGateCount").GetInt32() > 0);
        Assert.True(bundle.GetProperty("releaseCloseFinalBridgeRejectedNonProofStateCount").GetInt32() >= 10);

        string[] ids =
        {
            "post-publish-proof-validator-bridge",
            "release-close-final-bridge",
        };
        string[] sourceArtifacts = bundle.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string id in ids)
        {
            Assert.Contains($"artifacts/final-release/{id}.json", sourceArtifacts);
            Assert.Contains($"artifacts/final-release/{id}-validation.json", sourceArtifacts);
            JsonElement item = bundle.GetProperty("evidenceItems").EnumerateArray().Single(evidenceItem => evidenceItem.GetProperty("id").GetString() == id);
            Assert.False(item.GetProperty("passed").GetBoolean());
            Assert.Contains("not publish", item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Contains("not package push", item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument classificationDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement classification = classificationDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", classification.GetProperty("auditState").GetString());
        Assert.Equal(0, classification.GetProperty("findingCount").GetInt32());
        Assert.Contains(classification.GetProperty("requiredNonSubstituteMarkers").EnumerateArray(), static marker =>
            marker.GetString() == "post publish proof validator bridge");
        Assert.Contains(classification.GetProperty("requiredNonSubstituteMarkers").EnumerateArray(), static marker =>
            marker.GetString() == "release close final bridge");
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

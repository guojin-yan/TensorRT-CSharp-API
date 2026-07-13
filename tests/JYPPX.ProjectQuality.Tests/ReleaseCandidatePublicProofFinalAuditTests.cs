using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization.Metadata;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleaseCandidatePublicProofFinalAuditTests
{
    private static readonly JsonSerializerOptions IndentedJsonOptions = new()
    {
        WriteIndented = true,
        TypeInfoResolver = new DefaultJsonTypeInfoResolver(),
    };

    [Fact]
    public void FinalAuditKeepsPublicProofPathBlockedAndNonProof()
    {
        RunAuditPipeline();

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-candidate-public-proof-final-audit.json");
        JsonElement audit = auditDocument.RootElement;

        Assert.Equal("release-candidate-public-proof-final-audit", audit.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-release-candidate-public-proof-final-audit-owner-proof-required", audit.GetProperty("auditState").GetString());
        Assert.Equal(6, audit.GetProperty("finalProofChainCount").GetInt32());
        Assert.Equal(6, audit.GetProperty("blockedFinalProofChainCount").GetInt32());
        Assert.Equal(0, audit.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(6, audit.GetProperty("failedActionRequiredCount").GetInt32());
        AssertFalseProofPublishCloseFlags(audit);
        AssertBoundary(audit.GetProperty("boundary").GetString()!);

        AssertIds(audit, "finalProofChain", RequiredFinalProofStepIds);
        AssertIds(audit, "evidenceCoverage", RequiredEvidenceItemIds);
        foreach (JsonElement step in audit.GetProperty("finalProofChain").EnumerateArray())
        {
            Assert.True(step.GetProperty("blocked").GetBoolean());
            Assert.False(step.GetProperty("proofAccepted").GetBoolean());
            AssertFalseProofPublishCloseFlags(step);
            AssertBoundary(step.GetProperty("boundary").GetString()!);
        }

        foreach (JsonElement coverage in audit.GetProperty("evidenceCoverage").EnumerateArray())
        {
            Assert.True(coverage.GetProperty("present").GetBoolean());
            Assert.False(coverage.GetProperty("passed").GetBoolean());
            Assert.True(coverage.GetProperty("hasNonProofBoundary").GetBoolean());
        }

        foreach (JsonElement substitute in audit.GetProperty("forbiddenSubstituteAudit").EnumerateArray())
        {
            Assert.False(substitute.GetProperty("accepted").GetBoolean());
        }

        string[] sources = audit.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string requiredSource in RequiredSourceArtifacts)
        {
            Assert.Contains(requiredSource, sources);
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("release-candidate-public-proof-final-audit-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("release-candidate-public-proof-final-audit-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-release-candidate-public-proof-final-audit-owner-proof-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("failedActionRequiredCount").GetInt32());
        AssertFalseProofPublishCloseFlags(validation);
        AssertBoundary(validation.GetProperty("boundary").GetString()!);
    }

    [Fact]
    public void ReleaseEvidenceCarriesFinalAuditAsBlockedNonProof()
    {
        RunAuditPipeline();

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        JsonElement item = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static value => value.GetProperty("id").GetString() == "release-candidate-public-proof-final-audit");
        Assert.False(item.GetProperty("passed").GetBoolean());
        AssertBoundary(item.GetProperty("boundary").GetString()!);
        Assert.Contains("blocked-release-candidate-public-proof-final-audit-owner-proof-required", item.GetProperty("state").GetString(), StringComparison.Ordinal);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/release-candidate-public-proof-final-audit.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-candidate-public-proof-final-audit.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-candidate-public-proof-final-audit-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-candidate-public-proof-final-audit-validation.md", sourceArtifacts);

        Assert.Contains(evidence.GetProperty("nonSubstituteProofKinds").EnumerateArray(), static marker =>
            marker.GetString() == "release candidate public proof final audit");
        Assert.Contains(evidence.GetProperty("nonSubstituteProofKinds").EnumerateArray(), static marker =>
            marker.GetString() == "final public proof path");

        using JsonDocument classificationDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement classification = classificationDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", classification.GetProperty("auditState").GetString());
        Assert.Equal(0, classification.GetProperty("findingCount").GetInt32());
        Assert.Contains(classification.GetProperty("auditedItems").EnumerateArray(), static audited =>
            audited.GetProperty("id").GetString() == "release-candidate-public-proof-final-audit" &&
            audited.GetProperty("passed").GetBoolean() == false &&
            audited.GetProperty("hasNonProofBoundary").GetBoolean());
    }

    [Fact]
    public void FinalAuditValidatorRejectsAcceptedForbiddenSubstituteAndPromotedFlags()
    {
        RunAuditPipeline();

        string sourcePath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-candidate-public-proof-final-audit.json");
        JsonObject audit = JsonNode.Parse(File.ReadAllText(sourcePath))!.AsObject();
        audit["canCloseReleaseIssue"] = true;
        audit["canPublishPublicly"] = true;

        JsonObject firstStep = audit["finalProofChain"]!.AsArray()[0]!.AsObject();
        firstStep["proofAccepted"] = true;
        firstStep["canPublishPublicly"] = true;

        JsonObject firstForbiddenSubstitute = audit["forbiddenSubstituteAudit"]!.AsArray()[0]!.AsObject();
        firstForbiddenSubstitute["accepted"] = true;

        string misusePath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-candidate-public-proof-final-audit.misuse.json");
        File.WriteAllText(misusePath, audit.ToJsonString(IndentedJsonOptions));

        RunPowerShell(
            "Test-ReleaseCandidatePublicProofFinalAudit.ps1",
            "-InputPath",
            "artifacts/final-release/release-candidate-public-proof-final-audit.misuse.json");

        using JsonDocument validationDocument = ReadFinalReleaseJson("release-candidate-public-proof-final-audit-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("invalid-release-candidate-public-proof-final-audit", validation.GetProperty("validationState").GetString());
        Assert.True(validation.GetProperty("failedBlockerCount").GetInt32() > 0);
        AssertValidationItemFailed(validation, "top-level-non-proof-flags");
        AssertValidationItemFailed(validation, "final-proof-chain-blocked-non-proof");
        AssertValidationItemFailed(validation, "forbidden-substitutes-not-accepted");

        RunPowerShell("Export-ReleaseCandidatePublicProofFinalAudit.ps1");
        RunPowerShell("Test-ReleaseCandidatePublicProofFinalAudit.ps1", "-Strict");
    }

    private static void RunAuditPipeline()
    {
        RunPowerShell("Export-FinalOwnerExecutionOneScreenPack.ps1");
        RunPowerShell("Test-FinalOwnerExecutionOneScreenPack.ps1", "-Strict");
        RunPowerShell("Export-FinalPublicReleaseClosureBridge.ps1");
        RunPowerShell("Test-FinalPublicReleaseClosureBridge.ps1", "-Strict");
        RunPowerShell("Export-ReleaseIssueCloseOwnerDecisionInput.ps1");
        RunPowerShell("Test-ReleaseIssueCloseOwnerDecisionInput.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");
        RunPowerShell("Export-ReleaseCandidatePublicProofFinalAudit.ps1");
        RunPowerShell("Test-ReleaseCandidatePublicProofFinalAudit.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");
    }

    private static void AssertIds(JsonElement root, string propertyName, string[] expectedIds)
    {
        string[] ids = root.GetProperty(propertyName).EnumerateArray().Select(static item => item.GetProperty("id").GetString()!).ToArray();
        foreach (string expectedId in expectedIds)
        {
            Assert.Contains(expectedId, ids);
        }
    }

    private static void AssertValidationItemFailed(JsonElement validation, string id)
    {
        JsonElement item = validation.GetProperty("validationItems")
            .EnumerateArray()
            .Single(candidate => candidate.GetProperty("id").GetString() == id);

        Assert.False(item.GetProperty("passed").GetBoolean());
        Assert.Equal("blocker", item.GetProperty("severity").GetString());
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    private static void AssertFalseProofPublishCloseFlags(JsonElement element)
    {
        Assert.False(element.GetProperty("performsPublish").GetBoolean());
        Assert.False(element.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(element.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(element.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(element.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(element.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(element.GetProperty("isReleaseCloseProof").GetBoolean());
    }

    private static void AssertBoundary(string boundary)
    {
        Assert.Contains("not runtime proof", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not post-publish proof", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not publish approval", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not release close approval", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package push", boundary, StringComparison.OrdinalIgnoreCase);
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

    private static readonly string[] RequiredFinalProofStepIds =
    [
        "github-actions-run-evidence",
        "owner-public-publish-result",
        "public-package-download-proof",
        "post-publish-clean-consumer-proof-result",
        "final-public-release-closure-bridge",
        "release-issue-close-owner-decision-input",
    ];

    private static readonly string[] RequiredEvidenceItemIds =
    [
        "remote-ci-and-public-publish-proof-backfill-gate",
        "owner-public-publish-execution-result-candidate",
        "post-publish-clean-consumer-proof-result-import",
        "post-publish-clean-consumer-proof-result-candidate",
        "final-public-release-closure-bridge",
        "release-issue-close-owner-decision-input",
        "final-owner-execution-one-screen-pack",
    ];

    private static readonly string[] RequiredSourceArtifacts =
    [
        "artifacts/final-release/github-actions-run-evidence-import-validation.json",
        "artifacts/final-release/owner-public-publish-execution-result-candidate-validation.json",
        "artifacts/final-release/public-package-download-proof-candidate-validation.json",
        "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json",
        "artifacts/final-release/final-public-release-closure-bridge.json",
        "artifacts/final-release/final-public-release-closure-bridge-validation.json",
        "artifacts/final-release/release-issue-close-owner-decision-input.template.json",
        "artifacts/final-release/release-issue-close-owner-decision-input-validation.json",
        "artifacts/final-release/final-owner-execution-one-screen-pack.json",
        "artifacts/final-release/final-owner-execution-one-screen-pack-validation.json",
    ];
}

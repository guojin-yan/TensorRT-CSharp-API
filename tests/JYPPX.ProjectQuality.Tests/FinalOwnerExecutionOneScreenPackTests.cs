using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalOwnerExecutionOneScreenPackTests
{
    [Fact]
    public void OneScreenPackAggregatesFinalOwnerLanesWithoutPromotion()
    {
        RunPowerShell("Export-FinalOwnerExecutionOneScreenPack.ps1");
        RunPowerShell("Test-FinalOwnerExecutionOneScreenPack.ps1", "-Strict");

        using JsonDocument packDocument = ReadFinalReleaseJson("final-owner-execution-one-screen-pack.json");
        JsonElement pack = packDocument.RootElement;

        Assert.Equal("final-owner-execution-one-screen-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-execution-one-screen-real-owner-input-required", pack.GetProperty("packState").GetString());
        Assert.True(pack.GetProperty("ownerActionRequired").GetBoolean());
        Assert.False(pack.GetProperty("performsPublish").GetBoolean());
        Assert.False(pack.GetProperty("performsRuntimeExecution").GetBoolean());
        Assert.False(pack.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(pack.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(pack.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(pack.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(pack.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(pack.GetProperty("isReleaseCloseProof").GetBoolean());

        Assert.True(pack.GetProperty("laneCount").GetInt32() >= 6);
        Assert.True(pack.GetProperty("ownerInputGapCount").GetInt32() >= 17);
        Assert.Equal(9, pack.GetProperty("finalPublicProofPathCount").GetInt32());
        Assert.Equal(9, pack.GetProperty("blockedFinalPublicProofPathCount").GetInt32());
        Assert.Equal(pack.GetProperty("laneCount").GetInt32(), pack.GetProperty("blockedLaneCount").GetInt32());
        Assert.Equal(pack.GetProperty("ownerInputGapCount").GetInt32(), pack.GetProperty("blockedOwnerInputGapCount").GetInt32());
        Assert.Equal(2, pack.GetProperty("dualPackageRouteCount").GetInt32());
        Assert.Equal(2, pack.GetProperty("dualPackageFinalCloseBlockedLaneCount").GetInt32());
        Assert.False(pack.GetProperty("dualPackageAcceptsSubstituteProof").GetBoolean());

        AssertIds(pack, "lanes", new[]
        {
            "clean-external-package-consumer",
            "post-publish-owner-verification",
            "owner-public-publish-result-input",
            "post-publish-proof-record-contract",
            "final-release-close-owner-approval",
            "release-evidence-and-public-docs-freeze"
        });
        AssertIds(pack, "ownerInputGapTable", new[]
        {
            "clean-consumer-root",
            "package-source-url",
            "managed-package-identity",
            "runtime-package-identity",
            "native-asset-listing",
            "restore-build-run-logs",
            "log-sha256",
            "runtime-exit-and-smoke-status",
            "runtime-execution-timestamps",
            "host-metadata",
            "owner-review",
            "post-publish-downloaded-package-hash",
            "dual-package-nuget-route-owner-proof",
            "dual-package-github-runtime-route-owner-proof",
            "rollback-review",
            "final-close-decision",
            "strict-validator-chain"
        });
        AssertIds(pack, "finalPublicProofPath", new[]
        {
            "github-actions-run-evidence",
            "owner-public-publish-result",
            "public-package-download-proof",
            "public-package-download-owner-execution-pack",
            "post-publish-clean-consumer-proof-result",
            "post-publish-user-verification-pack",
            "final-public-release-closure-bridge",
            "release-issue-close-owner-decision-input",
            "dual-package-final-close-lanes"
        });

        JsonElement[] lanes = pack.GetProperty("lanes").EnumerateArray().ToArray();
        Assert.All(lanes, item =>
        {
            Assert.True(item.GetProperty("blocked").GetBoolean());
            Assert.True(item.GetProperty("ownerActionRequired").GetBoolean());
            Assert.False(item.GetProperty("performsPublish").GetBoolean());
            Assert.False(item.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.False(item.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
            AssertBoundary(item.GetProperty("boundary").GetString()!);
        });

        JsonElement[] finalPublicProofPath = pack.GetProperty("finalPublicProofPath").EnumerateArray().ToArray();
        Assert.All(finalPublicProofPath, item =>
        {
            Assert.True(item.GetProperty("blocked").GetBoolean());
            Assert.True(item.GetProperty("ownerActionRequired").GetBoolean());
            Assert.False(item.GetProperty("performsPublish").GetBoolean());
            Assert.False(item.GetProperty("performsRuntimeExecution").GetBoolean());
            Assert.False(item.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.False(item.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(item.GetProperty("isRuntimeExecutionProof").GetBoolean());
            Assert.False(item.GetProperty("isPostPublishProof").GetBoolean());
            Assert.False(item.GetProperty("isReleaseCloseProof").GetBoolean());
            AssertBoundary(item.GetProperty("boundary").GetString()!);
        });

        string[] sources = pack.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/clean-external-package-consumer-owner-runbook.json", sources);
        Assert.Contains("artifacts/final-release/post-publish-owner-verification-runbook.json", sources);
        Assert.Contains("artifacts/final-release/owner-public-publish-execution-result-input-contract.json", sources);
        Assert.Contains("artifacts/final-release/final-release-close-owner-approval-contract.json", sources);
        Assert.Contains("artifacts/final-release/public-proof-claim-boundary-audit.json", sources);
        Assert.Contains("artifacts/final-release/release-evidence-classification-audit.json", sources);
        Assert.Contains("artifacts/final-release/github-actions-run-evidence-import-validation.json", sources);
        Assert.Contains("artifacts/final-release/owner-public-publish-execution-result-candidate-validation.json", sources);
        Assert.Contains("artifacts/final-release/public-package-download-proof-candidate-validation.json", sources);
        Assert.Contains("artifacts/final-release/public-package-download-proof-owner-execution-pack-validation.json", sources);
        Assert.Contains("artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json", sources);
        Assert.Contains("artifacts/final-release/post-publish-user-verification-pack-validation.json", sources);
        Assert.Contains("artifacts/final-release/final-public-release-closure-bridge-validation.json", sources);
        Assert.Contains("artifacts/final-release/release-issue-close-owner-decision-input-validation.json", sources);
        Assert.Contains("artifacts/final-release/dual-package-publish-preflight-matrix.json", sources);
        Assert.Contains("artifacts/final-release/dual-package-publish-preflight-matrix-validation.json", sources);
        Assert.Contains("artifacts/final-release/final-close-gate-convergence.json", sources);
        Assert.Contains("artifacts/final-release/final-close-gate-convergence-validation.json", sources);

        string packText = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "final-owner-execution-one-screen-pack.json"));
        Assert.Contains("Test-PackageConsumerRuntimeProofRecord.ps1", packText, StringComparison.Ordinal);
        Assert.Contains("-RequireExistingLog", packText, StringComparison.Ordinal);
        Assert.Contains("-FailOnNotProof", packText, StringComparison.Ordinal);
        Assert.Contains("Test-PostPublishCleanConsumerProofRecordDraft.ps1", packText, StringComparison.Ordinal);
        Assert.Contains("Test-ReleaseEvidenceClassificationAudit.ps1", packText, StringComparison.Ordinal);
        Assert.Contains("Test-GitHubActionsRunEvidenceImport.ps1", packText, StringComparison.Ordinal);
        Assert.Contains("Test-OwnerPublicPublishExecutionResultCandidate.ps1", packText, StringComparison.Ordinal);
        Assert.Contains("Test-PublicPackageDownloadProofCandidate.ps1", packText, StringComparison.Ordinal);
        Assert.Contains("Test-PublicPackageDownloadProofOwnerExecutionPack.ps1", packText, StringComparison.Ordinal);
        Assert.Contains("Test-PostPublishUserVerificationPack.ps1", packText, StringComparison.Ordinal);
        Assert.Contains("Test-FinalPublicReleaseClosureBridge.ps1", packText, StringComparison.Ordinal);
        Assert.Contains("Test-ReleaseIssueCloseOwnerDecisionInput.ps1", packText, StringComparison.Ordinal);
        Assert.Contains("Test-DualPackagePublishPreflightMatrix.ps1", packText, StringComparison.Ordinal);
        Assert.Contains("dual-package-final-close-lanes", packText, StringComparison.Ordinal);
        Assert.Contains("owner input gap table", packText, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("final public proof path", packText, StringComparison.OrdinalIgnoreCase);

        using JsonDocument validationDocument = ReadFinalReleaseJson("final-owner-execution-one-screen-pack-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-owner-execution-one-screen-pack-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-execution-one-screen-real-owner-input-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(9, validation.GetProperty("finalPublicProofPathCount").GetInt32());
        Assert.Equal(2, validation.GetProperty("dualPackageRouteCount").GetInt32());
        Assert.Equal(2, validation.GetProperty("dualPackageFinalCloseBlockedLaneCount").GetInt32());
        Assert.False(validation.GetProperty("dualPackageAcceptsSubstituteProof").GetBoolean());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        AssertBoundary(validation.GetProperty("boundary").GetString()!);
    }

    [Fact]
    public void ReleaseEvidenceCarriesOneScreenPackAsBlockedNonProof()
    {
        RunPowerShell("Export-FinalOwnerExecutionOneScreenPack.ps1");
        RunPowerShell("Test-FinalOwnerExecutionOneScreenPack.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        JsonElement item = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static value => value.GetProperty("id").GetString() == "final-owner-execution-one-screen-pack");
        Assert.False(item.GetProperty("passed").GetBoolean());
        AssertBoundary(item.GetProperty("boundary").GetString()!);
        Assert.Contains("ownerInputGaps=", item.GetProperty("state").GetString(), StringComparison.Ordinal);

        Assert.Contains(evidence.GetProperty("nonSubstituteProofKinds").EnumerateArray(), static marker =>
            marker.GetString() == "final owner execution one-screen pack");
        Assert.Contains(evidence.GetProperty("nonSubstituteProofKinds").EnumerateArray(), static marker =>
            marker.GetString() == "owner input gap table");
        Assert.Contains(evidence.GetProperty("nonSubstituteProofKinds").EnumerateArray(), static marker =>
            marker.GetString() == "final public proof path");

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        Assert.Contains(audit.GetProperty("auditedItems").EnumerateArray(), static audited =>
            audited.GetProperty("id").GetString() == "final-owner-execution-one-screen-pack" &&
            audited.GetProperty("passed").GetBoolean() == false &&
            audited.GetProperty("hasNonProofBoundary").GetBoolean());
    }

    private static void AssertIds(JsonElement root, string propertyName, string[] expectedIds)
    {
        string[] ids = root.GetProperty(propertyName).EnumerateArray().Select(static item => item.GetProperty("id").GetString()!).ToArray();
        foreach (string expectedId in expectedIds)
        {
            Assert.Contains(expectedId, ids);
        }
    }

    private static void AssertBoundary(string boundary)
    {
        Assert.Contains("not runtime proof", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not post-publish proof", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not publish approval", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not release close approval", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package push", boundary, StringComparison.OrdinalIgnoreCase);
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
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

using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class StrictCloseRealInputValidationTests
{
    private static readonly string[] EvidenceIds =
    [
        "owner-real-input-json-contract",
        "owner-real-input-json-import",
        "owner-real-input-hash-and-path-validator",
        "owner-real-input-forbidden-substitute-validator",
        "strict-close-real-input-dry-run",
        "strict-close-real-input-finding-report",
        "strict-close-owner-action-pack",
        "release-close-real-input-final-blocker-ledger",
    ];

    private static readonly string[] SourceArtifacts =
    [
        "artifacts/final-release/owner-real-input-json-contract.json",
        "artifacts/final-release/owner-real-input-json-contract.md",
        "artifacts/final-release/owner-real-input-json-contract-validation.json",
        "artifacts/final-release/owner-real-input-json-contract-validation.md",
        "artifacts/final-release/owner-real-input-json-import.json",
        "artifacts/final-release/owner-real-input-json-import.md",
        "artifacts/final-release/owner-real-input-json-import-validation.json",
        "artifacts/final-release/owner-real-input-json-import-validation.md",
        "artifacts/final-release/owner-real-input-hash-and-path-validator.json",
        "artifacts/final-release/owner-real-input-hash-and-path-validator.md",
        "artifacts/final-release/owner-real-input-hash-and-path-validator-validation.json",
        "artifacts/final-release/owner-real-input-hash-and-path-validator-validation.md",
        "artifacts/final-release/owner-real-input-forbidden-substitute-validator.json",
        "artifacts/final-release/owner-real-input-forbidden-substitute-validator.md",
        "artifacts/final-release/owner-real-input-forbidden-substitute-validator-validation.json",
        "artifacts/final-release/owner-real-input-forbidden-substitute-validator-validation.md",
        "artifacts/final-release/strict-close-real-input-dry-run.json",
        "artifacts/final-release/strict-close-real-input-dry-run.md",
        "artifacts/final-release/strict-close-real-input-dry-run-validation.json",
        "artifacts/final-release/strict-close-real-input-dry-run-validation.md",
        "artifacts/final-release/strict-close-real-input-finding-report.json",
        "artifacts/final-release/strict-close-real-input-finding-report.md",
        "artifacts/final-release/strict-close-real-input-finding-report-validation.json",
        "artifacts/final-release/strict-close-real-input-finding-report-validation.md",
        "artifacts/final-release/strict-close-owner-action-pack.json",
        "artifacts/final-release/strict-close-owner-action-pack.md",
        "artifacts/final-release/strict-close-owner-action-pack-validation.json",
        "artifacts/final-release/strict-close-owner-action-pack-validation.md",
        "artifacts/final-release/release-close-real-input-final-blocker-ledger.json",
        "artifacts/final-release/release-close-real-input-final-blocker-ledger.md",
        "artifacts/final-release/release-close-real-input-final-blocker-ledger-validation.json",
        "artifacts/final-release/release-close-real-input-final-blocker-ledger-validation.md",
    ];

    [Fact]
    public void StrictCloseRealInputArtifactsStayBlockedNonProof()
    {
        RunPipeline();

        string[] requiredFinalBlockerIds =
        [
            "owner-authorization",
            "package-consumer-runtime",
            "linux-runner-proof",
            "real-model-runtime",
            "post-publish-verification",
        ];

        foreach (string id in EvidenceIds)
        {
            using JsonDocument artifact = ReadFinalReleaseJson(id + ".json");
            JsonElement root = artifact.RootElement;

            Assert.Equal(id, root.GetProperty("artifactId").GetString());
            Assert.False(root.GetProperty("passed").GetBoolean());
            Assert.False(root.GetProperty("performsPublish").GetBoolean());
            Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
            Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());
            Assert.False(root.GetProperty("isReleaseCloseProof").GetBoolean());
            Assert.False(root.GetProperty("isReleaseCloseRecordProof").GetBoolean());

            string[] forbiddenSubstitutes = root.GetProperty("forbiddenSubstitutes").EnumerateArray().Select(static item => item.GetString()!).ToArray();
            Assert.Contains("local feed", forbiddenSubstitutes);
            Assert.Contains("ProjectReference", forbiddenSubstitutes);
            Assert.Contains("direct nupkg", forbiddenSubstitutes);
            Assert.Contains("real proof readiness gate", forbiddenSubstitutes);
            Assert.Contains("pre-publish package", forbiddenSubstitutes);
            Assert.Contains("build-only", forbiddenSubstitutes);
            Assert.Contains("dependency probe", forbiddenSubstitutes);
            Assert.Contains("blocked-by-driver", forbiddenSubstitutes);
            AssertBoundary(root.GetProperty("boundary").GetString()!);

            Assert.Equal(5, root.GetProperty("finalBlockerLaneCount").GetInt32());
            string[] finalBlockerIds = root.GetProperty("finalBlockerIds").EnumerateArray().Select(static item => item.GetString()!).ToArray();
            foreach (string blockerId in requiredFinalBlockerIds)
            {
                Assert.Contains(blockerId, finalBlockerIds);
            }

            JsonElement[] lanes = root.GetProperty("finalBlockerLanes").EnumerateArray().ToArray();
            Assert.Equal(5, lanes.Length);
            foreach (JsonElement lane in lanes)
            {
                Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("ownerInputFile").GetString()));
                Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("validator").GetString()));
                Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("releaseCloseTarget").GetString()));
                Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("acceptanceRule").GetString()));

                string[] requiredEvidenceFields = lane.GetProperty("requiredEvidenceFields")
                    .EnumerateArray()
                    .Select(static item => item.GetString()!)
                    .ToArray();
                Assert.Contains("stdoutPath", requiredEvidenceFields);
                Assert.Contains("stderrPath", requiredEvidenceFields);
                Assert.Contains("transcriptPath", requiredEvidenceFields);
                Assert.Contains("logPath", requiredEvidenceFields);
                Assert.Contains("logSha256", requiredEvidenceFields);
                Assert.Contains("exitCode", requiredEvidenceFields);
                Assert.Contains("hostIdentity", requiredEvidenceFields);
                Assert.Contains("ownerReviewer", requiredEvidenceFields);
                Assert.Contains("nonSubstituteConfirmations", requiredEvidenceFields);
            }

            string[] releaseCloseTargets = root.GetProperty("releaseCloseTargetMapping")
                .EnumerateArray()
                .Select(static item => item.GetProperty("releaseCloseTarget").GetString()!)
                .ToArray();
            Assert.Contains("release-issue-close-strict-owner-decision-import", releaseCloseTargets);
            Assert.Contains("final-publish-proof-gate-report", releaseCloseTargets);
            Assert.Contains("release-close-real-proof-import-bridge", releaseCloseTargets);
            Assert.Contains("real-proof-record-candidate-from-owner-result-import", releaseCloseTargets);
            Assert.Contains("release-issue-close-record", releaseCloseTargets);

            string[] requiredOwnerFields = root.GetProperty("requiredOwnerFields")
                .EnumerateArray()
                .Select(static item => item.GetString()!)
                .ToArray();
            string joinedItems = string.Join("\n", root.EnumerateObject()
                .Where(static property => property.Name is "fields" or "importChecks" or "validationChecks" or "substituteChecks")
                .SelectMany(static property => property.Value.EnumerateArray())
                .Select(static item => item.GetProperty("title").GetString()));
            if (id == "owner-real-input-json-contract")
            {
                Assert.Contains("nugetPackageSource", requiredOwnerFields);
                Assert.Contains("githubRelease.releaseUrl", requiredOwnerFields);
                Assert.Contains("githubRelease.managedAssetSha256", requiredOwnerFields);
                Assert.Contains("githubRelease.runtimeAssetSha256", requiredOwnerFields);
                Assert.Contains("managedPackage.publicDownloadSha256", requiredOwnerFields);
                Assert.Contains("runtimePackage.publicDownloadSha256", requiredOwnerFields);
                Assert.Contains("publicPackageOwnerInput.cleanExternalConsumer.root", requiredOwnerFields);
                Assert.Contains("publicPackageOwnerInput.cleanExternalConsumer.restoreLogSha256", requiredOwnerFields);
                Assert.Contains("publicPackageOwnerInput.cleanExternalConsumer.stdoutLogSha256", requiredOwnerFields);
                Assert.Contains("publicPackageOwnerInput.hostMetadata", requiredOwnerFields);
                Assert.Contains("publicPackageOwnerInput.hostMetadata.tensorRtVersion", requiredOwnerFields);
                Assert.Contains("publicPackageOwnerInput.ownerReview", requiredOwnerFields);
                Assert.Contains("rollbackReview.reviewedAtUtc", requiredOwnerFields);
                Assert.Contains("finalCloseDecision.decidedAtUtc", requiredOwnerFields);
            }
            else if (id == "owner-real-input-json-import")
            {
                Assert.Contains("nugetPackageSource", requiredOwnerFields);
                Assert.Contains("githubRelease.releaseUrl", requiredOwnerFields);
                Assert.Contains("managedPackage.publicDownloadSha256", requiredOwnerFields);
                Assert.Contains("runtimePackage.publicDownloadSha256", requiredOwnerFields);
                Assert.Contains("publicPackageOwnerInput.cleanExternalConsumer.restoreLogSha256", requiredOwnerFields);
                Assert.Contains("publicPackageOwnerInput.cleanExternalConsumer.stdoutLogSha256", requiredOwnerFields);
                Assert.Contains("publicPackageOwnerInput.hostMetadata", requiredOwnerFields);
                Assert.Contains("publicPackageOwnerInput.ownerReview", requiredOwnerFields);
                Assert.Contains("rollbackReview.reviewedAtUtc", requiredOwnerFields);
                Assert.Contains("finalCloseDecision.decidedAtUtc", requiredOwnerFields);
                Assert.Contains("clean external consumer root/project/log SHA256", joinedItems, StringComparison.OrdinalIgnoreCase);
                Assert.Contains("rollbackReview", joinedItems, StringComparison.Ordinal);
                Assert.Contains("finalCloseDecision", joinedItems, StringComparison.Ordinal);
            }
            else if (id == "owner-real-input-hash-and-path-validator")
            {
                Assert.Contains("nugetPackageSource", requiredOwnerFields);
                Assert.Contains("githubRelease.releaseUrl", requiredOwnerFields);
                Assert.Contains("githubRelease.managedAssetSha256", requiredOwnerFields);
                Assert.Contains("githubRelease.runtimeAssetSha256", requiredOwnerFields);
                Assert.Contains("managedPackage.publicDownloadSha256", requiredOwnerFields);
                Assert.Contains("runtimePackage.publicDownloadSha256", requiredOwnerFields);
                Assert.Contains("publicPackageOwnerInput.cleanExternalConsumer.restoreLogSha256", requiredOwnerFields);
                Assert.Contains("publicPackageOwnerInput.cleanExternalConsumer.stdoutLogSha256", requiredOwnerFields);
                Assert.Contains("publicPackageOwnerInput.hostMetadata", requiredOwnerFields);
                Assert.Contains("GitHub Release", joinedItems, StringComparison.Ordinal);
                Assert.Contains("public download SHA256", joinedItems, StringComparison.OrdinalIgnoreCase);
                Assert.Contains("restore/build/smoke/stdout/stderr", joinedItems, StringComparison.OrdinalIgnoreCase);
            }
            else if (id == "owner-real-input-forbidden-substitute-validator")
            {
                Assert.Contains("prePublishPackageReferenceCount", requiredOwnerFields);
                Assert.Contains("dependencyProbeOnlyCount", requiredOwnerFields);
                Assert.Contains("blockedByDriverOnlyCount", requiredOwnerFields);
                Assert.Contains("build-only record count", joinedItems, StringComparison.OrdinalIgnoreCase);
            }

            using JsonDocument validation = ReadFinalReleaseJson(id + "-validation.json");
            JsonElement validationRoot = validation.RootElement;
            Assert.Equal("validation-passed-non-proof-strict-close-real-input-boundary-intact", validationRoot.GetProperty("validationState").GetString());
            Assert.Equal(0, validationRoot.GetProperty("findingCount").GetInt32());
            Assert.False(validationRoot.GetProperty("passed").GetBoolean());
            AssertBoundary(validationRoot.GetProperty("boundary").GetString()!);
        }
    }

    [Fact]
    public void EvidenceBundleAuditAndDocsIncludeStrictCloseRealInputArtifacts()
    {
        RunPipeline();

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.False(evidence.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("canCloseReleaseIssue").GetBoolean());

        foreach (string id in EvidenceIds)
        {
            AssertBlockedEvidenceItem(evidence, id);
        }

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string sourceArtifact in SourceArtifacts)
        {
            Assert.Contains(sourceArtifact, sourceArtifacts);
        }

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        foreach (string id in EvidenceIds)
        {
            AssertAuditedNonProofItem(audit, id);
        }

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string releaseEvidenceDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));

        foreach (string id in EvidenceIds)
        {
            Assert.Contains(id, docsIndex, StringComparison.Ordinal);
            Assert.Contains(id, docsToc, StringComparison.Ordinal);
            Assert.Contains(id, readme, StringComparison.Ordinal);
            Assert.Contains(id, readmeZh, StringComparison.Ordinal);
            Assert.Contains(id, releaseEvidenceDoc, StringComparison.Ordinal);
        }
    }

    private static void RunPipeline()
    {
        RunPowerShell("Export-OwnerRealInputJsonContract.ps1");
        RunPowerShell("Test-OwnerRealInputJsonContract.ps1", "-Strict");
        RunPowerShell("Export-OwnerRealInputJsonImport.ps1");
        RunPowerShell("Test-OwnerRealInputJsonImport.ps1", "-Strict");
        RunPowerShell("Export-OwnerRealInputHashAndPathValidator.ps1");
        RunPowerShell("Test-OwnerRealInputHashAndPathValidator.ps1", "-Strict");
        RunPowerShell("Export-OwnerRealInputForbiddenSubstituteValidator.ps1");
        RunPowerShell("Test-OwnerRealInputForbiddenSubstituteValidator.ps1", "-Strict");
        RunPowerShell("Export-StrictCloseRealInputDryRun.ps1");
        RunPowerShell("Test-StrictCloseRealInputDryRun.ps1", "-Strict");
        RunPowerShell("Export-StrictCloseRealInputFindingReport.ps1");
        RunPowerShell("Test-StrictCloseRealInputFindingReport.ps1", "-Strict");
        RunPowerShell("Export-StrictCloseOwnerActionPack.ps1");
        RunPowerShell("Test-StrictCloseOwnerActionPack.ps1", "-Strict");
        RunPowerShell("Export-ReleaseCloseRealInputFinalBlockerLedger.ps1");
        RunPowerShell("Test-ReleaseCloseRealInputFinalBlockerLedger.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    private static void AssertBoundary(string boundary)
    {
        Assert.Contains("not runtime proof", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not post-publish proof", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not publish approval", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not release close approval", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package push", boundary, StringComparison.OrdinalIgnoreCase);
    }

    private static void AssertBlockedEvidenceItem(JsonElement evidence, string id)
    {
        JsonElement item = evidence.GetProperty("evidenceItems").EnumerateArray().Single(item => item.GetProperty("id").GetString() == id);
        Assert.False(item.GetProperty("passed").GetBoolean());
        AssertBoundary(item.GetProperty("boundary").GetString()!);
    }

    private static void AssertAuditedNonProofItem(JsonElement audit, string id)
    {
        Assert.Contains(audit.GetProperty("auditedItems").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == id &&
            item.GetProperty("passed").GetBoolean() == false &&
            item.GetProperty("hasNonProofBoundary").GetBoolean());
    }

    private static void RunPowerShell(string scriptName, params string[] arguments)
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", scriptName);
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(scriptPath, arguments);
    }
}

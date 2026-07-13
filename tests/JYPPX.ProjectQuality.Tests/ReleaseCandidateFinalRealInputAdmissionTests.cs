using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleaseCandidateFinalRealInputAdmissionTests
{
    private static readonly string[] EvidenceIds =
    [
        "release-candidate-real-proof-final-freeze",
        "owner-real-input-import-preflight",
        "public-package-hash-cross-check-gate",
        "clean-consumer-runtime-proof-cross-check-gate",
        "post-publish-rollback-owner-decision-gate",
        "release-close-final-real-input-admission-pack",
    ];

    private static readonly string[] SourceArtifacts =
    [
        "artifacts/final-release/release-candidate-real-proof-final-freeze.json",
        "artifacts/final-release/release-candidate-real-proof-final-freeze.md",
        "artifacts/final-release/release-candidate-real-proof-final-freeze-validation.json",
        "artifacts/final-release/release-candidate-real-proof-final-freeze-validation.md",
        "artifacts/final-release/owner-real-input-import-preflight.json",
        "artifacts/final-release/owner-real-input-import-preflight.md",
        "artifacts/final-release/owner-real-input-import-preflight-validation.json",
        "artifacts/final-release/owner-real-input-import-preflight-validation.md",
        "artifacts/final-release/public-package-hash-cross-check-gate.json",
        "artifacts/final-release/public-package-hash-cross-check-gate.md",
        "artifacts/final-release/public-package-hash-cross-check-gate-validation.json",
        "artifacts/final-release/public-package-hash-cross-check-gate-validation.md",
        "artifacts/final-release/clean-consumer-runtime-proof-cross-check-gate.json",
        "artifacts/final-release/clean-consumer-runtime-proof-cross-check-gate.md",
        "artifacts/final-release/clean-consumer-runtime-proof-cross-check-gate-validation.json",
        "artifacts/final-release/clean-consumer-runtime-proof-cross-check-gate-validation.md",
        "artifacts/final-release/post-publish-rollback-owner-decision-gate.json",
        "artifacts/final-release/post-publish-rollback-owner-decision-gate.md",
        "artifacts/final-release/post-publish-rollback-owner-decision-gate-validation.json",
        "artifacts/final-release/post-publish-rollback-owner-decision-gate-validation.md",
        "artifacts/final-release/release-close-final-real-input-admission-pack.json",
        "artifacts/final-release/release-close-final-real-input-admission-pack.md",
        "artifacts/final-release/release-close-final-real-input-admission-pack-validation.json",
        "artifacts/final-release/release-close-final-real-input-admission-pack-validation.md",
    ];

    [Fact]
    public void FinalRealInputAdmissionArtifactsStayBlockedNonProof()
    {
        RunPipeline();

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
            AssertBoundary(root.GetProperty("boundary").GetString()!);

            if (id == "public-package-hash-cross-check-gate")
            {
                string[] requiredOwnerFields = root.GetProperty("requiredOwnerFields").EnumerateArray().Select(static item => item.GetString()!).ToArray();
                Assert.Contains("nugetPackageSource", requiredOwnerFields);
                Assert.Contains("githubRelease.managedAssetSha256", requiredOwnerFields);
                Assert.Contains("githubRelease.runtimeAssetSha256", requiredOwnerFields);
                Assert.Contains("managedPackage.publicDownloadSha256", requiredOwnerFields);
                Assert.Contains("runtimePackage.publicDownloadSha256", requiredOwnerFields);
                Assert.Contains("ownerReview.reviewedAtUtc", requiredOwnerFields);

                string joinedChecks = string.Join("\n", root.GetProperty("checks").EnumerateArray().Select(static item => item.GetProperty("title").GetString()));
                Assert.Contains("NuGet package source", joinedChecks, StringComparison.Ordinal);
                Assert.Contains("GitHub Release", joinedChecks, StringComparison.Ordinal);
                Assert.Contains("public download SHA256", joinedChecks, StringComparison.OrdinalIgnoreCase);
            }

            if (id == "clean-consumer-runtime-proof-cross-check-gate")
            {
                string[] requiredOwnerFields = root.GetProperty("requiredOwnerFields").EnumerateArray().Select(static item => item.GetString()!).ToArray();
                Assert.Contains("publicPackageOwnerInput.nugetPackageSource", requiredOwnerFields);
                Assert.Contains("publicPackageOwnerInput.cleanExternalConsumer.restoreLogSha256", requiredOwnerFields);
                Assert.Contains("publicPackageOwnerInput.cleanExternalConsumer.stdoutLogSha256", requiredOwnerFields);
                Assert.Contains("publicPackageOwnerInput.hostMetadata", requiredOwnerFields);
                Assert.Contains("publicPackageOwnerInput.ownerReview", requiredOwnerFields);

                string joinedChecks = string.Join("\n", root.GetProperty("checks").EnumerateArray().Select(static item => item.GetProperty("title").GetString()));
                Assert.Contains("package source 必须一致", joinedChecks, StringComparison.Ordinal);
                Assert.Contains("log SHA256", joinedChecks, StringComparison.OrdinalIgnoreCase);
                Assert.Contains("host metadata", joinedChecks, StringComparison.OrdinalIgnoreCase);
            }

            using JsonDocument validation = ReadFinalReleaseJson(id + "-validation.json");
            JsonElement validationRoot = validation.RootElement;
            Assert.Equal("validation-passed-non-proof-final-real-input-admission-boundary-intact", validationRoot.GetProperty("validationState").GetString());
            Assert.Equal(0, validationRoot.GetProperty("findingCount").GetInt32());
            Assert.False(validationRoot.GetProperty("passed").GetBoolean());
            AssertBoundary(validationRoot.GetProperty("boundary").GetString()!);
        }
    }

    [Fact]
    public void EvidenceBundleAuditAndDocsIncludeFinalRealInputAdmissionArtifacts()
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
        RunPowerShell("Export-ReleaseCandidateRealProofFinalFreeze.ps1");
        RunPowerShell("Test-ReleaseCandidateRealProofFinalFreeze.ps1", "-Strict");
        RunPowerShell("Export-OwnerRealInputImportPreflight.ps1");
        RunPowerShell("Test-OwnerRealInputImportPreflight.ps1", "-Strict");
        RunPowerShell("Export-PublicPackageHashCrossCheckGate.ps1");
        RunPowerShell("Test-PublicPackageHashCrossCheckGate.ps1", "-Strict");
        RunPowerShell("Export-CleanConsumerRuntimeProofCrossCheckGate.ps1");
        RunPowerShell("Test-CleanConsumerRuntimeProofCrossCheckGate.ps1", "-Strict");
        RunPowerShell("Export-PostPublishRollbackOwnerDecisionGate.ps1");
        RunPowerShell("Test-PostPublishRollbackOwnerDecisionGate.ps1", "-Strict");
        RunPowerShell("Export-ReleaseCloseFinalRealInputAdmissionPack.ps1");
        RunPowerShell("Test-ReleaseCloseFinalRealInputAdmissionPack.ps1", "-Strict");
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

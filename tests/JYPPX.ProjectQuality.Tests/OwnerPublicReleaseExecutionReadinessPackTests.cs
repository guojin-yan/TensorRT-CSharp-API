using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerPublicReleaseExecutionReadinessPackTests
{
    private static readonly string[] EvidenceIds =
    [
        "public-release-owner-execution-package",
        "external-clean-consumer-proof-kit",
        "runtime-proof-compatible-host-kit",
        "post-publish-owner-verification-kit",
        "owner-public-release-execution-readiness-pack",
    ];

    private static readonly string[] SourceArtifacts =
    [
        "artifacts/final-release/public-release-owner-execution-package.json",
        "artifacts/final-release/public-release-owner-execution-package.md",
        "artifacts/final-release/public-release-owner-execution-package-validation.json",
        "artifacts/final-release/public-release-owner-execution-package-validation.md",
        "artifacts/final-release/external-clean-consumer-proof-kit.json",
        "artifacts/final-release/external-clean-consumer-proof-kit.md",
        "artifacts/final-release/external-clean-consumer-proof-kit-validation.json",
        "artifacts/final-release/external-clean-consumer-proof-kit-validation.md",
        "artifacts/final-release/runtime-proof-compatible-host-kit.json",
        "artifacts/final-release/runtime-proof-compatible-host-kit.md",
        "artifacts/final-release/runtime-proof-compatible-host-kit-validation.json",
        "artifacts/final-release/runtime-proof-compatible-host-kit-validation.md",
        "artifacts/final-release/post-publish-owner-verification-kit.json",
        "artifacts/final-release/post-publish-owner-verification-kit.md",
        "artifacts/final-release/post-publish-owner-verification-kit-validation.json",
        "artifacts/final-release/post-publish-owner-verification-kit-validation.md",
        "artifacts/final-release/owner-public-release-execution-readiness-pack.json",
        "artifacts/final-release/owner-public-release-execution-readiness-pack.md",
        "artifacts/final-release/owner-public-release-execution-readiness-pack-validation.json",
        "artifacts/final-release/owner-public-release-execution-readiness-pack-validation.md",
    ];

    [Fact]
    public void OwnerExecutionArtifactsStayBlockedNonProof()
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
            AssertBoundary(root.GetProperty("boundary").GetString()!);

            using JsonDocument validation = ReadFinalReleaseJson(id + "-validation.json");
            JsonElement validationRoot = validation.RootElement;
            Assert.Equal("validation-passed-non-proof-owner-execution-boundary-intact", validationRoot.GetProperty("validationState").GetString());
            Assert.Equal(0, validationRoot.GetProperty("findingCount").GetInt32());
            Assert.False(validationRoot.GetProperty("passed").GetBoolean());
            AssertBoundary(validationRoot.GetProperty("boundary").GetString()!);
        }
    }

    [Fact]
    public void ReleaseEvidenceBundleAndClassificationAuditIncludeOwnerExecutionArtifacts()
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
        RunPowerShell("Export-PublicReleaseOwnerExecutionPackage.ps1");
        RunPowerShell("Test-PublicReleaseOwnerExecutionPackage.ps1", "-Strict");
        RunPowerShell("Export-ExternalCleanConsumerProofKit.ps1");
        RunPowerShell("Test-ExternalCleanConsumerProofKit.ps1", "-Strict");
        RunPowerShell("Export-RuntimeProofCompatibleHostKit.ps1");
        RunPowerShell("Test-RuntimeProofCompatibleHostKit.ps1", "-Strict");
        RunPowerShell("Export-PostPublishOwnerVerificationKit.ps1");
        RunPowerShell("Test-PostPublishOwnerVerificationKit.ps1", "-Strict");
        RunPowerShell("Export-OwnerPublicReleaseExecutionReadinessPack.ps1");
        RunPowerShell("Test-OwnerPublicReleaseExecutionReadinessPack.ps1", "-Strict");
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

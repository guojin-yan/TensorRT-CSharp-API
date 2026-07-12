using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleaseCloseStrictValidationBridgeTests
{
    [Fact]
    public void ReleaseCloseStrictBridgeExportsBlockedPrerequisiteAggregator()
    {
        RunReleaseCloseStrictBridgePipeline();

        using JsonDocument bridgeDocument = ReadFinalReleaseJson("release-close-strict-validation-bridge.json");
        JsonElement bridge = bridgeDocument.RootElement;

        Assert.Equal("release-close-strict-validation-bridge", bridge.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-release-close-strict-validation-required", bridge.GetProperty("bridgeState").GetString());
        Assert.Equal(6, bridge.GetProperty("bridgeItemCount").GetInt32());
        Assert.Equal(6, bridge.GetProperty("blockedBridgeItemCount").GetInt32());
        Assert.Equal(0, bridge.GetProperty("readyBridgeItemCount").GetInt32());
        Assert.False(bridge.GetProperty("runtimeProofInputReady").GetBoolean());
        Assert.False(bridge.GetProperty("ownerRuntimeRunbookReady").GetBoolean());
        Assert.False(bridge.GetProperty("postPublishReady").GetBoolean());
        Assert.False(bridge.GetProperty("releaseCloseRecordReady").GetBoolean());
        Assert.False(bridge.GetProperty("finalOwnerRunbookReady").GetBoolean());
        Assert.False(bridge.GetProperty("releaseEvidenceBundleReady").GetBoolean());
        Assert.False(bridge.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(bridge.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(bridge.GetProperty("canCloseReleaseIssue").GetBoolean());

        JsonElement firstItem = bridge.GetProperty("bridgeItems").EnumerateArray().First();
        Assert.Equal("blocked-release-close-strict-validation-required", firstItem.GetProperty("bridgeState").GetString());
        Assert.True(firstItem.GetProperty("blockedReasons").GetArrayLength() >= 4);
        Assert.Contains("Test-ReleaseIssueCloseRecord.ps1", firstItem.GetProperty("strictCloseCommand").GetString(), StringComparison.Ordinal);
        Assert.Contains("cannot substitute", firstItem.GetProperty("notProofBoundary").GetString(), StringComparison.OrdinalIgnoreCase);

        using JsonDocument validationDocument = ReadFinalReleaseJson("release-close-strict-validation-bridge-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("release-close-strict-validation-bridge-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-release-close-strict-validation-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(6, validation.GetProperty("bridgeItemCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("blockedBridgeItemCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("readyBridgeItemCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("failedActionRequiredCount").GetInt32());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    [Fact]
    public void ReleaseEvidenceIncludesBridgeAndRunbookWithoutPromotingProof()
    {
        RunReleaseCloseStrictBridgePipeline();
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-owner-runtime-proof-execution-required", evidence.GetProperty("ownerRuntimeProofExecutionRunbookState").GetString());
        Assert.Equal("blocked-owner-runtime-proof-execution-required", evidence.GetProperty("ownerRuntimeProofExecutionRunbookValidationState").GetString());
        Assert.Equal("blocked-release-close-strict-validation-required", evidence.GetProperty("releaseCloseStrictValidationBridgeState").GetString());
        Assert.Equal("blocked-release-close-strict-validation-required", evidence.GetProperty("releaseCloseStrictValidationBridgeValidationState").GetString());
        Assert.Equal(6, evidence.GetProperty("ownerRuntimeProofExecutionRunbookBlockedRunbookItemCount").GetInt32());
        Assert.Equal(6, evidence.GetProperty("releaseCloseStrictValidationBridgeBlockedBridgeItemCount").GetInt32());
        Assert.False(evidence.GetProperty("ownerRuntimeProofExecutionRunbookCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("releaseCloseStrictValidationBridgeCanCloseReleaseIssue").GetBoolean());

        string[] evidenceIds = evidence.GetProperty("evidenceItems").EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();
        Assert.Contains("runtime-proof-execution-input-record", evidenceIds);
        Assert.Contains("owner-runtime-proof-execution-runbook", evidenceIds);
        Assert.Contains("release-close-strict-validation-bridge", evidenceIds);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));

        Assert.Contains("runtime-proof-execution-input-record.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("owner-runtime-proof-execution-runbook.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("release-close-strict-validation-bridge", readme, StringComparison.Ordinal);
        Assert.Contains("release-close-strict-validation-bridge", readmeZh, StringComparison.Ordinal);
    }

    internal static void RunReleaseCloseStrictBridgePipelineForReuse()
    {
        OwnerRuntimeProofExecutionRunbookTests.RunOwnerRuntimeProofRunbookPipeline();
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseStrictValidationBridge.ps1"));
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseCloseStrictValidationBridge.ps1"), "-Strict");
    }

    private static void RunReleaseCloseStrictBridgePipeline()
    {
        RunReleaseCloseStrictBridgePipelineForReuse();
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }
}

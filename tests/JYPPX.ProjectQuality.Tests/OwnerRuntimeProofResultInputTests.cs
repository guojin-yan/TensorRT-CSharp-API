using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerRuntimeProofResultInputTests
{
    [Fact]
    public void OwnerRuntimeProofResultInputExportsBlockedOwnerFillTemplate()
    {
        RunOwnerRuntimeProofResultInputPipeline();

        using JsonDocument templateDocument = ReadFinalReleaseJson("owner-runtime-proof-result-input.template.json");
        JsonElement template = templateDocument.RootElement;

        Assert.Equal("owner-runtime-proof-result-input-template", template.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-runtime-proof-result-input-required", template.GetProperty("templateState").GetString());
        Assert.Equal(6, template.GetProperty("resultInputCount").GetInt32());
        Assert.Equal(6, template.GetProperty("blockedResultInputCount").GetInt32());
        Assert.Equal(0, template.GetProperty("readyResultInputCount").GetInt32());
        Assert.False(template.GetProperty("performsPublish").GetBoolean());
        Assert.False(template.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(template.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(template.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(template.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(template.GetProperty("isReleaseCloseProof").GetBoolean());

        JsonElement firstInput = template.GetProperty("resultInputs").EnumerateArray().First();
        Assert.Contains("owner-result-input", firstInput.GetProperty("resultInputId").GetString(), StringComparison.Ordinal);
        Assert.Contains("owner-fill", firstInput.GetProperty("hostMetadata").GetProperty("os").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("owner-fill", firstInput.GetProperty("packageIdentity").GetProperty("packageId").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.True(firstInput.GetProperty("requiredResultFields").GetArrayLength() >= 20);
        Assert.True(firstInput.GetProperty("missingResultFields").GetArrayLength() >= 20);
        Assert.True(firstInput.GetProperty("nonSubstituteConfirmations").GetArrayLength() >= 8);

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-runtime-proof-result-input-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("owner-runtime-proof-result-input-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-runtime-proof-result-input-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(6, validation.GetProperty("resultInputCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("blockedResultInputCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("readyResultInputCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("failedActionRequiredCount").GetInt32());
        Assert.True(validation.GetProperty("missingRealInputCount").GetInt32() >= 100);
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validation.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(validation.GetProperty("isReleaseCloseProof").GetBoolean());
    }

    [Fact]
    public void ReleaseEvidenceIncludesOwnerRuntimeProofResultInputWithoutPromotingProof()
    {
        RunOwnerRuntimeProofResultInputPipeline();
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-owner-runtime-proof-result-input-required", evidence.GetProperty("ownerRuntimeProofResultInputTemplateState").GetString());
        Assert.Equal("blocked-owner-runtime-proof-result-input-required", evidence.GetProperty("ownerRuntimeProofResultInputValidationState").GetString());
        Assert.Equal(6, evidence.GetProperty("ownerRuntimeProofResultInputResultInputCount").GetInt32());
        Assert.Equal(6, evidence.GetProperty("ownerRuntimeProofResultInputBlockedResultInputCount").GetInt32());
        Assert.True(evidence.GetProperty("ownerRuntimeProofResultInputMissingRealInputCount").GetInt32() >= 100);
        Assert.False(evidence.GetProperty("ownerRuntimeProofResultInputCanPromoteRuntimeProof").GetBoolean());
        Assert.False(evidence.GetProperty("ownerRuntimeProofResultInputCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("ownerRuntimeProofResultInputCanCloseReleaseIssue").GetBoolean());

        string[] evidenceIds = evidence.GetProperty("evidenceItems").EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();
        Assert.Contains("owner-runtime-proof-result-input-template", evidenceIds);
        Assert.Contains("owner-runtime-proof-result-input-validation", evidenceIds);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("artifacts/final-release/owner-runtime-proof-result-input.template.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-runtime-proof-result-input-validation.json", sourceArtifacts);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));

        Assert.Contains("owner-runtime-proof-result-input.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("owner-runtime-proof-result-input.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("owner-runtime-proof-result-input", readme, StringComparison.Ordinal);
        Assert.Contains("owner-runtime-proof-result-input", readmeZh, StringComparison.Ordinal);
    }

    internal static void RunOwnerRuntimeProofResultInputPipeline()
    {
        ReleaseCloseStrictValidationBridgeTests.RunReleaseCloseStrictBridgePipelineForReuse();
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerRuntimeProofResultInputTemplate.ps1"));
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerRuntimeProofResultInput.ps1"), "-Strict");
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

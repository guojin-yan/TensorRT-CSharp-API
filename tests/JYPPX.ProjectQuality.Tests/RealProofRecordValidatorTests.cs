using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class RealProofRecordValidatorTests
{
    [Fact]
    public void ValidatorExportsBlockedContractsWithoutPromotingProof()
    {
        RunRecordValidatorPipeline();

        using JsonDocument validatorDocument = ReadFinalReleaseJson("real-proof-record-validator.json");
        JsonElement validator = validatorDocument.RootElement;

        Assert.Equal("real-proof-record-validator", validator.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-proof-record-validation-input-required", validator.GetProperty("validatorState").GetString());
        Assert.Equal(6, validator.GetProperty("candidateCount").GetInt32());
        Assert.Equal(6, validator.GetProperty("blockedValidatorContractCount").GetInt32());
        Assert.Equal(0, validator.GetProperty("readyValidatorContractCount").GetInt32());
        Assert.False(validator.GetProperty("performsPublish").GetBoolean());
        Assert.False(validator.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validator.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validator.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validator.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(validator.GetProperty("isReleaseCloseProof").GetBoolean());

        JsonElement firstContract = validator.GetProperty("validatorContracts").EnumerateArray().First();
        Assert.Equal("blocked-real-proof-record-validation-input-required", firstContract.GetProperty("contractState").GetString());
        Assert.True(firstContract.GetProperty("requiredRuntimeEvidence").GetArrayLength() > 0);
        Assert.True(firstContract.GetProperty("requiredHostMetadata").GetArrayLength() > 0);
        Assert.True(firstContract.GetProperty("requiredPackageIdentity").GetArrayLength() > 0);
        Assert.True(firstContract.GetProperty("requiredCommandCapture").GetArrayLength() > 0);
        Assert.True(firstContract.GetProperty("requiredLogHash").GetArrayLength() > 0);
        Assert.True(firstContract.GetProperty("requiredValidatorOutput").GetArrayLength() > 0);
        Assert.True(firstContract.GetProperty("forbiddenSubstitutes").GetArrayLength() > 0);
        Assert.True(firstContract.GetProperty("ownerReviewRequired").GetBoolean());
        Assert.Contains("Test-RealProofRecordValidator.ps1", firstContract.GetProperty("validatorCommand").GetString(), StringComparison.Ordinal);

        using JsonDocument validationDocument = ReadFinalReleaseJson("real-proof-record-validator-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("real-proof-record-validator-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-proof-record-validation-input-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(6, validation.GetProperty("candidateCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("blockedValidatorContractCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("readyValidatorContractCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("failedActionRequiredCount").GetInt32());
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validation.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(validation.GetProperty("isReleaseCloseProof").GetBoolean());
    }

    [Fact]
    public void ReleaseEvidenceAndDocsIncludeValidator()
    {
        RunRecordValidatorPipeline();
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-real-proof-record-validation-input-required", evidence.GetProperty("realProofRecordValidatorState").GetString());
        Assert.Equal("blocked-real-proof-record-validation-input-required", evidence.GetProperty("realProofRecordValidatorValidationState").GetString());
        Assert.Equal(6, evidence.GetProperty("realProofRecordValidatorCandidateCount").GetInt32());
        Assert.Equal(6, evidence.GetProperty("realProofRecordValidatorBlockedValidatorContractCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("realProofRecordValidatorReadyValidatorContractCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("realProofRecordValidatorFailedBlockerCount").GetInt32());
        Assert.Equal(6, evidence.GetProperty("realProofRecordValidatorFailedActionRequiredCount").GetInt32());
        Assert.False(evidence.GetProperty("realProofRecordValidatorCanPromoteRuntimeProof").GetBoolean());
        Assert.False(evidence.GetProperty("realProofRecordValidatorCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("realProofRecordValidatorCanCloseReleaseIssue").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "real-proof-record-validator");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("strict future proof record contracts only", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("artifacts/final-release/real-proof-record-validator.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-proof-record-validator.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-proof-record-validator-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-proof-record-validator-validation.md", sourceArtifacts);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string releaseEvidenceDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("real-proof-record-validator.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("real-proof-record-validator.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("Export-RealProofRecordValidator.ps1", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("Test-RealProofRecordValidator.ps1 -Strict", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("real-proof-record-validator", readme, StringComparison.Ordinal);
        Assert.Contains("real-proof-record-validator", readmeZh, StringComparison.Ordinal);
        Assert.Contains("real proof record validator", evidenceMarkdown, StringComparison.OrdinalIgnoreCase);
    }

    internal static void RunRecordValidatorPipeline()
    {
        RealProofCandidatePromotionGuardTests.RunPromotionGuardPipelineForReuse();
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealProofRecordValidator.ps1"));
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-RealProofRecordValidator.ps1"), "-Strict");
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

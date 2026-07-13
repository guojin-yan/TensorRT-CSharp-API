using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class RuntimeProofExecutionInputRecordTests
{
    [Fact]
    public void RuntimeProofInputRecordExportsBlockedOwnerFillInputs()
    {
        RunRuntimeProofInputPipeline();

        using JsonDocument recordDocument = ReadFinalReleaseJson("runtime-proof-execution-input-record.json");
        JsonElement record = recordDocument.RootElement;

        Assert.Equal("runtime-proof-execution-input-record", record.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-runtime-proof-execution-input-required", record.GetProperty("inputState").GetString());
        Assert.Equal(6, record.GetProperty("executionInputCount").GetInt32());
        Assert.Equal(6, record.GetProperty("blockedExecutionInputCount").GetInt32());
        Assert.Equal(0, record.GetProperty("readyExecutionInputCount").GetInt32());
        Assert.False(record.GetProperty("performsPublish").GetBoolean());
        Assert.False(record.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(record.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(record.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(record.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(record.GetProperty("isReleaseCloseProof").GetBoolean());

        JsonElement firstInput = record.GetProperty("executionInputs").EnumerateArray().First();
        Assert.Equal("blocked-runtime-proof-execution-input-required", firstInput.GetProperty("inputState").GetString());
        Assert.True(firstInput.GetProperty("forbiddenSubstituteChecks").GetArrayLength() >= 7);
        Assert.True(firstInput.GetProperty("missingInputFields").GetArrayLength() > 20);
        Assert.Contains("owner-fill", firstInput.GetProperty("hostMetadata").GetProperty("os").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not runtime proof", firstInput.GetProperty("nonSubstituteBoundary").GetString(), StringComparison.OrdinalIgnoreCase);

        using JsonDocument validationDocument = ReadFinalReleaseJson("runtime-proof-execution-input-record-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("runtime-proof-execution-input-record-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-runtime-proof-execution-input-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(6, validation.GetProperty("executionInputCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("blockedExecutionInputCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("readyExecutionInputCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() >= 6);
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    [Fact]
    public void ReleaseEvidenceIncludesRuntimeProofInputRecord()
    {
        RunRuntimeProofInputPipeline();
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-runtime-proof-execution-input-required", evidence.GetProperty("runtimeProofExecutionInputRecordState").GetString());
        Assert.Equal("blocked-runtime-proof-execution-input-required", evidence.GetProperty("runtimeProofExecutionInputRecordValidationState").GetString());
        Assert.Equal(6, evidence.GetProperty("runtimeProofExecutionInputRecordExecutionInputCount").GetInt32());
        Assert.Equal(6, evidence.GetProperty("runtimeProofExecutionInputRecordBlockedExecutionInputCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("runtimeProofExecutionInputRecordReadyExecutionInputCount").GetInt32());
        Assert.False(evidence.GetProperty("runtimeProofExecutionInputRecordCanPromoteRuntimeProof").GetBoolean());
        Assert.False(evidence.GetProperty("runtimeProofExecutionInputRecordCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("runtimeProofExecutionInputRecordCanCloseReleaseIssue").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "runtime-proof-execution-input-record");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("owner-fill input surface only", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("artifacts/final-release/runtime-proof-execution-input-record.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/runtime-proof-execution-input-record-validation.json", sourceArtifacts);
    }

    internal static void RunRuntimeProofInputPipeline()
    {
        RealProofRecordValidatorTests.RunRecordValidatorPipeline();
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerRealProofExecutionClosurePack.ps1"));
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerRealProofExecutionClosurePack.ps1"), "-Strict");
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RuntimeProofExecutionInputRecord.ps1"));
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-RuntimeProofExecutionInputRecord.ps1"), "-Strict");
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

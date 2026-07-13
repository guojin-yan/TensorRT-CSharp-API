using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerRuntimeProofExecutionRunbookTests
{
    [Fact]
    public void OwnerRuntimeProofRunbookExportsBlockedCommandSequences()
    {
        RunOwnerRuntimeProofRunbookPipeline();

        using JsonDocument runbookDocument = ReadFinalReleaseJson("owner-runtime-proof-execution-runbook.json");
        JsonElement runbook = runbookDocument.RootElement;

        Assert.Equal("owner-runtime-proof-execution-runbook", runbook.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-runtime-proof-execution-required", runbook.GetProperty("runbookState").GetString());
        Assert.Equal(6, runbook.GetProperty("runbookItemCount").GetInt32());
        Assert.Equal(6, runbook.GetProperty("blockedRunbookItemCount").GetInt32());
        Assert.Equal(0, runbook.GetProperty("readyRunbookItemCount").GetInt32());
        Assert.False(runbook.GetProperty("performsPublish").GetBoolean());
        Assert.False(runbook.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(runbook.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(runbook.GetProperty("canCloseReleaseIssue").GetBoolean());

        JsonElement firstItem = runbook.GetProperty("runbookItems").EnumerateArray().First();
        Assert.Equal("blocked-owner-runtime-proof-execution-required", firstItem.GetProperty("runbookState").GetString());
        Assert.True(firstItem.GetProperty("commandSequence").GetArrayLength() >= 4);
        Assert.True(firstItem.GetProperty("requiredHashes").GetArrayLength() >= 5);
        Assert.True(firstItem.GetProperty("validatorCommands").GetArrayLength() > 0);
        Assert.Contains("not runtime proof", firstItem.GetProperty("nonSubstituteBoundary").GetString(), StringComparison.OrdinalIgnoreCase);

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-runtime-proof-execution-runbook-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("owner-runtime-proof-execution-runbook-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-runtime-proof-execution-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(6, validation.GetProperty("runbookItemCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("blockedRunbookItemCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("readyRunbookItemCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("failedActionRequiredCount").GetInt32());
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    internal static void RunOwnerRuntimeProofRunbookPipeline()
    {
        RuntimeProofExecutionInputRecordTests.RunRuntimeProofInputPipeline();
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerRuntimeProofExecutionRunbook.ps1"));
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerRuntimeProofExecutionRunbook.ps1"), "-Strict");
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

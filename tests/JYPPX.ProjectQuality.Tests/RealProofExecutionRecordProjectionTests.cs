using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class RealProofExecutionRecordProjectionTests
{
    [Fact]
    public void ProjectionExportsBlockedRecordsWithoutPromotingProof()
    {
        RunProjectionPipeline();

        using JsonDocument projectionDocument = ReadFinalReleaseJson("real-proof-execution-record-projection.json");
        JsonElement projection = projectionDocument.RootElement;

        Assert.Equal("real-proof-execution-record-projection", projection.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-proof-execution-record-input-required", projection.GetProperty("projectionState").GetString());
        Assert.Equal(6, projection.GetProperty("recordCount").GetInt32());
        Assert.Equal(6, projection.GetProperty("blockedRecordCount").GetInt32());
        Assert.Equal(0, projection.GetProperty("readyRecordCount").GetInt32());
        Assert.False(projection.GetProperty("performsPublish").GetBoolean());
        Assert.False(projection.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(projection.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(projection.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(projection.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(projection.GetProperty("isReleaseCloseProof").GetBoolean());

        JsonElement firstRecord = projection.GetProperty("proofExecutionRecords").EnumerateArray().First();
        Assert.True(firstRecord.TryGetProperty("hostMetadata", out _));
        Assert.True(firstRecord.TryGetProperty("execution", out JsonElement execution));
        Assert.True(execution.GetProperty("commands").GetArrayLength() >= 1);
        Assert.True(firstRecord.GetProperty("logs").GetArrayLength() >= 1);
        Assert.True(firstRecord.GetProperty("hashes").GetArrayLength() >= 1);
        Assert.True(firstRecord.GetProperty("validatorOutputs").GetArrayLength() >= 1);
        Assert.True(firstRecord.GetProperty("forbiddenSubstituteChecks").GetArrayLength() >= 1);
        Assert.False(firstRecord.GetProperty("promotionFlags").GetProperty("canPromoteRuntimeProof").GetBoolean());

        using JsonDocument validationDocument = ReadFinalReleaseJson("real-proof-execution-record-projection-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("real-proof-execution-record-projection-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-proof-execution-record-input-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(6, validation.GetProperty("recordCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("blockedRecordCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("readyRecordCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validation.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(validation.GetProperty("isReleaseCloseProof").GetBoolean());
        Assert.Contains(validation.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString()!.EndsWith("-hostOs-input-required", StringComparison.Ordinal) &&
            item.GetProperty("severity").GetString() == "action-required" &&
            item.GetProperty("passed").GetBoolean() == false);
    }

    [Fact]
    public void ReleaseEvidenceAndDocsIncludeProjection()
    {
        RunProjectionPipeline();
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-real-proof-execution-record-input-required", evidence.GetProperty("realProofExecutionRecordProjectionState").GetString());
        Assert.Equal("blocked-real-proof-execution-record-input-required", evidence.GetProperty("realProofExecutionRecordProjectionValidationState").GetString());
        Assert.Equal(6, evidence.GetProperty("realProofExecutionRecordProjectionRecordCount").GetInt32());
        Assert.Equal(6, evidence.GetProperty("realProofExecutionRecordProjectionBlockedRecordCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("realProofExecutionRecordProjectionReadyRecordCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("realProofExecutionRecordProjectionFailedBlockerCount").GetInt32());
        Assert.True(evidence.GetProperty("realProofExecutionRecordProjectionFailedActionRequiredCount").GetInt32() >= 1);
        Assert.False(evidence.GetProperty("realProofExecutionRecordProjectionCanPromoteRuntimeProof").GetBoolean());
        Assert.False(evidence.GetProperty("realProofExecutionRecordProjectionCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("realProofExecutionRecordProjectionCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("realProofExecutionRecordProjectionIsRuntimeExecutionProof").GetBoolean());
        Assert.False(evidence.GetProperty("realProofExecutionRecordProjectionIsReleaseCloseProof").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "real-proof-execution-record-projection");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("record-shape", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("artifacts/final-release/real-proof-execution-record-projection.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-proof-execution-record-projection.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-proof-execution-record-projection-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-proof-execution-record-projection-validation.md", sourceArtifacts);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string releaseEvidenceDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));
        string projectionDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "real-proof-execution-record-projection.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("real-proof-execution-record-projection.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("real-proof-execution-record-projection.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("Export-RealProofExecutionRecordProjection.ps1", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("Test-RealProofExecutionRecordProjection.ps1 -Strict", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("real-proof-execution-record-projection", projectionDoc, StringComparison.Ordinal);
        Assert.Contains("real proof execution record projection", evidenceMarkdown, StringComparison.OrdinalIgnoreCase);
    }

    private static void RunProjectionPipeline()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseProofWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealExternalProofBackfillExecutionBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealProofRunnerInputBackfill.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-RealProofRunnerInputBackfill.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealProofExecutionRecordProjection.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-RealProofExecutionRecordProjection.ps1"), "-Strict");
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
    {
        using Process process = new();
        process.StartInfo.FileName = "pwsh";
        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(scriptPath);
        foreach (string argument in arguments)
        {
            process.StartInfo.ArgumentList.Add(argument);
        }

        process.StartInfo.WorkingDirectory = RepositoryPaths.Root;
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.RedirectStandardError = true;
        process.StartInfo.UseShellExecute = false;

        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptPath}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        return stdout;
    }
}

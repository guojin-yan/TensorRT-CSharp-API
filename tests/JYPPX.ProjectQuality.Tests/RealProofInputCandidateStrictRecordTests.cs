using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class RealProofInputCandidateStrictRecordTests
{
    [Fact]
    public void StrictRecordExportsBlockedCandidatesWithoutPromotingProof()
    {
        RunStrictRecordPipeline();

        using JsonDocument recordDocument = ReadFinalReleaseJson("real-proof-input-candidate-strict-record.json");
        JsonElement record = recordDocument.RootElement;

        Assert.Equal("real-proof-input-candidate-strict-record", record.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-proof-input-candidate-required", record.GetProperty("candidateState").GetString());
        Assert.Equal(6, record.GetProperty("candidateCount").GetInt32());
        Assert.Equal(6, record.GetProperty("blockedCandidateCount").GetInt32());
        Assert.Equal(0, record.GetProperty("readyCandidateCount").GetInt32());
        Assert.False(record.GetProperty("performsPublish").GetBoolean());
        Assert.False(record.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(record.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(record.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(record.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(record.GetProperty("isReleaseCloseProof").GetBoolean());

        string[] lanes = record.GetProperty("strictCandidateRecords").EnumerateArray()
            .Select(static item => item.GetProperty("proofLane").GetString()!)
            .ToArray();
        Assert.Contains("package-consumer-runtime", lanes);
        Assert.Contains("post-publish-verification", lanes);
        Assert.Contains("linux-runner-proof", lanes);
        Assert.Contains("real-model-runtime", lanes);
        Assert.Contains("release-close-owner-input", lanes);
        Assert.Contains("strict-close-validation", lanes);

        JsonElement firstCandidate = record.GetProperty("strictCandidateRecords").EnumerateArray().First();
        Assert.Equal("blocked-real-proof-input-candidate-required", firstCandidate.GetProperty("candidateState").GetString());
        Assert.True(firstCandidate.GetProperty("fieldContracts").GetArrayLength() >= 7);
        Assert.True(firstCandidate.GetProperty("blockedFieldCount").GetInt32() >= 1);
        Assert.False(firstCandidate.GetProperty("readyForOwnerReview").GetBoolean());
        Assert.False(firstCandidate.GetProperty("readyForPromotion").GetBoolean());

        using JsonDocument validationDocument = ReadFinalReleaseJson("real-proof-input-candidate-strict-record-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("real-proof-input-candidate-strict-record-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-proof-input-candidate-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(6, validation.GetProperty("candidateCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("blockedCandidateCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("readyCandidateCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validation.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(validation.GetProperty("isReleaseCloseProof").GetBoolean());
    }

    [Fact]
    public void ReleaseEvidenceAndDocsIncludeStrictRecord()
    {
        RunStrictRecordPipeline();
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-real-proof-input-candidate-required", evidence.GetProperty("realProofInputCandidateStrictRecordState").GetString());
        Assert.Equal("blocked-real-proof-input-candidate-required", evidence.GetProperty("realProofInputCandidateStrictRecordValidationState").GetString());
        Assert.Equal(6, evidence.GetProperty("realProofInputCandidateStrictRecordCandidateCount").GetInt32());
        Assert.Equal(6, evidence.GetProperty("realProofInputCandidateStrictRecordBlockedCandidateCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("realProofInputCandidateStrictRecordReadyCandidateCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("realProofInputCandidateStrictRecordFailedBlockerCount").GetInt32());
        Assert.True(evidence.GetProperty("realProofInputCandidateStrictRecordFailedActionRequiredCount").GetInt32() >= 1);
        Assert.False(evidence.GetProperty("realProofInputCandidateStrictRecordCanPromoteRuntimeProof").GetBoolean());
        Assert.False(evidence.GetProperty("realProofInputCandidateStrictRecordCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("realProofInputCandidateStrictRecordCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("realProofInputCandidateStrictRecordIsRuntimeExecutionProof").GetBoolean());
        Assert.False(evidence.GetProperty("realProofInputCandidateStrictRecordIsReleaseCloseProof").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "real-proof-input-candidate-strict-record");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("candidate contracts only", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("artifacts/final-release/real-proof-input-candidate-strict-record.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-proof-input-candidate-strict-record.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-proof-input-candidate-strict-record-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-proof-input-candidate-strict-record-validation.md", sourceArtifacts);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string releaseEvidenceDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));
        string strictRecordDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "real-proof-input-candidate-strict-record.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("real-proof-input-candidate-strict-record.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("real-proof-input-candidate-strict-record.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("Export-RealProofInputCandidateStrictRecord.ps1", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("Test-RealProofInputCandidateStrictRecord.ps1 -Strict", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("real-proof-input-candidate-strict-record", strictRecordDoc, StringComparison.Ordinal);
        Assert.Contains("real proof input candidate strict record", evidenceMarkdown, StringComparison.OrdinalIgnoreCase);
    }

    internal static void RunStrictRecordPipelineForReuse()
    {
        RunStrictRecordPipeline();
    }

    private static void RunStrictRecordPipeline()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseProofWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealExternalProofBackfillExecutionBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealProofRunnerInputBackfill.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-RealProofRunnerInputBackfill.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealProofExecutionRecordProjection.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-RealProofExecutionRecordProjection.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerRealProofReportPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerRealProofReportPack.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealProofInputCandidateStrictRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-RealProofInputCandidateStrictRecord.ps1"), "-Strict");
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

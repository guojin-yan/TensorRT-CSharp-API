using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerRealProofFieldDeltaPackTests
{
    [Fact]
    public void DeltaPackExportsBlockedOwnerActionsWithoutPromotingProof()
    {
        RunDeltaPackPipeline();

        using JsonDocument packDocument = ReadFinalReleaseJson("owner-real-proof-field-delta-pack.json");
        JsonElement pack = packDocument.RootElement;

        Assert.Equal("owner-real-proof-field-delta-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-proof-field-delta-required", pack.GetProperty("deltaState").GetString());
        Assert.Equal(6, pack.GetProperty("candidateCount").GetInt32());
        Assert.True(pack.GetProperty("fieldDeltaCount").GetInt32() >= 6);
        Assert.True(pack.GetProperty("blockedFieldContractCount").GetInt32() >= 6);
        Assert.Equal(0, pack.GetProperty("readyFieldContractCount").GetInt32());
        Assert.False(pack.GetProperty("performsPublish").GetBoolean());
        Assert.False(pack.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(pack.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(pack.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(pack.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(pack.GetProperty("isReleaseCloseProof").GetBoolean());

        JsonElement firstDelta = pack.GetProperty("fieldDeltas").EnumerateArray().First();
        Assert.False(firstDelta.GetProperty("ready").GetBoolean());
        Assert.Equal("blocked-owner-real-proof-field-delta-required", firstDelta.GetProperty("deltaState").GetString());
        Assert.False(string.IsNullOrWhiteSpace(firstDelta.GetProperty("candidateId").GetString()));
        Assert.False(string.IsNullOrWhiteSpace(firstDelta.GetProperty("fieldName").GetString()));
        Assert.Contains("Test-RealProofInputCandidateStrictRecord.ps1", firstDelta.GetProperty("targetValidationCommand").GetString(), StringComparison.Ordinal);
        Assert.Contains("cannot satisfy", firstDelta.GetProperty("nonSubstituteBoundary").GetString(), StringComparison.OrdinalIgnoreCase);

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-real-proof-field-delta-pack-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("owner-real-proof-field-delta-pack-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-proof-field-delta-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(6, validation.GetProperty("candidateCount").GetInt32());
        Assert.True(validation.GetProperty("fieldDeltaCount").GetInt32() >= 6);
        Assert.True(validation.GetProperty("blockedFieldContractCount").GetInt32() >= 6);
        Assert.Equal(0, validation.GetProperty("readyFieldContractCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validation.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(validation.GetProperty("isReleaseCloseProof").GetBoolean());
    }

    [Fact]
    public void ReleaseEvidenceAndDocsIncludeDeltaPack()
    {
        RunDeltaPackPipeline();
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-owner-real-proof-field-delta-required", evidence.GetProperty("ownerRealProofFieldDeltaPackState").GetString());
        Assert.Equal("blocked-owner-real-proof-field-delta-required", evidence.GetProperty("ownerRealProofFieldDeltaPackValidationState").GetString());
        Assert.Equal(6, evidence.GetProperty("ownerRealProofFieldDeltaPackCandidateCount").GetInt32());
        Assert.True(evidence.GetProperty("ownerRealProofFieldDeltaPackFieldDeltaCount").GetInt32() >= 6);
        Assert.True(evidence.GetProperty("ownerRealProofFieldDeltaPackBlockedFieldContractCount").GetInt32() >= 6);
        Assert.Equal(0, evidence.GetProperty("ownerRealProofFieldDeltaPackReadyFieldContractCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("ownerRealProofFieldDeltaPackFailedBlockerCount").GetInt32());
        Assert.True(evidence.GetProperty("ownerRealProofFieldDeltaPackFailedActionRequiredCount").GetInt32() >= 1);
        Assert.False(evidence.GetProperty("ownerRealProofFieldDeltaPackCanPromoteRuntimeProof").GetBoolean());
        Assert.False(evidence.GetProperty("ownerRealProofFieldDeltaPackCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("ownerRealProofFieldDeltaPackCanCloseReleaseIssue").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "owner-real-proof-field-delta-pack");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("Owner action list only", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("artifacts/final-release/owner-real-proof-field-delta-pack.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-real-proof-field-delta-pack.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-real-proof-field-delta-pack-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-real-proof-field-delta-pack-validation.md", sourceArtifacts);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string releaseEvidenceDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "owner-real-proof-field-delta-pack.md"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("owner-real-proof-field-delta-pack.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("owner-real-proof-field-delta-pack.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("Export-OwnerRealProofFieldDeltaPack.ps1", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("Test-OwnerRealProofFieldDeltaPack.ps1 -Strict", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("owner-real-proof-field-delta-pack", article, StringComparison.Ordinal);
        Assert.Contains("owner-real-proof-field-delta-pack", readme, StringComparison.Ordinal);
        Assert.Contains("owner-real-proof-field-delta-pack", readmeZh, StringComparison.Ordinal);
        Assert.Contains("owner real proof field delta pack", evidenceMarkdown, StringComparison.OrdinalIgnoreCase);
    }

    internal static void RunDeltaPackPipeline()
    {
        RealProofInputCandidateStrictRecordTests.RunStrictRecordPipelineForReuse();
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerRealProofFieldDeltaPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerRealProofFieldDeltaPack.ps1"), "-Strict");
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    internal static string RunPowerShell(string scriptPath, params string[] arguments)
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

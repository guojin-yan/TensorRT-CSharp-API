using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class RealProofRunnerInputBackfillTests
{
    [Fact]
    public void RunnerInputBackfillExportsBlockedTemplateWithoutPromotingProof()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseProofWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealExternalProofBackfillExecutionBundle.ps1"));

        string exportOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealProofRunnerInputBackfill.ps1"));
        string validationOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-RealProofRunnerInputBackfill.ps1"), "-Strict");

        Assert.Contains("Real proof runner input backfill template written", exportOutput, StringComparison.Ordinal);
        Assert.Contains("ValidationState=blocked-owner-runner-input-required", validationOutput, StringComparison.Ordinal);

        using JsonDocument templateDocument = ReadFinalReleaseJson("real-proof-runner-input-backfill.template.json");
        JsonElement template = templateDocument.RootElement;

        Assert.Equal("real-proof-runner-input-backfill", template.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-runner-input-required", template.GetProperty("inputState").GetString());
        Assert.Equal(6, template.GetProperty("trackCount").GetInt32());
        Assert.Equal(6, template.GetProperty("blockedTrackCount").GetInt32());
        Assert.False(template.GetProperty("performsPublish").GetBoolean());
        Assert.False(template.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(template.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(template.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(template.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(template.GetProperty("isReleaseCloseProof").GetBoolean());

        string[] forbiddenSubstitutes = template.GetProperty("forbiddenSubstitutes")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("local feed", forbiddenSubstitutes);
        Assert.Contains("ProjectReference", forbiddenSubstitutes);
        Assert.Contains("direct .nupkg", forbiddenSubstitutes);
        Assert.Contains("DependencyProbe", forbiddenSubstitutes);
        Assert.Contains("build-only", forbiddenSubstitutes);
        Assert.Contains("template", forbiddenSubstitutes);
        Assert.Contains("Windows handoff for Linux proof", forbiddenSubstitutes);
        Assert.Contains("hash-only audit", forbiddenSubstitutes);

        using JsonDocument validationDocument = ReadFinalReleaseJson("real-proof-runner-input-backfill-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("real-proof-runner-input-backfill-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-runner-input-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(6, validation.GetProperty("trackCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validation.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(validation.GetProperty("isReleaseCloseProof").GetBoolean());

        Assert.Contains(validation.GetProperty("validationItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString()!.EndsWith("-host-input-required", StringComparison.Ordinal) &&
            item.GetProperty("severity").GetString() == "action-required" &&
            item.GetProperty("passed").GetBoolean() == false);
    }

    [Fact]
    public void ReleaseEvidenceAndDocsIncludeRunnerInputBackfill()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseProofWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealExternalProofBackfillExecutionBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealProofRunnerInputBackfill.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-RealProofRunnerInputBackfill.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-owner-runner-input-required", evidence.GetProperty("realProofRunnerInputBackfillState").GetString());
        Assert.Equal("blocked-owner-runner-input-required", evidence.GetProperty("realProofRunnerInputBackfillValidationState").GetString());
        Assert.Equal(6, evidence.GetProperty("realProofRunnerInputBackfillTrackCount").GetInt32());
        Assert.Equal(6, evidence.GetProperty("realProofRunnerInputBackfillBlockedTrackCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("realProofRunnerInputBackfillFailedBlockerCount").GetInt32());
        Assert.True(evidence.GetProperty("realProofRunnerInputBackfillFailedActionRequiredCount").GetInt32() >= 1);
        Assert.False(evidence.GetProperty("realProofRunnerInputBackfillCanPromoteRuntimeProof").GetBoolean());
        Assert.False(evidence.GetProperty("realProofRunnerInputBackfillCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("realProofRunnerInputBackfillCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("realProofRunnerInputBackfillIsRuntimeExecutionProof").GetBoolean());
        Assert.False(evidence.GetProperty("realProofRunnerInputBackfillIsReleaseCloseProof").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "real-proof-runner-input-backfill");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("template", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("artifacts/final-release/real-proof-runner-input-backfill.template.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-proof-runner-input-backfill.template.md", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-proof-runner-input-backfill-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-proof-runner-input-backfill-validation.md", sourceArtifacts);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string releaseEvidenceDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));
        string runnerInputDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "real-proof-runner-input-backfill.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("real-proof-runner-input-backfill.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("real-proof-runner-input-backfill.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("Export-RealProofRunnerInputBackfill.ps1", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("Test-RealProofRunnerInputBackfill.ps1 -Strict", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("failedActionRequiredCount>=1", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("real-proof-runner-input-backfill", runnerInputDoc, StringComparison.Ordinal);
        Assert.Contains("real proof runner input backfill", evidenceMarkdown, StringComparison.OrdinalIgnoreCase);
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

using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PackageConsumerRuntimeProofWorklistTests
{
    [Fact]
    public void WorklistAggregatesOwnerInputCandidateRecordAndScaffoldWithoutPromotingProof()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PackageConsumerRuntimeProofOwnerInput.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofCandidate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PackageConsumerRuntimeProofCandidate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofRecordFromOwnerInput.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PackageConsumerRuntimeProofRecord.ps1"));

        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofWorklist.ps1"));
        Assert.Contains("Package consumer runtime proof worklist written", output, StringComparison.Ordinal);

        using JsonDocument worklistDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-worklist.json");
        JsonElement worklist = worklistDocument.RootElement;

        Assert.Equal("package-consumer-runtime-proof-worklist", worklist.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-package-consumer-runtime-proof-required", worklist.GetProperty("worklistState").GetString());
        Assert.Equal(4, worklist.GetProperty("workItemCount").GetInt32());
        Assert.True(worklist.GetProperty("blockedWorkItemCount").GetInt32() >= 1);
        Assert.True(worklist.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        Assert.False(worklist.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(worklist.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(worklist.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(worklist.GetProperty("performsPublish").GetBoolean());
        Assert.False(worklist.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.Contains("not proof", worklist.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] itemIds = worklist.GetProperty("workItems").EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();
        Assert.Contains("owner-input-clean-consumer-fields", itemIds);
        Assert.Contains("candidate-clean-runtime-smoke", itemIds);
        Assert.Contains("strict-proof-record", itemIds);
        Assert.Contains("external-smoke-scaffold", itemIds);

        string markdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "package-consumer-runtime-proof-worklist.md"));
        Assert.Contains("## Work Items", markdown, StringComparison.Ordinal);
        Assert.Contains("owner-input-clean-consumer-fields", markdown, StringComparison.Ordinal);
        Assert.Contains("Test-PackageConsumerRuntimeProofRecord.ps1 -FailOnNotProof", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseEvidenceAndDocsIncludePackageConsumerRuntimeProofWorklist()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-real-package-consumer-runtime-proof-required", evidence.GetProperty("packageConsumerRuntimeProofWorklistState").GetString());
        Assert.True(evidence.GetProperty("packageConsumerRuntimeProofWorklistFailedActionRequiredCount").GetInt32() >= 1);
        Assert.False(evidence.GetProperty("packageConsumerRuntimeProofWorklistCanPromoteRuntimeProof").GetBoolean());
        Assert.False(evidence.GetProperty("packageConsumerRuntimeProofWorklistIsRuntimeExecutionProof").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "package-consumer-runtime-proof-worklist");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("not proof", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-worklist.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-worklist.md", sourceArtifacts);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string releaseEvidenceDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));
        string worklistDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "package-consumer-runtime-proof-worklist.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("package-consumer-runtime-proof-worklist.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime-proof-worklist.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("Export-PackageConsumerRuntimeProofWorklist.ps1", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime-proof-worklist", worklistDoc, StringComparison.Ordinal);
        Assert.Contains("package consumer runtime proof worklist", evidenceMarkdown, StringComparison.OrdinalIgnoreCase);
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

        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException(
                $"PowerShell script failed with exit code {process.ExitCode}:{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        }

        return stdout;
    }
}

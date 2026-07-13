using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PackageConsumerRuntimeProofCandidateTests
{
    [Fact]
    public void PackageConsumerRuntimeProofCandidateExportsBlockedOwnerInputSurface()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofBackfillExecutionPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofExecutionHandoff.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerExternalProofInputPreflight.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofInputRepairPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofInputDraftPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerExternalProofBackfillOrchestrator.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofCandidate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PackageConsumerRuntimeProofCandidate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument candidateDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-candidate.json");
        JsonElement candidate = candidateDocument.RootElement;

        Assert.Equal("package-consumer-runtime-proof-candidate", candidate.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-package-consumer-smoke-required", candidate.GetProperty("candidateState").GetString());
        Assert.Equal("package-consumer-runtime", candidate.GetProperty("proofLineId").GetString());
        Assert.False(candidate.GetProperty("canPromoteProof").GetBoolean());
        Assert.False(candidate.GetProperty("performsPublish").GetBoolean());
        Assert.False(candidate.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(candidate.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.True(candidate.GetProperty("smokeCommandIncludesRuntimePackageKey").GetBoolean());
        Assert.Contains("--runtime-package-key", candidate.GetProperty("smokeCommand").GetString(), StringComparison.Ordinal);

        string[] rules = candidate.GetProperty("requiredRealInputRules").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string rule in new[]
        {
            "cleanExternalConsumerIdentity",
            "noProjectReference",
            "noLocalFeedAsPublicProof",
            "managedNupkgSha256",
            "runtimeNupkgSha256",
            "runtimePackageKeyMatches",
            "compatibleHostMetadata",
            "smokeCommandIncludesRuntimePackageKey",
            "smokeLogPath",
            "smokeLogSha256",
        })
        {
            Assert.Contains(rule, rules);
        }

        Assert.True(candidate.GetProperty("consumerProjectUsesProjectReference").GetBoolean());
        Assert.True(candidate.GetProperty("consumerProjectUsesLocalFeed").GetBoolean());
        Assert.True(candidate.GetProperty("consumerProjectUsesDirectNupkg").GetBoolean());
        Assert.True(candidate.GetProperty("publicPackageSourceIsLocal").GetBoolean());

        using JsonDocument validationDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-candidate-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("package-consumer-runtime-proof-candidate-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-package-consumer-smoke-required", validation.GetProperty("validationState").GetString());
        Assert.True(validation.GetProperty("isValidCandidate").GetBoolean());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        Assert.False(validation.GetProperty("canPromoteProof").GetBoolean());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("blocked-real-package-consumer-smoke-required", evidence.GetProperty("packageConsumerRuntimeProofCandidateState").GetString());
        Assert.Equal("blocked-real-package-consumer-smoke-required", evidence.GetProperty("packageConsumerRuntimeProofCandidateValidationState").GetString());
        Assert.False(evidence.GetProperty("packageConsumerRuntimeProofCandidateCanPromoteProof").GetBoolean());
        Assert.False(evidence.GetProperty("packageConsumerRuntimeProofCandidateCleanExternalConsumerReady").GetBoolean());
        Assert.False(evidence.GetProperty("packageConsumerRuntimeProofCandidateNupkgSha256Ready").GetBoolean());
        Assert.False(evidence.GetProperty("packageConsumerRuntimeProofCandidateSmokeLogReady").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "package-consumer-runtime-proof-candidate");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("owner input surface only", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-candidate.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-candidate-validation.json", sourceArtifacts);

        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "package-consumer-runtime-proof-candidate.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("package-consumer-runtime-proof-candidate", readme, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime-proof-candidate", readmeZh, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/package-consumer-runtime-proof-candidate.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/package-consumer-runtime-proof-candidate.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("candidateState=blocked-real-package-consumer-smoke-required", article, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("package consumer runtime proof candidate: `blocked-real-package-consumer-smoke-required`", evidenceMarkdown, StringComparison.Ordinal);
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

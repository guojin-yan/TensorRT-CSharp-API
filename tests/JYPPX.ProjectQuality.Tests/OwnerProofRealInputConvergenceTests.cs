using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerProofRealInputConvergenceTests
{
    [Fact]
    public void OwnerProofRealInputConvergenceStaysBlockedAndAuditable()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerReleaseExecutionPackage.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerReleaseExecutionPackage.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseIssueFinalCloseDecisionTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseIssueFinalCloseDecision.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalEvidenceFreeze.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalEvidenceFreeze.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealExternalProofOverlayPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-RealExternalProofOverlayPack.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseIssueCloseRecordOverlayCandidate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseIssueCloseRecordOverlayCandidate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerExternalExecutionResultBackfillKit.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerExternalExecutionResultBackfillKit.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerInputCrossHashAudit.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerInputCrossHashAudit.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseStrictRecordCandidate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseCloseStrictRecordCandidate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofRealBackfillExecutionPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerProofRealBackfillExecutionPack.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseIssueCloseRecordRealInputMap.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseIssueCloseRecordRealInputMap.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofRealInputConvergence.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerProofRealInputConvergence.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument convergenceDocument = ReadFinalReleaseJson("owner-proof-real-input-convergence.json");
        JsonElement convergence = convergenceDocument.RootElement;
        Assert.Equal("owner-proof-real-input-convergence", convergence.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-input-convergence-required", convergence.GetProperty("convergenceState").GetString());
        Assert.True(convergence.GetProperty("inputMappingCount").GetInt32() >= 8);
        Assert.True(convergence.GetProperty("missingRealInputCount").GetInt32() >= 1);
        Assert.True(convergence.GetProperty("blockedValidatorCount").GetInt32() >= 1);
        Assert.True(convergence.GetProperty("blockedProofCount").GetInt32() >= 1);
        Assert.False(convergence.GetProperty("performsPublish").GetBoolean());
        Assert.False(convergence.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(convergence.GetProperty("canCloseReleaseIssue").GetBoolean());

        string sequence = string.Join('\n', convergence.GetProperty("recommendedOwnerSequence").EnumerateArray().Select(static item => item.GetString()));
        Assert.Contains("post-publish", sequence, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("clean consumer runtime smoke", sequence, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("final close decision", sequence, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("strict close validator", sequence, StringComparison.OrdinalIgnoreCase);

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-proof-real-input-convergence-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("owner-proof-real-input-convergence-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-input-convergence-required", validation.GetProperty("validationState").GetString());
        Assert.True(validation.GetProperty("isValidConvergenceShape").GetBoolean());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        Assert.True(validation.GetProperty("missingRealInputCount").GetInt32() >= 1);
        Assert.True(validation.GetProperty("blockedValidatorCount").GetInt32() >= 1);
        Assert.True(validation.GetProperty("blockedProofCount").GetInt32() >= 1);
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("blocked-owner-real-input-convergence-required", evidence.GetProperty("ownerProofRealInputConvergenceState").GetString());
        Assert.Equal("blocked-owner-real-input-convergence-required", evidence.GetProperty("ownerProofRealInputConvergenceValidationState").GetString());
        Assert.Equal(0, evidence.GetProperty("ownerProofRealInputConvergenceFailedBlockerCount").GetInt32());
        Assert.True(evidence.GetProperty("ownerProofRealInputConvergenceFailedActionRequiredCount").GetInt32() >= 1);
        Assert.True(evidence.GetProperty("ownerProofRealInputConvergenceMissingRealInputCount").GetInt32() >= 1);
        Assert.True(evidence.GetProperty("ownerProofRealInputConvergenceBlockedValidatorCount").GetInt32() >= 1);
        Assert.True(evidence.GetProperty("ownerProofRealInputConvergenceBlockedProofCount").GetInt32() >= 1);
        Assert.False(evidence.GetProperty("ownerProofRealInputConvergenceCanCloseReleaseIssue").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "owner-proof-real-input-convergence");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("validation matrix only", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/owner-proof-real-input-convergence.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-proof-real-input-convergence-validation.json", sourceArtifacts);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "owner-proof-real-input-convergence.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("articles/zh-cn/owner-proof-real-input-convergence.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/owner-proof-real-input-convergence.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("owner-proof-real-input-convergence", readme, StringComparison.Ordinal);
        Assert.Contains("owner-proof-real-input-convergence", readmeZh, StringComparison.Ordinal);
        Assert.Contains("convergenceState=blocked-owner-real-input-convergence-required", article, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("owner proof real input convergence validation: `blocked-owner-real-input-convergence-required`", evidenceMarkdown, StringComparison.Ordinal);
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

using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerProofRealBackfillExecutionPackTests
{
    [Fact]
    public void OwnerProofRealBackfillExecutionPackStaysBlockedAndAuditable()
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
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument packDocument = ReadFinalReleaseJson("owner-proof-real-backfill-execution-pack.json");
        JsonElement pack = packDocument.RootElement;
        Assert.Equal("owner-proof-real-backfill-execution-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-proof-backfill-required", pack.GetProperty("packState").GetString());
        Assert.True(pack.GetProperty("ownerInputTaskCount").GetInt32() >= 8);
        Assert.True(pack.GetProperty("realProofTaskCount").GetInt32() >= 5);
        Assert.True(pack.GetProperty("hashCheckTaskCount").GetInt32() >= 9);
        Assert.True(pack.GetProperty("blockedTaskCount").GetInt32() >= 1);
        Assert.False(pack.GetProperty("performsPublish").GetBoolean());
        Assert.False(pack.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(pack.GetProperty("canCloseReleaseIssue").GetBoolean());

        string[] nonSubstitutes = pack.GetProperty("nonSubstituteProofKinds").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("local feed", nonSubstitutes);
        Assert.Contains("ProjectReference", nonSubstitutes);
        Assert.Contains("direct .nupkg", nonSubstitutes);
        Assert.Contains("blocked-by-cuda-driver", nonSubstitutes);

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-proof-real-backfill-execution-pack-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("owner-proof-real-backfill-execution-pack-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-proof-backfill-required", validation.GetProperty("validationState").GetString());
        Assert.True(validation.GetProperty("isValidExecutionPackShape").GetBoolean());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        Assert.True(validation.GetProperty("blockedTaskCount").GetInt32() >= 1);
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("blocked-owner-real-proof-backfill-required", evidence.GetProperty("ownerProofRealBackfillExecutionPackState").GetString());
        Assert.Equal("blocked-owner-real-proof-backfill-required", evidence.GetProperty("ownerProofRealBackfillExecutionPackValidationState").GetString());
        Assert.Equal(0, evidence.GetProperty("ownerProofRealBackfillExecutionPackFailedBlockerCount").GetInt32());
        Assert.True(evidence.GetProperty("ownerProofRealBackfillExecutionPackFailedActionRequiredCount").GetInt32() >= 1);
        Assert.True(evidence.GetProperty("ownerProofRealBackfillExecutionPackBlockedTaskCount").GetInt32() >= 1);
        Assert.False(evidence.GetProperty("ownerProofRealBackfillExecutionPackCanCloseReleaseIssue").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "owner-proof-real-backfill-execution-pack");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("owner handoff only", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/owner-proof-real-backfill-execution-pack.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-proof-real-backfill-execution-pack-validation.json", sourceArtifacts);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "owner-proof-real-backfill-execution-pack.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("articles/zh-cn/owner-proof-real-backfill-execution-pack.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/owner-proof-real-backfill-execution-pack.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("owner-proof-real-backfill-execution-pack", readme, StringComparison.Ordinal);
        Assert.Contains("owner-proof-real-backfill-execution-pack", readmeZh, StringComparison.Ordinal);
        Assert.Contains("packState=blocked-owner-real-proof-backfill-required", article, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("owner proof real backfill execution pack validation: `blocked-owner-real-proof-backfill-required`", evidenceMarkdown, StringComparison.Ordinal);
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

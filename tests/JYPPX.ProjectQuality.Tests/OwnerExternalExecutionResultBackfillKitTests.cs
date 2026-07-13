using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerExternalExecutionResultBackfillKitTests
{
    [Fact]
    public void OwnerExternalExecutionResultBackfillKitStaysBlockedAndAuditable()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerExternalExecutionResultBackfillKit.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerExternalExecutionResultBackfillKit.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument kitDocument = ReadFinalReleaseJson("owner-external-execution-result-backfill-kit.json");
        JsonElement kit = kitDocument.RootElement;
        Assert.Equal("owner-external-execution-result-backfill-kit", kit.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-external-execution-results-required", kit.GetProperty("kitState").GetString());
        Assert.True(kit.GetProperty("requiredOwnerExecutionResultCount").GetInt32() >= 4);
        Assert.True(kit.GetProperty("placeholderFieldCount").GetInt32() >= 1);
        Assert.True(kit.GetProperty("missingExternalLogCount").GetInt32() >= 1);
        Assert.True(kit.GetProperty("missingSha256Count").GetInt32() >= 1);
        Assert.False(kit.GetProperty("performsPublish").GetBoolean());
        Assert.False(kit.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(kit.GetProperty("canCloseReleaseIssue").GetBoolean());

        string[] lineIds = kit.GetProperty("backfillLines")
            .EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();
        Assert.Contains("package-consumer-runtime-proof", lineIds);
        Assert.Contains("post-publish-verification", lineIds);
        Assert.Contains("release-issue-close-record-owner-input", lineIds);
        Assert.Contains("release-issue-final-close-decision", lineIds);

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-external-execution-result-backfill-kit-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("owner-external-execution-result-backfill-kit-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-external-execution-results-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("blocked-owner-external-execution-results-required", evidence.GetProperty("ownerExternalExecutionResultBackfillKitState").GetString());
        Assert.Equal("blocked-owner-external-execution-results-required", evidence.GetProperty("ownerExternalExecutionResultBackfillKitValidationState").GetString());
        Assert.Equal(0, evidence.GetProperty("ownerExternalExecutionResultBackfillKitFailedBlockerCount").GetInt32());
        Assert.True(evidence.GetProperty("ownerExternalExecutionResultBackfillKitFailedActionRequiredCount").GetInt32() >= 1);
        Assert.False(evidence.GetProperty("ownerExternalExecutionResultBackfillKitCanCloseReleaseIssue").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "owner-external-execution-result-backfill-kit");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("cannot collect proof by itself", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/owner-external-execution-result-backfill-kit.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-external-execution-result-backfill-kit-validation.json", sourceArtifacts);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "owner-external-execution-result-backfill-kit.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("articles/zh-cn/owner-external-execution-result-backfill-kit.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/owner-external-execution-result-backfill-kit.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("owner-external-execution-result-backfill-kit", readme, StringComparison.Ordinal);
        Assert.Contains("owner-external-execution-result-backfill-kit", readmeZh, StringComparison.Ordinal);
        Assert.Contains("kitState=blocked-owner-external-execution-results-required", article, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("owner external execution result backfill kit validation: `blocked-owner-external-execution-results-required`", evidenceMarkdown, StringComparison.Ordinal);
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

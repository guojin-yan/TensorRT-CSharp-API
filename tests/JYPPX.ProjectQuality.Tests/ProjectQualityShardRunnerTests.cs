using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ProjectQualityShardRunnerTests
{
    [Fact]
    public void InventoryExporterAndShardPreviewRemainBoundedAndNonPublishing()
    {
        string inventoryOutput = RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Export-ProjectQualityTestInventory.ps1"));
        Assert.Contains("ProjectQuality test inventory written.", inventoryOutput, StringComparison.Ordinal);

        string previewOutput = RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Invoke-ProjectQualityTestShards.ps1"),
            "-Shard",
            "G-M,T-Z",
            "-TimeoutSeconds",
            "120",
            "-RunId",
            "project-quality-shard-preview",
            "-PreviewOnly");
        Assert.Contains("ProjectQuality shard summary written.", previewOutput, StringComparison.Ordinal);
        Assert.Contains("RunState=preview Shards=2", previewOutput, StringComparison.Ordinal);

        string inventoryPath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "test-analysis",
            "project-quality-test-inventory.json");
        string summaryPath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "test-analysis",
            "project-quality-shards",
            "latest-summary.json");
        Assert.True(File.Exists(inventoryPath), $"Expected {inventoryPath} to exist.");
        Assert.True(File.Exists(summaryPath), $"Expected {summaryPath} to exist.");

        using JsonDocument inventoryDocument = JsonDocument.Parse(File.ReadAllText(inventoryPath));
        JsonElement inventory = inventoryDocument.RootElement;
        Assert.Equal("project-quality-test-inventory", inventory.GetProperty("recordKind").GetString());
        Assert.True(inventory.GetProperty("testCount").GetInt32() >= 1000);
        Assert.True(inventory.GetProperty("classCount").GetInt32() >= 300);
        Assert.Equal(0, inventory.GetProperty("unassignedClassCount").GetInt32());
        Assert.Equal(64, inventory.GetProperty("listTestsLogSha256").GetString()!.Length);

        JsonElement[] shards = inventory.GetProperty("shards").EnumerateArray().ToArray();
        Assert.Equal(new[] { "A-F", "G-M", "N-S", "T-Z" }, shards.Select(static item => item.GetProperty("id").GetString()).ToArray());
        Assert.All(shards, static item =>
        {
            Assert.True(item.GetProperty("classCount").GetInt32() > 0);
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("filterExpression").GetString()));
        });

        using JsonDocument summaryDocument = JsonDocument.Parse(File.ReadAllText(summaryPath));
        JsonElement summary = summaryDocument.RootElement;
        Assert.Equal("project-quality-test-shard-run-summary", summary.GetProperty("recordKind").GetString());
        Assert.Equal("preview", summary.GetProperty("runState").GetString());
        Assert.True(summary.GetProperty("previewOnly").GetBoolean());
        Assert.Equal(2, summary.GetProperty("shardCount").GetInt32());
        Assert.Equal(2, summary.GetProperty("previewShardCount").GetInt32());
        Assert.Equal(0, summary.GetProperty("totalExecutedTests").GetInt32());
        Assert.False(summary.GetProperty("performsPublish").GetBoolean());
        Assert.False(summary.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(summary.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains("not test passes", summary.GetProperty("boundary").GetString(), StringComparison.Ordinal);

        JsonElement[] previewResults = summary.GetProperty("results").EnumerateArray().ToArray();
        Assert.Equal(new[] { "G-M", "T-Z" }, previewResults.Select(static item => item.GetProperty("id").GetString()).ToArray());
        Assert.All(previewResults, static item =>
        {
            Assert.Equal("preview", item.GetProperty("state").GetString());
            Assert.True(item.GetProperty("classCount").GetInt32() > 0);
            Assert.True(item.GetProperty("filterExpressionLength").GetInt32() > 0);
            Assert.Contains("--filter", item.GetProperty("command").GetString(), StringComparison.Ordinal);
            Assert.False(item.GetProperty("timedOut").GetBoolean());
        });

        string runnerSource = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Invoke-ProjectQualityTestShards.ps1"));
        Assert.Contains("WaitForExitAsync", runnerSource, StringComparison.Ordinal);
        Assert.Contains("Kill($true)", runnerSource, StringComparison.Ordinal);
        Assert.Contains("processTreeCleanup", runnerSource, StringComparison.Ordinal);
        Assert.Contains("logSha256", runnerSource, StringComparison.Ordinal);
        Assert.Contains("trxSha256", runnerSource, StringComparison.Ordinal);
        Assert.Contains("BatchSize", runnerSource, StringComparison.Ordinal);
        Assert.Contains("ClassNamePattern", runnerSource, StringComparison.Ordinal);
        Assert.Contains("classNames", runnerSource, StringComparison.Ordinal);
        Assert.Contains("DOTNET_CLI_USE_MSBUILD_SERVER", runnerSource, StringComparison.Ordinal);
        Assert.Contains("MSBUILDDISABLENODEREUSE", runnerSource, StringComparison.Ordinal);

        string batchPreviewOutput = RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Invoke-ProjectQualityTestShards.ps1"),
            "-Shard",
            "A-F",
            "-TimeoutSeconds",
            "120",
            "-BatchSize",
            "15",
            "-Batch",
            "1",
            "-RunId",
            "project-quality-shard-batch-preview",
            "-PreviewOnly");
        Assert.Contains("RunState=preview Shards=1", batchPreviewOutput, StringComparison.Ordinal);

        string batchSummaryPath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "test-analysis",
            "project-quality-shards",
            "project-quality-shard-batch-preview",
            "summary.json");
        using JsonDocument batchSummaryDocument = JsonDocument.Parse(File.ReadAllText(batchSummaryPath));
        JsonElement batchSummary = batchSummaryDocument.RootElement;
        Assert.Equal(15, batchSummary.GetProperty("batchSize").GetInt32());
        Assert.Equal(1, batchSummary.GetProperty("executionUnitCount").GetInt32());
        JsonElement batchResult = batchSummary.GetProperty("results")[0];
        Assert.Equal("A-F", batchResult.GetProperty("parentShardId").GetString());
        Assert.Equal(1, batchResult.GetProperty("batchNumber").GetInt32());
        Assert.True(batchResult.GetProperty("batchCount").GetInt32() >= 7);
        Assert.Equal(15, batchResult.GetProperty("classCount").GetInt32());
        Assert.Equal(15, batchResult.GetProperty("classNames").GetArrayLength());
        Assert.StartsWith("JYPPX.ProjectQuality.Tests.", batchResult.GetProperty("firstClass").GetString(), StringComparison.Ordinal);
        Assert.StartsWith("JYPPX.ProjectQuality.Tests.", batchResult.GetProperty("lastClass").GetString(), StringComparison.Ordinal);
    }

    [Fact]
    public void ShardCoverageExporterRequiresHashVerifiedPassedTrxForEveryInventoryClass()
    {
        string inventoryOutput = RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Export-ProjectQualityTestInventory.ps1"));
        Assert.Contains("ProjectQuality test inventory written.", inventoryOutput, StringComparison.Ordinal);

        string shardOutput = RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Invoke-ProjectQualityTestShards.ps1"),
            "-Shard",
            "G-M,T-Z",
            "-TimeoutSeconds",
            "120",
            "-RunId",
            "project-quality-shard-coverage-self-check",
            "-ContinueOnFailure");
        Assert.Contains("ProjectQuality shard summary written.", shardOutput, StringComparison.Ordinal);

        string output = RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Export-ProjectQualityShardCoverage.ps1"));
        Assert.Contains("ProjectQuality shard coverage written.", output, StringComparison.Ordinal);
        Assert.Contains("CoverageState=complete-class-coverage", output, StringComparison.Ordinal);

        string coveragePath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "test-analysis",
            "project-quality-shard-class-coverage.json");
        using JsonDocument coverageDocument = JsonDocument.Parse(File.ReadAllText(coveragePath));
        JsonElement coverage = coverageDocument.RootElement;

        Assert.Equal("project-quality-shard-class-coverage", coverage.GetProperty("recordKind").GetString());
        Assert.Equal("complete-class-coverage", coverage.GetProperty("coverageState").GetString());
        Assert.True(coverage.GetProperty("inventoryTestCount").GetInt32() >= 1000);
        Assert.True(coverage.GetProperty("inventoryClassCount").GetInt32() >= 300);
        Assert.Equal(
            coverage.GetProperty("inventoryClassCount").GetInt32(),
            coverage.GetProperty("coveredClassCount").GetInt32());
        Assert.Equal(0, coverage.GetProperty("missingClassCount").GetInt32());
        Assert.True(coverage.GetProperty("allClassesCovered").GetBoolean());
        Assert.True(coverage.GetProperty("strictPassedTrxCount").GetInt32() > 0);
        Assert.Equal(0, coverage.GetProperty("invalidEvidenceCount").GetInt32());
        Assert.Equal(4, coverage.GetProperty("shardCoverage").GetArrayLength());
        Assert.All(coverage.GetProperty("shardCoverage").EnumerateArray(), static shard =>
        {
            Assert.Equal(
                shard.GetProperty("inventoryClassCount").GetInt32(),
                shard.GetProperty("coveredClassCount").GetInt32());
            Assert.Equal(0, shard.GetProperty("missingClassCount").GetInt32());
        });
        Assert.False(coverage.GetProperty("performsPublish").GetBoolean());
        Assert.False(coverage.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(coverage.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains("not a one-shot whole-suite run", coverage.GetProperty("boundary").GetString(), StringComparison.Ordinal);
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
    {
        ProcessStartInfo startInfo = new()
        {
            FileName = "pwsh",
            WorkingDirectory = RepositoryPaths.Root,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        startInfo.ArgumentList.Add("-NoProfile");
        startInfo.ArgumentList.Add("-ExecutionPolicy");
        startInfo.ArgumentList.Add("Bypass");
        startInfo.ArgumentList.Add("-File");
        startInfo.ArgumentList.Add(scriptPath);
        foreach (string argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        using Process process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("Failed to start PowerShell.");
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        Assert.True(process.WaitForExit(180_000), $"PowerShell timed out.{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        Assert.Equal(0, process.ExitCode);
        Assert.True(string.IsNullOrWhiteSpace(stderr), stderr);
        return stdout;
    }
}

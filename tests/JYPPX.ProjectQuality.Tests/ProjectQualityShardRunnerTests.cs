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

        JsonElement isolation = summary.GetProperty("sharedEvidenceIsolation");
        Assert.Equal("cross-process-exclusive-file-lock", isolation.GetProperty("mode").GetString());
        Assert.Equal("artifacts/final-release", isolation.GetProperty("scope").GetString());
        Assert.True(isolation.GetProperty("required").GetBoolean());
        Assert.False(isolation.GetProperty("acquired").GetBoolean());
        Assert.False(isolation.GetProperty("released").GetBoolean());
        Assert.Equal(1800, isolation.GetProperty("lockWaitTimeoutSeconds").GetInt32());
        Assert.True(isolation.GetProperty("testTimeoutStartsAfterLockAcquired").GetBoolean());
        Assert.True(isolation.GetProperty("previewDoesNotAcquireLock").GetBoolean());

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
        Assert.Contains("MissingOnly", runnerSource, StringComparison.Ordinal);
        Assert.Contains("coverage inventory SHA256", runnerSource, StringComparison.Ordinal);
        Assert.Contains("slowestExecutionUnits", runnerSource, StringComparison.Ordinal);
        Assert.Contains("classNames", runnerSource, StringComparison.Ordinal);
        Assert.Contains("DOTNET_CLI_USE_MSBUILD_SERVER", runnerSource, StringComparison.Ordinal);
        Assert.Contains("MSBUILDDISABLENODEREUSE", runnerSource, StringComparison.Ordinal);
        Assert.Contains("[IO.FileShare]::None", runnerSource, StringComparison.Ordinal);
        Assert.Contains("SharedEvidenceLockTimeoutSeconds", runnerSource, StringComparison.Ordinal);
        Assert.Contains("No tests were started by this runner", runnerSource, StringComparison.Ordinal);

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
    public void SharedEvidenceLockContentionFailsClosedBeforeStartingTests()
    {
        string fixtureRoot = Path.Combine(Path.GetTempPath(), $"project-quality-shared-evidence-lock-{Guid.NewGuid():N}");
        string outputRoot = Path.Combine(fixtureRoot, "output");
        string lockPath = Path.Combine(fixtureRoot, "shared-evidence.lock");
        string inventoryPath = Path.Combine(fixtureRoot, "inventory.json");
        Directory.CreateDirectory(fixtureRoot);
        File.WriteAllText(inventoryPath, JsonSerializer.Serialize(new
        {
            recordKind = "project-quality-test-inventory",
            shards = new object[]
            {
                new
                {
                    id = "A-F",
                    classes = new[] { "JYPPX.ProjectQuality.Tests.ArticleRoadmap30PlusTests" },
                },
            },
        }));

        try
        {
            using (FileStream heldLock = File.Open(lockPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None))
            {
                Stopwatch stopwatch = Stopwatch.StartNew();
                string failure = RunPowerShellExpectFailure(
                    Path.Combine(RepositoryPaths.Root, "eng", "Invoke-ProjectQualityTestShards.ps1"),
                    "-Shard",
                    "A-F",
                    "-InventoryPath",
                    inventoryPath,
                    "-OutputRoot",
                    outputRoot,
                    "-SharedEvidenceLockPath",
                    lockPath,
                    "-SharedEvidenceLockTimeoutSeconds",
                    "1",
                    "-TimeoutSeconds",
                    "120",
                    "-RunId",
                    "lock-contention");
                stopwatch.Stop();

                Assert.Contains("waiting for the ProjectQuality shared evidence lock", failure, StringComparison.Ordinal);
                Assert.Contains("No tests were started by this runner", failure, StringComparison.Ordinal);
                Assert.True(stopwatch.Elapsed < TimeSpan.FromSeconds(15), $"Lock admission took {stopwatch.Elapsed}.");
                Assert.False(File.Exists(Path.Combine(outputRoot, "lock-contention", "summary.json")));
            }
        }
        finally
        {
            Directory.Delete(fixtureRoot, recursive: true);
        }
    }

    [Fact]
    public void MissingOnlyPreviewUsesExactClassesAndFailsClosedOnStaleInventory()
    {
        string inventoryOutput = RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Export-ProjectQualityTestInventory.ps1"));
        Assert.Contains("ProjectQuality test inventory written.", inventoryOutput, StringComparison.Ordinal);

        string inventoryPath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "test-analysis",
            "project-quality-test-inventory.json");
        using JsonDocument inventoryDocument = JsonDocument.Parse(File.ReadAllText(inventoryPath));
        JsonElement shard = inventoryDocument.RootElement.GetProperty("shards")[0];
        string shardId = shard.GetProperty("id").GetString()!;
        string missingClass = shard.GetProperty("classes")[0].GetString()!;
        string inventorySha256 = Convert.ToHexString(
            System.Security.Cryptography.SHA256.HashData(File.ReadAllBytes(inventoryPath))).ToLowerInvariant();
        string coveragePath = Path.Combine(Path.GetTempPath(), $"project-quality-coverage-{Guid.NewGuid():N}.json");

        try
        {
            File.WriteAllText(coveragePath, JsonSerializer.Serialize(new
            {
                recordKind = "project-quality-shard-class-coverage",
                inventorySha256,
                missingClasses = new[] { missingClass },
            }));

            string previewOutput = RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Invoke-ProjectQualityTestShards.ps1"),
                "-Shard",
                shardId,
                "-CoveragePath",
                coveragePath,
                "-MissingOnly",
                "-TimeoutSeconds",
                "120",
                "-RunId",
                "project-quality-missing-only-preview",
                "-PreviewOnly");
            Assert.Contains("RunState=preview Shards=1", previewOutput, StringComparison.Ordinal);

            string summaryPath = Path.Combine(
                RepositoryPaths.Root,
                "artifacts",
                "test-analysis",
                "project-quality-shards",
                "project-quality-missing-only-preview",
                "summary.json");
            using JsonDocument summaryDocument = JsonDocument.Parse(File.ReadAllText(summaryPath));
            JsonElement summary = summaryDocument.RootElement;
            Assert.True(summary.GetProperty("missingOnly").GetBoolean());
            Assert.Equal(inventorySha256, summary.GetProperty("inventorySha256").GetString());
            Assert.Equal(1, summary.GetProperty("requestedMissingClassCount").GetInt32());
            Assert.Equal(1, summary.GetProperty("selectedMissingClassCount").GetInt32());
            Assert.Equal(1, summary.GetProperty("classLevelExecutionUnitCount").GetInt32());
            Assert.Equal(missingClass, summary.GetProperty("selectedClassNames")[0].GetString());
            Assert.Equal(1, summary.GetProperty("slowestExecutionUnits").GetArrayLength());

            File.WriteAllText(coveragePath, JsonSerializer.Serialize(new
            {
                recordKind = "project-quality-shard-class-coverage",
                inventorySha256 = new string('0', 64),
                missingClasses = new[] { missingClass },
            }));
            string failure = RunPowerShellExpectFailure(
                Path.Combine(RepositoryPaths.Root, "eng", "Invoke-ProjectQualityTestShards.ps1"),
                "-Shard",
                shardId,
                "-CoveragePath",
                coveragePath,
                "-MissingOnly",
                "-TimeoutSeconds",
                "120",
                "-RunId",
                "project-quality-missing-only-stale-preview",
                "-PreviewOnly");
            Assert.Contains("coverage inventory SHA256 does not match", failure, StringComparison.Ordinal);
        }
        finally
        {
            File.Delete(coveragePath);
        }
    }

    [Fact]
    public void ShardCoverageExporterRequiresHashVerifiedPassedTrxForEveryInventoryClass()
    {
        string fixtureRoot = Path.Combine(Path.GetTempPath(), $"project-quality-shard-coverage-{Guid.NewGuid():N}");
        string shardRoot = Path.Combine(fixtureRoot, "shards", "fixture-run");
        string outputDirectory = Path.Combine(fixtureRoot, "output");
        Directory.CreateDirectory(shardRoot);

        try
        {
            string firstClass = "JYPPX.ProjectQuality.Tests.AlphaFixtureTests";
            string secondClass = "JYPPX.ProjectQuality.Tests.ZuluFixtureTests";
            string historicalClass = "JYPPX.ProjectQuality.Tests.HistoricalFixtureTests";
            string inventoryPath = Path.Combine(fixtureRoot, "inventory.json");
            File.WriteAllText(inventoryPath, JsonSerializer.Serialize(new
            {
                recordKind = "project-quality-test-inventory",
                testCount = 2,
                classCount = 2,
                shards = new object[]
                {
                    new { id = "A-F", classes = new[] { firstClass } },
                    new { id = "G-M", classes = Array.Empty<string>() },
                    new { id = "N-S", classes = Array.Empty<string>() },
                    new { id = "T-Z", classes = new[] { secondClass } },
                },
            }));

            string firstTrxPath = Path.Combine(shardRoot, "first.trx");
            string secondTrxPath = Path.Combine(shardRoot, "second.trx");
            string historicalTrxPath = Path.Combine(shardRoot, "historical.trx");
            File.WriteAllText(firstTrxPath, $"<TestRun><TestDefinitions><UnitTest><TestMethod className=\"{firstClass}\" name=\"Pass\" /></UnitTest></TestDefinitions></TestRun>");
            File.WriteAllText(secondTrxPath, $"<TestRun><TestDefinitions><UnitTest><TestMethod className=\"{secondClass}\" name=\"Pass\" /></UnitTest></TestDefinitions></TestRun>");
            File.WriteAllText(historicalTrxPath, $"<TestRun><TestDefinitions><UnitTest><TestMethod className=\"{historicalClass}\" name=\"Pass\" /></UnitTest></TestDefinitions></TestRun>");

            static string Sha256(string path) => Convert.ToHexString(
                System.Security.Cryptography.SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
            object Counters() => new { available = true, total = 1, executed = 1, passed = 1, failed = 0, error = 0, timeout = 0, aborted = 0 };
            File.WriteAllText(Path.Combine(shardRoot, "summary.json"), JsonSerializer.Serialize(new
            {
                runId = "fixture-run",
                results = new object[]
                {
                    new { id = "A-F", parentShardId = "A-F", state = "passed", classCount = 1, classNames = new[] { firstClass }, durationSeconds = 1.25, endedAtUtc = "2026-07-17T00:00:01Z", trxPath = firstTrxPath, trxSha256 = Sha256(firstTrxPath), counters = Counters() },
                    new { id = "T-Z", parentShardId = "T-Z", state = "passed", classCount = 1, classNames = new[] { secondClass }, durationSeconds = 2.5, endedAtUtc = "2026-07-17T00:00:02Z", trxPath = secondTrxPath, trxSha256 = Sha256(secondTrxPath), counters = Counters() },
                    new { id = "historical", parentShardId = "", state = "passed", classCount = 1, classNames = new[] { historicalClass }, durationSeconds = 99.0, endedAtUtc = "2026-07-16T00:00:00Z", trxPath = historicalTrxPath, trxSha256 = Sha256(historicalTrxPath), counters = Counters() },
                },
            }));

            string exporterPath = Path.Combine(RepositoryPaths.Root, "eng", "Export-ProjectQualityShardCoverage.ps1");
            string output = RunPowerShell(exporterPath, "-InventoryPath", inventoryPath, "-ShardRoot", Path.GetDirectoryName(shardRoot)!, "-OutputDirectory", outputDirectory);
            Assert.Contains("CoverageState=complete-class-coverage Classes=2/2 Missing=0 ValidTrx=3 InvalidEvidence=0", output, StringComparison.Ordinal);

            string coveragePath = Path.Combine(outputDirectory, "project-quality-shard-class-coverage.json");
            using (JsonDocument coverageDocument = JsonDocument.Parse(File.ReadAllText(coveragePath)))
            {
                JsonElement coverage = coverageDocument.RootElement;
                Assert.Equal("complete-class-coverage", coverage.GetProperty("coverageState").GetString());
                Assert.Equal(2, coverage.GetProperty("coveredClassCount").GetInt32());
                Assert.Equal(3, coverage.GetProperty("singleClassPassedTrxCount").GetInt32());
                Assert.Equal(2, coverage.GetProperty("singleClassCoveredClassCount").GetInt32());
                Assert.Equal(0, coverage.GetProperty("invalidEvidenceCount").GetInt32());
                Assert.False(coverage.GetProperty("canPublishPublicly").GetBoolean());
                Assert.False(coverage.GetProperty("canCloseReleaseIssue").GetBoolean());
            }

            File.AppendAllText(secondTrxPath, "<!-- tampered -->");
            string tamperedOutput = RunPowerShell(exporterPath, "-InventoryPath", inventoryPath, "-ShardRoot", Path.GetDirectoryName(shardRoot)!, "-OutputDirectory", outputDirectory);
            Assert.Contains("CoverageState=incomplete-class-coverage Classes=1/2 Missing=1 ValidTrx=2 InvalidEvidence=1", tamperedOutput, StringComparison.Ordinal);

            using JsonDocument tamperedDocument = JsonDocument.Parse(File.ReadAllText(coveragePath));
            JsonElement tamperedCoverage = tamperedDocument.RootElement;
            Assert.Equal("incomplete-class-coverage", tamperedCoverage.GetProperty("coverageState").GetString());
            Assert.Equal(secondClass, tamperedCoverage.GetProperty("missingClasses")[0].GetString());
            Assert.Equal("trx-sha256-mismatch", tamperedCoverage.GetProperty("invalidEvidence")[0].GetProperty("reason").GetString());
        }
        finally
        {
            Directory.Delete(fixtureRoot, recursive: true);
        }
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
    {
        ProcessStartInfo startInfo = new()
        {
            FileName = PowerShellHost.ResolveExecutable(),
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

    private static string RunPowerShellExpectFailure(string scriptPath, params string[] arguments)
    {
        ProcessStartInfo startInfo = new()
        {
            FileName = PowerShellHost.ResolveExecutable(),
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
        Assert.NotEqual(0, process.ExitCode);
        return stdout + Environment.NewLine + stderr;
    }
}

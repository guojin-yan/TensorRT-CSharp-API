using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DeferredBTierWorkItemProofClosureLedgerTests
{
    [Fact]
    public void LedgerClosesAllStableWorkItemsWithoutChangingProofClass()
    {
        using JsonDocument ledger = ReadJson("artifacts", "interface-coverage", "deferred-btier-work-item-proof-closure-ledger.json");
        JsonElement root = ledger.RootElement;

        Assert.Equal("deferred-btier-work-item-proof-closure-ledger", root.GetProperty("recordKind").GetString());
        Assert.Equal("source-quality-proof-closed", root.GetProperty("closureState").GetString());
        Assert.Equal(51, root.GetProperty("closedWorkItemCount").GetInt32());

        string[] closedIds = root.GetProperty("closedWorkItemIds")
            .EnumerateArray()
            .Select(static value => value.GetString()!)
            .ToArray();
        Assert.Equal(51, closedIds.Length);
        Assert.Equal(51, closedIds.Distinct(StringComparer.Ordinal).Count());
        Assert.Equal("btier-001", closedIds[0]);
        Assert.Equal("btier-051", closedIds[^1]);

        JsonElement[] batches = root.GetProperty("closureBatches").EnumerateArray().ToArray();
        Assert.Equal(5, batches.Length);
        Assert.Equal(51, batches.Sum(static batch => batch.GetProperty("closedWorkItemCount").GetInt32()));
        Dictionary<string, int> expectedBatches = new(StringComparer.Ordinal)
        {
            ["btier-001-012"] = 12,
            ["btier-013-024"] = 12,
            ["btier-025-040"] = 16,
            ["btier-041-046"] = 6,
            ["btier-047-051"] = 5,
        };
        Assert.Equal(
            expectedBatches,
            batches.ToDictionary(
                static batch => batch.GetProperty("batch").GetString()!,
                static batch => batch.GetProperty("closedWorkItemCount").GetInt32(),
                StringComparer.Ordinal));
        Assert.All(batches, static batch =>
        {
            Assert.NotEmpty(batch.GetProperty("evidence").EnumerateArray());
            Assert.All(batch.GetProperty("evidence").EnumerateArray(), static evidence =>
                Assert.True(File.Exists(Path.Combine(RepositoryPaths.Root, evidence.GetString()!.Replace('/', Path.DirectorySeparatorChar)))));
        });

        string markdown = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "interface-coverage",
            "deferred-btier-work-item-proof-closure-ledger.md"));
        foreach ((string batch, int count) in expectedBatches)
        {
            Assert.Contains($"| `{batch}` | {count} |", markdown, StringComparison.Ordinal);
        }
        Assert.Contains("`btier-001` 到 `btier-051`", markdown, StringComparison.Ordinal);
        Assert.Contains("DeferredBTierWorkItemProofClosureLedgerTests", markdown, StringComparison.Ordinal);
        Assert.DoesNotContain("DeferredBTierImplementationWorkPackageTests", markdown, StringComparison.Ordinal);
        Assert.DoesNotContain("DeferredBTierWorkItemProofBatchTests", markdown, StringComparison.Ordinal);

        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("canDeleteDeferredRecords").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("canPromoteReleaseProof").GetBoolean());
        Assert.Contains("does not prove runtime execution", root.GetProperty("boundary").GetString()!, StringComparison.OrdinalIgnoreCase);
    }

    private static JsonDocument ReadJson(params string[] pathParts) =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray())));

    private static string RunPowerShell(string scriptName)
    {
        ProcessStartInfo startInfo = new()
        {
            FileName = PowerShellHost.ResolveExecutable(),
            WorkingDirectory = RepositoryPaths.Root,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        startInfo.ArgumentList.Add("-NoProfile");
        startInfo.ArgumentList.Add("-ExecutionPolicy");
        startInfo.ArgumentList.Add("Bypass");
        startInfo.ArgumentList.Add("-File");
        startInfo.ArgumentList.Add(Path.Combine(RepositoryPaths.Root, "eng", scriptName));

        using Process process = Process.Start(startInfo)!;
        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();
        Assert.True(process.ExitCode == 0, $"PowerShell command failed with exit code {process.ExitCode}:{Environment.NewLine}{output}{error}");
        return output + error;
    }
}

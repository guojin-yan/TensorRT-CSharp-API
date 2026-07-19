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
        Assert.Equal(45, root.GetProperty("closedWorkItemCount").GetInt32());

        string[] closedIds = root.GetProperty("closedWorkItemIds")
            .EnumerateArray()
            .Select(static value => value.GetString()!)
            .ToArray();
        Assert.Equal(45, closedIds.Length);
        Assert.Equal(45, closedIds.Distinct(StringComparer.Ordinal).Count());
        Assert.Equal("btier-001", closedIds[0]);
        Assert.Equal("btier-045", closedIds[^1]);

        JsonElement[] batches = root.GetProperty("closureBatches").EnumerateArray().ToArray();
        Assert.Equal(4, batches.Length);
        Assert.Equal(45, batches.Sum(static batch => batch.GetProperty("closedWorkItemCount").GetInt32()));
        Assert.All(batches, static batch =>
        {
            Assert.NotEmpty(batch.GetProperty("evidence").EnumerateArray());
            Assert.All(batch.GetProperty("evidence").EnumerateArray(), static evidence =>
                Assert.True(File.Exists(Path.Combine(RepositoryPaths.Root, evidence.GetString()!.Replace('/', Path.DirectorySeparatorChar)))));
        });

        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("canDeleteDeferredRecords").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("canPromoteReleaseProof").GetBoolean());
        Assert.Contains("does not prove runtime execution", root.GetProperty("boundary").GetString()!, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void GeneratedWorkPackageConsumesLedgerAndLeavesNoClosedItemPending()
    {
        string output = RunPowerShell("Export-DeferredBTierImplementationWorkPackage.ps1");
        Assert.Contains("ClosedWorkItemCount=45", output, StringComparison.Ordinal);
        Assert.Contains("RemainingWorkItemCount=0", output, StringComparison.Ordinal);

        using JsonDocument ledger = ReadJson("artifacts", "interface-coverage", "deferred-btier-work-item-proof-closure-ledger.json");
        using JsonDocument workPackage = ReadJson("artifacts", "interface-coverage", "deferred-btier-implementation-work-package.json");
        string[] closedIds = ledger.RootElement.GetProperty("closedWorkItemIds").EnumerateArray().Select(static value => value.GetString()!).ToArray();
        Dictionary<string, JsonElement> workItems = workPackage.RootElement.GetProperty("workItems")
            .EnumerateArray()
            .ToDictionary(static item => item.GetProperty("workItemId").GetString()!, static item => item);

        Assert.Equal(closedIds, workItems.Keys);
        Assert.All(closedIds, id =>
        {
            JsonElement item = workItems[id];
            Assert.Equal("source-quality-proof-closed", item.GetProperty("workItemState").GetString());
            Assert.Equal("artifacts/interface-coverage/deferred-btier-work-item-proof-closure-ledger.json", item.GetProperty("closureProofRecord").GetString());
            Assert.Contains("do not schedule this item as new implementation work", item.GetProperty("implementationAction").GetString()!, StringComparison.Ordinal);
            Assert.True(item.GetProperty("safeAlternativeManifestIds").GetArrayLength() > 0);
            Assert.True(item.GetProperty("deferredHistoryManifestIds").GetArrayLength() > 0);
            Assert.False(item.GetProperty("canDeleteDeferredRecord").GetBoolean());
            Assert.False(item.GetProperty("canPromoteReleaseProof").GetBoolean());
        });

        string markdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-btier-work-item-proof-closure-ledger.md"));
        Assert.Contains("source-quality-proof-closed", markdown, StringComparison.Ordinal);
        Assert.Contains("不得再次选择", markdown, StringComparison.Ordinal);
    }

    private static JsonDocument ReadJson(params string[] pathParts) =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray())));

    private static string RunPowerShell(string scriptName)
    {
        ProcessStartInfo startInfo = new()
        {
            FileName = "pwsh",
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

using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DeferredBTierImplementationWorkPackageTests
{
    [Fact]
    public void ImplementationWorkPackageProjectsProofClosedItemsWithoutPromotingRelease()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredReadOnlyApiCandidatePlan.ps1"), "-IncludeMediumRisk", "-MaxItems", "60");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredBTierProofClosureDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredBTierAliasProofClosureRecord.ps1"));

        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredBTierImplementationWorkPackage.ps1"));
        Assert.Contains("Deferred B-tier implementation work package written", output, StringComparison.Ordinal);

        string aliasPath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-btier-alias-proof-closure-record.json");
        string packagePath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-btier-implementation-work-package.json");
        string markdownPath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-btier-implementation-work-package.md");

        Assert.True(File.Exists(packagePath), $"Expected {packagePath} to exist.");
        Assert.True(File.Exists(markdownPath), $"Expected {markdownPath} to exist.");

        using JsonDocument alias = JsonDocument.Parse(File.ReadAllText(aliasPath));
        using JsonDocument package = JsonDocument.Parse(File.ReadAllText(packagePath));

        JsonElement aliasRoot = alias.RootElement;
        JsonElement root = package.RootElement;

        Assert.Equal("deferred-btier-implementation-work-package", root.GetProperty("recordKind").GetString());
        Assert.Equal("source-quality-proof-closed", root.GetProperty("workPackageState").GetString());
        Assert.Equal(aliasRoot.GetProperty("closureCandidateCount").GetInt32(), root.GetProperty("sourceAliasClosureCandidateCount").GetInt32());
        Assert.Equal("stable-v1-existing-40-then-deterministic-append", root.GetProperty("workItemOrderingPolicy").GetString());
        Assert.Equal(40, root.GetProperty("stableWorkItemKeyCount").GetInt32());
        Assert.Equal(60, root.GetProperty("workItemTargetCount").GetInt32());
        Assert.Equal(aliasRoot.GetProperty("closureCandidateCount").GetInt32(), root.GetProperty("workItemCount").GetInt32());
        Assert.Equal(45, root.GetProperty("workItemCount").GetInt32());
        Assert.Equal(45, root.GetProperty("closedWorkItemCount").GetInt32());
        Assert.Equal(0, root.GetProperty("remainingWorkItemCount").GetInt32());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("canDeleteDeferredRecords").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.Contains("not runtime proof", root.GetProperty("boundary").GetString()!, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("permission to delete deferred records", root.GetProperty("boundary").GetString()!, StringComparison.OrdinalIgnoreCase);

        JsonElement[] phases = root.GetProperty("phaseSummaries").EnumerateArray().ToArray();
        Assert.NotEmpty(phases);
        Assert.Contains(phases, static phase =>
            phase.GetProperty("phase").GetString() == "phase-2-wrapper-docs-quality-proof" ||
            phase.GetProperty("phase").GetString() == "phase-1-safe-alternative-proof");

        JsonElement[] workItems = root.GetProperty("workItems").EnumerateArray().ToArray();
        Assert.Equal(root.GetProperty("workItemCount").GetInt32(), workItems.Length);
        Assert.All(workItems, static item =>
        {
            foreach (string propertyName in new[]
            {
                "workItemId",
                "workItemState",
                "closureProofRecord",
                "phase",
                "interface",
                "version",
                "class",
                "method",
                "safeAlternativeManifestIds",
                "deferredHistoryManifestIds",
                "sourceEvidence",
                "requiredFilesToInspect",
                "implementationAction",
                "acceptanceCriteria",
                "validationCommands",
            })
            {
                Assert.True(item.TryGetProperty(propertyName, out _), $"Expected work item property '{propertyName}'.");
            }

            Assert.StartsWith("btier-", item.GetProperty("workItemId").GetString(), StringComparison.Ordinal);
            Assert.Equal("source-quality-proof-closed", item.GetProperty("workItemState").GetString());
            Assert.Equal("artifacts/interface-coverage/deferred-btier-work-item-proof-closure-ledger.json", item.GetProperty("closureProofRecord").GetString());
            Assert.Equal("B - safe-alternative-or-alias", item.GetProperty("safetyTier").GetString());
            Assert.True(item.GetProperty("safeAlternativeManifestIds").GetArrayLength() > 0);
            Assert.True(item.GetProperty("deferredHistoryManifestIds").GetArrayLength() > 0);
            Assert.True(item.GetProperty("sourceEvidence").GetArrayLength() >= 2);
            Assert.True(item.GetProperty("requiredFilesToInspect").GetArrayLength() > 0);
            Assert.Contains("deferred history manifest IDs remain present", item.GetProperty("acceptanceCriteria").EnumerateArray().Select(static criterion => criterion.GetString()!));
            Assert.Contains("dotnet", string.Join(" ", item.GetProperty("validationCommands").EnumerateArray().Select(static command => command.GetString()!)), StringComparison.OrdinalIgnoreCase);
            Assert.False(item.GetProperty("canDeleteDeferredRecord").GetBoolean());
            Assert.False(item.GetProperty("canPromoteReleaseProof").GetBoolean());
            Assert.False(item.GetProperty("isRuntimeExecutionProof").GetBoolean());
            Assert.False(item.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        });

        Assert.Contains("artifacts/interface-coverage/deferred-btier-alias-proof-closure-record.json", root.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!));
        Assert.Contains("artifacts/interface-coverage/deferred-btier-work-item-proof-closure-ledger.json", root.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!));
        Assert.Contains("Do not select workItems whose workItemState is source-quality-proof-closed.", root.GetProperty("nextBatchPromptFocus").EnumerateArray().Select(static item => item.GetString()!));

        string markdown = File.ReadAllText(markdownPath);
        foreach (string marker in new[]
        {
            "Deferred B-tier Implementation Work Package",
            "source-quality-proof-closed",
            "remaining work item count",
            "Work Items",
            "Safe alternative manifests",
            "Deferred history manifests",
            "canPublishPublicly",
            "canCloseReleaseIssue",
            "canDeleteDeferredRecords",
        })
        {
            Assert.Contains(marker, markdown, StringComparison.OrdinalIgnoreCase);
        }
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
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
        startInfo.ArgumentList.Add(scriptPath);

        foreach (string argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        using Process process = Process.Start(startInfo)!;
        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"PowerShell command failed with exit code {process.ExitCode}:{Environment.NewLine}{output}{error}");
        return output + error;
    }
}

using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DeferredBTierAliasProofClosureTests
{
    [Fact]
    public void AliasProofClosureRecordCapturesSelectedBTierCandidatesWithoutDeletingDeferredRecords()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredReadOnlyApiCandidatePlan.ps1"), "-IncludeMediumRisk", "-MaxItems", "60");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredBTierProofClosureDashboard.ps1"));

        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredBTierAliasProofClosureRecord.ps1"));
        Assert.Contains("Deferred B-tier alias proof closure record written", output, StringComparison.Ordinal);

        string dashboardPath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-btier-proof-closure-dashboard.json");
        string recordPath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-btier-alias-proof-closure-record.json");
        string markdownPath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-btier-alias-proof-closure-record.md");
        Assert.True(File.Exists(recordPath), $"Expected {recordPath} to exist.");
        Assert.True(File.Exists(markdownPath), $"Expected {markdownPath} to exist.");

        using JsonDocument dashboard = JsonDocument.Parse(File.ReadAllText(dashboardPath));
        using JsonDocument record = JsonDocument.Parse(File.ReadAllText(recordPath));

        JsonElement dashboardRoot = dashboard.RootElement;
        JsonElement root = record.RootElement;
        int expectedCandidateCount = dashboardRoot.GetProperty("selectedAliasProofReadyCandidateCount").GetInt32();

        Assert.Equal("deferred-btier-alias-proof-closure-record", root.GetProperty("recordKind").GetString());
        Assert.Equal("alias-proof-ready-engineering-record", root.GetProperty("closureState").GetString());
        Assert.Equal(expectedCandidateCount, root.GetProperty("closureCandidateCount").GetInt32());
        Assert.Equal(expectedCandidateCount, root.GetProperty("closureCandidates").GetArrayLength());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("canDeleteDeferredRecords").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.Contains("does not delete deferred records", root.GetProperty("boundary").GetString()!, StringComparison.Ordinal);

        JsonElement[] candidates = root.GetProperty("closureCandidates").EnumerateArray().ToArray();
        Assert.All(candidates, static candidate =>
        {
            Assert.Equal("B - safe-alternative-or-alias", candidate.GetProperty("safetyTier").GetString());
            Assert.Equal("alias-proof-ready", candidate.GetProperty("closureProofState").GetString());
            Assert.True(candidate.GetProperty("safeAlternativeManifestIds").GetArrayLength() > 0);
            Assert.True(candidate.GetProperty("deferredHistoryManifestIds").GetArrayLength() > 0);
            Assert.True(candidate.GetProperty("sourceEvidence").GetArrayLength() >= 2);
            Assert.False(candidate.GetProperty("canDeleteDeferredRecord").GetBoolean());
            Assert.False(candidate.GetProperty("canPromoteReleaseProof").GetBoolean());
            Assert.False(candidate.GetProperty("isRuntimeExecutionProof").GetBoolean());
            Assert.False(candidate.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
            Assert.Contains("keep-deferred-history", candidate.GetProperty("closureDecision").GetString()!, StringComparison.Ordinal);

            foreach (string propertyName in new[]
            {
                "interface",
                "version",
                "class",
                "method",
                "closureBucket",
                "closureProofState",
                "safeAlternativeManifestIds",
                "deferredHistoryManifestIds",
                "sourceEvidence",
                "publicApiExposurePolicy",
                "recommendedAction",
                "closureDecision",
            })
            {
                Assert.True(candidate.TryGetProperty(propertyName, out _), $"Expected closure candidate property '{propertyName}'.");
            }
        });

        Assert.DoesNotContain(candidates, static candidate => candidate.GetProperty("safetyTier").GetString() == "C - design-gate-required");
        Assert.DoesNotContain(candidates, static candidate => candidate.GetProperty("safetyTier").GetString() == "D - keep-deferred");

        string markdown = File.ReadAllText(markdownPath);
        foreach (string marker in new[]
        {
            "deferred-btier-alias-proof-closure-record",
            "alias-proof-ready-engineering-record",
            "safe alternative",
            "deferred history",
            "keep-deferred-history",
            "canPublishPublicly",
            "canCloseReleaseIssue",
            "canDeleteDeferredRecords",
            "C/D",
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

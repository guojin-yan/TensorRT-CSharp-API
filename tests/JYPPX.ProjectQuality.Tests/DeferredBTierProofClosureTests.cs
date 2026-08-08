using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DeferredBTierProofClosureTests
{
    [Fact]
    public void DeferredBTierProofClosureDashboardBucketsBTierWithoutPromotingReleaseOrDeletingDeferredRecords()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredReadOnlyApiCandidatePlan.ps1"), "-IncludeMediumRisk", "-MaxItems", "60");

        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredBTierProofClosureDashboard.ps1"));
        Assert.Contains("Deferred B-tier proof closure dashboard written", output, StringComparison.Ordinal);

        string triagePath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-candidate-safety-triage.json");
        string dashboardPath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-btier-proof-closure-dashboard.json");
        string markdownPath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-btier-proof-closure-dashboard.md");
        Assert.True(File.Exists(dashboardPath), $"Expected {dashboardPath} to exist.");
        Assert.True(File.Exists(markdownPath), $"Expected {markdownPath} to exist.");

        using JsonDocument triage = JsonDocument.Parse(File.ReadAllText(triagePath));
        using JsonDocument dashboard = JsonDocument.Parse(File.ReadAllText(dashboardPath));

        JsonElement triageRoot = triage.RootElement;
        JsonElement root = dashboard.RootElement;
        JsonElement[] triageRows = triageRoot.GetProperty("rows").EnumerateArray().ToArray();
        int expectedBTierCount = triageRows.Count(static row => row.GetProperty("safetyTier").GetString() == "B - safe-alternative-or-alias");
        int expectedUniqueBTierCount = triageRows
            .Where(static row => row.GetProperty("safetyTier").GetString() == "B - safe-alternative-or-alias")
            .Select(static row =>
                row.GetProperty("tensorRtLine").GetString() + "|" +
                row.GetProperty("interface").GetString() + "|" +
                row.GetProperty("matchedManifestIds").GetString())
            .Distinct(StringComparer.Ordinal)
            .Count();
        int expectedCTierCount = triageRows.Count(static row => row.GetProperty("safetyTier").GetString() == "C - design-gate-required");
        int expectedDTierCount = triageRows.Count(static row => row.GetProperty("safetyTier").GetString() == "D - keep-deferred");

        Assert.Equal("deferred-btier-proof-closure-dashboard", root.GetProperty("recordKind").GetString());
        Assert.Equal("planning-input-only", root.GetProperty("closureState").GetString());
        Assert.Equal(expectedBTierCount, root.GetProperty("totalBTierCount").GetInt32());
        Assert.Equal(expectedUniqueBTierCount, root.GetProperty("uniqueBTierCandidateCount").GetInt32());
        Assert.Equal(expectedUniqueBTierCount, root.GetProperty("totalClosureCandidateCount").GetInt32());
        Assert.Equal(60, root.GetProperty("selectedCandidateTargetCount").GetInt32());
        Assert.Equal(60, root.GetProperty("selectedCandidateCount").GetInt32());
        Assert.True(root.GetProperty("aliasProofReadyCandidateCount").GetInt32() > 0);
        Assert.True(root.GetProperty("selectedAliasProofReadyCandidateCount").GetInt32() > 0);
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("canDeleteDeferredRecords").GetBoolean());
        Assert.Contains("permission to delete deferred records", root.GetProperty("boundary").GetString()!, StringComparison.Ordinal);

        string[] expectedBuckets =
        {
            "already-safe-alternative-proof",
            "alias-closure-needed",
            "docs-test-proof-needed",
            "runtime-smoke-backfill-needed",
            "manual-review-required",
        };
        string[] buckets = root
            .GetProperty("closureBuckets")
            .EnumerateArray()
            .Select(static item => item.GetProperty("closureBucket").GetString()!)
            .Order(StringComparer.Ordinal)
            .ToArray();
        Assert.Equal(expectedBuckets.Order(StringComparer.Ordinal).ToArray(), buckets);
        Assert.Equal(expectedUniqueBTierCount, root.GetProperty("closureBuckets").EnumerateArray().Sum(static item => item.GetProperty("candidateCount").GetInt32()));
        Assert.Contains(root.GetProperty("closureBuckets").EnumerateArray(), static item =>
            item.GetProperty("closureBucket").GetString() == "docs-test-proof-needed" &&
            item.GetProperty("candidateCount").GetInt32() > 0);
        Assert.Contains(root.GetProperty("closureBuckets").EnumerateArray(), static item =>
            item.GetProperty("closureBucket").GetString() == "runtime-smoke-backfill-needed" &&
            item.GetProperty("candidateCount").GetInt32() > 0);
        Assert.Contains(root.GetProperty("closureBuckets").EnumerateArray(), static item =>
            item.GetProperty("closureBucket").GetString() == "manual-review-required" &&
            item.GetProperty("candidateCount").GetInt32() > 0);

        JsonElement[] excludedTiers = root.GetProperty("excludedTierSummaries").EnumerateArray().ToArray();
        Assert.Contains(excludedTiers, item =>
            item.GetProperty("safetyTier").GetString() == "C - design-gate-required" &&
            item.GetProperty("count").GetInt32() == expectedCTierCount &&
            item.GetProperty("closurePolicy").GetString()!.Contains("requires design gate", StringComparison.Ordinal));
        Assert.Contains(excludedTiers, item =>
            item.GetProperty("safetyTier").GetString() == "D - keep-deferred" &&
            item.GetProperty("count").GetInt32() == expectedDTierCount &&
            item.GetProperty("closurePolicy").GetString()!.Contains("must remain deferred", StringComparison.Ordinal));

        JsonElement[] selectedCandidates = root.GetProperty("selectedClosureCandidates").EnumerateArray().ToArray();
        Assert.Equal(root.GetProperty("selectedCandidateCount").GetInt32(), selectedCandidates.Length);
        Assert.All(selectedCandidates, static candidate =>
        {
            Assert.Equal("B - safe-alternative-or-alias", candidate.GetProperty("safetyTier").GetString());
            Assert.NotEqual("C - design-gate-required", candidate.GetProperty("safetyTier").GetString());
            Assert.NotEqual("D - keep-deferred", candidate.GetProperty("safetyTier").GetString());
            Assert.False(candidate.GetProperty("canDeleteDeferredRecord").GetBoolean());
            Assert.False(candidate.GetProperty("canPromoteReleaseProof").GetBoolean());

            foreach (string propertyName in new[]
            {
                "interface",
                "version",
                "safetyTier",
                "closureBucket",
                "evidenceKind",
                "closureProofState",
                "safeAlternativeManifestIds",
                "deferredHistoryManifestIds",
                "sourceEvidence",
                "recommendedAction",
                "publicApiExposurePolicy",
            })
            {
                Assert.True(candidate.TryGetProperty(propertyName, out _), $"Expected selected candidate property '{propertyName}'.");
            }

            Assert.True(candidate.GetProperty("sourceEvidence").GetArrayLength() >= 2);
            Assert.Contains(candidate.GetProperty("closureBucket").GetString()!, new[]
            {
                "already-safe-alternative-proof",
                "alias-closure-needed",
                "docs-test-proof-needed",
                "runtime-smoke-backfill-needed",
                "manual-review-required",
            });
        });
        Assert.Contains(selectedCandidates, static candidate =>
            candidate.GetProperty("closureProofState").GetString() == "alias-proof-ready" &&
            candidate.GetProperty("safeAlternativeManifestIds").GetArrayLength() > 0 &&
            candidate.GetProperty("deferredHistoryManifestIds").GetArrayLength() > 0);

        Assert.Contains("deferred safety triage", root.GetProperty("nonSubstituteProofKinds").EnumerateArray().Select(static item => item.GetString()!));
        Assert.Contains("artifacts/interface-coverage/deferred-candidate-safety-triage.json", root.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!));

        string markdown = File.ReadAllText(markdownPath);
        foreach (string marker in new[]
        {
            "deferred-btier-proof-closure-dashboard",
            "planning-input-only",
            "already-safe-alternative-proof",
            "alias-closure-needed",
            "docs-test-proof-needed",
            "runtime-smoke-backfill-needed",
            "manual-review-required",
            "canPublishPublicly",
            "canCloseReleaseIssue",
            "canDeleteDeferredRecords",
            "C/D 不进入",
        })
        {
            Assert.Contains(marker, markdown, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void ReleaseFreezeFinalVerificationConsumesBTierClosureDashboardAsPlanningInputOnly()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredReadOnlyApiCandidatePlan.ps1"), "-IncludeMediumRisk", "-MaxItems", "60");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredBTierProofClosureDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseFreezeFinalVerification.ps1"));

        string dashboardPath = Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-btier-proof-closure-dashboard.json");
        string freezePath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-freeze-final-verification.json");
        string freezeMarkdownPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-freeze-final-verification.md");

        using JsonDocument dashboard = JsonDocument.Parse(File.ReadAllText(dashboardPath));
        using JsonDocument freeze = JsonDocument.Parse(File.ReadAllText(freezePath));
        JsonElement dashboardRoot = dashboard.RootElement;
        JsonElement freezeRoot = freeze.RootElement;

        Assert.Equal("blocked-real-proof-required", freezeRoot.GetProperty("verificationState").GetString());
        Assert.False(freezeRoot.GetProperty("performsPublish").GetBoolean());
        Assert.False(freezeRoot.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(freezeRoot.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal("planning-input-only", freezeRoot.GetProperty("deferredBTierProofClosureDashboardState").GetString());
        Assert.Equal(dashboardRoot.GetProperty("totalBTierCount").GetInt32(), freezeRoot.GetProperty("deferredBTierProofClosureDashboardTotalCount").GetInt32());
        Assert.Equal(dashboardRoot.GetProperty("selectedCandidateCount").GetInt32(), freezeRoot.GetProperty("deferredBTierProofClosureDashboardSelectedCount").GetInt32());
        Assert.False(freezeRoot.GetProperty("deferredBTierProofClosureDashboardIsReleaseProof").GetBoolean());
        Assert.False(freezeRoot.GetProperty("deferredBTierProofClosureDashboardCanPublishPublicly").GetBoolean());
        Assert.False(freezeRoot.GetProperty("deferredBTierProofClosureDashboardCanCloseReleaseIssue").GetBoolean());
        Assert.False(freezeRoot.GetProperty("deferredBTierProofClosureDashboardCanDeleteDeferredRecords").GetBoolean());
        Assert.Contains("artifacts/interface-coverage/deferred-btier-proof-closure-dashboard.json", freezeRoot.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!));
        Assert.Contains("B-tier proof closure dashboard", freezeRoot.GetProperty("nonSubstituteProofKinds").EnumerateArray().Select(static item => item.GetString()!));

        string markdown = File.ReadAllText(freezeMarkdownPath);
        Assert.Contains("B-tier Proof Closure Dashboard", markdown, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly=false", markdown, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue=false", markdown, StringComparison.Ordinal);
        Assert.Contains("canDeleteDeferredRecords=false", markdown, StringComparison.Ordinal);
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
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

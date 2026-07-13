using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerProofBackfillExecutionPackTests
{
    private static readonly string[] RequiredBackfillIds =
    {
        "owner-authorization",
        "package-consumer-runtime",
        "linux-runner-proof",
        "real-model-runtime",
        "post-publish-verification",
        "release-issue-close-record",
    };

    [Fact]
    public void OwnerProofBackfillExecutionPackExportsBlockedOwnerGuidance()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseIssueCloseRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseIssueCloseRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseClosePreflight.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerReleaseExecutionPackage.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofBackfillExecutionPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument document = ReadFinalReleaseJson("owner-proof-backfill-execution-pack.json");
        JsonElement root = document.RootElement;

        Assert.Equal("owner-proof-backfill-execution-pack", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-proof-required", root.GetProperty("packageState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal("blocked-template-only", root.GetProperty("releaseIssueCloseRecordValidationState").GetString());
        Assert.False(root.GetProperty("releaseIssueCloseRecordCanPromote").GetBoolean());
        Assert.Equal(RequiredBackfillIds.Length, root.GetProperty("backfillItemCount").GetInt32());
        Assert.Equal(0, root.GetProperty("readyBackfillItemCount").GetInt32());

        string[] ids = root.GetProperty("backfillItems")
            .EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .Order(StringComparer.Ordinal)
            .ToArray();
        Assert.Equal(RequiredBackfillIds.Order(StringComparer.Ordinal).ToArray(), ids);

        string[] nonSubstitutes = root.GetProperty("nonSubstituteProofKinds")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        foreach (string marker in new[]
        {
            "local feed",
            "ProjectReference",
            "direct .nupkg reference",
            "template-only release issue close record",
            "release-issue-close-record-template.json",
            "preflight-only release issue close record",
            "readiness snapshot",
            "dependency-probe-only",
        })
        {
            Assert.Contains(marker, nonSubstitutes);
        }

        foreach (JsonElement item in root.GetProperty("backfillItems").EnumerateArray())
        {
            Assert.False(item.GetProperty("performsPublish").GetBoolean());
            Assert.False(item.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("validatorCommand").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("firstCommand").GetString()));
            Assert.True(item.GetProperty("requiredRealInputCount").GetInt32() >= 6);
            Assert.True(item.GetProperty("expectedArtifacts").GetArrayLength() >= 2);
            Assert.True(item.GetProperty("sourceArtifacts").GetArrayLength() >= 2);
            Assert.True(item.GetProperty("cannotUse").GetArrayLength() >= nonSubstitutes.Length);
        }

        JsonElement closeItem = root.GetProperty("backfillItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "release-issue-close-record");
        Assert.Contains("Test-ReleaseIssueCloseRecord.ps1", closeItem.GetProperty("validatorCommand").GetString(), StringComparison.Ordinal);
        Assert.Contains("releaseEvidenceBundle.sha256", closeItem.GetProperty("requiredRealInputs").EnumerateArray().Select(static item => item.GetString()));
        Assert.Contains("rollbackPlan.packageYankOrDeprecatePlan", closeItem.GetProperty("requiredRealInputs").EnumerateArray().Select(static item => item.GetString()));

        JsonElement postPublishItem = root.GetProperty("backfillItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "post-publish-verification");
        Assert.Contains("Test-PostPublishVerificationRecord.ps1", postPublishItem.GetProperty("validatorCommand").GetString(), StringComparison.Ordinal);
        Assert.Contains("--runtime-package-key", string.Join(" ", postPublishItem.GetProperty("requiredRealInputs").EnumerateArray().Select(static item => item.GetString())));

        string markdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "owner-proof-backfill-execution-pack.md"));
        Assert.Contains("Owner Proof Backfill Execution Pack", markdown, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue=false", markdown, StringComparison.Ordinal);
        Assert.Contains("Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady", markdown, StringComparison.Ordinal);
        Assert.Contains("release-issue-close-record-template.json", markdown, StringComparison.Ordinal);

        using JsonDocument evidenceBundle = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidenceRoot = evidenceBundle.RootElement;

        JsonElement ownerProofBackfillEvidenceItem = evidenceRoot.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "owner-proof-backfill-execution-pack");
        Assert.False(ownerProofBackfillEvidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("blocked-real-proof-required", ownerProofBackfillEvidenceItem.GetProperty("state").GetString(), StringComparison.Ordinal);
        Assert.Contains("guidance only", ownerProofBackfillEvidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceEvidence = evidenceRoot.GetProperty("sourceEvidence")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        string[] sourceArtifacts = evidenceRoot.GetProperty("sourceArtifacts")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        foreach (string artifactPath in new[]
        {
            "artifacts/final-release/owner-proof-backfill-execution-pack.json",
            "artifacts/final-release/owner-proof-backfill-execution-pack.md",
        })
        {
            Assert.Contains(artifactPath, sourceEvidence);
            Assert.Contains(artifactPath, sourceArtifacts);
        }

        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));
        Assert.Contains("owner-proof-backfill-execution-pack", evidenceMarkdown, StringComparison.Ordinal);
        Assert.Contains("owner proof backfill execution pack", evidenceMarkdown, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("owner proof ready backfill items: `0`", evidenceMarkdown, StringComparison.Ordinal);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "owner-proof-backfill-execution-pack.md"));
        string releaseEvidenceArticle = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));
        string ownerReleaseArticle = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "owner-release-execution-package.md"));

        Assert.Contains("articles/zh-cn/owner-proof-backfill-execution-pack.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/owner-proof-backfill-execution-pack.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("Owner proof backfill execution pack: `artifacts/final-release/owner-proof-backfill-execution-pack.md`", docsIndex, StringComparison.Ordinal);
        Assert.Contains("Owner proof backfill execution pack artifact: `artifacts/final-release/owner-proof-backfill-execution-pack.json`", readme, StringComparison.Ordinal);
        Assert.Contains("Owner proof backfill execution pack artifact：`artifacts/final-release/owner-proof-backfill-execution-pack.json`", readmeZh, StringComparison.Ordinal);
        Assert.Contains("owner-proof-backfill-execution-pack", releaseEvidenceArticle, StringComparison.Ordinal);
        Assert.Contains("owner-proof-backfill-execution-pack", ownerReleaseArticle, StringComparison.Ordinal);

        foreach (string marker in new[]
        {
            "recordKind=owner-proof-backfill-execution-pack",
            "packageState=blocked-real-proof-required",
            "performsPublish=false",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "owner-authorization",
            "package-consumer-runtime",
            "linux-runner-proof",
            "real-model-runtime",
            "post-publish-verification",
            "release-issue-close-record",
            "release-issue-close-record-template.json",
            "Test-ReleaseIssueCloseRecord.ps1",
        })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
        }
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static string RunPowerShell(string scriptPath)
    {
        using Process process = new();
        process.StartInfo.FileName = "pwsh";
        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(scriptPath);
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

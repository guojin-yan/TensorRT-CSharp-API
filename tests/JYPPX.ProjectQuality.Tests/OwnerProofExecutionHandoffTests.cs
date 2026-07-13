using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OwnerProofExecutionHandoffTests
{
    private static readonly string[] RequiredLineIds =
    {
        "owner-authorization",
        "package-consumer-runtime",
        "linux-runner-proof",
        "real-model-runtime",
        "post-publish-verification",
        "release-issue-close-record",
    };

    [Fact]
    public void OwnerProofExecutionHandoffExportsDashboardAndKeepsReleaseGateBlocked()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseIssueCloseRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseIssueCloseRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseClosePreflight.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofBackfillExecutionPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofExecutionHandoff.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument handoffDocument = ReadFinalReleaseJson("owner-proof-execution-handoff.json");
        JsonElement root = handoffDocument.RootElement;

        Assert.Equal("owner-proof-execution-handoff", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-proof-required", root.GetProperty("handoffState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal(RequiredLineIds.Length, root.GetProperty("handoffLineCount").GetInt32());
        Assert.Equal(0, root.GetProperty("readyHandoffLineCount").GetInt32());
        Assert.Equal(RequiredLineIds.Length, root.GetProperty("blockedHandoffLineCount").GetInt32());
        Assert.False(root.GetProperty("releaseEvidenceBundleCanPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("releaseEvidenceBundleCanCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("releaseClosePreflightCanCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("releaseIssueCloseRecordCanPromote").GetBoolean());
        Assert.False(root.GetProperty("releaseIssueCloseRecordCanCloseReleaseIssue").GetBoolean());

        string[] ids = root.GetProperty("handoffLines")
            .EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .Order(StringComparer.Ordinal)
            .ToArray();
        Assert.Equal(RequiredLineIds.Order(StringComparer.Ordinal).ToArray(), ids);

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
            "Windows handoff for Linux proof",
        })
        {
            Assert.Contains(marker, nonSubstitutes);
        }

        foreach (JsonElement line in root.GetProperty("handoffLines").EnumerateArray())
        {
            Assert.False(line.GetProperty("performsPublish").GetBoolean());
            Assert.False(line.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(line.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(line.GetProperty("canPromoteProof").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(line.GetProperty("ownerNextAction").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(line.GetProperty("firstCommand").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(line.GetProperty("validatorCommand").GetString()));
            Assert.True(line.GetProperty("requiredRealInputs").GetArrayLength() >= 6);
            Assert.True(line.GetProperty("missingRealInputCount").GetInt32() >= 6);
            Assert.True(line.GetProperty("expectedArtifacts").GetArrayLength() >= 2);
            Assert.True(line.GetProperty("sourceArtifacts").GetArrayLength() >= 2);
            Assert.True(line.GetProperty("cannotUse").GetArrayLength() >= nonSubstitutes.Length);
        }

        JsonElement closeLine = root.GetProperty("handoffLines")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "release-issue-close-record");
        Assert.Contains("Test-ReleaseIssueCloseRecord.ps1", closeLine.GetProperty("validatorCommand").GetString(), StringComparison.Ordinal);
        Assert.Contains("releaseEvidenceBundle.sha256", closeLine.GetProperty("requiredRealInputs").EnumerateArray().Select(static item => item.GetString()));
        Assert.Contains("rollbackPlan.packageYankOrDeprecatePlan", closeLine.GetProperty("requiredRealInputs").EnumerateArray().Select(static item => item.GetString()));

        JsonElement packageConsumerLine = root.GetProperty("handoffLines")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "package-consumer-runtime");
        Assert.Contains("Test-ExternalRuntimeProofRecord.ps1", packageConsumerLine.GetProperty("validatorCommand").GetString(), StringComparison.Ordinal);
        Assert.Contains("clean external consumer", packageConsumerLine.GetProperty("ownerNextAction").GetString(), StringComparison.OrdinalIgnoreCase);

        string markdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "owner-proof-execution-handoff.md"));
        Assert.Contains("Owner Proof Execution Handoff", markdown, StringComparison.Ordinal);
        Assert.Contains("owner-proof-execution-handoff", markdown, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue=false", markdown, StringComparison.Ordinal);
        Assert.Contains("release-issue-close-record-template.json", markdown, StringComparison.Ordinal);
        Assert.Contains("Test-ReleaseIssueCloseRecord.ps1", markdown, StringComparison.Ordinal);

        using JsonDocument evidenceBundle = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidenceRoot = evidenceBundle.RootElement;

        JsonElement handoffEvidenceItem = evidenceRoot.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "owner-proof-execution-handoff");
        Assert.False(handoffEvidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("blocked-real-proof-required", handoffEvidenceItem.GetProperty("state").GetString(), StringComparison.Ordinal);
        Assert.Contains("guidance only", handoffEvidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

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
            "artifacts/final-release/owner-proof-execution-handoff.json",
            "artifacts/final-release/owner-proof-execution-handoff.md",
        })
        {
            Assert.Contains(artifactPath, sourceEvidence);
            Assert.Contains(artifactPath, sourceArtifacts);
        }

        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));
        Assert.Contains("owner-proof-execution-handoff", evidenceMarkdown, StringComparison.Ordinal);
        Assert.Contains("owner proof execution handoff ready lines: `0`", evidenceMarkdown, StringComparison.Ordinal);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "owner-proof-execution-handoff.md"));
        string releaseEvidenceArticle = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));

        Assert.Contains("articles/zh-cn/owner-proof-execution-handoff.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/owner-proof-execution-handoff.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("Owner proof execution handoff: `artifacts/final-release/owner-proof-execution-handoff.md`", docsIndex, StringComparison.Ordinal);
        Assert.Contains("Owner proof execution handoff artifact: `artifacts/final-release/owner-proof-execution-handoff.json`", readme, StringComparison.Ordinal);
        Assert.Contains("Owner proof execution handoff artifact：`artifacts/final-release/owner-proof-execution-handoff.json`", readmeZh, StringComparison.Ordinal);
        Assert.Contains("owner-proof-execution-handoff", releaseEvidenceArticle, StringComparison.Ordinal);

        foreach (string marker in new[]
        {
            "recordKind=owner-proof-execution-handoff",
            "handoffState=blocked-real-proof-required",
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

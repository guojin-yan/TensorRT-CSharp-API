using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OwnerExternalProofInputPreflightTests
{
    private static readonly string[] RequiredProofLineIds =
    {
        "owner-authorization",
        "package-consumer-runtime",
        "linux-runner-proof",
        "real-model-runtime",
        "post-publish-verification",
        "release-issue-close-record",
    };

    [Fact]
    public void OwnerExternalProofInputPreflightExportsCandidateAuditAndKeepsReleaseGateBlocked()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseIssueCloseRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseIssueCloseRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseClosePreflight.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofBackfillExecutionPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofExecutionHandoff.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerExternalProofInputPreflight.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument preflightDocument = ReadFinalReleaseJson("owner-external-proof-input-preflight.json");
        JsonElement root = preflightDocument.RootElement;

        Assert.Equal("owner-external-proof-input-preflight", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-proof-required", root.GetProperty("preflightState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal(RequiredProofLineIds.Length, root.GetProperty("proofLineCount").GetInt32());
        Assert.Equal(0, root.GetProperty("readyProofLineCount").GetInt32());
        Assert.Equal(RequiredProofLineIds.Length, root.GetProperty("blockedProofLineCount").GetInt32());
        JsonElement[] proofLines = root.GetProperty("proofLines").EnumerateArray().ToArray();
        int templateOnlyLineCount = proofLines.Count(static line =>
            line.GetProperty("candidateClassification").GetString() == "template-only");
        Assert.InRange(templateOnlyLineCount, 4, RequiredProofLineIds.Length);
        Assert.Equal(templateOnlyLineCount, root.GetProperty("templateOnlyLineCount").GetInt32());
        Assert.False(root.GetProperty("releaseEvidenceBundleCanPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("releaseEvidenceBundleCanCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("releaseClosePreflightCanCloseReleaseIssue").GetBoolean());

        string[] ids = proofLines
            .Select(static item => item.GetProperty("id").GetString()!)
            .Order(StringComparer.Ordinal)
            .ToArray();
        Assert.Equal(RequiredProofLineIds.Order(StringComparer.Ordinal).ToArray(), ids);

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

        foreach (JsonElement line in proofLines)
        {
            Assert.Contains(
                line.GetProperty("candidateClassification").GetString(),
                new[] { "template-only", "candidate-needs-owner-review" });
            Assert.False(line.GetProperty("performsPublish").GetBoolean());
            Assert.False(line.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(line.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(line.GetProperty("canPromoteProof").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(line.GetProperty("ownerNextAction").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(line.GetProperty("validatorCommand").GetString()));
            Assert.True(line.GetProperty("requiredRealInputCount").GetInt32() >= 6);
            Assert.True(line.GetProperty("missingRealInputCount").GetInt32() >= 6);
            Assert.True(line.GetProperty("expectedArtifactCount").GetInt32() >= 2);
            Assert.True(line.GetProperty("sourceArtifactCount").GetInt32() >= 2);

            string[] riskMarkers = line.GetProperty("riskMarkers")
                .EnumerateArray()
                .Select(static item => item.GetString()!)
                .ToArray();
            foreach (string risk in new[]
            {
                "local feed",
                "ProjectReference",
                "direct .nupkg reference",
                "missing log hash",
                "mismatched SHA256",
                "missing owner final close decision",
                "missing rollback plan",
            })
            {
                Assert.Contains(risk, riskMarkers);
            }
        }

        JsonElement closeLine = root.GetProperty("proofLines")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "release-issue-close-record");
        Assert.Contains("Test-ReleaseIssueCloseRecord.ps1", closeLine.GetProperty("validatorCommand").GetString(), StringComparison.Ordinal);
        Assert.Contains("releaseEvidenceBundle.sha256", closeLine.GetProperty("requiredRealInputs").EnumerateArray().Select(static item => item.GetString()));
        Assert.Contains("rollbackPlan.packageYankOrDeprecatePlan", closeLine.GetProperty("requiredRealInputs").EnumerateArray().Select(static item => item.GetString()));

        string markdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "owner-external-proof-input-preflight.md"));
        Assert.Contains("Owner External Proof Input Preflight", markdown, StringComparison.Ordinal);
        Assert.Contains("owner-external-proof-input-preflight", markdown, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue=false", markdown, StringComparison.Ordinal);
        Assert.Contains("local feed", markdown, StringComparison.Ordinal);
        Assert.Contains("ProjectReference", markdown, StringComparison.Ordinal);
        Assert.Contains("missing rollback plan", markdown, StringComparison.Ordinal);

        using JsonDocument evidenceBundle = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidenceRoot = evidenceBundle.RootElement;

        JsonElement preflightEvidenceItem = evidenceRoot.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "owner-external-proof-input-preflight");
        Assert.False(preflightEvidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("blocked-real-proof-required", preflightEvidenceItem.GetProperty("state").GetString(), StringComparison.Ordinal);
        Assert.Contains("candidate audit guidance only", preflightEvidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

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
            "artifacts/final-release/owner-external-proof-input-preflight.json",
            "artifacts/final-release/owner-external-proof-input-preflight.md",
        })
        {
            Assert.Contains(artifactPath, sourceEvidence);
            Assert.Contains(artifactPath, sourceArtifacts);
        }

        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));
        Assert.Contains("owner-external-proof-input-preflight", evidenceMarkdown, StringComparison.Ordinal);
        Assert.Contains("owner external proof input preflight ready lines: `0`", evidenceMarkdown, StringComparison.Ordinal);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "owner-external-proof-input-preflight.md"));
        string releaseEvidenceArticle = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));
        string handoffArticle = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "owner-proof-execution-handoff.md"));

        Assert.Contains("articles/zh-cn/owner-external-proof-input-preflight.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/owner-external-proof-input-preflight.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("Owner external proof input preflight: `artifacts/final-release/owner-external-proof-input-preflight.md`", docsIndex, StringComparison.Ordinal);
        Assert.Contains("Owner external proof input preflight artifact: `artifacts/final-release/owner-external-proof-input-preflight.json`", readme, StringComparison.Ordinal);
        Assert.Contains("Owner external proof input preflight artifact：`artifacts/final-release/owner-external-proof-input-preflight.json`", readmeZh, StringComparison.Ordinal);
        Assert.Contains("owner-external-proof-input-preflight", releaseEvidenceArticle, StringComparison.Ordinal);
        Assert.Contains("owner-external-proof-input-preflight", handoffArticle, StringComparison.Ordinal);

        foreach (string marker in new[]
        {
            "recordKind=owner-external-proof-input-preflight",
            "preflightState=blocked-real-proof-required",
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

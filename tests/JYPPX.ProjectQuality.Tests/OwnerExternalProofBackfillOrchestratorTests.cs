using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OwnerExternalProofBackfillOrchestratorTests
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
    public void OwnerExternalProofBackfillOrchestratorExportsBlockedCommandPlan()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofBackfillExecutionPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofExecutionHandoff.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerExternalProofInputPreflight.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofInputRepairPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofInputDraftPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerProofInputDraftPack.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerExternalProofBackfillOrchestrator.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerExternalProofBackfillOrchestrator.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument orchestratorDocument = ReadFinalReleaseJson("owner-external-proof-backfill-orchestrator.json");
        JsonElement root = orchestratorDocument.RootElement;

        Assert.Equal("owner-external-proof-backfill-orchestrator", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-external-proof-required", root.GetProperty("orchestratorState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal(RequiredLineIds.Length, root.GetProperty("backfillLineCount").GetInt32());
        Assert.Equal(RequiredLineIds.Length, root.GetProperty("blockedBackfillLineCount").GetInt32());

        string[] ids = root.GetProperty("backfillLines")
            .EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .Order(StringComparer.Ordinal)
            .ToArray();
        Assert.Equal(RequiredLineIds.Order(StringComparer.Ordinal).ToArray(), ids);

        foreach (JsonElement line in root.GetProperty("backfillLines").EnumerateArray())
        {
            Assert.Equal("blocked-owner-external-proof-required", line.GetProperty("backfillState").GetString());
            Assert.False(line.GetProperty("canPromoteProof").GetBoolean());
            Assert.False(line.GetProperty("performsPublish").GetBoolean());
            Assert.False(line.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(line.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(line.GetProperty("inputDraftPath").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(line.GetProperty("targetProofRecordPath").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(line.GetProperty("strictValidationCommand").GetString()));
            Assert.True(line.GetProperty("nextOwnerCommands").GetArrayLength() >= 3);
            Assert.Contains("template", line.GetProperty("blockedByNonProofMarkers").EnumerateArray().Select(static marker => marker.GetString()));
            Assert.Contains("ProjectReference", line.GetProperty("blockedByNonProofMarkers").EnumerateArray().Select(static marker => marker.GetString()));
        }

        JsonElement packageConsumer = root.GetProperty("backfillLines")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "package-consumer-runtime");
        string[] packageRules = packageConsumer.GetProperty("requiredRealInputRules")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        foreach (string rule in new[]
        {
            "cleanExternalConsumerIdentity",
            "noProjectReference",
            "noLocalFeedAsPublicProof",
            "managedNupkgSha256",
            "runtimeNupkgSha256",
            "runtimePackageKeyMatches",
            "compatibleHostMetadata",
            "smokeCommandIncludesRuntimePackageKey",
            "smokeLogPath",
            "smokeLogSha256",
        })
        {
            Assert.Contains(rule, packageRules);
        }
        Assert.Contains("outside this repository", string.Join(" ", packageConsumer.GetProperty("nextOwnerCommands").EnumerateArray().Select(static item => item.GetString())), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("-FailOnNotProof", packageConsumer.GetProperty("strictValidationCommand").GetString(), StringComparison.Ordinal);

        JsonElement closeRecord = root.GetProperty("backfillLines")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "release-issue-close-record");
        string[] closeRules = closeRecord.GetProperty("requiredRealInputRules")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        foreach (string rule in new[]
        {
            "releaseEvidenceBundleSha256",
            "releaseClosePreflightPathAndHash",
            "staleClaimsAuditPathAndHash",
            "postPublishProofValidationPathAndHash",
            "rollbackPlan",
            "ownerFinalCloseDecision",
            "strictCloseValidatorCommand",
        })
        {
            Assert.Contains(rule, closeRules);
        }
        Assert.Contains("-FailOnNotCloseReady", closeRecord.GetProperty("strictValidationCommand").GetString(), StringComparison.Ordinal);

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-external-proof-backfill-orchestrator-validation.json");
        JsonElement validationRoot = validationDocument.RootElement;
        Assert.Equal("owner-external-proof-backfill-orchestrator-validation", validationRoot.GetProperty("recordKind").GetString());
        Assert.Equal("valid-orchestrator-blocked-guidance", validationRoot.GetProperty("validationState").GetString());
        Assert.True(validationRoot.GetProperty("isValidOrchestrator").GetBoolean());
        Assert.False(validationRoot.GetProperty("performsPublish").GetBoolean());
        Assert.False(validationRoot.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validationRoot.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal(0, validationRoot.GetProperty("failedBlockerCount").GetInt32());

        using JsonDocument evidenceBundle = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidenceRoot = evidenceBundle.RootElement;
        Assert.Equal("blocked-owner-external-proof-required", evidenceRoot.GetProperty("ownerExternalProofBackfillOrchestratorState").GetString());
        Assert.Equal("valid-orchestrator-blocked-guidance", evidenceRoot.GetProperty("ownerExternalProofBackfillOrchestratorValidationState").GetString());
        Assert.Equal(RequiredLineIds.Length, evidenceRoot.GetProperty("ownerExternalProofBackfillOrchestratorLineCount").GetInt32());
        Assert.Equal(RequiredLineIds.Length, evidenceRoot.GetProperty("ownerExternalProofBackfillOrchestratorBlockedLineCount").GetInt32());
        Assert.False(evidenceRoot.GetProperty("ownerExternalProofBackfillOrchestratorCanPublishPublicly").GetBoolean());
        Assert.False(evidenceRoot.GetProperty("ownerExternalProofBackfillOrchestratorCanCloseReleaseIssue").GetBoolean());

        JsonElement evidenceItem = evidenceRoot.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "owner-external-proof-backfill-orchestrator");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("blocked-owner-external-proof-required", evidenceItem.GetProperty("state").GetString(), StringComparison.Ordinal);
        Assert.Contains("owner action guidance only", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceEvidence = evidenceRoot.GetProperty("sourceEvidence").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        string[] sourceArtifacts = evidenceRoot.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string artifactPath in new[]
        {
            "artifacts/final-release/owner-external-proof-backfill-orchestrator.json",
            "artifacts/final-release/owner-external-proof-backfill-orchestrator.md",
            "artifacts/final-release/owner-external-proof-backfill-orchestrator-validation.json",
            "artifacts/final-release/owner-external-proof-backfill-orchestrator-validation.md",
        })
        {
            Assert.Contains(artifactPath, sourceEvidence);
            Assert.Contains(artifactPath, sourceArtifacts);
        }

        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));
        Assert.Contains("owner-external-proof-backfill-orchestrator", evidenceMarkdown, StringComparison.Ordinal);
        Assert.Contains("owner external proof backfill blocked lines: `6`", evidenceMarkdown, StringComparison.Ordinal);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "owner-external-proof-backfill-orchestrator.md"));
        string releaseEvidenceArticle = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));

        Assert.Contains("articles/zh-cn/owner-external-proof-backfill-orchestrator.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/owner-external-proof-backfill-orchestrator.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("Owner external proof backfill orchestrator: `artifacts/final-release/owner-external-proof-backfill-orchestrator.md`", docsIndex, StringComparison.Ordinal);
        Assert.Contains("Owner external proof backfill orchestrator artifact: `artifacts/final-release/owner-external-proof-backfill-orchestrator.json`", readme, StringComparison.Ordinal);
        Assert.Contains("Owner external proof backfill orchestrator artifact：`artifacts/final-release/owner-external-proof-backfill-orchestrator.json`", readmeZh, StringComparison.Ordinal);
        Assert.Contains("owner-external-proof-backfill-orchestrator", releaseEvidenceArticle, StringComparison.Ordinal);

        foreach (string marker in new[]
        {
            "recordKind=owner-external-proof-backfill-orchestrator",
            "orchestratorState=blocked-owner-external-proof-required",
            "performsPublish=false",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "package-consumer-runtime",
            "release-issue-close-record",
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

    private static string RunPowerShell(string scriptPath, params string[] arguments)
    {
        using Process process = new();
        process.StartInfo.FileName = "pwsh";
        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(scriptPath);
        foreach (string argument in arguments)
        {
            process.StartInfo.ArgumentList.Add(argument);
        }

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

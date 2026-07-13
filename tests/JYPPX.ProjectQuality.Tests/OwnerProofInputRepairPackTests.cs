using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OwnerProofInputRepairPackTests
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
    public void OwnerProofInputRepairPackExportsRepairChecklistAndKeepsReleaseGateBlocked()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseIssueCloseRecordTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseIssueCloseRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseClosePreflight.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofBackfillExecutionPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofExecutionHandoff.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerExternalProofInputPreflight.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerProofInputRepairPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument repairDocument = ReadFinalReleaseJson("owner-proof-input-repair-pack.json");
        JsonElement root = repairDocument.RootElement;

        Assert.Equal("owner-proof-input-repair-pack", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-input-repair-required", root.GetProperty("repairPackState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal(RequiredLineIds.Length, root.GetProperty("repairItemCount").GetInt32());
        Assert.Equal(0, root.GetProperty("readyRepairItemCount").GetInt32());
        Assert.Equal(RequiredLineIds.Length, root.GetProperty("blockedRepairItemCount").GetInt32());
        JsonElement[] repairItems = root.GetProperty("repairItems").EnumerateArray().ToArray();
        int templateOnlyRepairItemCount = repairItems.Count(static item =>
            item.GetProperty("currentCandidateClassification").GetString() == "template-only");
        Assert.InRange(templateOnlyRepairItemCount, 4, RequiredLineIds.Length);
        Assert.Equal(templateOnlyRepairItemCount, root.GetProperty("templateOnlyRepairItemCount").GetInt32());
        Assert.True(root.GetProperty("requiredRealInputCount").GetInt32() >= RequiredLineIds.Length * 6);
        Assert.True(root.GetProperty("missingRealInputCount").GetInt32() >= RequiredLineIds.Length * 6);

        string[] ids = repairItems
            .Select(static item => item.GetProperty("id").GetString()!)
            .Order(StringComparer.Ordinal)
            .ToArray();
        Assert.Equal(RequiredLineIds.Order(StringComparer.Ordinal).ToArray(), ids);

        foreach (JsonElement item in repairItems)
        {
            Assert.False(item.GetProperty("canPromoteProof").GetBoolean());
            Assert.False(item.GetProperty("inputDraftIsProof").GetBoolean());
            Assert.False(item.GetProperty("performsPublish").GetBoolean());
            Assert.False(item.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.Contains(
                item.GetProperty("currentCandidateClassification").GetString(),
                new[] { "template-only", "candidate-needs-owner-review" });
            Assert.Equal("blocked-real-input-repair-required", item.GetProperty("repairState").GetString());
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("firstRepairCommand").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("validatorCommand").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("inputDraftPath").GetString()));
            Assert.True(item.GetProperty("requiredRealInputs").GetArrayLength() >= 6);
            Assert.True(item.GetProperty("fieldsRequiringExistingFiles").GetArrayLength() >= 1);
            Assert.True(item.GetProperty("cannotUseMarkers").GetArrayLength() >= 8);
            Assert.Contains("template", item.GetProperty("cannotUseMarkers").EnumerateArray().Select(static marker => marker.GetString()));
            Assert.Contains("ProjectReference", item.GetProperty("cannotUseMarkers").EnumerateArray().Select(static marker => marker.GetString()));
            Assert.Contains("release-issue-close-record-template.json", item.GetProperty("cannotUseMarkers").EnumerateArray().Select(static marker => marker.GetString()));
        }

        JsonElement packageConsumer = root.GetProperty("repairItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "package-consumer-runtime");
        Assert.Contains("Test-ExternalRuntimeProofRecord.ps1", packageConsumer.GetProperty("validatorCommand").GetString(), StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-record.input-draft.json", packageConsumer.GetProperty("inputDraftPath").GetString(), StringComparison.Ordinal);
        Assert.True(packageConsumer.GetProperty("fieldsRequiringCleanConsumerEvidence").GetArrayLength() >= 1);
        Assert.True(packageConsumer.GetProperty("fieldsRequiringSha256").GetArrayLength() >= 1);

        JsonElement closeRecord = root.GetProperty("repairItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "release-issue-close-record");
        Assert.Contains("Test-ReleaseIssueCloseRecord.ps1", closeRecord.GetProperty("validatorCommand").GetString(), StringComparison.Ordinal);
        Assert.Contains("releaseEvidenceBundle.sha256", closeRecord.GetProperty("requiredRealInputs").EnumerateArray().Select(static value => value.GetString()));
        Assert.True(closeRecord.GetProperty("fieldsRequiringRollbackPlan").GetArrayLength() >= 1);
        Assert.True(closeRecord.GetProperty("fieldsRequiringOwnerDecision").GetArrayLength() >= 1);

        string markdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "owner-proof-input-repair-pack.md"));
        Assert.Contains("Owner Proof Input Repair Pack", markdown, StringComparison.Ordinal);
        Assert.Contains("owner-proof-input-repair-pack", markdown, StringComparison.Ordinal);
        Assert.Contains("inputDraftIsProof", markdown, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue", markdown, StringComparison.Ordinal);
        Assert.Contains("repair pack、repair markdown、input draft", markdown, StringComparison.Ordinal);

        using JsonDocument evidenceBundle = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidenceRoot = evidenceBundle.RootElement;
        Assert.Equal("blocked-real-input-repair-required", evidenceRoot.GetProperty("ownerProofInputRepairPackState").GetString());
        Assert.Equal(RequiredLineIds.Length, evidenceRoot.GetProperty("ownerProofInputRepairPackItemCount").GetInt32());
        Assert.Equal(RequiredLineIds.Length, evidenceRoot.GetProperty("ownerProofInputRepairPackBlockedItemCount").GetInt32());
        Assert.False(evidenceRoot.GetProperty("ownerProofInputRepairPackPerformsPublish").GetBoolean());
        Assert.False(evidenceRoot.GetProperty("ownerProofInputRepairPackCanPublishPublicly").GetBoolean());
        Assert.False(evidenceRoot.GetProperty("ownerProofInputRepairPackCanCloseReleaseIssue").GetBoolean());

        JsonElement repairEvidenceItem = evidenceRoot.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "owner-proof-input-repair-pack");
        Assert.False(repairEvidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("blocked-real-input-repair-required", repairEvidenceItem.GetProperty("state").GetString(), StringComparison.Ordinal);
        Assert.Contains("repair guidance only", repairEvidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

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
            "artifacts/final-release/owner-proof-input-repair-pack.json",
            "artifacts/final-release/owner-proof-input-repair-pack.md",
        })
        {
            Assert.Contains(artifactPath, sourceEvidence);
            Assert.Contains(artifactPath, sourceArtifacts);
        }

        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));
        Assert.Contains("owner-proof-input-repair-pack", evidenceMarkdown, StringComparison.Ordinal);
        Assert.Contains("owner proof input repair blocked items: `6`", evidenceMarkdown, StringComparison.Ordinal);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "owner-proof-input-repair-pack.md"));
        string releaseEvidenceArticle = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));

        Assert.Contains("articles/zh-cn/owner-proof-input-repair-pack.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/owner-proof-input-repair-pack.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("Owner proof input repair pack: `artifacts/final-release/owner-proof-input-repair-pack.md`", docsIndex, StringComparison.Ordinal);
        Assert.Contains("Owner proof input repair pack artifact: `artifacts/final-release/owner-proof-input-repair-pack.json`", readme, StringComparison.Ordinal);
        Assert.Contains("Owner proof input repair pack artifact：`artifacts/final-release/owner-proof-input-repair-pack.json`", readmeZh, StringComparison.Ordinal);
        Assert.Contains("owner-proof-input-repair-pack", releaseEvidenceArticle, StringComparison.Ordinal);

        foreach (string marker in new[]
        {
            "recordKind=owner-proof-input-repair-pack",
            "repairPackState=blocked-real-input-repair-required",
            "performsPublish=false",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "placeholder",
            "SHA256",
            "clean consumer",
            "rollback plan",
            "input draft",
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

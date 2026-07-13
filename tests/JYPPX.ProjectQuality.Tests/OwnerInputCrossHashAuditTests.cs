using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerInputCrossHashAuditTests
{
    [Fact]
    public void OwnerInputCrossHashAuditStaysBlockedAndAuditable()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerReleaseExecutionPackage.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerReleaseExecutionPackage.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseIssueFinalCloseDecisionTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseIssueFinalCloseDecision.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalEvidenceFreeze.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalEvidenceFreeze.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealExternalProofOverlayPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-RealExternalProofOverlayPack.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseIssueCloseRecordOverlayCandidate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseIssueCloseRecordOverlayCandidate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerExternalExecutionResultBackfillKit.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerExternalExecutionResultBackfillKit.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerInputCrossHashAudit.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerInputCrossHashAudit.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument auditDocument = ReadFinalReleaseJson("owner-input-cross-hash-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("owner-input-cross-hash-audit", audit.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-input-cross-hash-audit-owner-proof-required", audit.GetProperty("auditState").GetString());
        Assert.True(audit.GetProperty("auditLineCount").GetInt32() >= 4);
        Assert.Equal(0, audit.GetProperty("mismatchedHashCount").GetInt32());
        Assert.False(audit.GetProperty("performsPublish").GetBoolean());
        Assert.False(audit.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(audit.GetProperty("canCloseReleaseIssue").GetBoolean());

        string[] lineIds = audit.GetProperty("auditLines")
            .EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();
        Assert.Contains("release-evidence-bundle", lineIds);
        Assert.Contains("final-evidence-freeze", lineIds);
        Assert.Contains("real-external-proof-overlay-pack-validation", lineIds);
        Assert.Contains("owner-external-execution-result-backfill-kit-validation", lineIds);

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-input-cross-hash-audit-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("owner-input-cross-hash-audit-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-input-cross-hash-audit-owner-proof-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("mismatchedHashCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("blocked-owner-input-cross-hash-audit-owner-proof-required", evidence.GetProperty("ownerInputCrossHashAuditState").GetString());
        Assert.Equal("blocked-owner-input-cross-hash-audit-owner-proof-required", evidence.GetProperty("ownerInputCrossHashAuditValidationState").GetString());
        Assert.Equal(0, evidence.GetProperty("ownerInputCrossHashAuditFailedBlockerCount").GetInt32());
        Assert.True(evidence.GetProperty("ownerInputCrossHashAuditFailedActionRequiredCount").GetInt32() >= 1);
        Assert.Equal(0, evidence.GetProperty("ownerInputCrossHashAuditMismatchedHashCount").GetInt32());
        Assert.False(evidence.GetProperty("ownerInputCrossHashAuditCanCloseReleaseIssue").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "owner-input-cross-hash-audit");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("hash matches cannot substitute real external proof", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/owner-input-cross-hash-audit.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-input-cross-hash-audit-validation.json", sourceArtifacts);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "owner-input-cross-hash-audit.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("articles/zh-cn/owner-input-cross-hash-audit.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/owner-input-cross-hash-audit.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("owner-input-cross-hash-audit", readme, StringComparison.Ordinal);
        Assert.Contains("owner-input-cross-hash-audit", readmeZh, StringComparison.Ordinal);
        Assert.Contains("mismatchedHashCount=0", article, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("owner input cross-hash audit mismatched hashes: `0`", evidenceMarkdown, StringComparison.Ordinal);
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

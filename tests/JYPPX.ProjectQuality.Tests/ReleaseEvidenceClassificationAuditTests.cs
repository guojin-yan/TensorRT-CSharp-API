using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleaseEvidenceClassificationAuditTests
{
    [Fact]
    public void ScriptDocsAndReleaseBundleDescribeClassificationAuditBoundary()
    {
        string script = ReadSource("eng", "Test-ReleaseEvidenceClassificationAudit.ps1");
        string releaseEvidenceDoc = ReadSource("docs", "articles", "zh-cn", "release-evidence-bundle.md");
        string auditDoc = ReadSource("docs", "articles", "zh-cn", "release-evidence-classification-audit.md");
        string index = ReadSource("docs", "index.md");
        string toc = ReadSource("docs", "toc.yml");

        Assert.Contains("release-evidence-classification-audit.json", script, StringComparison.Ordinal);
        Assert.Contains("release-evidence-classification-audit.md", script, StringComparison.Ordinal);
        Assert.Contains("recordKind = \"release-evidence-classification-audit\"", script, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly = $false", script, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue = $false", script, StringComparison.Ordinal);
        Assert.Contains("isRuntimeExecutionProof = $false", script, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", script, StringComparison.Ordinal);
        Assert.Contains("approvesPublicRelease = $false", script, StringComparison.Ordinal);
        Assert.Contains("design-gate-required planning input", script, StringComparison.Ordinal);
        Assert.Contains("keep-deferred boundary disclosure", script, StringComparison.Ordinal);
        Assert.Contains("not runtime proof", script, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not permission to delete deferred records", script, StringComparison.OrdinalIgnoreCase);

        Assert.Contains("Test-ReleaseEvidenceClassificationAudit.ps1 -Strict", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("release-evidence-classification-audit.json", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("classification-audit-passed-non-proof-boundaries-intact", releaseEvidenceDoc, StringComparison.Ordinal);
        Assert.Contains("release-evidence-classification-audit", auditDoc, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly=false", auditDoc, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue=false", auditDoc, StringComparison.Ordinal);
        Assert.Contains("isRuntimeExecutionProof=false", auditDoc, StringComparison.Ordinal);
        Assert.Contains("Release Evidence Classification Audit", index, StringComparison.Ordinal);
        Assert.Contains("release-evidence-classification-audit.md", index, StringComparison.Ordinal);
        Assert.Contains("Release Evidence Classification Audit", toc, StringComparison.Ordinal);
        Assert.Contains("release-evidence-classification-audit.md", toc, StringComparison.Ordinal);
    }

    [Fact]
    public void StrictAuditPassesOnlyAsNonProofReleaseClassificationGate()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseEvidenceClassificationAudit.ps1"), "-Strict");
        Assert.Contains("Release evidence classification audit written", output, StringComparison.Ordinal);
        Assert.Contains("AuditState=classification-audit-passed-non-proof-boundaries-intact", output, StringComparison.Ordinal);

        using JsonDocument bundleDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement bundle = bundleDocument.RootElement;
        Assert.True(bundle.GetProperty("ownerRealProofStagingWorkspaceContractLaneCount").GetInt32() >= 5);
        Assert.True(bundle.GetProperty("ownerRealProofStagingWorkspaceContractFileCount").GetInt32() >= 37);
        Assert.True(bundle.GetProperty("ownerRealProofStagingWorkspaceContractSha256RequiredFileCount").GetInt32() >= 37);
        Assert.True(bundle.GetProperty("ownerRealProofStagingWorkspaceContractForbiddenSubstituteCount").GetInt32() >= 300);
        Assert.True(bundle.GetProperty("ownerRealProofStagingWorkspaceContractExternalWorkspaceRequired").GetBoolean());
        Assert.False(bundle.GetProperty("ownerRealProofStagingWorkspaceReadyForStrictImport").GetBoolean());
        Assert.True(bundle.GetProperty("ownerRealProofStagingWorkspaceImportLaneCount").GetInt32() >= 5);
        Assert.True(bundle.GetProperty("ownerRealProofStagingWorkspaceMappingCount").GetInt32() >= 37);
        Assert.Equal(0, bundle.GetProperty("ownerRealProofStagingWorkspaceExistingFileCount").GetInt32());
        Assert.True(bundle.GetProperty("ownerRealProofStagingWorkspaceImportSha256RequiredFileCount").GetInt32() >= 37);
        Assert.Equal(0, bundle.GetProperty("ownerRealProofStagingWorkspaceSha256ValidFileCount").GetInt32());
        Assert.False(bundle.GetProperty("ownerRealProofStagingWorkspaceRootOutsideRepository").GetBoolean());
        Assert.False(bundle.GetProperty("ownerRealProofStagingWorkspaceRequireExistingFiles").GetBoolean());
        Assert.False(bundle.GetProperty("ownerRealProofStagingWorkspaceRequireHashMatch").GetBoolean());
        Assert.True(bundle.GetProperty("ownerRealProofStagingWorkspaceFailedActionRequiredCount").GetInt32() >= 38);
        Assert.False(bundle.GetProperty("articlePublicationProofFromStagingWorkspaceShapeValid").GetBoolean());
        Assert.Equal(0, bundle.GetProperty("articlePublicationProofFromStagingWorkspaceRecordCount").GetInt32());
        Assert.True(bundle.GetProperty("articlePublicationProofFromStagingWorkspaceFailedActionRequiredCount").GetInt32() >= 3);
        Assert.False(bundle.GetProperty("yoloVisionRealModelProofFromStagingWorkspaceShapeValid").GetBoolean());
        Assert.Equal(12, bundle.GetProperty("yoloVisionRealModelProofFromStagingWorkspaceAssetFileCount").GetInt32());
        Assert.Equal(0, bundle.GetProperty("yoloVisionRealModelProofFromStagingWorkspaceExistingAssetFileCount").GetInt32());
        Assert.True(bundle.GetProperty("yoloVisionRealModelProofFromStagingWorkspaceFailedActionRequiredCount").GetInt32() >= 12);
        JsonElement stagingContractItem = bundle.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "owner-real-proof-staging-workspace-contract");
        Assert.Contains("lanes=", stagingContractItem.GetProperty("state").GetString(), StringComparison.Ordinal);
        Assert.Contains("sha256Required=", stagingContractItem.GetProperty("state").GetString(), StringComparison.Ordinal);
        JsonElement stagingImportItem = bundle.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "owner-real-proof-staging-workspace-import");
        Assert.Contains("mappings=", stagingImportItem.GetProperty("state").GetString(), StringComparison.Ordinal);
        Assert.Contains("rootOutsideRepository=False", stagingImportItem.GetProperty("state").GetString(), StringComparison.Ordinal);

        using JsonDocument audit = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement root = audit.RootElement;

        Assert.Equal("release-evidence-classification-audit", root.GetProperty("recordKind").GetString());
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", root.GetProperty("auditState").GetString());
        Assert.True(root.GetProperty("auditPassed").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("approvesPublicRelease").GetBoolean());
        Assert.Equal(0, root.GetProperty("findingCount").GetInt32());
        Assert.True(root.GetProperty("requiredNonProofItemCount").GetInt32() >= 50);
        Assert.True(root.GetProperty("auditedNonProofItemCount").GetInt32() >= 50);
        Assert.Contains("not runtime proof", root.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("permission to delete deferred records", root.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        Assert.Contains(root.GetProperty("requiredNonSubstituteMarkers").EnumerateArray(), static marker =>
            marker.GetString() == "design-gate-required planning input");
        Assert.Contains(root.GetProperty("requiredNonSubstituteMarkers").EnumerateArray(), static marker =>
            marker.GetString() == "keep-deferred boundary disclosure");
        Assert.Contains(root.GetProperty("auditedItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "calibrator-metadata-design-gate" &&
            item.GetProperty("passed").GetBoolean() == false &&
            item.GetProperty("designGateOrPrecheck").GetBoolean() &&
            item.GetProperty("hasNonProofBoundary").GetBoolean());
        Assert.Contains(root.GetProperty("auditedItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "package-consumer-runtime-proof-record" &&
            item.GetProperty("passed").GetBoolean() == false &&
            item.GetProperty("hasNonProofBoundary").GetBoolean());
        Assert.Contains(root.GetProperty("auditedItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "cuda-device-initialization-local-smoke-classification" &&
            item.GetProperty("passed").GetBoolean() == false &&
            item.GetProperty("hasNonProofBoundary").GetBoolean());
        Assert.Contains(root.GetProperty("auditedItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "clean-consumer-proof-execution-bundle" &&
            item.GetProperty("passed").GetBoolean() == false &&
            item.GetProperty("hasNonProofBoundary").GetBoolean());
        Assert.Contains(root.GetProperty("auditedItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "clean-consumer-external-proof-closure-pack" &&
            item.GetProperty("passed").GetBoolean() == false &&
            item.GetProperty("hasNonProofBoundary").GetBoolean());
        foreach (string id in new[]
        {
            "external-clean-consumer-execution-workspace-contract",
            "external-clean-consumer-owner-command-pack",
            "external-clean-consumer-execution-result-import",
            "external-clean-consumer-execution-result-candidate",
            "post-publish-clean-consumer-proof-result-import",
            "post-publish-clean-consumer-proof-result-candidate",
            "final-owner-real-proof-execution-package",
            "final-owner-real-proof-gap-matrix",
            "final-owner-real-proof-convergence-gate",
            "owner-real-proof-evidence-backfill-package",
            "owner-real-proof-staging-workspace-contract",
            "owner-real-proof-staging-workspace-import",
            "article-publication-proof-from-staging-workspace",
            "yolovision-real-model-proof-from-staging-workspace",
            "final-owner-rollback-review-import",
            "final-owner-close-decision-import"
        })
        {
            Assert.Contains(root.GetProperty("auditedItems").EnumerateArray(), item =>
                item.GetProperty("id").GetString() == id &&
                item.GetProperty("passed").GetBoolean() == false &&
                item.GetProperty("hasNonProofBoundary").GetBoolean());
        }
        Assert.Contains(root.GetProperty("requiredNonSubstituteMarkers").EnumerateArray(), static marker =>
            marker.GetString() == "clean-consumer-proof-execution-bundle");
        Assert.Contains(root.GetProperty("requiredNonSubstituteMarkers").EnumerateArray(), static marker =>
            marker.GetString() == "clean-consumer-external-proof-closure-pack");
        Assert.Contains(root.GetProperty("requiredNonSubstituteMarkers").EnumerateArray(), static marker =>
            marker.GetString() == "external clean consumer execution workspace contract");
        Assert.Contains(root.GetProperty("requiredNonSubstituteMarkers").EnumerateArray(), static marker =>
            marker.GetString() == "post-publish clean consumer proof result import");
        Assert.Contains(root.GetProperty("requiredNonSubstituteMarkers").EnumerateArray(), static marker =>
            marker.GetString() == "final owner real proof execution package");
        Assert.Contains(root.GetProperty("requiredNonSubstituteMarkers").EnumerateArray(), static marker =>
            marker.GetString() == "final owner real proof gap matrix");
        Assert.Contains(root.GetProperty("requiredNonSubstituteMarkers").EnumerateArray(), static marker =>
            marker.GetString() == "final owner real proof convergence gate");
        Assert.Contains(root.GetProperty("requiredNonSubstituteMarkers").EnumerateArray(), static marker =>
            marker.GetString() == "owner real proof evidence backfill package");
        Assert.Contains(root.GetProperty("requiredNonSubstituteMarkers").EnumerateArray(), static marker =>
            marker.GetString() == "owner real proof staging workspace contract");
        Assert.Contains(root.GetProperty("requiredNonSubstituteMarkers").EnumerateArray(), static marker =>
            marker.GetString() == "owner real proof staging workspace import");
        Assert.Contains(root.GetProperty("requiredNonSubstituteMarkers").EnumerateArray(), static marker =>
            marker.GetString() == "article publication proof from staging workspace");
        Assert.Contains(root.GetProperty("requiredNonSubstituteMarkers").EnumerateArray(), static marker =>
            marker.GetString() == "yolovision real model proof from staging workspace");
        Assert.Contains(root.GetProperty("requiredNonSubstituteMarkers").EnumerateArray(), static marker =>
            marker.GetString() == "final owner rollback review import");
        Assert.Contains(root.GetProperty("requiredNonSubstituteMarkers").EnumerateArray(), static marker =>
            marker.GetString() == "final owner close decision import");
        Assert.Contains(root.GetProperty("requiredNonSubstituteMarkers").EnumerateArray(), static marker =>
            marker.GetString() == "owner-action-required");

        string markdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-classification-audit.md"));
        Assert.Contains("## Audited Non-Proof Items", markdown, StringComparison.Ordinal);
        Assert.Contains("No classification findings", markdown, StringComparison.Ordinal);
        Assert.Contains("calibrator-metadata-design-gate", markdown, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime-proof-record", markdown, StringComparison.Ordinal);
        Assert.Contains("cuda-device-initialization-local-smoke-classification", markdown, StringComparison.Ordinal);
        Assert.Contains("clean-consumer-proof-execution-bundle", markdown, StringComparison.Ordinal);
        Assert.Contains("clean-consumer-external-proof-closure-pack", markdown, StringComparison.Ordinal);
        Assert.Contains("external-clean-consumer-execution-result-import", markdown, StringComparison.Ordinal);
        Assert.Contains("post-publish-clean-consumer-proof-result-import", markdown, StringComparison.Ordinal);
        Assert.Contains("final-owner-real-proof-execution-package", markdown, StringComparison.Ordinal);
        Assert.Contains("final-owner-real-proof-gap-matrix", markdown, StringComparison.Ordinal);
        Assert.Contains("final-owner-real-proof-convergence-gate", markdown, StringComparison.Ordinal);
        Assert.Contains("owner-real-proof-evidence-backfill-package", markdown, StringComparison.Ordinal);
        Assert.Contains("owner-real-proof-staging-workspace-contract", markdown, StringComparison.Ordinal);
        Assert.Contains("owner-real-proof-staging-workspace-import", markdown, StringComparison.Ordinal);
        Assert.Contains("article-publication-proof-from-staging-workspace", markdown, StringComparison.Ordinal);
        Assert.Contains("yolovision-real-model-proof-from-staging-workspace", markdown, StringComparison.Ordinal);
        Assert.Contains("final-owner-rollback-review-import", markdown, StringComparison.Ordinal);
        Assert.Contains("final-owner-close-decision-import", markdown, StringComparison.Ordinal);
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static string ReadSource(params string[] segments)
    {
        return File.ReadAllText(Path.Combine(RepositoryPaths.Root, Path.Combine(segments)));
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
    {
        using Process process = new()
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "pwsh",
                WorkingDirectory = RepositoryPaths.Root,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            },
        };

        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(scriptPath);
        foreach (string argument in arguments)
        {
            process.StartInfo.ArgumentList.Add(argument);
        }

        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException(
                $"PowerShell script failed with exit code {process.ExitCode}:{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        }

        return stdout;
    }
}

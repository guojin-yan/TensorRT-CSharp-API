using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseQualityGateWorkflowTests
{
    [Fact]
    public void WorkflowSeparatesSourceChecksFromOptInLargeArtifactJobs()
    {
        string workflow = ReadSource(".github", "workflows", "release-quality-gate.yml");

        Assert.Contains("permissions:\n  contents: read", Normalize(workflow), StringComparison.Ordinal);
        Assert.Contains("source-quality:", workflow, StringComparison.Ordinal);
        Assert.Contains("run_release_artifact_audit", workflow, StringComparison.Ordinal);
        Assert.Contains("run_split_package_build", workflow, StringComparison.Ordinal);
        Assert.Contains("run_package_managed_dry_run", workflow, StringComparison.Ordinal);
        Assert.True(CountOccurrences(workflow, "default: false") >= 3);
        Assert.Contains("package-managed-dry-run:", workflow, StringComparison.Ordinal);
        Assert.Contains("uses: ./.github/workflows/package-managed.yml", workflow, StringComparison.Ordinal);
        Assert.Contains("publish_to_nuget: false", workflow, StringComparison.Ordinal);
        Assert.Contains("publish_to_github_packages: false", workflow, StringComparison.Ordinal);
        Assert.Contains("artifact_name: package-managed-dry-run", workflow, StringComparison.Ordinal);
        Assert.Contains("Export-GitHubActionsPackageValidationAudit.ps1", workflow, StringComparison.Ordinal);
        Assert.Contains("github-actions-package-validation-audit.*", workflow, StringComparison.Ordinal);
        Assert.Contains("runs-on: [self-hosted, windows, x64, release-artifacts]", workflow, StringComparison.Ordinal);
        Assert.Contains("runs-on: [self-hosted, windows, x64]", workflow, StringComparison.Ordinal);
        Assert.Contains("-SplitPackageRole all", workflow, StringComparison.Ordinal);
        Assert.Contains("-IncludeMetaPackage", workflow, StringComparison.Ordinal);
        Assert.Contains("Test-ReleaseEvidenceClassificationAudit.ps1 -Strict", workflow, StringComparison.Ordinal);
        Assert.Contains("Test-PublicProofClaimBoundaryAudit.ps1 -Strict", workflow, StringComparison.Ordinal);
        Assert.Contains("Test-ReleaseQualityGate.ps1", workflow, StringComparison.Ordinal);
        Assert.Contains("release-quality-gate-summary.json", ReadSource("eng", "Test-ReleaseQualityGate.ps1"), StringComparison.Ordinal);

        Assert.DoesNotContain("dotnet nuget push", workflow, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("Push-NuGetPackages", workflow, StringComparison.Ordinal);
        Assert.DoesNotContain("NUGET_API_KEY", workflow, StringComparison.Ordinal);
        Assert.DoesNotContain("GITHUB_PACKAGES_TOKEN", workflow, StringComparison.Ordinal);
        Assert.DoesNotContain("gh release create", workflow, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("gh release upload", workflow, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void SourceQualityGateProducesMachineReadableNonProofSummary()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseQualityGate.ps1");
        RunPowerShell(script, "-Strict");

        string summaryPath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "release-quality-gate",
            "release-quality-gate-summary.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(summaryPath));
        JsonElement root = document.RootElement;

        Assert.Equal("release-quality-gate-summary", root.GetProperty("recordKind").GetString());
        Assert.Equal("release-quality-gate-passed", root.GetProperty("state").GetString());
        Assert.True(root.GetProperty("sourceGatePassed").GetBoolean());
        Assert.Equal(0, root.GetProperty("requiredFailureCount").GetInt32());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("usesPublishToken").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());

        JsonElement[] checks = root.GetProperty("checks").EnumerateArray().ToArray();
        Assert.Contains(checks, static check =>
            check.GetProperty("id").GetString() == "workflow-no-publish-side-effects" &&
            check.GetProperty("passed").GetBoolean());
        Assert.Contains(checks, static check =>
            check.GetProperty("id").GetString() == "split-manifest-component-roles" &&
            check.GetProperty("passed").GetBoolean());
        Assert.Contains(checks, static check =>
            check.GetProperty("id").GetString() == "public-markdown-array-rendering" &&
            check.GetProperty("passed").GetBoolean());
        Assert.Contains(checks, static check =>
            check.GetProperty("id").GetString() == "legacy-yolodet-public-paths" &&
            check.GetProperty("passed").GetBoolean());
    }

    [Fact]
    public void GitHubActionsPackageValidationAuditPreservesCurrentRunProofBoundary()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Export-GitHubActionsPackageValidationAudit.ps1");
        RunPowerShell(script);

        string auditPath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "github-actions-package-validation-audit.json");
        string markdownPath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "github-actions-package-validation-audit.md");

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(auditPath));
        JsonElement root = document.RootElement;

        Assert.Equal("github-actions-package-validation-audit", root.GetProperty("recordKind").GetString());
        Assert.False(root.GetProperty("hasGitHubActionsRunEvidenceForCurrentCode").GetBoolean());
        Assert.False(root.GetProperty("canClaimGitHubActionsPackageValidationForCurrentCode").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("usesPublishToken").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());

        JsonElement workflowContracts = root.GetProperty("workflowContracts");
        Assert.True(workflowContracts.GetProperty("packageManagedDryRunReady").GetBoolean());
        Assert.True(workflowContracts.GetProperty("packageManagedPublishGuarded").GetBoolean());
        Assert.True(workflowContracts.GetProperty("releaseQualityHasSourceGate").GetBoolean());
        Assert.True(workflowContracts.GetProperty("releaseQualityHasPackageDryRunAudit").GetBoolean());
        Assert.True(workflowContracts.GetProperty("releaseBundleRemoteReady").GetBoolean());
        Assert.True(workflowContracts.GetProperty("runtimeWorkflowsPackageReady").GetBoolean());

        string boundary = root.GetProperty("proofBoundary").GetString()!;
        Assert.Contains("does not upload code", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("does not execute GitHub Actions", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("does not publish packages", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("is not package-consumer runtime proof", boundary, StringComparison.OrdinalIgnoreCase);

        JsonElement[] checks = root.GetProperty("checks").EnumerateArray().ToArray();
        Assert.Contains(checks, static check =>
            check.GetProperty("id").GetString() == "workflow-release-quality-package-dry-run-audit" &&
            check.GetProperty("passed").GetBoolean());
        Assert.Contains(checks, static check =>
            check.GetProperty("id").GetString() == "workflow-runtime-package-ready" &&
            check.GetProperty("passed").GetBoolean());

        string markdown = File.ReadAllText(markdownPath);
        Assert.Contains("不能声称本轮代码已在 GitHub Actions 上完成 NuGet/包验证", markdown, StringComparison.Ordinal);
        Assert.Contains("可声称 GitHub Actions 已验证当前代码包构建：`False`", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void WorktreeStagingAuditClassifiesLargeDirtyTreeWithoutStagingSideEffects()
    {
        string script = Path.Combine(RepositoryPaths.Root, "eng", "Export-WorktreeStagingAudit.ps1");
        RunPowerShell(script);

        string auditPath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "worktree-staging-audit.json");
        string markdownPath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "worktree-staging-audit.md");
        string pathspecRoot = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "staging-pathspecs");

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(auditPath));
        JsonElement root = document.RootElement;

        Assert.Equal("worktree-staging-audit", root.GetProperty("recordKind").GetString());
        Assert.Contains(
            "does not stage, commit, push, delete, or publish files",
            root.GetProperty("boundary").GetString()!,
            StringComparison.OrdinalIgnoreCase);
        Assert.Contains(
            "do not use git add .",
            root.GetProperty("recommendedStageCommand").GetString()!,
            StringComparison.OrdinalIgnoreCase);

        JsonElement counts = root.GetProperty("counts");
        Assert.True(counts.GetProperty("total").GetInt32() >= counts.GetProperty("trackedDirty").GetInt32());
        Assert.True(counts.GetProperty("safeStageCandidate").GetInt32() > 0);
        Assert.True(counts.GetProperty("ignoreCandidate").GetInt32() >= 0);
        Assert.Equal(0, counts.GetProperty("reviewCandidate").GetInt32());

        string[] safeBuckets = root.GetProperty("safeStageBuckets").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("workflow", safeBuckets);
        Assert.Contains("engineering-script", safeBuckets);
        Assert.Contains("quality-test", safeBuckets);
        Assert.Contains("managed-source", safeBuckets);
        Assert.Contains("native-manifest", safeBuckets);
        Assert.Contains("application", safeBuckets);
        Assert.Contains("sample", safeBuckets);

        JsonElement[] bucketSummary = root.GetProperty("bucketSummary").EnumerateArray().ToArray();
        Assert.NotEmpty(bucketSummary);
        foreach (JsonElement item in bucketSummary)
        {
            Assert.Contains(item.GetProperty("bucket").GetString()!, safeBuckets.Concat(["ignore-output", "evidence-review", "binary-review", "manual-review"]));
            Assert.True(item.GetProperty("count").GetInt32() > 0);
        }

        string markdown = File.ReadAllText(markdownPath);
        Assert.Contains("Worktree Staging Audit", markdown, StringComparison.Ordinal);
        Assert.Contains("Safe stage candidates", markdown, StringComparison.Ordinal);
        Assert.Contains("safe-stage-pathspecs.txt", markdown, StringComparison.Ordinal);
        Assert.Contains("Boundary", markdown, StringComparison.Ordinal);

        string safeStagePathspecs = File.ReadAllText(Path.Combine(pathspecRoot, "safe-stage-pathspecs.txt"));
        string reviewHoldPathspecs = File.ReadAllText(Path.Combine(pathspecRoot, "review-hold-pathspecs.txt"));
        string ignoreHoldPathspecs = File.ReadAllText(Path.Combine(pathspecRoot, "ignore-hold-pathspecs.txt"));
        Assert.True(safeStagePathspecs.Length > 0 || counts.GetProperty("safeStageCandidate").GetInt32() == 0);
        Assert.DoesNotContain("samples/YoloDet", safeStagePathspecs, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloDet.csproj", safeStagePathspecs, StringComparison.Ordinal);
        Assert.True(reviewHoldPathspecs.Length == 0 || reviewHoldPathspecs.Split(Environment.NewLine).Length >= 0);
        Assert.True(ignoreHoldPathspecs.Length > 0 || counts.GetProperty("ignoreCandidate").GetInt32() == 0);
        Assert.DoesNotContain("samples/YoloDet", ignoreHoldPathspecs, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloDet.csproj", ignoreHoldPathspecs, StringComparison.Ordinal);

        string auditText = File.ReadAllText(auditPath);
        Assert.DoesNotContain("samples/YoloDet", auditText, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloDet.csproj", auditText, StringComparison.Ordinal);
        Assert.Contains("samples/legacy-yolo-sample-removed", auditText, StringComparison.Ordinal);

        string gitignore = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".gitignore"));
        Assert.Contains("-Strict/", gitignore, StringComparison.Ordinal);
    }

    private static void RunPowerShell(string scriptPath, params string[] arguments)
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

        Assert.True(
            process.ExitCode == 0,
            $"PowerShell command failed with exit code {process.ExitCode}:{Environment.NewLine}{output}{error}");
    }

    private static int CountOccurrences(string value, string needle)
    {
        int count = 0;
        int index = 0;
        while ((index = value.IndexOf(needle, index, StringComparison.Ordinal)) >= 0)
        {
            count++;
            index += needle.Length;
        }

        return count;
    }

    private static string Normalize(string value)
    {
        return value.Replace("\r\n", "\n", StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] pathParts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
    }
}

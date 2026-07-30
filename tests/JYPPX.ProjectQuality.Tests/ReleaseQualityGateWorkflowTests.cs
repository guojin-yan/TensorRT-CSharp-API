using System.Diagnostics;
using System.IO.Compression;
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
        Assert.Contains("push:", workflow, StringComparison.Ordinal);
        Assert.Contains("- TensorRtSharp4.0", workflow, StringComparison.Ordinal);
        Assert.Contains("source-quality:", workflow, StringComparison.Ordinal);
        Assert.Contains("run_release_artifact_audit", workflow, StringComparison.Ordinal);
        Assert.Contains("run_split_package_build", workflow, StringComparison.Ordinal);
        Assert.Contains("run_package_managed_dry_run", workflow, StringComparison.Ordinal);
        Assert.True(CountOccurrences(workflow, "default: false") >= 3);
        Assert.Contains("package-managed-dry-run:", workflow, StringComparison.Ordinal);
        Assert.Contains("uses: ./.github/workflows/package-managed.yml", workflow, StringComparison.Ordinal);
        Assert.Contains("github.repository_owner == 'guojin-yan'", workflow, StringComparison.Ordinal);
        Assert.Contains("publish_to_nuget: false", workflow, StringComparison.Ordinal);
        Assert.Contains("publish_to_github_packages: false", workflow, StringComparison.Ordinal);
        Assert.Contains("artifact_name: package-managed-dry-run", workflow, StringComparison.Ordinal);
        Assert.Contains("Export-GitHubActionsPackageValidationAudit.ps1", workflow, StringComparison.Ordinal);
        Assert.Contains("github-actions-package-validation-audit.*", workflow, StringComparison.Ordinal);
        Assert.Contains("Enforce public API documentation", workflow, StringComparison.Ordinal);
        Assert.Contains("Test-PublicApiBilingualDocumentation.ps1", workflow, StringComparison.Ordinal);
        Assert.DoesNotContain("Test-PublicApiBilingualDocumentation.ps1 -SkipBuild", workflow, StringComparison.Ordinal);
        Assert.Contains("public-api-documentation-closure.*", workflow, StringComparison.Ordinal);
        Assert.Contains("Run source-only release quality tests", workflow, StringComparison.Ordinal);
        Assert.Contains("--filter \"FullyQualifiedName~ReleaseAutomationTests|FullyQualifiedName~ReleaseQualityGateWorkflowTests\"", workflow, StringComparison.Ordinal);
        Assert.DoesNotContain("FinalReleaseMarkdownRenderingTests", workflow, StringComparison.Ordinal);
        Assert.DoesNotContain("RnnV2BorrowedStateDesignGateTests", workflow, StringComparison.Ordinal);
        Assert.DoesNotContain("EngineAndRnnReadonlyDiagnosticsTests", workflow, StringComparison.Ordinal);
        Assert.DoesNotContain("RuntimePackageReadinessTests", workflow, StringComparison.Ordinal);
        Assert.Contains("Run bounded ProjectQuality shard smoke", workflow, StringComparison.Ordinal);
        Assert.Contains("Invoke-ProjectQualityTestShards.ps1", workflow, StringComparison.Ordinal);
        Assert.Contains("-Shard N-S", workflow, StringComparison.Ordinal);
        Assert.Contains("PluginInventorySourceOnlySmoke|PublicApiDocumentationClosure|PublicApiHandleExposureAudit|ReleaseQualityGateWorkflow", workflow, StringComparison.Ordinal);
        Assert.Contains("artifacts/test-analysis/project-quality-test-inventory.*", workflow, StringComparison.Ordinal);
        Assert.Contains("artifacts/test-analysis/project-quality-shards/**", workflow, StringComparison.Ordinal);
        Assert.Contains("Record split runner availability", workflow, StringComparison.Ordinal);
        Assert.Contains("continue-on-error: true", workflow, StringComparison.Ordinal);
        Assert.Contains("GH_TOKEN: ${{ github.token }}", workflow, StringComparison.Ordinal);
        Assert.Contains("Test-GitHubRunnerAvailability.ps1", workflow, StringComparison.Ordinal);
        Assert.Contains("-RequiredLabelSet \"self-hosted,windows,x64\"", workflow, StringComparison.Ordinal);
        Assert.Contains("-WarnOnly", workflow, StringComparison.Ordinal);
        Assert.Contains("artifacts/runner-availability/**", workflow, StringComparison.Ordinal);
        string runnerScript = ReadSource("eng", "Test-GitHubRunnerAvailability.ps1");
        Assert.Contains("recordKind = \"github-runner-availability\"", runnerScript, StringComparison.Ordinal);
        Assert.Contains("querySucceeded", runnerScript, StringComparison.Ordinal);
        Assert.Contains("queryError", runnerScript, StringComparison.Ordinal);
        Assert.Contains("if (-not $WarnOnly.IsPresent)", runnerScript, StringComparison.Ordinal);
        Assert.Contains("runs-on: [self-hosted, windows, x64, release-artifacts]", workflow, StringComparison.Ordinal);
        Assert.Contains("runs-on: [self-hosted, windows, x64]", workflow, StringComparison.Ordinal);
        Assert.Contains("-SplitPackageRole bridge", workflow, StringComparison.Ordinal);
        Assert.DoesNotContain("-SplitPackageRole all", workflow, StringComparison.Ordinal);
        Assert.DoesNotContain("-IncludeMetaPackage", workflow, StringComparison.Ordinal);
        Assert.Contains("Test-ExternalVendorRuntimePackagePolicy.ps1", workflow, StringComparison.Ordinal);
        Assert.Contains("Test-ReleaseEvidenceClassificationAudit.ps1 -Strict", workflow, StringComparison.Ordinal);
        Assert.Contains("Test-PublicProofClaimBoundaryAudit.ps1 -Strict", workflow, StringComparison.Ordinal);
        Assert.Contains("Test-ReleaseQualityGate.ps1", workflow, StringComparison.Ordinal);
        Assert.Contains("release-quality-gate-summary.json", ReadSource("eng", "Test-ReleaseQualityGate.ps1"), StringComparison.Ordinal);
        Assert.Contains("workflow-bridge-package-contract", ReadSource("eng", "Test-ReleaseQualityGate.ps1"), StringComparison.Ordinal);

        Assert.DoesNotContain("dotnet nuget push", workflow, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("Push-NuGetPackages", workflow, StringComparison.Ordinal);
        Assert.DoesNotContain("NUGET_API_KEY", workflow, StringComparison.Ordinal);
        Assert.DoesNotContain("GITHUB_PACKAGES_TOKEN", workflow, StringComparison.Ordinal);
        Assert.DoesNotContain("gh release create", workflow, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("gh release upload", workflow, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void PackageDryRunRequiresManualDispatchAndDisabledPublishFlags()
    {
        string workflow = Normalize(ReadSource(".github", "workflows", "release-quality-gate.yml"));
        string packageManagedWorkflow = Normalize(ReadSource(".github", "workflows", "package-managed.yml"));
        string actionsAudit = ReadSource("eng", "Export-GitHubActionsPackageValidationAudit.ps1");
        string remoteBundleScript = ReadSource("eng", "Invoke-RemoteReleaseBundle.ps1");

        Assert.Contains("workflow_dispatch:", workflow, StringComparison.Ordinal);
        Assert.Contains("run_package_managed_dry_run:", workflow, StringComparison.Ordinal);
        Assert.Contains("default: false", workflow, StringComparison.Ordinal);
        Assert.Contains("package-managed-dry-run:", workflow, StringComparison.Ordinal);
        Assert.Contains("if: ${{ github.repository_owner == 'guojin-yan' && github.event_name == 'workflow_dispatch' && inputs.run_package_managed_dry_run }}", workflow, StringComparison.Ordinal);
        Assert.DoesNotContain("package-managed-dry-run:\n    if: ${{ github.event_name == 'push'", workflow, StringComparison.Ordinal);
        Assert.Contains("publish_to_nuget: false", workflow, StringComparison.Ordinal);
        Assert.Contains("publish_to_github_packages: false", workflow, StringComparison.Ordinal);
        Assert.Contains("attach_to_github_release: false", workflow, StringComparison.Ordinal);
        Assert.Contains("release_tag: \"\"", workflow, StringComparison.Ordinal);
        Assert.Contains("artifact_name: package-managed-dry-run", workflow, StringComparison.Ordinal);

        Assert.Contains("publish_to_nuget:", packageManagedWorkflow, StringComparison.Ordinal);
        Assert.Contains("publish_to_github_packages:", packageManagedWorkflow, StringComparison.Ordinal);
        Assert.True(CountOccurrences(packageManagedWorkflow, "default: false") >= 4);
        Assert.Contains("if: ${{ inputs.publish_to_nuget && github.repository_owner == 'guojin-yan' }}", packageManagedWorkflow, StringComparison.Ordinal);
        Assert.Contains("if: ${{ inputs.publish_to_github_packages && github.repository_owner == 'guojin-yan' }}", packageManagedWorkflow, StringComparison.Ordinal);
        Assert.Contains("inputs.publish_to_nuget && github.repository_owner", actionsAudit, StringComparison.Ordinal);
        Assert.Contains("inputs.publish_to_github_packages && github.repository_owner", actionsAudit, StringComparison.Ordinal);
        Assert.Contains("grape-yan repository is validation-only", actionsAudit, StringComparison.Ordinal);

        Assert.Contains("[object]$PublishManagedToNuGet = $false", remoteBundleScript, StringComparison.Ordinal);
        Assert.Contains("[object]$PublishRuntimeToGitHubPackages = $false", remoteBundleScript, StringComparison.Ordinal);
        Assert.Contains("Add-WorkflowInput -ArgumentList $arguments -Name \"publish_managed_to_nuget\"", remoteBundleScript, StringComparison.Ordinal);
        Assert.Contains("Add-WorkflowInput -ArgumentList $arguments -Name \"publish_managed_to_github_packages\"", remoteBundleScript, StringComparison.Ordinal);
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
            check.GetProperty("id").GetString() == "workflow-push-current-branch" &&
            check.GetProperty("passed").GetBoolean());
        Assert.Contains(checks, static check =>
            check.GetProperty("id").GetString() == "workflow-project-quality-shard-smoke" &&
            check.GetProperty("passed").GetBoolean());
        Assert.Contains(checks, static check =>
            check.GetProperty("id").GetString() == "workflow-public-api-documentation" &&
            check.GetProperty("passed").GetBoolean());
        Assert.Contains(checks, static check =>
            check.GetProperty("id").GetString() == "workflow-source-only-test-filter" &&
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
        Assert.True(workflowContracts.GetProperty("releaseQualitySourceOnlyFilterClean").GetBoolean());
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
            check.GetProperty("id").GetString() == "workflow-release-quality-source-only-filter" &&
            check.GetProperty("passed").GetBoolean());
        Assert.Contains(checks, static check =>
            check.GetProperty("id").GetString() == "workflow-runtime-package-ready" &&
            check.GetProperty("passed").GetBoolean());

        string markdown = File.ReadAllText(markdownPath);
        Assert.True(
            markdown.Contains("仍需用 GitHub Actions run URL/artifact 补充包验证证据", StringComparison.Ordinal) ||
            markdown.Contains("不能声称本轮代码已在 GitHub Actions 上完成 NuGet/包验证", StringComparison.Ordinal),
            "Expected the audit Markdown to keep the current-run package validation boundary.");
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
        Assert.True(counts.GetProperty("safeStageCandidate").GetInt32() >= 0);
        Assert.True(counts.GetProperty("ignoreCandidate").GetInt32() >= 0);
        Assert.Equal(0, counts.GetProperty("reviewCandidate").GetInt32());

        string[] safeBuckets = root.GetProperty("safeStageBuckets").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        if (counts.GetProperty("safeStageCandidate").GetInt32() > 0)
        {
            Assert.Contains("workflow", safeBuckets);
            Assert.Contains("engineering-script", safeBuckets);
            Assert.Contains("quality-test", safeBuckets);
            Assert.Contains("managed-source", safeBuckets);
            Assert.Contains("native-manifest", safeBuckets);
            Assert.Contains("application", safeBuckets);
            Assert.Contains("sample", safeBuckets);
        }

        JsonElement[] bucketSummary = root.GetProperty("bucketSummary").EnumerateArray().ToArray();
        Assert.True(bucketSummary.Length > 0 || counts.GetProperty("total").GetInt32() == 0);
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
        Assert.DoesNotContain("samples/legacy-yolo-sample-removed/YoloDet", auditText, StringComparison.Ordinal);

        string gitignore = File.ReadAllText(Path.Combine(RepositoryPaths.Root, ".gitignore"));
        Assert.Contains("-Strict/", gitignore, StringComparison.Ordinal);
    }

    [Fact]
    public void GitHubActionsRunEvidenceImportPromotesDryRunPackWithoutPublishingClaims()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "trtsharp-run-evidence-" + Guid.NewGuid().ToString("N"));
        string runId = "29160655818";
        string headSha = "72d65909a120e1e550568e01796bbbdb5ec2b2e4";
        string artifactsRoot = Path.Combine(tempRoot, "artifacts", "github-actions-runs", runId);
        string packageRoot = Path.Combine(artifactsRoot, "package-managed-dry-run");
        string releaseGateRoot = Path.Combine(artifactsRoot, "release-quality-gate", "release-quality-gate");
        string finalReleaseRoot = Path.Combine(artifactsRoot, "release-quality-gate", "final-release");
        string outputPath = Path.Combine(tempRoot, "github-actions-run-evidence-import.json");
        string markdownPath = Path.Combine(tempRoot, "github-actions-run-evidence-import.md");
        string runMetadataPath = Path.Combine(artifactsRoot, "github-run-view.json");

        try
        {
            Directory.CreateDirectory(packageRoot);
            Directory.CreateDirectory(releaseGateRoot);
            Directory.CreateDirectory(finalReleaseRoot);

            File.WriteAllText(
                Path.Combine(releaseGateRoot, "release-quality-gate-summary.json"),
                $$"""
                {
                  "recordKind": "release-quality-gate-summary",
                  "state": "release-quality-gate-passed",
                  "performsPublish": false,
                  "usesPublishToken": false,
                  "isRuntimeExecutionProof": false,
                  "isPackageConsumerRuntimeProof": false,
                  "canPublishPublicly": false,
                  "canCloseReleaseIssue": false,
                  "checks": []
                }
                """);
            File.WriteAllText(
                Path.Combine(finalReleaseRoot, "github-actions-package-validation-audit.json"),
                $$"""
                {
                  "recordKind": "github-actions-package-validation-audit",
                  "headSha": "{{headSha}}",
                  "performsPublish": false,
                  "usesPublishToken": false,
                  "isPackageConsumerRuntimeProof": false,
                  "isPostPublishProof": false,
                  "canPublishPublicly": false
                }
                """);
            File.WriteAllText(
                runMetadataPath,
                $$"""
                {
                  "databaseId": 29160655818,
                  "headSha": "{{headSha}}",
                  "status": "completed",
                  "conclusion": "success",
                  "url": "https://github.com/guojin-yan/TensorRT-CSharp-API/actions/runs/29160655818",
                  "jobs": [
                    { "name": "source-quality", "status": "completed", "conclusion": "success" },
                    { "name": "package-managed-dry-run / pack", "status": "completed", "conclusion": "success" },
                    { "name": "package-managed-dry-run / publish-nuget", "status": "completed", "conclusion": "skipped" },
                    { "name": "package-managed-dry-run / publish-github-packages", "status": "completed", "conclusion": "skipped" }
                  ]
                }
                """);

            CreateMinimalManagedNupkg(Path.Combine(packageRoot, "JYPPX.TensorRT.CSharp.API.4.0.0.nupkg"));

            string script = Path.Combine(RepositoryPaths.Root, "eng", "Export-GitHubActionsRunEvidenceImport.ps1");
            RunPowerShell(
                script,
                "-RunId",
                runId,
                "-ArtifactsRoot",
                artifactsRoot,
                "-RunMetadataPath",
                runMetadataPath,
                "-ExpectedHeadSha",
                headSha,
                "-OutputPath",
                outputPath,
                "-MarkdownOutputPath",
                markdownPath);

            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(outputPath));
            JsonElement root = document.RootElement;

            Assert.Equal("github-actions-run-evidence-import", root.GetProperty("recordKind").GetString());
            Assert.Equal(runId, root.GetProperty("runId").GetString());
            Assert.Equal(headSha, root.GetProperty("headSha").GetString());
            Assert.Equal("success", root.GetProperty("runConclusion").GetString());
            Assert.Equal("success", root.GetProperty("sourceQualityConclusion").GetString());
            Assert.Equal("success", root.GetProperty("packageManagedDryRunPackConclusion").GetString());
            Assert.Equal("skipped", root.GetProperty("publishNugetConclusion").GetString());
            Assert.Equal("skipped", root.GetProperty("publishGitHubPackagesConclusion").GetString());
            Assert.Equal(0, root.GetProperty("failedBlockerCount").GetInt32());
            Assert.True(root.GetProperty("canClaimGitHubActionsPackageDryRunPackForRun").GetBoolean());
            Assert.False(root.GetProperty("canClaimNuGetPublished").GetBoolean());
            Assert.False(root.GetProperty("canClaimGitHubPackagesPublished").GetBoolean());
            Assert.False(root.GetProperty("performsPublish").GetBoolean());
            Assert.False(root.GetProperty("usesPublishToken").GetBoolean());
            Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
            Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());

            JsonElement package = Assert.Single(root.GetProperty("nupkgPackages").EnumerateArray());
            Assert.Equal("JYPPX.TensorRT.CSharp.API.4.0.0.nupkg", package.GetProperty("fileName").GetString());
            Assert.True(package.GetProperty("hasNuspec").GetBoolean());
            Assert.True(package.GetProperty("hasReadme").GetBoolean());
            Assert.True(package.GetProperty("dllEntryCount").GetInt32() > 0);
            Assert.True(package.GetProperty("xmlEntryCount").GetInt32() > 0);

            string markdown = File.ReadAllText(markdownPath);
            Assert.Contains("Can claim package dry-run pack: `True`", markdown, StringComparison.Ordinal);
            Assert.Contains("Can claim NuGet published: `False`", markdown, StringComparison.Ordinal);
            Assert.Contains("Can claim GitHub Packages published: `False`", markdown, StringComparison.Ordinal);
            Assert.Contains("is not package-consumer runtime proof", markdown, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
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

    private static void CreateMinimalManagedNupkg(string path)
    {
        using ZipArchive archive = ZipFile.Open(path, ZipArchiveMode.Create);
        AddZipEntry(archive, "_rels/.rels", "<Relationships />");
        AddZipEntry(archive, "JYPPX.TensorRT.CSharp.API.nuspec", "<package><metadata><id>JYPPX.TensorRT.CSharp.API</id><version>4.0.0</version></metadata></package>");
        AddZipEntry(archive, "README.md", "# JYPPX TensorRT CSharp API");
        AddZipEntry(archive, "lib/net8.0/JYPPX.CudaSharp.dll", "binary");
        AddZipEntry(archive, "lib/net8.0/JYPPX.CudaSharp.xml", "<doc />");
        AddZipEntry(archive, "lib/net8.0/JYPPX.Shared.dll", "binary");
        AddZipEntry(archive, "lib/net8.0/JYPPX.Shared.xml", "<doc />");
        AddZipEntry(archive, "lib/net8.0/JYPPX.TensorRtSharp.dll", "binary");
        AddZipEntry(archive, "lib/net8.0/JYPPX.TensorRtSharp.xml", "<doc />");
    }

    private static void AddZipEntry(ZipArchive archive, string entryName, string content)
    {
        ZipArchiveEntry entry = archive.CreateEntry(entryName);
        using Stream stream = entry.Open();
        using StreamWriter writer = new(stream);
        writer.Write(content);
    }
}

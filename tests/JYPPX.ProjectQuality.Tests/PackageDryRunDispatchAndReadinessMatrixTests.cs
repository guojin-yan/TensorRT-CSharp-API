using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class PackageDryRunDispatchAndReadinessMatrixTests
{
    [Fact]
    public void OwnerDispatchPackGeneratesNonPublishCommandWithoutExecutingWorkflow()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "trtsharp-dry-run-dispatch-pack-" + Guid.NewGuid().ToString("N"));
        string preflightPath = Path.Combine(tempRoot, "current-head-package-dry-run-preflight.json");
        string packPath = Path.Combine(tempRoot, "current-head-package-dry-run-owner-dispatch-pack.json");
        string packMarkdownPath = Path.Combine(tempRoot, "current-head-package-dry-run-owner-dispatch-pack.md");
        string validationPath = Path.Combine(tempRoot, "current-head-package-dry-run-owner-dispatch-pack-validation.json");
        string validationMarkdownPath = Path.Combine(tempRoot, "current-head-package-dry-run-owner-dispatch-pack-validation.md");

        try
        {
            Directory.CreateDirectory(tempRoot);
            WriteBlockedPreflight(preflightPath);

            RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Export-CurrentHeadPackageDryRunOwnerDispatchPack.ps1"),
                "-CurrentHeadPackageDryRunPreflightPath",
                preflightPath,
                "-OutputPath",
                packPath,
                "-MarkdownOutputPath",
                packMarkdownPath);
            RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Test-CurrentHeadPackageDryRunOwnerDispatchPack.ps1"),
                "-InputPath",
                packPath,
                "-OutputPath",
                validationPath,
                "-MarkdownOutputPath",
                validationMarkdownPath,
                "-Strict");

            using JsonDocument packDocument = JsonDocument.Parse(File.ReadAllText(packPath));
            JsonElement pack = packDocument.RootElement;
            Assert.Equal("current-head-package-dry-run-owner-dispatch-pack", pack.GetProperty("recordKind").GetString());
            Assert.Equal("owner-authorization-required-before-workflow-dispatch", pack.GetProperty("packState").GetString());
            Assert.True(pack.GetProperty("requiresOwnerAuthorization").GetBoolean());
            Assert.False(pack.GetProperty("workflowDispatchExecuted").GetBoolean());
            Assert.False(pack.GetProperty("currentHeadPackageDryRunReady").GetBoolean());
            Assert.False(pack.GetProperty("performsPublish").GetBoolean());
            Assert.False(pack.GetProperty("usesPublishToken").GetBoolean());
            Assert.False(pack.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(pack.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(pack.GetProperty("canPromoteProof").GetBoolean());
            Assert.False(pack.GetProperty("isPackageDryRunProof").GetBoolean());
            Assert.False(pack.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
            Assert.False(pack.GetProperty("isPostPublishProof").GetBoolean());

            string command = pack.GetProperty("dispatchCommand").GetString()!;
            Assert.Contains("gh workflow run release-quality-gate.yml", command, StringComparison.Ordinal);
            Assert.Contains("-f run_package_managed_dry_run=true", command, StringComparison.Ordinal);
            Assert.Contains("-f run_release_artifact_audit=false", command, StringComparison.Ordinal);
            Assert.Contains("-f run_split_package_build=false", command, StringComparison.Ordinal);
            Assert.DoesNotContain("dotnet nuget push", command, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("gh release upload", command, StringComparison.OrdinalIgnoreCase);

            JsonElement effectiveInputs = pack.GetProperty("effectivePackageManagedInputs");
            Assert.False(effectiveInputs.GetProperty("publish_to_nuget").GetBoolean());
            Assert.False(effectiveInputs.GetProperty("publish_to_github_packages").GetBoolean());
            Assert.False(effectiveInputs.GetProperty("attach_to_github_release").GetBoolean());
            Assert.Equal("package-managed-dry-run", effectiveInputs.GetProperty("artifact_name").GetString());

            using JsonDocument validationDocument = JsonDocument.Parse(File.ReadAllText(validationPath));
            JsonElement validation = validationDocument.RootElement;
            Assert.Equal("current-head-package-dry-run-owner-dispatch-pack-validation", validation.GetProperty("recordKind").GetString());
            Assert.Equal("current-head-package-dry-run-dispatch-pack-ready-for-owner", validation.GetProperty("validationState").GetString());
            Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
            Assert.Equal(0, validation.GetProperty("failedActionRequiredCount").GetInt32());
            Assert.False(validation.GetProperty("workflowDispatchExecuted").GetBoolean());
            Assert.False(validation.GetProperty("performsPublish").GetBoolean());
            Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void PreReleaseReadinessMatrixKeepsPublicRuntimeAndPostPublishProofBlocked()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "trtsharp-pre-release-readiness-matrix-" + Guid.NewGuid().ToString("N"));
        string preflightPath = Path.Combine(tempRoot, "current-head-package-dry-run-preflight.json");
        string sourceValidationPath = Path.Combine(tempRoot, "source-quality-validation.json");
        string dispatchValidationPath = Path.Combine(tempRoot, "dispatch-validation.json");
        string publicDownloadValidationPath = Path.Combine(tempRoot, "public-download-validation.json");
        string consumerRuntimeValidationPath = Path.Combine(tempRoot, "consumer-runtime-validation.json");
        string postPublishValidationPath = Path.Combine(tempRoot, "post-publish-validation.json");
        string matrixPath = Path.Combine(tempRoot, "pre-release-package-proof-readiness-matrix.json");
        string matrixMarkdownPath = Path.Combine(tempRoot, "pre-release-package-proof-readiness-matrix.md");

        try
        {
            Directory.CreateDirectory(tempRoot);
            WriteBlockedPreflight(preflightPath);
            File.WriteAllText(
                sourceValidationPath,
                """
                {
                  "recordKind": "github-actions-run-evidence-import-validation",
                  "validationState": "source-quality-run-evidence-ready",
                  "sourceQualityRunEvidenceReady": true,
                  "packageDryRunEvidenceReady": false,
                  "githubActionsRunEvidenceReady": false
                }
                """);
            File.WriteAllText(
                dispatchValidationPath,
                """
                {
                  "recordKind": "current-head-package-dry-run-owner-dispatch-pack-validation",
                  "validationState": "current-head-package-dry-run-dispatch-pack-ready-for-owner",
                  "failedBlockerCount": 0,
                  "failedActionRequiredCount": 0,
                  "workflowDispatchExecuted": false,
                  "performsPublish": false,
                  "canPublishPublicly": false,
                  "canCloseReleaseIssue": false
                }
                """);
            File.WriteAllText(
                publicDownloadValidationPath,
                """
                {
                  "recordKind": "public-package-download-proof-input-validation",
                  "validationState": "blocked-public-package-download-proof-required",
                  "publicPackageDownloadProofReady": false
                }
                """);
            File.WriteAllText(
                consumerRuntimeValidationPath,
                """
                {
                  "recordKind": "package-consumer-runtime-proof-owner-input-validation",
                  "validationState": "blocked-owner-input-required",
                  "cleanOwnerInputReady": false,
                  "ownerInputBlockedReason": "clean external consumer runtime owner input is incomplete"
                }
                """);
            File.WriteAllText(
                postPublishValidationPath,
                """
                {
                  "recordKind": "post-publish-clean-consumer-proof-result-validation",
                  "validationState": "blocked-post-publish-proof-required",
                  "postPublishProofReady": false
                }
                """);

            RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Export-PreReleasePackageProofReadinessMatrix.ps1"),
                "-CurrentHeadPackageDryRunPreflightPath",
                preflightPath,
                "-DispatchPackValidationPath",
                dispatchValidationPath,
                "-SourceQualityEvidenceValidationPath",
                sourceValidationPath,
                "-PublicPackageDownloadProofInputValidationPath",
                publicDownloadValidationPath,
                "-PackageConsumerRuntimeOwnerInputValidationPath",
                consumerRuntimeValidationPath,
                "-PostPublishProofValidationPath",
                postPublishValidationPath,
                "-OutputPath",
                matrixPath,
                "-MarkdownOutputPath",
                matrixMarkdownPath);

            using JsonDocument matrixDocument = JsonDocument.Parse(File.ReadAllText(matrixPath));
            JsonElement matrix = matrixDocument.RootElement;
            Assert.Equal("pre-release-package-proof-readiness-matrix", matrix.GetProperty("recordKind").GetString());
            Assert.Equal("blocked-real-public-package-and-runtime-proof-required", matrix.GetProperty("matrixState").GetString());
            Assert.Equal(2, matrix.GetProperty("readyLaneCount").GetInt32());
            Assert.Equal(4, matrix.GetProperty("blockedLaneCount").GetInt32());
            Assert.False(matrix.GetProperty("currentHeadPackageDryRunReady").GetBoolean());
            Assert.True(matrix.GetProperty("ownerDispatchPackReadyForOwner").GetBoolean());
            Assert.False(matrix.GetProperty("publicPackageDownloadProofReady").GetBoolean());
            Assert.False(matrix.GetProperty("packageConsumerRuntimeProofReady").GetBoolean());
            Assert.False(matrix.GetProperty("postPublishProofReady").GetBoolean());
            Assert.False(matrix.GetProperty("performsPublish").GetBoolean());
            Assert.False(matrix.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(matrix.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(matrix.GetProperty("canPromoteProof").GetBoolean());

            JsonElement[] lanes = matrix.GetProperty("lanes").EnumerateArray().ToArray();
            AssertLane(lanes, "source-quality-ci", ready: true);
            AssertLane(lanes, "owner-dispatch-pack", ready: true);
            AssertLane(lanes, "current-head-package-dry-run", ready: false);
            AssertLane(lanes, "public-package-download", ready: false);
            AssertLane(lanes, "clean-external-package-consumer-runtime", ready: false);
            AssertLane(lanes, "post-publish-clean-consumer-proof", ready: false);

            JsonElement publicDownloadLane = Assert.Single(lanes, item => item.GetProperty("id").GetString() == "public-package-download");
            Assert.Equal("eng\\Test-PublicPackageDownloadProofInput.ps1 -Strict", publicDownloadLane.GetProperty("validatorPath").GetString());
            Assert.Contains("Public NuGet/GitHub Packages package URLs", publicDownloadLane.GetProperty("requiredEvidence").GetString(), StringComparison.Ordinal);
            Assert.Equal(0, publicDownloadLane.GetProperty("failedBlockerCount").GetInt32());
            Assert.Equal(1, publicDownloadLane.GetProperty("failedActionRequiredCount").GetInt32());
            Assert.False(publicDownloadLane.GetProperty("canPromotePublicProof").GetBoolean());

            JsonElement runtimeLane = Assert.Single(lanes, item => item.GetProperty("id").GetString() == "clean-external-package-consumer-runtime");
            Assert.Equal("eng\\Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict", runtimeLane.GetProperty("validatorPath").GetString());
            Assert.False(runtimeLane.GetProperty("canPromoteRuntimeProof").GetBoolean());

            JsonElement postPublishLane = Assert.Single(lanes, item => item.GetProperty("id").GetString() == "post-publish-clean-consumer-proof");
            Assert.Equal("eng\\Test-PostPublishCleanConsumerProofResult.ps1 -Strict", postPublishLane.GetProperty("validatorPath").GetString());
            Assert.False(postPublishLane.GetProperty("canPromotePostPublishProof").GetBoolean());

            string markdown = File.ReadAllText(matrixMarkdownPath);
            Assert.Contains("Pre-Release Package Proof Readiness Matrix", markdown, StringComparison.Ordinal);
            Assert.Contains("blocked-real-public-package-and-runtime-proof-required", markdown, StringComparison.Ordinal);
            Assert.Contains("Validator", markdown, StringComparison.Ordinal);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    private static void AssertLane(JsonElement[] lanes, string id, bool ready)
    {
        JsonElement lane = Assert.Single(lanes, item => item.GetProperty("id").GetString() == id);
        Assert.Equal(ready, lane.GetProperty("ready").GetBoolean());
        Assert.False(lane.GetProperty("performsPublish").GetBoolean());
        Assert.False(lane.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(lane.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(lane.GetProperty("canPromoteProof").GetBoolean());
        Assert.False(lane.GetProperty("canPromotePublicProof").GetBoolean());
        Assert.False(lane.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(lane.GetProperty("canPromotePostPublishProof").GetBoolean());
        Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("requiredEvidence").GetString()));
        Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("validatorPath").GetString()));
    }

    private static void WriteBlockedPreflight(string path)
    {
        File.WriteAllText(
            path,
            """
            {
              "recordKind": "current-head-package-dry-run-preflight",
              "state": "blocked-owner-authorization-required",
              "currentHead": "faa27f4574166b4fab4812373f6e2cd36d24793a",
              "sourceQualityRunId": "29232339732",
              "sourceQualityRunEvidenceReady": true,
              "packageDryRunRunId": "29160655818",
              "packageDryRunHeadSha": "72d65909a120e1e550568e01796bbbdb5ec2b2e4",
              "packageDryRunHeadMatchesCurrentHead": false,
              "packageDryRunCanClaimPackForRun": true,
              "canClaimGitHubActionsPackageDryRunPackForCurrentHead": false,
              "packageDryRunRequiresOwnerAuthorization": true,
              "manualWorkflowDispatchNotPerformed": true,
              "blockedReason": "Existing package dry-run evidence is not for the current HEAD; Owner must explicitly authorize a new workflow_dispatch dry-run with publishing disabled.",
              "performsPublish": false,
              "canPublishPublicly": false,
              "canCloseReleaseIssue": false,
              "isPackageConsumerRuntimeProof": false,
              "isPostPublishProof": false
            }
            """);
    }

    private static void RunPowerShell(string scriptPath, params string[] arguments)
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
    }
}

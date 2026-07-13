using System.Diagnostics;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class GitHubActionsRunEvidenceImportTests
{
    [Fact]
    public void MissingInputStaysBlockedWithoutBlockers()
    {
        string tempRoot = CreateTempRoot();
        string outputRoot = Path.Combine(tempRoot, "final-release");

        try
        {
            RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Test-GitHubActionsRunEvidenceImport.ps1"),
                "-InputPath",
                Path.Combine(tempRoot, "missing-github-actions-run-evidence-import.json"),
                "-OutputRoot",
                outputRoot,
                "-Strict");

            using JsonDocument document = ReadJson(Path.Combine(outputRoot, "github-actions-run-evidence-import-validation.json"));
            JsonElement root = document.RootElement;

            Assert.Equal("github-actions-run-evidence-import-validation", root.GetProperty("recordKind").GetString());
            Assert.Equal("blocked-github-actions-run-evidence-required", root.GetProperty("validationState").GetString());
            Assert.False(root.GetProperty("githubActionsRunEvidenceReady").GetBoolean());
            Assert.False(root.GetProperty("sourceQualityRunEvidenceReady").GetBoolean());
            Assert.False(root.GetProperty("packageDryRunEvidenceReady").GetBoolean());
            Assert.Equal(0, root.GetProperty("failedBlockerCount").GetInt32());
            Assert.True(root.GetProperty("failedActionRequiredCount").GetInt32() > 0);
            Assert.False(root.GetProperty("isGitHubActionsProof").GetBoolean());
            Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
            AssertValidationItem(root, "input-present", passed: false);
            AssertValidationItem(root, "forbidden-substitutes-absent", passed: true);
        }
        finally
        {
            DeleteTempRoot(tempRoot);
        }
    }

    [Fact]
    public void SourceOnlyPushRunValidatesAsSourceQualityEvidenceWithoutPackageClaims()
    {
        string tempRoot = CreateTempRoot();
        string runId = "29229700998";
        string headSha = "ab618c5cb6e37c7cb882063d9782991fa03b3170";
        string artifactsRoot = Path.Combine(tempRoot, "artifacts", "github-actions-runs", runId);
        string releaseGateRoot = Path.Combine(artifactsRoot, "release-quality-gate", "release-quality-gate");
        string finalReleaseRoot = Path.Combine(artifactsRoot, "release-quality-gate", "final-release");
        string evidenceRoot = Path.Combine(artifactsRoot, "owner-run-evidence");
        string runMetadataPath = Path.Combine(artifactsRoot, "github-run-view.json");
        string workflowRunLogPath = Path.Combine(evidenceRoot, "workflow-run.log");
        string artifactManifestPath = Path.Combine(evidenceRoot, "artifact-manifest.json");
        string outputRoot = Path.Combine(tempRoot, "final-release");
        string importPath = Path.Combine(outputRoot, "github-actions-run-evidence-import.json");
        string importMarkdownPath = Path.Combine(outputRoot, "github-actions-run-evidence-import.md");

        try
        {
            Directory.CreateDirectory(releaseGateRoot);
            Directory.CreateDirectory(finalReleaseRoot);
            Directory.CreateDirectory(evidenceRoot);
            Directory.CreateDirectory(outputRoot);

            File.WriteAllText(
                Path.Combine(releaseGateRoot, "release-quality-gate-summary.json"),
                """
                {
                  "recordKind": "release-quality-gate-summary",
                  "state": "release-quality-gate-passed",
                  "sourceGatePassed": true,
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
                  "hasGitHubActionsRunEvidenceForCurrentCode": false,
                  "canClaimGitHubActionsPackageValidationForCurrentCode": false,
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
                  "databaseId": 29229700998,
                  "headSha": "{{headSha}}",
                  "status": "completed",
                  "conclusion": "success",
                  "url": "https://github.com/guojin-yan/TensorRT-CSharp-API/actions/runs/29229700998",
                  "runAttempt": 1,
                  "workflowName": "release-quality-gate",
                  "workflowFile": ".github/workflows/release-quality-gate.yml",
                  "event": "push",
                  "headBranch": "TensorRtSharp4.0",
                  "ref": "refs/heads/TensorRtSharp4.0",
                  "startedAtUtc": "2026-07-13T06:30:00Z",
                  "completedAtUtc": "2026-07-13T06:45:12Z",
                  "jobs": [
                    { "name": "source-quality", "status": "completed", "conclusion": "success" }
                  ]
                }
                """);

            File.WriteAllText(
                workflowRunLogPath,
                """
                release-quality-gate push run 29229700998
                source-quality: success
                package-managed-dry-run: skipped by push event
                publish jobs: not created
                """);
            File.WriteAllText(
                artifactManifestPath,
                $$"""
                {
                  "runId": "{{runId}}",
                  "headSha": "{{headSha}}",
                  "artifacts": [
                    "release-quality-gate/release-quality-gate-summary.json",
                    "final-release/github-actions-package-validation-audit.json"
                  ],
                  "packageManagedDryRunArtifactPresent": false
                }
                """);

            RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Export-GitHubActionsRunEvidenceImport.ps1"),
                "-RunId",
                runId,
                "-ArtifactsRoot",
                artifactsRoot,
                "-RunMetadataPath",
                runMetadataPath,
                "-ExpectedHeadSha",
                headSha,
                "-WorkflowRunLogPath",
                workflowRunLogPath,
                "-ArtifactManifestPath",
                artifactManifestPath,
                "-OwnerReviewer",
                "guojin-yan",
                "-CapturedAtUtc",
                "2026-07-13T06:50:00Z",
                "-SourceQualityOnly",
                "-OutputPath",
                importPath,
                "-MarkdownOutputPath",
                importMarkdownPath);

            RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Test-GitHubActionsRunEvidenceImport.ps1"),
                "-InputPath",
                importPath,
                "-OutputRoot",
                outputRoot,
                "-Strict");

            using JsonDocument importDocument = ReadJson(importPath);
            JsonElement importRoot = importDocument.RootElement;
            Assert.Equal("source-quality-only", importRoot.GetProperty("importMode").GetString());
            Assert.Equal("source-quality-run-evidence-ready", importRoot.GetProperty("evidenceState").GetString());
            Assert.True(importRoot.GetProperty("canClaimGitHubActionsSourceQualityForRun").GetBoolean());
            Assert.False(importRoot.GetProperty("canClaimGitHubActionsPackageDryRunPackForRun").GetBoolean());
            Assert.False(importRoot.GetProperty("canClaimNuGetPublished").GetBoolean());
            Assert.False(importRoot.GetProperty("canClaimGitHubPackagesPublished").GetBoolean());
            Assert.Empty(importRoot.GetProperty("nupkgPackages").EnumerateArray());

            using JsonDocument validationDocument = ReadJson(Path.Combine(outputRoot, "github-actions-run-evidence-import-validation.json"));
            JsonElement validation = validationDocument.RootElement;
            Assert.Equal("source-quality-run-evidence-ready", validation.GetProperty("validationState").GetString());
            Assert.True(validation.GetProperty("sourceQualityRunEvidenceReady").GetBoolean());
            Assert.False(validation.GetProperty("packageDryRunEvidenceReady").GetBoolean());
            Assert.False(validation.GetProperty("githubActionsRunEvidenceReady").GetBoolean());
            Assert.True(validation.GetProperty("canClaimGitHubActionsSourceQualityForRun").GetBoolean());
            Assert.False(validation.GetProperty("canClaimGitHubActionsPackageDryRunPackForRun").GetBoolean());
            Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
            Assert.Equal(0, validation.GetProperty("failedActionRequiredCount").GetInt32());
            Assert.False(validation.GetProperty("performsPublish").GetBoolean());
            Assert.False(validation.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
            Assert.False(validation.GetProperty("isPostPublishProof").GetBoolean());
            AssertValidationItem(validation, "source-quality-claim-ready", passed: true);
            AssertValidationItem(validation, "package-dry-run-pack-success", passed: true);
            AssertValidationItem(validation, "nupkg-packages-present", passed: true);
            AssertValidationItem(validation, "dry-run-pack-claim-ready", passed: true);
        }
        finally
        {
            DeleteTempRoot(tempRoot);
        }
    }

    [Fact]
    public void ReadyFixtureValidatesRealRunEvidenceWithoutProofClaims()
    {
        string tempRoot = CreateTempRoot();
        string runId = "29160655818";
        string headSha = "72d65909a120e1e550568e01796bbbdb5ec2b2e4";
        string artifactsRoot = Path.Combine(tempRoot, "artifacts", "github-actions-runs", runId);
        string packageRoot = Path.Combine(artifactsRoot, "package-managed-dry-run");
        string releaseGateRoot = Path.Combine(artifactsRoot, "release-quality-gate", "release-quality-gate");
        string finalReleaseRoot = Path.Combine(artifactsRoot, "release-quality-gate", "final-release");
        string evidenceRoot = Path.Combine(artifactsRoot, "owner-run-evidence");
        string runMetadataPath = Path.Combine(artifactsRoot, "github-run-view.json");
        string workflowRunLogPath = Path.Combine(evidenceRoot, "workflow-run.log");
        string artifactManifestPath = Path.Combine(evidenceRoot, "artifact-manifest.json");
        string outputRoot = Path.Combine(tempRoot, "final-release");
        string importPath = Path.Combine(outputRoot, "github-actions-run-evidence-import.json");
        string importMarkdownPath = Path.Combine(outputRoot, "github-actions-run-evidence-import.md");

        try
        {
            Directory.CreateDirectory(packageRoot);
            Directory.CreateDirectory(releaseGateRoot);
            Directory.CreateDirectory(finalReleaseRoot);
            Directory.CreateDirectory(evidenceRoot);
            Directory.CreateDirectory(outputRoot);

            File.WriteAllText(
                Path.Combine(releaseGateRoot, "release-quality-gate-summary.json"),
                """
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
                  "runAttempt": 2,
                  "workflowName": "release-quality-gate",
                  "workflowFile": ".github/workflows/release-quality-gate.yml",
                  "event": "workflow_dispatch",
                  "headBranch": "TensorRtSharp4.0",
                  "ref": "refs/heads/TensorRtSharp4.0",
                  "startedAtUtc": "2026-07-13T02:00:00Z",
                  "completedAtUtc": "2026-07-13T02:18:00Z",
                  "jobs": [
                    { "name": "source-quality", "status": "completed", "conclusion": "success" },
                    { "name": "package-managed-dry-run / pack", "status": "completed", "conclusion": "success" },
                    { "name": "package-managed-dry-run / publish-nuget", "status": "completed", "conclusion": "skipped" },
                    { "name": "package-managed-dry-run / publish-github-packages", "status": "completed", "conclusion": "skipped" }
                  ]
                }
                """);

            string nupkgPath = Path.Combine(packageRoot, "JYPPX.TensorRT.CSharp.API.4.0.0.nupkg");
            CreateMinimalManagedNupkg(nupkgPath);

            File.WriteAllText(
                workflowRunLogPath,
                """
                release-quality-gate run 29160655818
                source-quality: success
                package-managed-dry-run / pack: success
                publish jobs: skipped
                """);
            File.WriteAllText(
                artifactManifestPath,
                $$"""
                {
                  "runId": "{{runId}}",
                  "headSha": "{{headSha}}",
                  "packages": [
                    {
                      "fileName": "JYPPX.TensorRT.CSharp.API.4.0.0.nupkg",
                      "sha256": "{{Sha256(nupkgPath)}}"
                    }
                  ]
                }
                """);

            RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Export-GitHubActionsRunEvidenceImport.ps1"),
                "-RunId",
                runId,
                "-ArtifactsRoot",
                artifactsRoot,
                "-RunMetadataPath",
                runMetadataPath,
                "-ExpectedHeadSha",
                headSha,
                "-WorkflowRunLogPath",
                workflowRunLogPath,
                "-ArtifactManifestPath",
                artifactManifestPath,
                "-OwnerReviewer",
                "guojin-yan",
                "-CapturedAtUtc",
                "2026-07-13T02:30:00Z",
                "-OutputPath",
                importPath,
                "-MarkdownOutputPath",
                importMarkdownPath);

            RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Test-GitHubActionsRunEvidenceImport.ps1"),
                "-InputPath",
                importPath,
                "-OutputRoot",
                outputRoot,
                "-Strict");

            using JsonDocument importDocument = ReadJson(importPath);
            JsonElement importRoot = importDocument.RootElement;
            Assert.Equal(Sha256(workflowRunLogPath), importRoot.GetProperty("workflowRunLogSha256").GetString());
            Assert.Equal(Sha256(artifactManifestPath), importRoot.GetProperty("artifactManifestSha256").GetString());
            Assert.Equal("guojin-yan", importRoot.GetProperty("ownerReviewer").GetString());
            Assert.False(importRoot.GetProperty("isGitHubActionsProof").GetBoolean());

            using JsonDocument validationDocument = ReadJson(Path.Combine(outputRoot, "github-actions-run-evidence-import-validation.json"));
            JsonElement validation = validationDocument.RootElement;
            Assert.Equal("github-actions-run-evidence-ready", validation.GetProperty("validationState").GetString());
            Assert.True(validation.GetProperty("githubActionsRunEvidenceReady").GetBoolean());
            Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
            Assert.Equal(0, validation.GetProperty("failedActionRequiredCount").GetInt32());
            Assert.Equal(runId, validation.GetProperty("runId").GetString());
            Assert.Equal("release-quality-gate", validation.GetProperty("workflowName").GetString());
            Assert.Equal(".github/workflows/release-quality-gate.yml", validation.GetProperty("workflowFile").GetString());
            Assert.Equal("2", validation.GetProperty("runAttempt").GetString());
            Assert.Equal(Sha256(workflowRunLogPath), validation.GetProperty("workflowRunLogSha256").GetString());
            Assert.Equal(Sha256(artifactManifestPath), validation.GetProperty("artifactManifestSha256").GetString());
            Assert.False(validation.GetProperty("performsPublish").GetBoolean());
            Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(validation.GetProperty("isRuntimeExecutionProof").GetBoolean());
            Assert.False(validation.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
            Assert.False(validation.GetProperty("isPostPublishProof").GetBoolean());
            Assert.False(validation.GetProperty("isReleaseCloseProof").GetBoolean());
            Assert.False(validation.GetProperty("isGitHubActionsProof").GetBoolean());
            AssertValidationItem(validation, "workflow-run-log-hash-match", passed: true);
            AssertValidationItem(validation, "artifact-manifest-hash-match", passed: true);
            AssertValidationItem(validation, "forbidden-substitutes-absent", passed: true);
        }
        finally
        {
            DeleteTempRoot(tempRoot);
        }
    }

    [Fact]
    public void ForbiddenSubstitutesBecomeBlockers()
    {
        string tempRoot = CreateTempRoot();
        string outputRoot = Path.Combine(tempRoot, "final-release");
        string importPath = Path.Combine(tempRoot, "github-actions-run-evidence-import.json");

        try
        {
            Directory.CreateDirectory(outputRoot);
            File.WriteAllText(
                importPath,
                """
                {
                  "recordKind": "github-actions-run-evidence-import",
                  "runId": "29160655818",
                  "runUrl": "https://github.com/guojin-yan/TensorRT-CSharp-API/actions",
                  "runStatus": "queued",
                  "runConclusion": "success",
                  "runAttempt": "1",
                  "workflowName": "release-quality-gate",
                  "workflowFile": ".github/workflows/release-quality-gate.yml",
                  "runEvent": "workflow_dispatch",
                  "runBranch": "TensorRtSharp4.0",
                  "runRef": "refs/heads/TensorRtSharp4.0",
                  "startedAtUtc": "2026-07-13T02:00:00Z",
                  "completedAtUtc": "2026-07-13T02:18:00Z",
                  "headSha": "72d65909a120e1e550568e01796bbbdb5ec2b2e4",
                  "expectedHeadSha": "72d65909a120e1e550568e01796bbbdb5ec2b2e4",
                  "sourceQualityConclusion": "success",
                  "packageManagedDryRunPackConclusion": "success",
                  "publishNugetConclusion": "skipped",
                  "publishGitHubPackagesConclusion": "skipped",
                  "failedBlockerCount": 0,
                  "canClaimGitHubActionsPackageDryRunPackForRun": true,
                  "workflowRunLogPath": "artifacts/package-managed-dry-run/JYPPX.TensorRT.CSharp.API.4.0.0.nupkg",
                  "workflowRunLogSha256": "0000000000000000000000000000000000000000000000000000000000000000",
                  "artifactManifestPath": "artifacts/package-managed-dry-run/artifact-manifest.json",
                  "artifactManifestSha256": "1111111111111111111111111111111111111111111111111111111111111111",
                  "ownerReviewer": "manual approval",
                  "capturedAtUtc": "2026-07-13T02:30:00Z",
                  "nupkgPackages": [
                    {
                      "fileName": "JYPPX.TensorRT.CSharp.API.4.0.0.nupkg",
                      "sha256": "2222222222222222222222222222222222222222222222222222222222222222"
                    }
                  ],
                  "notExecutedByAutomation": true,
                  "ownerExecutionOnly": true,
                  "performsPublish": false,
                  "usesPublishToken": false,
                  "canPromoteRuntimeProof": false,
                  "canPublishPublicly": false,
                  "canCloseReleaseIssue": false,
                  "isRuntimeExecutionProof": false,
                  "isPackageConsumerRuntimeProof": false,
                  "isPostPublishProof": false,
                  "isReleaseCloseProof": false,
                  "isGitHubActionsProof": false
                }
                """);

            RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Test-GitHubActionsRunEvidenceImport.ps1"),
                "-InputPath",
                importPath,
                "-OutputRoot",
                outputRoot);

            using JsonDocument document = ReadJson(Path.Combine(outputRoot, "github-actions-run-evidence-import-validation.json"));
            JsonElement validation = document.RootElement;
            Assert.Equal("invalid-github-actions-run-evidence-import", validation.GetProperty("validationState").GetString());
            Assert.True(validation.GetProperty("failedBlockerCount").GetInt32() > 0);
            AssertValidationItem(validation, "forbidden-substitutes-absent", passed: false);

            string[] findings = validation.GetProperty("forbiddenSubstituteFindings")
                .EnumerateArray()
                .Select(static item => item.GetString()!)
                .ToArray();
            Assert.Contains(findings, static finding => finding.Contains("queued", StringComparison.OrdinalIgnoreCase));
            Assert.Contains(findings, static finding => finding.Contains("dashboard", StringComparison.OrdinalIgnoreCase));
            Assert.Contains(findings, static finding => finding.Contains("package-managed-dry-run", StringComparison.OrdinalIgnoreCase));
            Assert.Contains(findings, static finding => finding.Contains("manual-approval", StringComparison.OrdinalIgnoreCase));
        }
        finally
        {
            DeleteTempRoot(tempRoot);
        }
    }

    private static JsonDocument ReadJson(string path)
    {
        return JsonDocument.Parse(File.ReadAllText(path));
    }

    private static void AssertValidationItem(JsonElement validation, string id, bool passed)
    {
        Assert.Contains(validation.GetProperty("validationItems").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == id &&
            item.GetProperty("passed").GetBoolean() == passed);
    }

    private static string CreateTempRoot()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "trtsharp-gha-run-evidence-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);
        return tempRoot;
    }

    private static void DeleteTempRoot(string tempRoot)
    {
        if (Directory.Exists(tempRoot))
        {
            Directory.Delete(tempRoot, recursive: true);
        }
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

    private static string Sha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
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
        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException($"PowerShell command failed with exit code {process.ExitCode}: {scriptPath}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        }

        return stdout;
    }
}

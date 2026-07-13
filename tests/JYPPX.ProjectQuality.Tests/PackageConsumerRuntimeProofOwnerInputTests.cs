using System.Diagnostics;
using System.IO.Compression;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PackageConsumerRuntimeProofOwnerInputTests
{
    [Fact]
    public void PackageConsumerRuntimeProofOwnerInputExportsBlockedOverlaySurface()
    {
        const string sourceQualityContextPath = "artifacts/final-release/package-consumer-runtime-proof-owner-input.source-quality-context.json";
        string sourceFixtureRoot = PrepareSourceQualityRunEvidenceImportFixture(out string sourceRunId, out string sourceHeadSha, "package-consumer-runtime-proof-owner-input.source-quality-context.json");
        string fixtureRoot = PrepareGitHubActionsRunEvidenceImportFixture(out string runId, out string headSha);
        try
        {
            RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1"),
                "-SourceQualityRunEvidenceImportPath",
                sourceQualityContextPath);
            RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PackageConsumerRuntimeProofOwnerInput.ps1"), "-Strict");
        }
        finally
        {
            if (Directory.Exists(fixtureRoot))
            {
                Directory.Delete(fixtureRoot, recursive: true);
            }

            if (Directory.Exists(sourceFixtureRoot))
            {
                Directory.Delete(sourceFixtureRoot, recursive: true);
            }
        }

        using JsonDocument templateDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-owner-input.template.json");
        JsonElement template = templateDocument.RootElement;
        Assert.Equal("package-consumer-runtime-proof-owner-input", template.GetProperty("recordKind").GetString());
        Assert.Equal("template-owner-input-required", template.GetProperty("ownerInputState").GetString());
        Assert.Equal("package-consumer-runtime", template.GetProperty("proofLineId").GetString());
        Assert.Equal(ReadGitHead(), template.GetProperty("currentHead").GetString());
        Assert.Equal(sourceRunId, template.GetProperty("sourceQualityRunId").GetString());
        Assert.Equal(sourceHeadSha, template.GetProperty("sourceQualityHeadSha").GetString());
        Assert.True(template.GetProperty("sourceQualityRunEvidenceReady").GetBoolean());
        Assert.True(template.GetProperty("sourceQualityHeadMatchesCurrentHead").GetBoolean());
        Assert.Equal(runId, template.GetProperty("packageDryRunRunId").GetString());
        Assert.Equal(headSha, template.GetProperty("packageDryRunHeadSha").GetString());
        Assert.False(template.GetProperty("packageDryRunHeadMatchesCurrentHead").GetBoolean());
        Assert.Equal(runId, template.GetProperty("sourceGitHubActionsRunId").GetString());
        Assert.Equal(headSha, template.GetProperty("sourceHeadSha").GetString());
        Assert.True(template.GetProperty("packageDryRunCanClaimPack").GetBoolean());
        Assert.False(template.GetProperty("packageDryRunCanClaimCurrentHeadPack").GetBoolean());
        Assert.True(template.GetProperty("packageDryRunRequiresOwnerAuthorization").GetBoolean());
        Assert.True(template.GetProperty("manualWorkflowDispatchNotPerformed").GetBoolean());
        Assert.True(template.GetProperty("isDryRunOnly").GetBoolean());
        Assert.False(template.GetProperty("isPublishedPackageProof").GetBoolean());
        Assert.False(template.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.Equal(
            template.GetProperty("packageDryRunManagedNupkgSha256").GetString(),
            ReadFinalReleaseJson("github-actions-run-evidence-import.json").RootElement
                .GetProperty("nupkgPackages")
                .EnumerateArray()
                .Single()
                .GetProperty("sha256")
                .GetString());
        Assert.Contains("--runtime-package-key", template.GetProperty("smokeCommand").GetString(), StringComparison.Ordinal);
        Assert.False(template.GetProperty("canPromoteProof").GetBoolean());
        Assert.False(template.GetProperty("performsPublish").GetBoolean());
        Assert.False(template.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(template.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument validationDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-owner-input-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("package-consumer-runtime-proof-owner-input-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-input-required", validation.GetProperty("validationState").GetString());
        Assert.True(validation.GetProperty("isValidOwnerInputShape").GetBoolean());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        Assert.False(validation.GetProperty("canPromoteProof").GetBoolean());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.True(validation.GetProperty("sourceQualityRunEvidenceImportPresent").GetBoolean());
        Assert.True(validation.GetProperty("sourceQualityRunEvidenceReady").GetBoolean());
        Assert.True(validation.GetProperty("sourceQualityHeadMatchesCurrentHead").GetBoolean());
        Assert.True(validation.GetProperty("packageDryRunEvidenceImportPresent").GetBoolean());
        Assert.True(validation.GetProperty("packageDryRunEvidenceCanClaimPackForRun").GetBoolean());
        Assert.False(validation.GetProperty("packageDryRunHeadMatchesCurrentHead").GetBoolean());
        Assert.False(validation.GetProperty("packageDryRunCanClaimCurrentHeadPack").GetBoolean());
        Assert.False(validation.GetProperty("packageDryRunCurrentHeadClaimReady").GetBoolean());
        Assert.True(validation.GetProperty("packageDryRunRequiresOwnerAuthorization").GetBoolean());
        Assert.True(validation.GetProperty("manualWorkflowDispatchNotPerformed").GetBoolean());
        Assert.True(validation.GetProperty("sourceGitHubActionsRunEvidenceImportPresent").GetBoolean());
        Assert.True(validation.GetProperty("sourceGitHubActionsDryRunPackClaimReady").GetBoolean());
        Assert.True(validation.GetProperty("dryRunOnlyNotProof").GetBoolean());
        Assert.False(validation.GetProperty("isPublishedPackageProof").GetBoolean());
        Assert.False(validation.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());

        string[] validationItemIds = validation.GetProperty("validationItems")
            .EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();
        Assert.Contains("source-quality-run-evidence-present", validationItemIds);
        Assert.Contains("source-quality-run-evidence-ready", validationItemIds);
        Assert.Contains("source-quality-current-head-match", validationItemIds);
        Assert.Contains("package-dry-run-evidence-present", validationItemIds);
        Assert.Contains("package-dry-run-current-head-match", validationItemIds);
        Assert.Contains("package-dry-run-current-head-claim-ready", validationItemIds);
        Assert.Contains("source-github-actions-run-evidence-import-present", validationItemIds);
        Assert.Contains("source-github-actions-dry-run-pack-claim-ready", validationItemIds);
        Assert.Contains("dry-run-only-not-proof", validationItemIds);
        Assert.Contains("published-package-proof-false", validationItemIds);
        Assert.Contains("package-consumer-runtime-proof-false", validationItemIds);
        Assert.Contains("managed-nupkg-not-dry-run-artifact", validationItemIds);
        Assert.Contains("clean-root-outside-repository", validationItemIds);
        Assert.Contains("public-package-source-not-local", validationItemIds);
        Assert.Contains("no-project-reference-to-repository", validationItemIds);
        Assert.Contains("smoke-command-runtime-key", validationItemIds);
    }

    [Fact]
    public void SourceOnlyEvidenceDoesNotSatisfyPackageDryRunClaim()
    {
        string sourceFixtureRoot = PrepareSourceQualityRunEvidenceImportFixture(
            out _,
            out _,
            "package-consumer-runtime-proof-owner-input.source-only-evidence.json");
        const string sourceOnlyPath = "artifacts/final-release/package-consumer-runtime-proof-owner-input.source-only-evidence.json";

        try
        {
            RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1"),
                "-SourceQualityRunEvidenceImportPath",
                sourceOnlyPath,
                "-PackageDryRunEvidenceImportPath",
                sourceOnlyPath);
            RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PackageConsumerRuntimeProofOwnerInput.ps1"), "-Strict");
        }
        finally
        {
            if (Directory.Exists(sourceFixtureRoot))
            {
                Directory.Delete(sourceFixtureRoot, recursive: true);
            }
        }

        using JsonDocument templateDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-owner-input.template.json");
        JsonElement template = templateDocument.RootElement;
        Assert.True(template.GetProperty("sourceQualityRunEvidenceReady").GetBoolean());
        Assert.False(template.GetProperty("packageDryRunCanClaimPack").GetBoolean());
        Assert.False(template.GetProperty("packageDryRunCanClaimCurrentHeadPack").GetBoolean());
        Assert.True(template.GetProperty("packageDryRunRequiresOwnerAuthorization").GetBoolean());
        Assert.Equal("<no-package-managed-dry-run-artifact>", template.GetProperty("packageDryRunArtifactPath").GetString());
        Assert.Equal("<no-package-managed-dry-run-sha256>", template.GetProperty("packageDryRunManagedNupkgSha256").GetString());

        using JsonDocument validationDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-owner-input-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.True(validation.GetProperty("sourceQualityRunEvidenceReady").GetBoolean());
        Assert.True(validation.GetProperty("packageDryRunEvidenceImportPresent").GetBoolean());
        Assert.False(validation.GetProperty("packageDryRunEvidenceCanClaimPackForRun").GetBoolean());
        Assert.False(validation.GetProperty("sourceGitHubActionsDryRunPackClaimReady").GetBoolean());
        Assert.False(validation.GetProperty("packageDryRunCurrentHeadClaimReady").GetBoolean());
        Assert.True(validation.GetProperty("packageDryRunRequiresOwnerAuthorization").GetBoolean());
        Assert.False(FindValidationItem(validation, "source-github-actions-dry-run-pack-claim-ready").GetProperty("passed").GetBoolean());
        Assert.False(FindValidationItem(validation, "package-dry-run-current-head-claim-ready").GetProperty("passed").GetBoolean());
    }

    [Fact]
    public void OwnerInputValidatorRejectsDryRunArtifactAsManagedPackageProof()
    {
        string fixtureRoot = PrepareGitHubActionsRunEvidenceImportFixture(out _, out _);
        try
        {
            RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1"));

            using JsonDocument templateDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-owner-input.template.json");
            Dictionary<string, object?> values = templateDocument.RootElement
                .EnumerateObject()
                .ToDictionary(
                    static property => property.Name,
                    static property => property.Value.ValueKind switch
                    {
                        JsonValueKind.True => (object?)true,
                        JsonValueKind.False => (object?)false,
                        JsonValueKind.Array => property.Value.EnumerateArray().Select(static item => item.GetString()).ToArray(),
                        _ => property.Value.GetString()
                    });

            values["managedNupkgPath"] = templateDocument.RootElement.GetProperty("packageDryRunArtifactPath").GetString();
            values["managedNupkgSha256"] = templateDocument.RootElement.GetProperty("packageDryRunManagedNupkgSha256").GetString();

            string misusePath = Path.Combine(
                RepositoryPaths.Root,
                "artifacts",
                "final-release",
                "package-consumer-runtime-proof-owner-input.dryrun-misuse.json");
            File.WriteAllText(misusePath, JsonSerializer.Serialize(values, new JsonSerializerOptions { WriteIndented = true }));

            RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Test-PackageConsumerRuntimeProofOwnerInput.ps1"),
                "-InputPath",
                "artifacts/final-release/package-consumer-runtime-proof-owner-input.dryrun-misuse.json",
                "-Strict");

            using JsonDocument validationDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-owner-input-validation.json");
            JsonElement dryRunMisuseItem = validationDocument.RootElement
                .GetProperty("validationItems")
                .EnumerateArray()
                .Single(static item => item.GetProperty("id").GetString() == "managed-nupkg-not-dry-run-artifact");

            Assert.False(dryRunMisuseItem.GetProperty("passed").GetBoolean());
            Assert.Equal("blocked-owner-input-required", validationDocument.RootElement.GetProperty("validationState").GetString());
            Assert.False(validationDocument.RootElement.GetProperty("canPromoteProof").GetBoolean());
        }
        finally
        {
            if (Directory.Exists(fixtureRoot))
            {
                Directory.Delete(fixtureRoot, recursive: true);
            }
        }
    }

    private static JsonElement FindValidationItem(JsonElement validation, string id)
    {
        return validation.GetProperty("validationItems")
            .EnumerateArray()
            .Single(item => item.GetProperty("id").GetString() == id);
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static string PrepareSourceQualityRunEvidenceImportFixture(
        out string runId,
        out string headSha,
        string outputFileName = "github-actions-source-quality-run-evidence-import.json")
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "trtsharp-owner-input-source-quality-" + Guid.NewGuid().ToString("N"));
        runId = "29229700998";
        headSha = ReadGitHead();
        string artifactsRoot = Path.Combine(tempRoot, "artifacts", "github-actions-runs", runId);
        string releaseGateRoot = Path.Combine(artifactsRoot, "release-quality-gate", "release-quality-gate");
        string finalReleaseRoot = Path.Combine(artifactsRoot, "release-quality-gate", "final-release");
        string evidenceRoot = Path.Combine(artifactsRoot, "owner-run-evidence");
        string runMetadataPath = Path.Combine(artifactsRoot, "github-run-view.json");
        string workflowRunLogPath = Path.Combine(evidenceRoot, "workflow-run.log");
        string artifactManifestPath = Path.Combine(evidenceRoot, "artifact-manifest.json");

        Directory.CreateDirectory(releaseGateRoot);
        Directory.CreateDirectory(finalReleaseRoot);
        Directory.CreateDirectory(evidenceRoot);

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
            $"artifacts/final-release/{outputFileName}",
            "-MarkdownOutputPath",
            $"artifacts/final-release/{Path.ChangeExtension(outputFileName, ".md")}");

        return tempRoot;
    }

    private static string PrepareGitHubActionsRunEvidenceImportFixture(out string runId, out string headSha)
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "trtsharp-owner-input-run-evidence-" + Guid.NewGuid().ToString("N"));
        runId = "29160655818";
        headSha = "72d65909a120e1e550568e01796bbbdb5ec2b2e4";
        string artifactsRoot = Path.Combine(tempRoot, "artifacts", "github-actions-runs", runId);
        string packageRoot = Path.Combine(artifactsRoot, "package-managed-dry-run");
        string releaseGateRoot = Path.Combine(artifactsRoot, "release-quality-gate", "release-quality-gate");
        string finalReleaseRoot = Path.Combine(artifactsRoot, "release-quality-gate", "final-release");
        string runMetadataPath = Path.Combine(artifactsRoot, "github-run-view.json");

        Directory.CreateDirectory(packageRoot);
        Directory.CreateDirectory(releaseGateRoot);
        Directory.CreateDirectory(finalReleaseRoot);

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
              "jobs": [
                { "name": "source-quality", "status": "completed", "conclusion": "success" },
                { "name": "package-managed-dry-run / pack", "status": "completed", "conclusion": "success" },
                { "name": "package-managed-dry-run / publish-nuget", "status": "completed", "conclusion": "skipped" },
                { "name": "package-managed-dry-run / publish-github-packages", "status": "completed", "conclusion": "skipped" }
              ]
            }
            """);

        CreateMinimalManagedNupkg(Path.Combine(packageRoot, "JYPPX.TensorRT.CSharp.API.4.0.0.nupkg"));

        RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Export-GitHubActionsRunEvidenceImport.ps1"),
            "-RunId",
            runId,
            "-ArtifactsRoot",
            artifactsRoot,
            "-RunMetadataPath",
            runMetadataPath,
            "-OutputPath",
            "artifacts/final-release/github-actions-run-evidence-import.json",
            "-MarkdownOutputPath",
            "artifacts/final-release/github-actions-run-evidence-import.md");

        return tempRoot;
    }

    private static string ReadGitHead()
    {
        using Process process = new();
        process.StartInfo.FileName = "git";
        process.StartInfo.ArgumentList.Add("rev-parse");
        process.StartInfo.ArgumentList.Add("HEAD");
        process.StartInfo.WorkingDirectory = RepositoryPaths.Root;
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.RedirectStandardError = true;
        process.StartInfo.UseShellExecute = false;

        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"git rev-parse HEAD failed:{Environment.NewLine}{stdout}{stderr}");
        return stdout.Trim();
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

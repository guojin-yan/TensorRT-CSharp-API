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
        string fixtureRoot = PrepareGitHubActionsRunEvidenceImportFixture(out string runId, out string headSha);
        try
        {
            RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1"));
            RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PackageConsumerRuntimeProofOwnerInput.ps1"), "-Strict");
            RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofCandidate.ps1"),
                "-OwnerInputPath",
                "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json");
            RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PackageConsumerRuntimeProofCandidate.ps1"), "-Strict");
            RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        }
        finally
        {
            if (Directory.Exists(fixtureRoot))
            {
                Directory.Delete(fixtureRoot, recursive: true);
            }
        }

        using JsonDocument templateDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-owner-input.template.json");
        JsonElement template = templateDocument.RootElement;
        Assert.Equal("package-consumer-runtime-proof-owner-input", template.GetProperty("recordKind").GetString());
        Assert.Equal("template-owner-input-required", template.GetProperty("ownerInputState").GetString());
        Assert.Equal("package-consumer-runtime", template.GetProperty("proofLineId").GetString());
        Assert.Equal(runId, template.GetProperty("sourceGitHubActionsRunId").GetString());
        Assert.Equal(headSha, template.GetProperty("sourceHeadSha").GetString());
        Assert.True(template.GetProperty("packageDryRunCanClaimPack").GetBoolean());
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
        Assert.True(validation.GetProperty("sourceGitHubActionsRunEvidenceImportPresent").GetBoolean());
        Assert.True(validation.GetProperty("sourceGitHubActionsDryRunPackClaimReady").GetBoolean());
        Assert.True(validation.GetProperty("dryRunOnlyNotProof").GetBoolean());
        Assert.False(validation.GetProperty("isPublishedPackageProof").GetBoolean());
        Assert.False(validation.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());

        string[] validationItemIds = validation.GetProperty("validationItems")
            .EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();
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

        using JsonDocument candidateDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-candidate.json");
        JsonElement candidate = candidateDocument.RootElement;
        Assert.True(candidate.GetProperty("ownerInputOverlayApplied").GetBoolean());
        Assert.Equal("artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json", candidate.GetProperty("ownerInputPath").GetString());
        Assert.False(candidate.GetProperty("canPromoteProof").GetBoolean());
        Assert.False(candidate.GetProperty("performsPublish").GetBoolean());
        Assert.False(candidate.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(candidate.GetProperty("canCloseReleaseIssue").GetBoolean());

        string[] candidateSourceArtifacts = candidate.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json", candidateSourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json", candidateSourceArtifacts);

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("blocked-owner-input-required", evidence.GetProperty("packageConsumerRuntimeProofOwnerInputValidationState").GetString());
        Assert.True(evidence.GetProperty("packageConsumerRuntimeProofOwnerInputFailedActionRequiredCount").GetInt32() >= 1);
        Assert.Equal(0, evidence.GetProperty("packageConsumerRuntimeProofOwnerInputFailedBlockerCount").GetInt32());
        Assert.False(evidence.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("performsPublish").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "package-consumer-runtime-proof-owner-input");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        Assert.Contains("not proof", evidenceItem.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json", sourceArtifacts);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "package-consumer-runtime-proof-owner-input.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("articles/zh-cn/package-consumer-runtime-proof-owner-input.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/package-consumer-runtime-proof-owner-input.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime-proof-owner-input", article, StringComparison.Ordinal);
        Assert.Contains("package consumer runtime proof owner input validation: `blocked-owner-input-required`", evidenceMarkdown, StringComparison.Ordinal);
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

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
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

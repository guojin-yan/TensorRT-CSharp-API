using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PackageConsumerRuntimeProofRecordTests
{
    [Fact]
    public void PackageConsumerRuntimeProofRecordExportsTemplateOnlyStrictRecordAndBridge()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PackageConsumerRuntimeProofOwnerInput.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Import-PackageConsumerRuntimeProofOwnerInput.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofRecordTemplate.ps1"));
        RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Test-PackageConsumerRuntimeProofRecord.ps1"),
            "-InputPath",
            "artifacts/final-release/package-consumer-runtime-proof-record.template.json",
            "-Strict");
        RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofRecordFromOwnerInput.ps1"),
            "-OwnerInputPath",
            "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json");
        RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Test-PackageConsumerRuntimeProofRecord.ps1"),
            "-InputPath",
            "artifacts/final-release/package-consumer-runtime-proof-record.json",
            "-Strict");
        string failOnNotProofOutputRoot = Path.Combine(
            Path.GetTempPath(),
            "jyppx-package-proof-fail-" + Guid.NewGuid().ToString("N"));

        try
        {
            string failOnNotProofOutput = RunPowerShellExpectFailure(
                Path.Combine(RepositoryPaths.Root, "eng", "Test-PackageConsumerRuntimeProofRecord.ps1"),
                "-InputPath",
                "artifacts/final-release/package-consumer-runtime-proof-record.json",
                "-OutputRoot",
                failOnNotProofOutputRoot,
                "-Strict",
                "-RequireExistingLog",
                "-FailOnNotProof");
            Assert.Contains("not promotable", failOnNotProofOutput, StringComparison.OrdinalIgnoreCase);

            using JsonDocument failValidationDocument = JsonDocument.Parse(File.ReadAllText(Path.Combine(
                failOnNotProofOutputRoot,
                "package-consumer-runtime-proof-record-validation.json")));
            Assert.True(failValidationDocument.RootElement.GetProperty("failOnNotProofRequested").GetBoolean());
        }
        finally
        {
            if (Directory.Exists(failOnNotProofOutputRoot))
            {
                Directory.Delete(failOnNotProofOutputRoot, recursive: true);
            }
        }

        using JsonDocument ownerInputDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-owner-input.template.json");
        JsonElement ownerInput = ownerInputDocument.RootElement;
        Assert.Equal("package-consumer-runtime-proof-owner-input", ownerInput.GetProperty("recordKind").GetString());
        AssertOwnerInputTemplateContainsRuntimeProofFields(ownerInput);

        using JsonDocument ownerInputValidationDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-owner-input-validation.json");
        JsonElement ownerInputValidation = ownerInputValidationDocument.RootElement;
        Assert.Equal("package-consumer-runtime-proof-owner-input-validation", ownerInputValidation.GetProperty("recordKind").GetString());
        Assert.False(ownerInputValidation.GetProperty("performsPublish").GetBoolean());
        Assert.False(ownerInputValidation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(ownerInputValidation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(ownerInputValidation.GetProperty("canPromoteProof").GetBoolean());
        AssertValidationContains(ownerInputValidation, "field-ownerName");
        AssertValidationContains(ownerInputValidation, "field-machineName");
        AssertValidationContains(ownerInputValidation, "field-gpuName");
        AssertValidationContains(ownerInputValidation, "field-cudaDriverSupportedRuntime");
        AssertValidationContains(ownerInputValidation, "field-cudnnVersion");
        AssertValidationContains(ownerInputValidation, "field-tensorRtLine");
        AssertValidationContains(ownerInputValidation, "field-restoreCommand");
        AssertValidationContains(ownerInputValidation, "field-buildCommand");
        AssertValidationContains(ownerInputValidation, "exit-code-zero");
        AssertValidationContains(ownerInputValidation, "started-at-utc-parseable");
        AssertValidationContains(ownerInputValidation, "finished-at-utc-parseable");
        AssertValidationContains(ownerInputValidation, "dependency-probe-status-passed");
        AssertValidationContains(ownerInputValidation, "smoke-status-passed");
        AssertValidationContains(ownerInputValidation, "native-assets-copied-true");

        using JsonDocument ownerInputImportDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-owner-input-import.json");
        JsonElement ownerInputImport = ownerInputImportDocument.RootElement;
        Assert.Equal("package-consumer-runtime-proof-owner-input-import", ownerInputImport.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-input-import-action-required", ownerInputImport.GetProperty("importState").GetString());
        Assert.Equal("blocked-owner-input-required", ownerInputImport.GetProperty("ownerInputValidationState").GetString());
        Assert.True(ownerInputImport.GetProperty("placeholderFieldCount").GetInt32() >= 1);
        Assert.False(ownerInputImport.GetProperty("performsPublish").GetBoolean());
        Assert.False(ownerInputImport.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(ownerInputImport.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(ownerInputImport.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains("local feed, ProjectReference, direct .nupkg, template, dry-run, and build-only substitutes blocked", ownerInputImport.GetProperty("safetyBoundary").GetString(), StringComparison.Ordinal);
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ExternalRuntimeProofRecordFromPackageConsumerProof.ps1"));
        RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Test-ExternalRuntimeProofRecord.ps1"),
            "-InputPath",
            "artifacts/final-release/external-runtime-proof-record.json");
        RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofCandidate.ps1"),
            "-OwnerInputPath",
            "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PackageConsumerRuntimeProofCandidate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));

        using JsonDocument templateDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-record.template.json");
        JsonElement template = templateDocument.RootElement;
        Assert.Equal("package-consumer-runtime-proof-record-template", template.GetProperty("recordKind").GetString());
        Assert.True(template.GetProperty("templateOnly").GetBoolean());
        Assert.False(template.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(template.GetProperty("performsPublish").GetBoolean());
        Assert.False(template.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(template.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument recordDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-record.json");
        JsonElement record = recordDocument.RootElement;
        Assert.Equal("package-consumer-runtime-proof-record", record.GetProperty("recordKind").GetString());
        Assert.Equal("template-only", record.GetProperty("proofClassification").GetString());
        Assert.Equal("win-x64-trt11.0-cuda13.2-cudnn9.22", record.GetProperty("runtimePackageKey").GetString());
        Assert.False(record.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.Contains("--runtime-package-key", record.GetProperty("command").GetProperty("smokeCommand").GetString(), StringComparison.Ordinal);

        using JsonDocument validationDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-record-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("package-consumer-runtime-proof-record-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("template-only", validation.GetProperty("validationState").GetString());
        Assert.Equal("template-only", validation.GetProperty("proofClassification").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedProofItemCount").GetInt32() >= 1);
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("failOnNotProofRequested").GetBoolean());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument externalDocument = ReadFinalReleaseJson("external-runtime-proof-record.json");
        JsonElement external = externalDocument.RootElement;
        Assert.Equal("external-runtime-proof-record", external.GetProperty("recordKind").GetString());
        Assert.Equal("template-only", external.GetProperty("proofClassification").GetString());
        Assert.False(external.GetProperty("canPromoteRuntimeProof").GetBoolean());

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("template-only", evidence.GetProperty("packageConsumerRuntimeProofRecordValidationState").GetString());
        Assert.False(evidence.GetProperty("packageConsumerRuntimeProofRecordCanPromoteRuntimeProof").GetBoolean());
        Assert.True(evidence.GetProperty("packageConsumerRuntimeProofRecordFailedActionRequiredCount").GetInt32() >= 1);

        Assert.Contains(evidence.GetProperty("evidenceItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "package-consumer-runtime-proof-record" &&
            item.GetProperty("passed").GetBoolean() == false);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-record.template.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-record.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-record-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/external-runtime-proof-record.json", sourceArtifacts);

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "package-consumer-runtime-proof-record.md"));
        string evidenceMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-evidence-bundle.md"));

        Assert.Contains("articles/zh-cn/package-consumer-runtime-proof-record.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/package-consumer-runtime-proof-record.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("external-runtime-proof-record", article, StringComparison.Ordinal);
        Assert.Contains("local feed、ProjectReference、direct `.nupkg` 不是 proof", article, StringComparison.Ordinal);
        Assert.Contains("package consumer runtime proof record validation: `template-only`", evidenceMarkdown, StringComparison.Ordinal);

        string validator = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-PackageConsumerRuntimeProofRecord.ps1"));
        Assert.Contains("[switch]$FailOnNotProof", validator, StringComparison.Ordinal);
        Assert.Contains("if ($FailOnNotProof -and -not $canPromoteRuntimeProof)", validator, StringComparison.Ordinal);
        Assert.Contains("function Test-PublicPackageSourceIsLocal", validator, StringComparison.Ordinal);
        Assert.Contains("no-project-reference", validator, StringComparison.Ordinal);
        Assert.Contains("no-local-feed", validator, StringComparison.Ordinal);
        Assert.Contains("no-direct-nupkg", validator, StringComparison.Ordinal);
        Assert.Contains("Public package source and restore sources must not be local feed/folder evidence.", validator, StringComparison.Ordinal);
        Assert.Contains("Direct .nupkg references cannot be public proof.", validator, StringComparison.Ordinal);

        string ownerInputTemplateExporter = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1"));
        string ownerInputValidator = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-PackageConsumerRuntimeProofOwnerInput.ps1"));
        string recordTemplateExporter = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofRecordTemplate.ps1"));
        Assert.Contains("Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof", recordTemplateExporter, StringComparison.Ordinal);
        Assert.Contains("ownerName", ownerInputTemplateExporter, StringComparison.Ordinal);
        Assert.Contains("machineName", ownerInputTemplateExporter, StringComparison.Ordinal);
        Assert.Contains("gpuName", ownerInputTemplateExporter, StringComparison.Ordinal);
        Assert.Contains("cudaDriverSupportedRuntime", ownerInputTemplateExporter, StringComparison.Ordinal);
        Assert.Contains("cudnnVersion", ownerInputTemplateExporter, StringComparison.Ordinal);
        Assert.Contains("tensorRtLine", ownerInputTemplateExporter, StringComparison.Ordinal);
        Assert.Contains("restoreCommand", ownerInputTemplateExporter, StringComparison.Ordinal);
        Assert.Contains("buildCommand", ownerInputTemplateExporter, StringComparison.Ordinal);
        Assert.Contains("dependencyProbeStatus", ownerInputTemplateExporter, StringComparison.Ordinal);
        Assert.Contains("nativeAssetsCopied", ownerInputTemplateExporter, StringComparison.Ordinal);
        Assert.Contains("function Test-IntZero", ownerInputValidator, StringComparison.Ordinal);
        Assert.Contains("function Test-BoolTrue", ownerInputValidator, StringComparison.Ordinal);
        Assert.Contains("function Test-DateTimeOffsetFormat", ownerInputValidator, StringComparison.Ordinal);
        Assert.Contains("dependency-probe-status-passed", ownerInputValidator, StringComparison.Ordinal);
        Assert.Contains("native-assets-copied-true", ownerInputValidator, StringComparison.Ordinal);
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static void AssertOwnerInputTemplateContainsRuntimeProofFields(JsonElement ownerInput)
    {
        foreach (string fieldName in new[]
        {
            "ownerName",
            "machineName",
            "gpuName",
            "cudaDriverSupportedRuntime",
            "cudnnVersion",
            "tensorRtLine",
            "restoreCommand",
            "buildCommand",
            "exitCode",
            "startedAtUtc",
            "finishedAtUtc",
            "dependencyProbeStatus",
            "smokeStatus",
            "nativeAssetsCopied",
            "failureDiagnostic"
        })
        {
            Assert.True(ownerInput.TryGetProperty(fieldName, out _), $"Owner input template is missing {fieldName}.");
        }
    }

    private static void AssertValidationContains(JsonElement validation, string id)
    {
        Assert.Contains(validation.GetProperty("validationItems").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == id);
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

    private static string RunPowerShellExpectFailure(string scriptPath, params string[] arguments)
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

        Assert.NotEqual(0, process.ExitCode);
        return stdout + Environment.NewLine + stderr;
    }
}

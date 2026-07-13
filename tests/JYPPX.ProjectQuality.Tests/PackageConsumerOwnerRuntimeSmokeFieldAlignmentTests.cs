using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PackageConsumerOwnerRuntimeSmokeFieldAlignmentTests
{
    [Fact]
    public void FieldAlignmentExportsValidNonProofOwnerRuntimeSmokeMatrix()
    {
        string exportScript = ReadSource("eng", "Export-PackageConsumerOwnerRuntimeSmokeFieldAlignment.ps1");
        string testScript = ReadSource("eng", "Test-PackageConsumerOwnerRuntimeSmokeFieldAlignment.ps1");

        Assert.Contains("package-consumer-owner-runtime-smoke-field-alignment.json", exportScript, StringComparison.Ordinal);
        Assert.Contains("blocked-owner-compatible-host-runtime-smoke-field-alignment", exportScript, StringComparison.Ordinal);
        Assert.Contains("requiredOwnerInputFields", exportScript, StringComparison.Ordinal);
        Assert.Contains("Smoke=not-requested", exportScript, StringComparison.Ordinal);
        Assert.Contains("canPromoteRuntimeProof = $false", exportScript, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly = $false", exportScript, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue = $false", exportScript, StringComparison.Ordinal);

        Assert.Contains("package-consumer-owner-runtime-smoke-field-alignment-validation.json", testScript, StringComparison.Ordinal);
        Assert.Contains("schema-runbook-collection-coverage", testScript, StringComparison.Ordinal);
        Assert.Contains("final-owner-surface-coverage", testScript, StringComparison.Ordinal);
        Assert.Contains("canPromoteRuntimeProof = $false", testScript, StringComparison.Ordinal);

        RunAlignmentPipeline();

        using JsonDocument alignmentDocument = ReadFinalReleaseJson("package-consumer-owner-runtime-smoke-field-alignment.json");
        JsonElement alignment = alignmentDocument.RootElement;

        Assert.Equal("package-consumer-owner-runtime-smoke-field-alignment", alignment.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-compatible-host-runtime-smoke-field-alignment", alignment.GetProperty("alignmentState").GetString());
        Assert.Equal("blocked-owner-compatible-host-runtime-smoke", alignment.GetProperty("ownerRuntimeSmokeRunbookState").GetString());
        Assert.True(alignment.GetProperty("fieldCount").GetInt32() >= 30);
        Assert.Equal("Smoke=not-requested", alignment.GetProperty("runtimeSmokeStatus").GetString());
        Assert.Equal(0, alignment.GetProperty("missingRequiredFieldCount").GetInt32());
        Assert.False(alignment.GetProperty("performsPublish").GetBoolean());
        Assert.False(alignment.GetProperty("approvesPublicRelease").GetBoolean());
        Assert.False(alignment.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(alignment.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(alignment.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(alignment.GetProperty("isRuntimeExecutionProof").GetBoolean());

        JsonElement[] fields = alignment.GetProperty("fields").EnumerateArray().ToArray();
        foreach (string name in new[]
        {
            "cleanExternalConsumerRoot",
            "consumerProjectPath",
            "publicPackageSource",
            "managedNupkgSha256",
            "runtimeNupkgSha256",
            "smokeLogSha256",
            "stdoutSummary",
            "stderrSummary",
            "gpuName",
            "cudaRuntimeVersion",
            "tensorRtVersion",
            "smokeStatus",
        })
        {
            JsonElement field = Assert.Single(fields, item => item.GetProperty("name").GetString() == name);
            Assert.True(field.GetProperty("presentInOwnerSchema").GetBoolean(), name);
            Assert.True(field.GetProperty("presentInRunbookRequiredOwnerInputFields").GetBoolean(), name);
            Assert.True(field.GetProperty("presentInCollectionBundle").GetBoolean(), name);
            Assert.False(string.IsNullOrWhiteSpace(field.GetProperty("validator").GetString()));
        }

        string[] forbidden = alignment.GetProperty("forbiddenRuntimeSmokeSubstitutes").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("local feed", forbidden);
        Assert.Contains("ProjectReference", forbidden);
        Assert.Contains("direct .nupkg", forbidden);
        Assert.Contains("Smoke=not-requested", forbidden);
        Assert.Contains("dependency-probe-only", forbidden);
        Assert.Contains("blocked-by-cuda-driver", forbidden);

        using JsonDocument validationDocument = ReadFinalReleaseJson("package-consumer-owner-runtime-smoke-field-alignment-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("package-consumer-owner-runtime-smoke-field-alignment-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-compatible-host-runtime-smoke-field-alignment-valid", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("blocked-owner-compatible-host-runtime-smoke-field-alignment", evidence.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentState").GetString());
        Assert.Equal("blocked-owner-compatible-host-runtime-smoke-field-alignment-valid", evidence.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState").GetString());
        Assert.Equal(0, evidence.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount").GetInt32());
        Assert.False(evidence.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentCanPromoteRuntimeProof").GetBoolean());
        Assert.False(evidence.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentCanCloseReleaseIssue").GetBoolean());
        Assert.Contains(evidence.GetProperty("evidenceItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "package-consumer-owner-runtime-smoke-field-alignment" &&
            item.GetProperty("passed").GetBoolean() == false &&
            item.GetProperty("boundary").GetString()!.Contains("not runtime proof", StringComparison.Ordinal));
        Assert.Contains(evidence.GetProperty("sourceArtifacts").EnumerateArray(), static item =>
            item.GetString() == "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment.json");
        Assert.Contains(evidence.GetProperty("sourceArtifacts").EnumerateArray(), static item =>
            item.GetString() == "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment-validation.json");

        string markdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "package-consumer-owner-runtime-smoke-field-alignment.md"));
        Assert.Contains("Package Consumer Owner Runtime Smoke Field Alignment", markdown, StringComparison.Ordinal);
        Assert.Contains("Smoke=not-requested", markdown, StringComparison.Ordinal);
        Assert.Contains("canPromoteRuntimeProof", markdown, StringComparison.Ordinal);
    }

    internal static void RunAlignmentPipeline()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofOwnerInputSchema.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PackageConsumerRuntimeProofOwnerInput.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CompatibleHostRuntimeProofRunbook.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-CompatibleHostRuntimeProofRunbook.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CompatibleHostRuntimeProofCollectionBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerOwnerRuntimeSmokeFieldAlignment.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PackageConsumerOwnerRuntimeSmokeFieldAlignment.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
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
        return stdout + stderr;
    }
}

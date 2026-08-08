using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class PackageConsumerRuntimeProofOwnerInputImportTests
{
    [Fact]
    public void OwnerInputImportKeepsPublishBlockedAndProjectsRecordArtifacts()
    {
        string importerPath = Path.Combine(RepositoryPaths.Root, "eng", "Import-PackageConsumerRuntimeProofOwnerInput.ps1");
        Assert.True(File.Exists(importerPath), "Owner input import script must exist.");

        string script = File.ReadAllText(importerPath);
        Assert.Contains("package-consumer-runtime-proof-owner-input.imported.json", script, StringComparison.Ordinal);
        Assert.Contains("Test-PackageConsumerRuntimeProofOwnerInput.ps1", script, StringComparison.Ordinal);
        Assert.Contains("Export-PackageConsumerRuntimeProofOwnerInputSchema.ps1", script, StringComparison.Ordinal);
        Assert.Contains("Export-PackageConsumerRuntimeProofForbiddenSubstituteScan.ps1", script, StringComparison.Ordinal);
        Assert.Contains("Export-PackageConsumerRuntimeProofRecordFromOwnerInput.ps1", script, StringComparison.Ordinal);
        Assert.Contains("Test-PackageConsumerRuntimeProofRecord.ps1", script, StringComparison.Ordinal);
        Assert.Contains("does not publish packages", script, StringComparison.Ordinal);
        Assert.Contains("canPromoteRuntimeProof = $false", script, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly = $false", script, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue = $false", script, StringComparison.Ordinal);

        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1"));
        RunPowerShell(importerPath, "-Strict");

        using JsonDocument importDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-owner-input-import.json");
        JsonElement import = importDocument.RootElement;
        Assert.Equal("package-consumer-runtime-proof-owner-input-import", import.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-input-import-action-required", import.GetProperty("importState").GetString());
        Assert.Equal("blocked-owner-input-required", import.GetProperty("ownerInputValidationState").GetString());
        Assert.Equal("template-only", import.GetProperty("projectedRecordProofClassification").GetString());
        Assert.True(import.GetProperty("placeholderFieldCount").GetInt32() >= 1);
        Assert.False(import.GetProperty("performsPublish").GetBoolean());
        Assert.False(import.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(import.GetProperty("canPromoteProof").GetBoolean());
        Assert.False(import.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(import.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal("blocked-forbidden-substitute-detected", import.GetProperty("forbiddenSubstituteScanState").GetString());
        Assert.True(import.GetProperty("detectedForbiddenSubstituteCount").GetInt32() >= 1);
        string safetyBoundary = import.GetProperty("safetyBoundary").GetString()!;
        Assert.Contains("local feed, ProjectReference, direct .nupkg, template, dry-run, and build-only substitutes blocked", safetyBoundary, StringComparison.Ordinal);
        Assert.Contains("queued GitHub Actions run", safetyBoundary, StringComparison.Ordinal);
        Assert.Contains("missing self-hosted runner", safetyBoundary, StringComparison.Ordinal);
        Assert.Contains("repository path leakage", safetyBoundary, StringComparison.Ordinal);
        Assert.Contains("GUI screenshots", safetyBoundary, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec build reports", safetyBoundary, StringComparison.Ordinal);

        string[] sourceArtifacts = import.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts\\final-release\\package-consumer-runtime-proof-owner-input.template.json", sourceArtifacts);
        Assert.Contains("artifacts\\final-release\\package-consumer-runtime-proof-owner-input.imported.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-record.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-record-validation.json", sourceArtifacts);

        using JsonDocument importedOwnerInputDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-owner-input.imported.json");
        Assert.Equal("package-consumer-runtime-proof-owner-input", importedOwnerInputDocument.RootElement.GetProperty("recordKind").GetString());
        Assert.True(importedOwnerInputDocument.RootElement.TryGetProperty("publicPackageSourceKind", out _));
        Assert.True(importedOwnerInputDocument.RootElement.TryGetProperty("publicPackageFeedUrl", out _));
        Assert.True(importedOwnerInputDocument.RootElement.TryGetProperty("managedPackageUrl", out _));
        Assert.True(importedOwnerInputDocument.RootElement.TryGetProperty("runtimePackageUrl", out _));
        Assert.True(importedOwnerInputDocument.RootElement.TryGetProperty("sourceRunnerQueueStatus", out _));
        Assert.True(importedOwnerInputDocument.RootElement.TryGetProperty("sourceRunnerInfrastructureStatus", out _));
        Assert.True(importedOwnerInputDocument.RootElement.TryGetProperty("sourceRunnerOwnerAction", out _));

        string importMarkdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "package-consumer-runtime-proof-owner-input-import.md"));
        Assert.Contains("Package Consumer Runtime Proof Owner Input Import", importMarkdown, StringComparison.Ordinal);
        Assert.Contains("blocked-owner-input-import-action-required", importMarkdown, StringComparison.Ordinal);
        Assert.Contains("forbiddenSubstituteScanState", importMarkdown, StringComparison.Ordinal);
        Assert.Contains("detectedForbiddenSubstituteCount", importMarkdown, StringComparison.Ordinal);
        Assert.Contains("canPromoteRuntimeProof", importMarkdown, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue", importMarkdown, StringComparison.Ordinal);
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
        process.StartInfo.FileName = PowerShellHost.ResolveExecutable();
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

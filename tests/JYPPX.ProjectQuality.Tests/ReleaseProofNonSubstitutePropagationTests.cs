using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseProofNonSubstitutePropagationTests
{
    private static readonly string[] RequiredNonSubstitutes =
    {
        "managed-readiness",
        "managed-readiness-only",
        "callback-allocator-readiness-snapshot",
        "CallbackAllocatorReadinessSnapshot",
        "TensorRtCallbackAllocatorReadinessSnapshot",
        "precheck-only",
        "dry-run-only",
        "schema-only",
    };

    [Fact]
    public void ReleaseRuntimeProofMatrixPropagatesReadinessPrecheckDryRunAndSchemaOnlyBoundaries()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-PackageConsumerRuntimeProofExecutionPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PackageConsumerRuntimeProofPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CallbackRuntimeProofExecutionPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-CallbackRuntimeProofExecutionPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseRuntimeProofExecutionMatrix.ps1"));

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "release-runtime-proof-execution-matrix.json")));

        JsonElement root = document.RootElement;
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isReleaseProofComplete").GetBoolean());

        string[] sourceArtifacts = root.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/callback-runtime-proof-execution-pack.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/callback-runtime-proof-execution-pack-validation.json", sourceArtifacts);

        string[] rootNonSubstitutes = root.GetProperty("nonSubstituteProofKinds").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string item in RequiredNonSubstitutes)
        {
            Assert.Contains(item, rootNonSubstitutes);
        }

        foreach (JsonElement proofItem in root.GetProperty("proofItems").EnumerateArray())
        {
            string[] itemNonSubstitutes = proofItem.GetProperty("nonSubstituteProofKinds").EnumerateArray().Select(static item => item.GetString()!).ToArray();
            foreach (string item in RequiredNonSubstitutes)
            {
                Assert.Contains(item, itemNonSubstitutes);
            }
        }

        Assert.Contains("CallbackAllocatorReadinessSnapshot", root.GetProperty("boundary").GetString()!, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseReadinessAndClosePreflightPropagateReadinessPrecheckDryRunAndSchemaOnlyBoundaries()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseRuntimeProofExecutionMatrix.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseProofReadinessSnapshot.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseClosePreflight.ps1"));

        using JsonDocument readiness = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "release-proof-readiness-snapshot.json")));
        using JsonDocument preflight = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "release-close-preflight.json")));

        foreach (JsonElement root in new[] { readiness.RootElement, preflight.RootElement })
        {
            string[] nonSubstitutes = root.GetProperty("nonSubstituteProofKinds").EnumerateArray().Select(static item => item.GetString()!).ToArray();
            foreach (string item in RequiredNonSubstitutes)
            {
                Assert.Contains(item, nonSubstitutes);
            }

            Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        }
    }

    [Fact]
    public void StaleReleaseClaimsAuditContainsManagedReadinessProofOverclaimRules()
    {
        string script = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-StaleReleaseClaims.ps1"));
        foreach (string marker in new[]
        {
            "managed-readiness means runtime proof",
            "CallbackAllocatorReadinessSnapshot means runtime proof",
            "precheck-only means runtime proof",
            "dry-run-only means runtime proof",
            "schema-only means runtime proof",
        })
        {
            Assert.Contains(marker, script, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ReleaseEvidenceBundleAndOwnerExecutionPackagePropagateReadinessBoundaries()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CallbackRuntimeProofExecutionPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-CallbackRuntimeProofExecutionPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerReleaseExecutionPackage.ps1"));

        using JsonDocument evidence = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "release-evidence-bundle.json")));
        using JsonDocument ownerPackage = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "owner-release-execution-package.json")));

        string[] evidenceNonSubstitutes = evidence.RootElement.GetProperty("nonSubstituteProofKinds")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        string[] ownerMustNotSubstitute = ownerPackage.RootElement.GetProperty("mustNotSubstitute")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        foreach (string item in RequiredNonSubstitutes)
        {
            Assert.Contains(item, evidenceNonSubstitutes);
            Assert.Contains(item, ownerMustNotSubstitute);
        }

        string[] evidenceSources = evidence.RootElement.GetProperty("sourceArtifacts")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        string[] ownerSources = ownerPackage.RootElement.GetProperty("sourceArtifacts")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        Assert.Contains("artifacts/final-release/callback-runtime-proof-execution-pack.json", evidenceSources);
        Assert.Contains("artifacts/final-release/callback-runtime-proof-execution-pack-validation.json", evidenceSources);
        Assert.Contains("artifacts/final-release/callback-runtime-proof-execution-pack.json", ownerSources);
        Assert.Contains("artifacts/final-release/callback-runtime-proof-execution-pack-validation.json", ownerSources);

        string ownerSafetyNotes = string.Join("\n", ownerPackage.RootElement.GetProperty("safetyNotes").EnumerateArray().Select(static item => item.GetString()));
        string evidenceSafetyNotes = string.Join("\n", evidence.RootElement.GetProperty("safetyNotes").EnumerateArray().Select(static item => item.GetString()));
        Assert.Contains("CallbackAllocatorReadinessSnapshot", ownerSafetyNotes, StringComparison.Ordinal);
        Assert.Contains("RuntimeEvidenceKind=managed-readiness", evidenceSafetyNotes, StringComparison.Ordinal);
        Assert.False(evidence.RootElement.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(ownerPackage.RootElement.GetProperty("canPublishPublicly").GetBoolean());
    }

    private static string RunPowerShell(string scriptPath)
    {
        using Process process = new();
        process.StartInfo.FileName = "pwsh";
        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(scriptPath);
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

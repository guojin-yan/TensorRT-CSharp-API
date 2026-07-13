using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class RuntimeProofEvidenceBackfillTests
{
    private static readonly string[] RuntimeProofNonSubstitutes =
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
    public void ExternalRuntimeProofValidatorRejectsReadinessPrecheckDryRunAndSchemaOnlyEvidence()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ExternalRuntimeProofRecord.ps1"));

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "external-runtime-proof-validation.json")));

        JsonElement root = document.RootElement;
        string[] nonSubstitutes = root.GetProperty("nonSubstituteProofKinds")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        foreach (string item in RuntimeProofNonSubstitutes)
        {
            Assert.Contains(item, nonSubstitutes);
        }

        string[] promotionRules = root.GetProperty("promotionRules")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        Assert.Contains(promotionRules, static rule => rule.Contains("RuntimeEvidenceKind=managed-readiness", StringComparison.Ordinal));
        Assert.Contains(promotionRules, static rule => rule.Contains("precheck-only, dry-run-only, schema-only, and managed-readiness-only", StringComparison.Ordinal));
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionEvidence").GetBoolean());
    }

    [Fact]
    public void PackageConsumerAndRealCaseValidatorsKeepReadinessPrecheckDryRunAndSchemaOnlyBlocked()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PackageConsumerRuntimeProofPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-RealCaseEvidenceRecord.ps1"));

        using JsonDocument packageConsumer = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "package-consumer-runtime-proof-pack-validation.json")));
        using JsonDocument realCase = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "real-case-evidence-record-validation.json")));

        string packageBoundary = packageConsumer.RootElement.GetProperty("boundary").GetString()!;
        foreach (string item in new[]
        {
            "managed-readiness-only",
            "precheck-only",
            "dry-run-only",
            "schema-only",
            "CallbackAllocatorReadinessSnapshot",
        })
        {
            Assert.Contains(item, packageBoundary, StringComparison.Ordinal);
        }

        string realCaseBoundary = realCase.RootElement.GetProperty("boundary").GetString()!;
        foreach (string item in new[] { "managed-readiness", "precheck-only", "dry-run-only", "schema-only" })
        {
            Assert.Contains(item, realCaseBoundary, StringComparison.Ordinal);
        }

        Assert.False(packageConsumer.RootElement.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.False(realCase.RootElement.GetProperty("canPromoteRealModelRuntime").GetBoolean());
    }

    [Fact]
    public void RuntimeProofDocumentsNameReadinessPrecheckDryRunAndSchemaOnlyAsNonSubstitutes()
    {
        string callbackEvidence = ReadSource("docs", "articles", "zh-cn", "real-callback-runtime-evidence-schema.md");
        string packageConsumerPlaybook = ReadSource("docs", "articles", "zh-cn", "package-consumer-runtime-proof-playbook.md");
        string nonSubstitutes = ReadSource("docs", "articles", "zh-cn", "release-proof-non-substitutes.md");

        foreach (string item in RuntimeProofNonSubstitutes)
        {
            Assert.Contains(item, nonSubstitutes, StringComparison.Ordinal);
        }

        foreach (string item in new[]
        {
            "TensorRtCallbackAllocatorReadinessSnapshot",
            "RuntimeEvidenceKind=managed-readiness",
            "managed-readiness-only",
            "precheck-only",
            "dry-run-only",
            "schema-only",
        })
        {
            Assert.Contains(item, callbackEvidence, StringComparison.Ordinal);
            Assert.Contains(item, packageConsumerPlaybook, StringComparison.Ordinal);
        }
    }

    private static string ReadSource(params string[] segments)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(segments).ToArray()));
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

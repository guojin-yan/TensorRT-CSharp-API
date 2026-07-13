using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class CleanConsumerProofExecutionBundleTests
{
    [Fact]
    public void CleanConsumerProofExecutionBundleMapsOwnerProofLanesWithoutPromotingSubstitutes()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CleanConsumerProofExecutionBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-CleanConsumerProofExecutionBundle.ps1"), "-Strict");

        using JsonDocument bundleDocument = ReadFinalReleaseJson("clean-consumer-proof-execution-bundle.json");
        JsonElement bundle = bundleDocument.RootElement;

        Assert.Equal("clean-consumer-proof-execution-bundle", bundle.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-clean-consumer-runtime-and-post-publish-proof-required", bundle.GetProperty("bundleState").GetString());
        Assert.True(bundle.GetProperty("ownerActionRequired").GetBoolean());
        Assert.False(bundle.GetProperty("performsPublish").GetBoolean());
        Assert.False(bundle.GetProperty("performsRuntimeExecution").GetBoolean());
        Assert.False(bundle.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(bundle.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(bundle.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(bundle.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(bundle.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(bundle.GetProperty("isPostPublishProof").GetBoolean());

        string[] sourceArtifacts = bundle.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/cuda-device-initialization-local-smoke-classification.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-evidence-bundle.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-preflight-matrix.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/external-runtime-proof-record.input-template.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/post-publish-clean-consumer-proof-record-contract.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-post-publish-clean-consumer-proof-record-contract.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/clean-consumer-runtime-proof-execution-checklist.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/clean-consumer-proof-owner-execution-pack.json", sourceArtifacts);

        JsonElement[] lanes = bundle.GetProperty("lanes").EnumerateArray().ToArray();
        Assert.Contains(lanes, item => item.GetProperty("id").GetString() == "local-smoke" && item.GetProperty("evidenceKind").GetString() == "non-proof");
        Assert.Contains(lanes, item => item.GetProperty("id").GetString() == "local-feed" && item.GetProperty("evidenceKind").GetString() == "non-proof");
        Assert.Contains(lanes, item => item.GetProperty("id").GetString() == "clean-external-package-consumer" && item.GetProperty("state").GetString() == "owner-action-required");
        Assert.Contains(lanes, item => item.GetProperty("id").GetString() == "compatible-cuda-host-runtime" && item.GetProperty("ownerActionRequired").GetBoolean());
        Assert.Contains(lanes, item => item.GetProperty("id").GetString() == "post-publish-clean-consumer" && item.GetProperty("state").GetString() == "owner-action-required-after-publication");
        Assert.All(lanes, item =>
        {
            Assert.False(item.GetProperty("performsPublish").GetBoolean());
            Assert.False(item.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.False(item.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.Contains("not", item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        });

        string[] forbidden = bundle.GetProperty("forbiddenSubstitutes").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string expected in new[]
        {
            "Skipped=True",
            "local-smoke",
            "local feed",
            "ProjectReference",
            "direct nupkg",
            "build-only",
            "dependency-probe",
            "blocked-by-cuda-driver",
            "article roadmap",
            "dashboard",
            "runbook",
            "candidate",
            "draft",
            "template",
            "preflight-only",
            "dry-run-only",
            "schema-only",
            "owner input without strict validator pass"
        })
        {
            Assert.Contains(expected, forbidden);
        }

        string[] nonSubstitutes = bundle.GetProperty("nonSubstituteProofKinds").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("clean-consumer-proof-execution-bundle", nonSubstitutes);
        Assert.Contains("owner-action-required", nonSubstitutes);
        Assert.Contains("local-smoke-not-external-proof", nonSubstitutes);
        Assert.Contains("Skipped=True", nonSubstitutes);
        Assert.Contains("ProjectReference", nonSubstitutes);

        AssertIds(bundle, "promotionRequirements", new[]
        {
            "repository-external-clean-consumer",
            "no-project-reference",
            "public-or-approved-package-source",
            "native-assets-and-sha256",
            "runtime-smoke-log-sha256",
            "host-metadata",
            "strict-external-runtime-proof-validator",
            "post-publish-separate-gate"
        });
        AssertIds(bundle, "executionCommands", new[]
        {
            "export-local-smoke-classification",
            "validate-local-smoke-classification",
            "prepare-owner-clean-consumer",
            "strict-runtime-proof-validation",
            "forbidden-substitute-scan",
            "refresh-release-evidence",
            "post-publish-owner-gate"
        });

        string bundleText = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "clean-consumer-proof-execution-bundle.json"));
        Assert.Contains("Test-PackageConsumerRuntimeProofRecord.ps1", bundleText, StringComparison.Ordinal);
        Assert.Contains("-RequireExistingLog", bundleText, StringComparison.Ordinal);
        Assert.Contains("-FailOnNotProof", bundleText, StringComparison.Ordinal);
        Assert.Contains("Test-ExternalRuntimeProofRecord.ps1", bundleText, StringComparison.Ordinal);
        Assert.Contains("native asset listing", bundleText, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("host OS/GPU/driver/CUDA/TensorRT/cuDNN metadata", bundleText, StringComparison.Ordinal);
        Assert.Contains("does not run runtime smoke", bundle.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("does not publish packages", bundle.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        using JsonDocument validationDocument = ReadFinalReleaseJson("clean-consumer-proof-execution-bundle-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("clean-consumer-proof-execution-bundle-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("clean-consumer-proof-execution-bundle-ready-non-proof-boundaries-intact", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("performsRuntimeExecution").GetBoolean());
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validation.GetProperty("isRuntimeExecutionProof").GetBoolean());
    }

    private static void AssertIds(JsonElement root, string propertyName, string[] expectedIds)
    {
        string[] ids = root.GetProperty(propertyName).EnumerateArray().Select(static item => item.GetProperty("id").GetString()!).ToArray();
        foreach (string expectedId in expectedIds)
        {
            Assert.Contains(expectedId, ids);
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

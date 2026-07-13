using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class CleanConsumerExternalProofClosurePackTests
{
    [Fact]
    public void CleanConsumerExternalProofClosurePackLinksOwnerProofLanesWithoutPromotion()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CleanConsumerExternalProofClosurePack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-CleanConsumerExternalProofClosurePack.ps1"), "-Strict");

        using JsonDocument packDocument = ReadFinalReleaseJson("clean-consumer-external-proof-closure-pack.json");
        JsonElement pack = packDocument.RootElement;

        Assert.Equal("clean-consumer-external-proof-closure-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-external-clean-consumer-proof-closure-required", pack.GetProperty("closureState").GetString());
        Assert.True(pack.GetProperty("ownerActionRequired").GetBoolean());
        Assert.False(pack.GetProperty("performsPublish").GetBoolean());
        Assert.False(pack.GetProperty("performsRuntimeExecution").GetBoolean());
        Assert.False(pack.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(pack.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(pack.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(pack.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(pack.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(pack.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(pack.GetProperty("isReleaseCloseProof").GetBoolean());

        AssertIds(pack, "closureLanes", new[]
        {
            "owner-input-contracts",
            "clean-external-package-consumer-runtime",
            "compatible-cuda-host-runtime",
            "owner-external-proof-result-import",
            "post-publish-clean-consumer-proof",
            "strict-close-admission"
        });
        AssertIds(pack, "requiredOwnerFields", new[]
        {
            "clean-consumer-root",
            "package-source",
            "managed-runtime-package-identity",
            "native-assets",
            "runtime-smoke-logs",
            "host-metadata",
            "owner-review",
            "post-publish-evidence"
        });
        AssertIds(pack, "executionSteps", new[]
        {
            "select-runtime-proof-preflight-option",
            "create-clean-consumer-outside-repository",
            "restore-managed-runtime-packages",
            "run-clean-consumer-runtime-smoke",
            "hash-logs-packages-native-assets",
            "import-owner-external-result",
            "refresh-release-evidence",
            "post-publish-clean-consumer-after-publication"
        });

        JsonElement[] lanes = pack.GetProperty("closureLanes").EnumerateArray().ToArray();
        Assert.All(lanes, item =>
        {
            Assert.True(item.GetProperty("ownerActionRequired").GetBoolean());
            Assert.False(item.GetProperty("performsPublish").GetBoolean());
            Assert.False(item.GetProperty("performsRuntimeExecution").GetBoolean());
            Assert.False(item.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.Contains("not", item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        });

        string[] sourceArtifacts = pack.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/clean-consumer-proof-execution-bundle.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-external-proof-execution-bundle.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/external-clean-consumer-proof-kit.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-external-real-proof-input-contract.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-external-real-proof-import-validator.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/runtime-compatible-host-real-proof-gate.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/post-publish-clean-consumer-real-proof-gate.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-close-real-proof-readiness-gate.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-preflight-matrix.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/external-runtime-proof-record.input-template.json", sourceArtifacts);

        string packText = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "clean-consumer-external-proof-closure-pack.json"));
        Assert.Contains("Test-PackageConsumerRuntimeProofRecord.ps1", packText, StringComparison.Ordinal);
        Assert.Contains("-RequireExistingLog", packText, StringComparison.Ordinal);
        Assert.Contains("-FailOnNotProof", packText, StringComparison.Ordinal);
        Assert.Contains("Test-ExternalRuntimeProofRecord.ps1", packText, StringComparison.Ordinal);
        Assert.Contains("Test-PostPublishCleanConsumerProofRecordDraft.ps1", packText, StringComparison.Ordinal);
        Assert.Contains("Test-ReleaseIssueCloseRecord.ps1", packText, StringComparison.Ordinal);
        Assert.Contains("repository-external", packText, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("native asset listing", packText, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("host metadata", packText, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("owner review", packText, StringComparison.OrdinalIgnoreCase);

        string[] forbidden = pack.GetProperty("forbiddenSubstitutes").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string expected in new[]
        {
            "Skipped=True",
            "local smoke",
            "local feed",
            "ProjectReference",
            "direct nupkg",
            "build-only",
            "dependency-probe",
            "blocked-by-cuda-driver",
            "dashboard",
            "runbook",
            "candidate",
            "draft",
            "template",
            "owner input without strict validator pass",
            "pre-publish smoke reused as post-publish proof"
        })
        {
            Assert.Contains(expected, forbidden);
        }

        string[] nonSubstitutes = pack.GetProperty("nonSubstituteProofKinds").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("clean-consumer-external-proof-closure-pack", nonSubstitutes);
        Assert.Contains("external proof closure guidance", nonSubstitutes);
        Assert.Contains("owner-action closure pack", nonSubstitutes);
        Assert.Contains("ProjectReference", nonSubstitutes);

        using JsonDocument validationDocument = ReadFinalReleaseJson("clean-consumer-external-proof-closure-pack-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("clean-consumer-external-proof-closure-pack-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("clean-consumer-external-proof-closure-pack-ready-non-proof-boundaries-intact", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("performsRuntimeExecution").GetBoolean());
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
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

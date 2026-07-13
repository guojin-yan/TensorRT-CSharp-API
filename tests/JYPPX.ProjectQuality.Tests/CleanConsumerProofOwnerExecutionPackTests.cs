using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class CleanConsumerProofOwnerExecutionPackTests
{
    [Fact]
    public void CleanConsumerProofOwnerExecutionPackKeepsOwnerRuntimeProofBlockedAndActionable()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CleanConsumerRuntimeProofExecutionChecklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-CleanConsumerRuntimeProofExecutionChecklist.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CleanConsumerProofOwnerExecutionPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-CleanConsumerProofOwnerExecutionPack.ps1"), "-Strict");

        using JsonDocument packDocument = ReadFinalReleaseJson("clean-consumer-proof-owner-execution-pack.json");
        JsonElement pack = packDocument.RootElement;

        Assert.Equal("clean-consumer-proof-owner-execution-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-compatible-host-runtime-proof-required", pack.GetProperty("packState").GetString());
        Assert.Equal("artifacts/final-release/clean-consumer-runtime-proof-execution-checklist.json", pack.GetProperty("sourceChecklist").GetString());
        Assert.True(pack.GetProperty("ownerCommandGroupCount").GetInt32() >= 8);
        Assert.False(pack.GetProperty("performsPublish").GetBoolean());
        Assert.False(pack.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(pack.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(pack.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(pack.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(pack.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(pack.GetProperty("isPostPublishProof").GetBoolean());

        string[] sourceArtifacts = pack.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/clean-consumer-runtime-proof-execution-checklist.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-proof-owner-handoff-pack.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-evidence-bundle.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-release-close-blocker-dashboard.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-record-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.json", sourceArtifacts);

        JsonElement[] commandGroups = pack.GetProperty("ownerCommandGroups").EnumerateArray().ToArray();
        foreach (string groupId in new[]
        {
            "prepare-clean-consumer-workspace",
            "configure-public-source-and-install-packages",
            "restore-build-smoke-and-capture-logs",
            "hash-logs-packages-and-project",
            "capture-host-package-and-owner-metadata",
            "fill-import-and-validate-owner-input",
            "scan-forbidden-substitutes-and-refresh-release-state",
            "post-publish-remains-separate-owner-gate"
        })
        {
            Assert.Contains(commandGroups, item => item.GetProperty("id").GetString() == groupId);
        }

        Assert.All(commandGroups, item =>
        {
            Assert.True(item.GetProperty("ownerActionRequired").GetBoolean());
            Assert.False(item.GetProperty("performsPublish").GetBoolean());
            Assert.False(item.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("proofBoundary").GetString()));
        });

        string packText = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "clean-consumer-proof-owner-execution-pack.json"));
        Assert.DoesNotContain("dotnet nuget push", packText, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("repository-external clean consumer smoke", pack.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("public package source", pack.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("strict validation", pack.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        AssertIds(pack, "requiredOwnerFiles", new[]
        {
            "clean-consumer-project",
            "restore-log",
            "build-log",
            "smoke-stdout-log",
            "smoke-stderr-log",
            "managed-nupkg",
            "runtime-nupkg",
            "owner-input"
        });
        AssertIds(pack, "requiredOwnerInputs", new[]
        {
            "public-package-source-url",
            "package-ids-and-versions",
            "consumer-project-identity",
            "commands",
            "logs-and-hashes",
            "host-metadata",
            "runtime-status",
            "owner-review"
        });
        AssertIds(pack, "proofRecordProjection", new[]
        {
            "classification",
            "runtime-execution",
            "package-consumer",
            "publishability",
            "release-close"
        });

        string[] validators = pack.GetProperty("strictValidatorCommands").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains(validators, item => item.Contains("Test-PackageConsumerRuntimeProofRecord.ps1", StringComparison.Ordinal) && item.Contains("-Strict", StringComparison.Ordinal) && item.Contains("-RequireExistingLog", StringComparison.Ordinal) && item.Contains("-FailOnNotProof", StringComparison.Ordinal));
        Assert.Contains(validators, item => item.Contains("Import-PackageConsumerRuntimeProofOwnerInput.ps1", StringComparison.Ordinal));
        Assert.Contains(validators, item => item.Contains("Export-PackageConsumerRuntimeProofForbiddenSubstituteScan.ps1", StringComparison.Ordinal));
        Assert.Contains(validators, item => item.Contains("Test-ReleaseIssueCloseRecord.ps1", StringComparison.Ordinal) && item.Contains("-FailOnNotCloseReady", StringComparison.Ordinal));

        string[] forbiddenRules = pack.GetProperty("forbiddenSubstituteRules").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("local feed is not clean public package source proof", forbiddenRules);
        Assert.Contains("ProjectReference is not package-consumer-runtime proof", forbiddenRules);
        Assert.Contains("TensorRtExec report is not package-consumer-runtime proof", forbiddenRules);
        Assert.Contains("owner input without strict validator pass is not proof", forbiddenRules);

        JsonElement handoff = pack.GetProperty("ownerHandoffSummary");
        Assert.Equal("blocked-final-release-close-owner-action-required", handoff.GetProperty("closeDashboardState").GetString());
        Assert.False(handoff.GetProperty("cleanOwnerInputReady").GetBoolean());
        Assert.False(handoff.GetProperty("ownerInputCanPromoteRuntimeProof").GetBoolean());
        Assert.Contains("Run real repository-external clean consumer smoke", handoff.GetProperty("nextOwnerAction").GetString(), StringComparison.Ordinal);

        using JsonDocument validationDocument = ReadFinalReleaseJson("clean-consumer-proof-owner-execution-pack-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("clean-consumer-proof-owner-execution-pack-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("clean-consumer-proof-owner-execution-pack-ready", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
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

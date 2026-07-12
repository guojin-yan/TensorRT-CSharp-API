using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerRealPublishEvidenceIntakeDryRunTests
{
    [Fact]
    public void OwnerRealPublishEvidenceIntakeDryRunGroupsRequiredFieldsAndBlocksFakeReadyInputs()
    {
        RunIntakePipeline();

        using JsonDocument document = ReadFinalReleaseJson("owner-real-publish-evidence-intake-dry-run-pack.json");
        JsonElement pack = document.RootElement;
        Assert.Equal("owner-real-publish-evidence-intake-dry-run-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-publish-evidence-intake-dry-run-owner-action-required", pack.GetProperty("intakeState").GetString());
        Assert.True(pack.GetProperty("contractRequiredFieldCount").GetInt32() >= 178);
        Assert.Equal(6, pack.GetProperty("intakeGroupCount").GetInt32());
        Assert.Equal(0, pack.GetProperty("declaredButMissingInContractCount").GetInt32());
        Assert.Equal(10, pack.GetProperty("blockedFakeReadyCaseCount").GetInt32());
        Assert.True(pack.GetProperty("dryRunOnly").GetBoolean());
        Assert.False(pack.GetProperty("performsPublish").GetBoolean());
        Assert.False(pack.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(pack.GetProperty("canCloseReleaseIssue").GetBoolean());
        AssertGroupExists(pack, "managed-runtime-public-package");
        AssertGroupExists(pack, "publish-command-and-transcripts");
        AssertGroupExists(pack, "owner-authorization");
        AssertGroupExists(pack, "source-runner-boundary");
        AssertGroupExists(pack, "forbidden-substitute-scan");
        AssertGroupExists(pack, "post-publish-clean-consumer");

        foreach (JsonElement fakeReady in pack.GetProperty("fakeReadyCases").EnumerateArray())
        {
            Assert.True(fakeReady.GetProperty("fakeReadyBlocked").GetBoolean());
            Assert.False(fakeReady.GetProperty("isProof").GetBoolean());
            Assert.False(fakeReady.GetProperty("canCloseReleaseIssue").GetBoolean());
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-real-publish-evidence-intake-dry-run-pack-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("owner-real-publish-evidence-intake-dry-run-pack-validation-ready-non-proof", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    [Fact]
    public void PostPublishStrictCrossCheckPackBlocksPlaceholderAndLocalSubstitutePaths()
    {
        RunIntakePipeline();
        RunPowerShell("Export-PostPublishStrictCrossCheckPack.ps1");
        RunPowerShell("Test-PostPublishStrictCrossCheckPack.ps1", "-Strict");

        using JsonDocument document = ReadFinalReleaseJson("post-publish-strict-cross-check-pack.json");
        JsonElement pack = document.RootElement;
        Assert.Equal("post-publish-strict-cross-check-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-post-publish-strict-cross-check-owner-evidence-required", pack.GetProperty("crossCheckState").GetString());
        Assert.Equal(7, pack.GetProperty("crossCheckCount").GetInt32());
        Assert.True(pack.GetProperty("failedCrossCheckCount").GetInt32() >= 1);
        Assert.False(pack.GetProperty("performsPublish").GetBoolean());
        Assert.False(pack.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(pack.GetProperty("canCloseReleaseIssue").GetBoolean());
        AssertCrossCheckExists(pack, "package-page-url-owner-input-to-record");
        AssertCrossCheckExists(pack, "managed-download-url-owner-input-to-validation");
        AssertCrossCheckExists(pack, "runtime-download-url-owner-input-to-validation");
        AssertCrossCheckExists(pack, "managed-sha256-owner-input-to-validation");
        AssertCrossCheckExists(pack, "runtime-sha256-owner-input-to-validation");
        AssertCrossCheckExists(pack, "clean-consumer-root-outside-repository");
        AssertCrossCheckExists(pack, "no-project-reference-local-feed-direct-nupkg");

        using JsonDocument validationDocument = ReadFinalReleaseJson("post-publish-strict-cross-check-pack-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("post-publish-strict-cross-check-pack-validation-ready-non-proof", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedCrossCheckCount").GetInt32() >= 1);
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    private static void RunIntakePipeline()
    {
        RunPowerShell("Export-FinalReadonlyPublishAuditPack.ps1");
        RunPowerShell("Test-FinalReadonlyPublishAuditPack.ps1", "-Strict");
        RunPowerShell("Export-FinalOwnerOneScreenExecutionManual.ps1");
        RunPowerShell("Test-FinalOwnerOneScreenExecutionManual.ps1", "-Strict");
        RunPowerShell("Export-OwnerPublicPublishExecutionResultInputContract.ps1");
        RunPowerShell("Export-OwnerPublicPublishExecutionResultInputTemplate.ps1");
        RunPowerShell("Test-OwnerPublicPublishExecutionResultPreflight.ps1", "-Strict");
        RunPowerShell("Import-OwnerPublicPublishExecutionResultCandidate.ps1");
        RunPowerShell("Test-OwnerPublicPublishExecutionResultCandidate.ps1", "-Strict");
        RunPowerShell("Export-PostPublishVerificationOwnerInputTemplate.ps1");
        RunPowerShell("Test-PostPublishVerificationOwnerInput.ps1", "-Strict");
        RunPowerShell("Export-PostPublishVerificationRecordFromOwnerInput.ps1", "-OwnerInputPath", "artifacts/final-release/post-publish-verification-owner-input.template.json");
        RunPowerShell("Test-PostPublishVerificationRecord.ps1", "-InputPath", "artifacts/final-release/post-publish-verification-record.json");
        RunPowerShell("Export-ReleaseCloseStrictEvidenceClosure.ps1");
        RunPowerShell("Test-ReleaseCloseStrictEvidenceClosure.ps1", "-Strict");
        RunPowerShell("Export-OwnerRealPublishEvidenceIntakeDryRunPack.ps1");
        RunPowerShell("Test-OwnerRealPublishEvidenceIntakeDryRunPack.ps1", "-Strict");
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    private static void AssertGroupExists(JsonElement pack, string id)
    {
        Assert.Contains(pack.GetProperty("intakeGroups").EnumerateArray(), group =>
            group.GetProperty("id").GetString() == id &&
            group.GetProperty("requiredFieldCount").GetInt32() > 0 &&
            !group.GetProperty("performsPublish").GetBoolean() &&
            !group.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    private static void AssertCrossCheckExists(JsonElement pack, string id)
    {
        Assert.Contains(pack.GetProperty("crossChecks").EnumerateArray(), check =>
            check.GetProperty("id").GetString() == id &&
            check.GetProperty("ownerActionRequired").GetBoolean() &&
            !check.GetProperty("isProof").GetBoolean());
    }

    private static string RunPowerShell(string scriptName, params string[] arguments)
    {
        using Process process = new()
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "pwsh",
                WorkingDirectory = RepositoryPaths.Root,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            },
        };

        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(Path.Combine(RepositoryPaths.Root, "eng", scriptName));
        foreach (string argument in arguments)
        {
            process.StartInfo.ArgumentList.Add(argument);
        }

        process.Start();
        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptName} {string.Join(' ', arguments)}{Environment.NewLine}{output}{Environment.NewLine}{error}");
        return output;
    }
}

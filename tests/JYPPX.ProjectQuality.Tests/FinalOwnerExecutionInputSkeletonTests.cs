using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalOwnerExecutionInputSkeletonTests
{
    [Fact]
    public void InputSkeletonExpandsOwnerGapsWithoutPromotion()
    {
        RunPowerShell("Export-FinalOwnerExecutionOneScreenPack.ps1");
        RunPowerShell("Test-FinalOwnerExecutionOneScreenPack.ps1", "-Strict");
        RunPowerShell("Export-FinalOwnerExecutionInputSkeleton.ps1");
        RunPowerShell("Test-FinalOwnerExecutionInputSkeleton.ps1", "-Strict");

        using JsonDocument skeletonDocument = ReadFinalReleaseJson("final-owner-execution-input-skeleton.json");
        JsonElement skeleton = skeletonDocument.RootElement;

        Assert.Equal("final-owner-execution-input-skeleton", skeleton.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-real-input-required", skeleton.GetProperty("skeletonState").GetString());
        Assert.True(skeleton.GetProperty("ownerActionRequired").GetBoolean());
        Assert.False(skeleton.GetProperty("performsPublish").GetBoolean());
        Assert.False(skeleton.GetProperty("performsRuntimeExecution").GetBoolean());
        Assert.False(skeleton.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(skeleton.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(skeleton.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(skeleton.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(skeleton.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(skeleton.GetProperty("isReleaseCloseProof").GetBoolean());
        Assert.Equal("blocked-owner-compatible-host-runtime-smoke-field-alignment", skeleton.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentState").GetString());
        Assert.Equal("blocked-owner-compatible-host-runtime-smoke-field-alignment-valid", skeleton.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState").GetString());
        Assert.Equal("Smoke=not-requested", skeleton.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus").GetString());
        Assert.True(skeleton.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount").GetInt32() >= 30);
        Assert.Equal(0, skeleton.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount").GetInt32());
        Assert.Equal(0, skeleton.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount").GetInt32());

        int requiredFieldCount = skeleton.GetProperty("requiredFieldCount").GetInt32();
        Assert.Equal(49, requiredFieldCount);
        Assert.Equal(9, skeleton.GetProperty("laneCount").GetInt32());
        Assert.True(skeleton.GetProperty("laneRequiredFieldCount").GetInt32() >= 60);
        Assert.Equal(10, skeleton.GetProperty("fieldGroupCount").GetInt32());
        Assert.Equal(requiredFieldCount, skeleton.GetProperty("missingFieldCount").GetInt32());
        Assert.Equal(requiredFieldCount, skeleton.GetProperty("placeholderFieldCount").GetInt32());
        Assert.Equal(0, skeleton.GetProperty("readyForImportFieldCount").GetInt32());

        string skeletonText = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "final-owner-execution-input-skeleton.json"));
        foreach (string expected in new[]
        {
            "cleanConsumer.projectRoot",
            "packageSource.url",
            "managedPackage.sha256",
            "runtimePackage.key",
            "nativeAssetListing.path",
            "runtimeSmoke.mergedTranscriptSha256",
            "host.cudaRuntimeToolkit",
            "host.tensorrt",
            "postPublish.downloadedPackageSha256",
            "dualPackageRoutes.nugetSmallBridgeCore.ownerAuthorizationUrl",
            "dualPackageRoutes.nugetSmallBridgeCore.publicPackageDownloadUrl",
            "dualPackageRoutes.githubPackagesFullRuntime.restoreSourceUrl",
            "dualPackageRoutes.githubPackagesFullRuntime.runtimeDllResolutionReportPath",
            "dual package route proof",
            "dual package final close lanes",
            "rollback.review",
            "finalClose.decision",
            "strictValidators.outputSha256",
            "local feed",
            "ProjectReference",
            "direct .nupkg",
            "pre-publish smoke reused as post-publish proof"
        })
        {
            Assert.Contains(expected, skeletonText, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("final-owner-execution-input-skeleton-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-owner-execution-input-skeleton-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("final-owner-execution-input-skeleton-validation-ready-non-proof", validation.GetProperty("validationState").GetString());
        Assert.Equal(49, validation.GetProperty("compatibilityFieldCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal("blocked-owner-compatible-host-runtime-smoke-field-alignment", validation.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentState").GetString());
        Assert.Equal("blocked-owner-compatible-host-runtime-smoke-field-alignment-valid", validation.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState").GetString());
        Assert.Equal(0, validation.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount").GetInt32());
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
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
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptName}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        return stdout;
    }
}

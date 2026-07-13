using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerInputPreflightBundleAndDashboardValidatorTests
{
    [Fact]
    public void OwnerInputPreflightBundleAggregatesOwnerLanesWithoutPromotion()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionOwnerProofFieldDeltaRepairPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseProofLaneWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-TensorRtExecAdvancedProofReadinessChecklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalPublishProofGate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseProofDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerInputPreflightBundle.ps1"));

        using JsonDocument document = ReadJson("artifacts", "final-release", "owner-input-preflight-bundle.json");
        JsonElement bundle = document.RootElement;

        Assert.Equal("owner-input-preflight-bundle", bundle.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-input-required", bundle.GetProperty("bundleState").GetString());
        Assert.False(bundle.GetProperty("performsPublish").GetBoolean());
        Assert.False(bundle.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(bundle.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(bundle.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.Equal(3, bundle.GetProperty("laneCount").GetInt32());
        Assert.Equal(3, bundle.GetProperty("blockedLaneCount").GetInt32());

        JsonElement[] lanes = bundle.GetProperty("ownerInputLanes").EnumerateArray().ToArray();
        string[] laneIds = lanes.Select(static lane => lane.GetProperty("id").GetString()!).ToArray();
        Assert.Contains("real-model-runtime", laneIds);
        Assert.Contains("package-consumer-runtime", laneIds);
        Assert.Contains("post-publish-verification", laneIds);
        Assert.All(lanes, lane =>
        {
            Assert.True(lane.GetProperty("blocked").GetBoolean());
            Assert.True(lane.GetProperty("requiredFieldCount").GetInt32() > 0);
            Assert.NotEmpty(lane.GetProperty("failFastOrder").EnumerateArray());
            Assert.NotEmpty(lane.GetProperty("ownerCommands").EnumerateArray());
            Assert.NotEmpty(lane.GetProperty("forbiddenSubstitutes").EnumerateArray());
            Assert.False(lane.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.False(lane.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(lane.GetProperty("canCloseReleaseIssue").GetBoolean());
        });

        string raw = bundle.GetRawText();
        foreach (string marker in new[]
        {
            "ownerHandoffCommands",
            "real-model-runtime",
            "package-consumer-runtime",
            "post-publish-verification",
            "publicPackageSource",
            "runtimeSmokeLogSha256",
            "selectedChannel",
            "managedNupkgSha256",
            "local feed",
            "ProjectReference",
            "direct .nupkg",
            "blocked-by-cuda-driver",
            "does not promote runtime proof"
        })
        {
            Assert.Contains(marker, raw, StringComparison.OrdinalIgnoreCase);
        }

        string markdown = ReadText("artifacts", "final-release", "owner-input-preflight-bundle.md");
        Assert.Contains("Owner Input Preflight Bundle", markdown, StringComparison.Ordinal);
        Assert.Contains("Owner Input Lanes", markdown, StringComparison.Ordinal);
        Assert.Contains("Owner Handoff Commands", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseProofDashboardStrictValidatorBlocksPromotionAndFeedsFinalGate()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionOwnerProofFieldDeltaRepairPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseProofLaneWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-TensorRtExecAdvancedProofReadinessChecklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalPublishProofGate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseProofDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseProofDashboard.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalPublishProofGate.ps1"), "-Strict");

        using JsonDocument validationDocument = ReadJson("artifacts", "final-release", "release-proof-dashboard-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("release-proof-dashboard-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-proof-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.Equal(4, validation.GetProperty("laneCount").GetInt32());
        Assert.Equal(4, validation.GetProperty("blockedLaneCount").GetInt32());

        string rawValidation = validation.GetRawText();
        foreach (string marker in new[]
        {
            "supporting-evidence-only-not-proof",
            "lane-real-model-runtime-present-blocked",
            "lane-package-consumer-runtime-present-blocked",
            "lane-post-publish-verification-present-blocked",
            "lane-public-owner-confirmation-present-blocked",
            "local feed",
            "ProjectReference",
            "direct .nupkg",
            "build-only",
            "dry-run",
            "template",
            "skipped run",
            "OnnxToEngine report is build/report evidence only and is not runtime proof"
        })
        {
            Assert.Contains(marker, rawValidation, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument gateDocument = ReadJson("artifacts", "final-release", "final-publish-proof-gate-report.json");
        JsonElement gate = gateDocument.RootElement;
        JsonElement[] gateItems = gate.GetProperty("validationItems").EnumerateArray().ToArray();
        Assert.Contains(gateItems, item => item.GetProperty("id").GetString() == "release-dashboard-validation-non-proof-passed" && item.GetProperty("passed").GetBoolean());

        string gateRaw = gate.GetRawText();
        Assert.Contains("release-proof-dashboard-validation.json", gateRaw, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("release-proof-dashboard-validation or public-docs-package-metadata-gate exists", gate.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    private static JsonDocument ReadJson(params string[] pathParts)
    {
        return JsonDocument.Parse(ReadText(pathParts));
    }

    private static string ReadText(params string[] pathParts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
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

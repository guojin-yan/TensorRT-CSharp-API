using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleaseProofDashboardAndOwnerFieldDeltaTests
{
    [Fact]
    public void YoloVisionOwnerProofFieldDeltaRepairPackListsMissingFieldsWithoutPromotion()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionOwnerProofFieldDeltaRepairPack.ps1"));

        using JsonDocument document = ReadJson("artifacts", "user-acceptance", "yolovision-owner-proof-field-delta-repair-pack.json");
        JsonElement pack = document.RootElement;

        Assert.Equal("yolovision-owner-proof-field-delta-repair-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-action-required", pack.GetProperty("repairPackState").GetString());
        Assert.Equal("owner-action-required", pack.GetProperty("validationState").GetString());
        Assert.Equal(6, pack.GetProperty("caseCount").GetInt32());
        Assert.True(pack.GetProperty("missingFieldCount").GetInt32() >= 100);
        Assert.True(pack.GetProperty("globalMissingFieldCount").GetInt32() >= 8);
        Assert.False(pack.GetProperty("canAutoFillOwnerFields").GetBoolean());
        Assert.False(pack.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.False(pack.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.False(pack.GetProperty("canCloseReleaseIssue").GetBoolean());

        JsonElement[] cases = pack.GetProperty("caseSummaries").EnumerateArray().ToArray();
        Assert.Equal(6, cases.Length);
        Assert.Contains(cases, item => item.GetProperty("task").GetString() == "sem");
        Assert.All(cases, item => Assert.True(item.GetProperty("missingFieldCount").GetInt32() > 0));

        string raw = pack.GetRawText();
        foreach (string marker in new[]
        {
            "globalMissingFieldCount",
            "hostMetadata",
            "global-host-metadata",
            "owner-review",
            "hash",
            "path-or-log",
            "validatorItemId",
            "recommendedRepairOrder",
            "cannot auto-fill owner fields"
        })
        {
            Assert.Contains(marker, raw, StringComparison.OrdinalIgnoreCase);
        }

        string markdown = ReadText("artifacts", "user-acceptance", "yolovision-owner-proof-field-delta-repair-pack.md");
        Assert.Contains("Case Summary", markdown, StringComparison.Ordinal);
        Assert.Contains("Field Deltas", markdown, StringComparison.Ordinal);
        Assert.Contains("canAutoFillOwnerFields", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseProofDashboardAggregatesLanesGateAndSupportingEvidence()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionOwnerProofFieldDeltaRepairPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseProofLaneWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-TensorRtExecAdvancedProofReadinessChecklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalPublishProofGate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseProofDashboard.ps1"));

        using JsonDocument document = ReadJson("artifacts", "final-release", "release-proof-dashboard.json");
        JsonElement dashboard = document.RootElement;

        Assert.Equal("release-proof-dashboard", dashboard.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-proof-required", dashboard.GetProperty("dashboardState").GetString());
        Assert.False(dashboard.GetProperty("performsPublish").GetBoolean());
        Assert.False(dashboard.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(dashboard.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(dashboard.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.Equal(0, dashboard.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(dashboard.GetProperty("actionRequiredCount").GetInt32() >= 5);
        Assert.Equal(4, dashboard.GetProperty("ownerProofActionRequiredLaneCount").GetInt32());
        Assert.Equal(4, dashboard.GetProperty("laneCount").GetInt32());
        Assert.Equal(4, dashboard.GetProperty("blockedLaneCount").GetInt32());

        string[] actionSources = dashboard.GetProperty("actionRequiredSources").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("real-model-runtime-owner-proof-required", actionSources);
        Assert.Contains("package-consumer-runtime-owner-proof-required", actionSources);
        Assert.Contains("post-publish-verification-owner-proof-required", actionSources);

        JsonElement[] lanes = dashboard.GetProperty("lanes").EnumerateArray().ToArray();
        string[] laneIds = lanes.Select(static lane => lane.GetProperty("id").GetString()!).ToArray();
        Assert.Contains("real-model-runtime", laneIds);
        Assert.Contains("package-consumer-runtime", laneIds);
        Assert.Contains("post-publish-verification", laneIds);
        Assert.Contains("public-owner-confirmation", laneIds);
        Assert.All(lanes, lane =>
        {
            Assert.True(lane.GetProperty("blocked").GetBoolean());
            Assert.False(lane.GetProperty("canPromoteProof").GetBoolean());
            Assert.False(lane.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(lane.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.NotEmpty(lane.GetProperty("validatorCommands").EnumerateArray());
            Assert.NotEmpty(lane.GetProperty("nextOwnerCommands").EnumerateArray());
        });

        string raw = dashboard.GetRawText();
        foreach (string marker in new[]
        {
            "release-close-proof-lane-worklist.json",
            "final-publish-proof-gate-report.json",
            "yolovision-real-asset-owner-proof-execution-pack.json",
            "package-consumer-runtime-proof-owner-input-validation.json",
            "post-publish-verification-validation.json",
            "tensorrtexec-advanced-proof-readiness-checklist.json",
            "publicPackageSource",
            "exitCode=0",
            "local feed",
            "ProjectReference",
            "direct .nupkg",
            "supporting-evidence-only",
            "TensorRtExec report is not runtime proof"
        })
        {
            Assert.Contains(marker, raw, StringComparison.OrdinalIgnoreCase);
        }

        string markdown = ReadText("artifacts", "final-release", "release-proof-dashboard.md");
        Assert.Contains("Release Proof Dashboard", markdown, StringComparison.Ordinal);
        Assert.Contains("Proof Lanes", markdown, StringComparison.Ordinal);
        Assert.Contains("Supporting Evidence Only", markdown, StringComparison.Ordinal);
        Assert.Contains("Next Owner Commands", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void PackageConsumerProofOwnerInputValidatorCarriesRequiredCrossChecks()
    {
        string validator = ReadText("eng", "Test-PackageConsumerRuntimeProofOwnerInput.ps1");
        string template = ReadText("eng", "Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1");
        string combined = validator + template;

        foreach (string marker in new[]
        {
            "publicPackageSource",
            "managedPackageId",
            "managedPackageVersion",
            "runtimePackageVersion",
            "hostOs",
            "gpuName",
            "cudaDriverVersion",
            "cudaRuntimeVersion",
            "tensorRtVersion",
            "exitCode",
            "stdout",
            "stderr",
            "smokeLogSha256",
            "local feed",
            "ProjectReference",
            "direct .nupkg",
            "build-only",
            "dry-run",
            "template",
            "skipped run"
        })
        {
            Assert.Contains(marker, combined, StringComparison.OrdinalIgnoreCase);
        }

        Assert.DoesNotContain("dotnet nuget push", combined, StringComparison.OrdinalIgnoreCase);
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

using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class YoloVisionOwnerRealEvidenceIntakeTests
{
    [Fact]
    public void OwnerRealEvidenceIntakeDashboardMapsSixTasksWithoutPromotingProof()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionSixTaskRealProofChainDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionOwnerRealEvidenceIntakeDashboard.ps1"));

        using JsonDocument document = ReadJson("artifacts", "user-acceptance", "yolovision-owner-real-evidence-intake-dashboard.json");
        JsonElement dashboard = document.RootElement;

        Assert.Equal("yolovision-owner-real-evidence-intake-dashboard", dashboard.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-evidence-required", dashboard.GetProperty("dashboardState").GetString());
        Assert.Equal(6, dashboard.GetProperty("taskCount").GetInt32());
        Assert.True(dashboard.GetProperty("totalMissingFieldCount").GetInt32() >= 190);
        Assert.True(dashboard.GetProperty("globalMissingFieldCount").GetInt32() >= 20);
        Assert.Equal(0, dashboard.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(6, dashboard.GetProperty("ownerActionRequiredTaskCount").GetInt32());
        Assert.Equal(6, dashboard.GetProperty("finalGateActionRequiredCount").GetInt32());
        Assert.False(dashboard.GetProperty("performsPublish").GetBoolean());
        Assert.False(dashboard.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(dashboard.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(dashboard.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.False(dashboard.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());

        JsonElement[] tasks = dashboard.GetProperty("tasks").EnumerateArray().ToArray();
        foreach (string task in new[] { "det", "seg", "pose", "obb", "cls", "sem" })
        {
            JsonElement item = Assert.Single(tasks, candidate => candidate.GetProperty("task").GetString() == task);
            Assert.Equal($"yolov8n-{task}", item.GetProperty("caseId").GetString());
            Assert.True(item.GetProperty("missingFieldCount").GetInt32() > 0);
            Assert.NotEmpty(item.GetProperty("missingSha256Fields").EnumerateArray());
            Assert.NotEmpty(item.GetProperty("missingHostMetadata").EnumerateArray());
            Assert.NotEmpty(item.GetProperty("missingPackageMetadata").EnumerateArray());
            Assert.NotEmpty(item.GetProperty("missingOwnerReview").EnumerateArray());
            Assert.False(item.GetProperty("canPromoteRealModelRuntime").GetBoolean());
            Assert.False(item.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
            Assert.Contains("strict validation", item.GetProperty("blockingReason").GetString(), StringComparison.OrdinalIgnoreCase);
        }

        string markdown = ReadText("artifacts", "user-acceptance", "yolovision-owner-real-evidence-intake-dashboard.md");
        Assert.Contains("Owner Real Evidence Intake", markdown, StringComparison.Ordinal);
        Assert.Contains("finalGateActionRequiredCount", markdown, StringComparison.Ordinal);
        Assert.Contains("canPromotePackageConsumerRuntime", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void OwnerEvidenceBatchBackfillPackGroupsOwnerWorkWithoutAutofill()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionOwnerRealEvidenceIntakeDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionOwnerEvidenceBatchBackfillPack.ps1"));

        using JsonDocument document = ReadJson("artifacts", "user-acceptance", "yolovision-owner-evidence-batch-backfill-pack.json");
        JsonElement pack = document.RootElement;

        Assert.Equal("yolovision-owner-evidence-batch-backfill-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-evidence-backfill-required", pack.GetProperty("packState").GetString());
        Assert.Equal(6, pack.GetProperty("taskCount").GetInt32());
        Assert.True(pack.GetProperty("totalMissingFieldCount").GetInt32() >= 190);
        Assert.Equal(8, pack.GetProperty("groupCount").GetInt32());
        Assert.False(pack.GetProperty("performsPublish").GetBoolean());
        Assert.False(pack.GetProperty("canAutoFillOwnerFields").GetBoolean());
        Assert.False(pack.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.False(pack.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.False(pack.GetProperty("canCloseReleaseIssue").GetBoolean());

        string[] ids = pack.GetProperty("groups").EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();

        foreach (string expected in new[]
        {
            "01-model-source-license",
            "02-labels-input-preprocessed",
            "03-tensorrtexec-build-report-engine-logs",
            "04-yolovision-run-output-json-logs",
            "05-sha256-verification",
            "06-host-metadata",
            "07-package-metadata",
            "08-owner-review-acceptance"
        })
        {
            Assert.Contains(expected, ids);
        }

        foreach (JsonElement group in pack.GetProperty("groups").EnumerateArray())
        {
            Assert.False(group.GetProperty("canAutoFill").GetBoolean());
            Assert.False(group.GetProperty("canPromoteProof").GetBoolean());
            string validatorCommand = group.GetProperty("validatorCommand").GetString() ?? string.Empty;
            Assert.NotEmpty(validatorCommand);
        }

        string markdown = ReadText("artifacts", "user-acceptance", "yolovision-owner-evidence-batch-backfill-pack.md");
        Assert.Contains("YoloVision Owner Evidence Batch Backfill Pack", markdown, StringComparison.Ordinal);
        Assert.Contains("canAutoFillOwnerFields", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void FinalPublishActionRequiredEvidenceMapExplainsAllSixActionsWithoutPromotion()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionOwnerRealEvidenceIntakeDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionOwnerEvidenceBatchBackfillPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalPublishActionRequiredEvidenceMap.ps1"));

        using JsonDocument document = ReadJson("artifacts", "final-release", "final-publish-action-required-evidence-map.json");
        JsonElement map = document.RootElement;

        Assert.Equal("final-publish-action-required-evidence-map", map.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-owner-evidence-required", map.GetProperty("mapState").GetString());
        Assert.Equal("blocked-final-publish-real-proof-required", map.GetProperty("finalGateState").GetString());
        Assert.Equal(0, map.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(6, map.GetProperty("actionRequiredCount").GetInt32());
        Assert.Equal(6, map.GetProperty("yoloVisionIntakeTaskCount").GetInt32());
        Assert.Equal(8, map.GetProperty("yoloVisionBackfillGroupCount").GetInt32());
        Assert.True(map.GetProperty("sourceTreeRealModelRuntimeReady").GetBoolean());
        Assert.Equal(6, map.GetProperty("sourceTreeRealModelRuntimeReadyTaskCount").GetInt32());
        Assert.Equal(0, map.GetProperty("sourceTreeRealModelRuntimeMissingTaskCount").GetInt32());
        Assert.False(map.GetProperty("sourceTreeRealModelRuntimeCanPromotePackageConsumer").GetBoolean());
        Assert.False(map.GetProperty("performsPublish").GetBoolean());
        Assert.False(map.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(map.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(map.GetProperty("canPromoteRuntimeProof").GetBoolean());

        string[] ids = map.GetProperty("actions").EnumerateArray()
            .Select(static item => item.GetProperty("id").GetString()!)
            .ToArray();

        foreach (string expected in new[]
        {
            "real-model-runtime-owner-proof-required",
            "package-consumer-runtime-owner-proof-required",
            "post-publish-verification-owner-proof-required",
            "final-owner-real-input-template-pack-owner-input-required",
            "owner-external-proof-result-import-owner-proof-required",
            "owner-result-candidate-bridge-real-proof-required"
        })
        {
            Assert.Contains(expected, ids);
        }

        foreach (JsonElement action in map.GetProperty("actions").EnumerateArray())
        {
            Assert.Equal("action-required", action.GetProperty("currentState").GetString());
            Assert.False(action.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(action.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(action.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.NotEmpty(action.GetProperty("forbiddenSubstitutes").EnumerateArray());
            Assert.Contains("not passed strict validation", action.GetProperty("whyNotPublishableYet").GetString(), StringComparison.OrdinalIgnoreCase);
        }
    }

    private static JsonDocument ReadJson(params string[] pathParts)
    {
        return JsonDocument.Parse(ReadText(pathParts));
    }

    private static string ReadText(params string[] pathParts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
    }

    private static void RunPowerShell(string scriptPath, params string[] arguments)
    {
        ProcessStartInfo psi = new("pwsh")
        {
            WorkingDirectory = RepositoryPaths.Root,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };

        psi.ArgumentList.Add("-NoProfile");
        psi.ArgumentList.Add("-ExecutionPolicy");
        psi.ArgumentList.Add("Bypass");
        psi.ArgumentList.Add("-File");
        psi.ArgumentList.Add(scriptPath);

        foreach (string argument in arguments)
        {
            psi.ArgumentList.Add(argument);
        }

        using Process process = Process.Start(psi)!;
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"PowerShell script failed: {scriptPath}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}");
    }
}

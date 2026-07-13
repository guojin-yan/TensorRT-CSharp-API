using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class YoloVisionSixTaskRealProofChainDashboardTests
{
    [Fact]
    public void DashboardExportsSixTaskAlignmentWithoutPromotingRuntimeProof()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionSixTaskRealProofChainDashboard.ps1"));

        using JsonDocument document = ReadJson("artifacts", "yolovision", "yolovision-six-task-real-proof-chain-dashboard.json");
        JsonElement dashboard = document.RootElement;

        Assert.Equal("yolovision-six-task-real-proof-chain-dashboard", dashboard.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-owner-proof-required", dashboard.GetProperty("dashboardState").GetString());
        Assert.Equal(6, dashboard.GetProperty("taskCount").GetInt32());
        Assert.Equal(0, dashboard.GetProperty("failedAlignmentCount").GetInt32());
        Assert.Equal(6, dashboard.GetProperty("ownerActionRequiredTaskCount").GetInt32());
        Assert.False(dashboard.GetProperty("performsPublish").GetBoolean());
        Assert.False(dashboard.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(dashboard.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(dashboard.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.False(dashboard.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.Contains("does not run models", dashboard.GetProperty("proofBoundary").GetString(), StringComparison.OrdinalIgnoreCase);

        JsonElement[] tasks = dashboard.GetProperty("tasks").EnumerateArray().ToArray();
        foreach (string task in new[] { "det", "seg", "pose", "obb", "cls", "sem" })
        {
            JsonElement item = Assert.Single(tasks, candidate => candidate.GetProperty("task").GetString() == task);
            Assert.True(item.GetProperty("candidateTemplateAligned").GetBoolean());
            Assert.True(item.GetProperty("ownerBackfillAligned").GetBoolean());
            Assert.True(item.GetProperty("ownerProofInputAligned").GetBoolean());
            Assert.True(item.GetProperty("candidateEvidenceAligned").GetBoolean());
            Assert.False(item.GetProperty("realOwnerEvidenceReady").GetBoolean());
            Assert.False(item.GetProperty("canPromoteRealModelRuntime").GetBoolean());
            Assert.False(item.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
            Assert.NotEmpty(item.GetProperty("contractRequiredMetadata").EnumerateArray());
            Assert.Contains("Owner must provide real logs", item.GetProperty("blockingReason").GetString(), StringComparison.Ordinal);
        }

        string markdown = ReadText("artifacts", "yolovision", "yolovision-six-task-real-proof-chain-dashboard.md");
        Assert.Contains("YoloVision Six Task Real Proof Chain Dashboard", markdown, StringComparison.Ordinal);
        Assert.Contains("sem", markdown, StringComparison.Ordinal);
        Assert.Contains("canPromotePackageConsumerRuntime", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void LiveDocsDoNotRetainLegacyFiveCaseSemanticGapWording()
    {
        string combined = string.Join(
            "\n",
            ReadText("samples", "YoloVision", "README.md"),
            ReadText("applications", "TensorRtExec", "README.md"),
            ReadText("docs", "articles", "zh-cn", "owner-real-proof-import-master-pack.md"),
            ReadText("docs", "articles", "zh-cn", "real-external-execution-backfill-final-freeze.md"));

        Assert.DoesNotContain("five YOLOv8n", combined, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("five generated evidence rows", combined, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("does not cover sem", combined, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("不覆盖 `sem`", combined, StringComparison.Ordinal);
        Assert.Contains("six YOLOv8n", combined, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("det/seg/pose/obb/cls/sem", combined, StringComparison.Ordinal);
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

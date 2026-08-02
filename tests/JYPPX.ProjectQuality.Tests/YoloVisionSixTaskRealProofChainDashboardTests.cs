using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class YoloVisionSixTaskRealProofChainDashboardTests
{
    [Fact]
    public void DashboardExportsSixTaskSourceTreeRuntimeProofWithoutPromotingPackageOrRelease()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionSixTaskRealProofChainDashboard.ps1"));

        using JsonDocument document = ReadJson("artifacts", "yolovision", "yolovision-six-task-real-proof-chain-dashboard.json");
        JsonElement dashboard = document.RootElement;

        Assert.Equal("yolovision-six-task-real-proof-chain-dashboard", dashboard.GetProperty("recordKind").GetString());
        Assert.Equal("source-tree-real-model-runtime-ready-package-proof-required", dashboard.GetProperty("dashboardState").GetString());
        Assert.Equal(6, dashboard.GetProperty("taskCount").GetInt32());
        Assert.Equal(0, dashboard.GetProperty("failedAlignmentCount").GetInt32());
        Assert.Equal(6, dashboard.GetProperty("realModelRuntimeReadyTaskCount").GetInt32());
        Assert.Equal(0, dashboard.GetProperty("realModelRuntimeMissingTaskCount").GetInt32());
        Assert.Equal(0, dashboard.GetProperty("ownerActionRequiredTaskCount").GetInt32());
        Assert.Equal(6, dashboard.GetProperty("packageConsumerProofRequiredTaskCount").GetInt32());
        Assert.False(dashboard.GetProperty("performsPublish").GetBoolean());
        Assert.False(dashboard.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(dashboard.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.True(dashboard.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.False(dashboard.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.Contains("does not itself run models", dashboard.GetProperty("proofBoundary").GetString(), StringComparison.OrdinalIgnoreCase);

        JsonElement[] tasks = dashboard.GetProperty("tasks").EnumerateArray().ToArray();
        foreach (string task in new[] { "det", "seg", "pose", "obb", "cls", "sem" })
        {
            JsonElement item = Assert.Single(tasks, candidate => candidate.GetProperty("task").GetString() == task);
            Assert.True(item.GetProperty("candidateTemplateAligned").GetBoolean());
            Assert.True(item.GetProperty("ownerBackfillAligned").GetBoolean());
            Assert.True(item.GetProperty("ownerProofInputAligned").GetBoolean());
            Assert.True(item.GetProperty("candidateEvidenceAligned").GetBoolean());
            Assert.True(item.GetProperty("runtimeReferenceValidated").GetBoolean());
            Assert.True(item.GetProperty("controlledNegativeValidated").GetBoolean());
            Assert.True(item.GetProperty("releaseBoundaryHeld").GetBoolean());
            Assert.True(item.GetProperty("sourceTreeRealModelEvidenceReady").GetBoolean());
            Assert.True(item.GetProperty("realOwnerEvidenceReady").GetBoolean());
            Assert.Equal(0, item.GetProperty("ownerActionRequiredCount").GetInt32());
            Assert.True(item.GetProperty("canPromoteRealModelRuntime").GetBoolean());
            Assert.False(item.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
            Assert.NotEmpty(item.GetProperty("contractRequiredMetadata").EnumerateArray());
            Assert.EndsWith("real-model-runtime-evidence.json", item.GetProperty("realModelEvidencePath").GetString(), StringComparison.Ordinal);
            Assert.Contains("Package-consumer/public/release promotion remains blocked", item.GetProperty("blockingReason").GetString(), StringComparison.Ordinal);
        }

        string markdown = ReadText("artifacts", "yolovision", "yolovision-six-task-real-proof-chain-dashboard.md");
        Assert.Contains("YoloVision Six Task Real Proof Chain Dashboard", markdown, StringComparison.Ordinal);
        Assert.Contains("sem", markdown, StringComparison.Ordinal);
        Assert.Contains("canPromotePackageConsumerRuntime", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void DashboardFailsClosedWhenARealModelRecordCrossesItsReleaseBoundary()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "jyppx-yolovision-proof-dashboard-" + Guid.NewGuid().ToString("N"));
        string evidenceRoot = Path.Combine(tempRoot, "evidence");
        string outputRoot = Path.Combine(tempRoot, "output");
        Directory.CreateDirectory(evidenceRoot);

        try
        {
            string sourceRoot = Path.Combine(RepositoryPaths.Root, "samples", "assets");
            string[] evidenceFiles = Directory.GetFiles(sourceRoot, "yolovision-*-real-model-runtime-evidence.json");
            Assert.Equal(6, evidenceFiles.Length);
            foreach (string sourcePath in evidenceFiles)
            {
                File.Copy(sourcePath, Path.Combine(evidenceRoot, Path.GetFileName(sourcePath)));
            }

            string detectionPath = Path.Combine(evidenceRoot, "yolovision-yolov8n-det-real-model-runtime-evidence.json");
            JsonObject detection = JsonNode.Parse(File.ReadAllText(detectionPath))!.AsObject();
            detection["proofBoundary"]!.AsObject()["packageConsumerRuntimeProof"] = true;
            File.WriteAllText(detectionPath, detection.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));

            RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionSixTaskRealProofChainDashboard.ps1"),
                "-EvidenceRoot",
                evidenceRoot,
                "-OutputRoot",
                outputRoot);

            using JsonDocument document = JsonDocument.Parse(
                File.ReadAllText(Path.Combine(outputRoot, "yolovision-six-task-real-proof-chain-dashboard.json")));
            JsonElement dashboard = document.RootElement;
            Assert.Equal("blocked-source-tree-real-model-runtime-proof-required", dashboard.GetProperty("dashboardState").GetString());
            Assert.Equal(5, dashboard.GetProperty("realModelRuntimeReadyTaskCount").GetInt32());
            Assert.Equal(1, dashboard.GetProperty("realModelRuntimeMissingTaskCount").GetInt32());
            Assert.False(dashboard.GetProperty("canPromoteRealModelRuntime").GetBoolean());

            JsonElement detectionTask = Assert.Single(
                dashboard.GetProperty("tasks").EnumerateArray(),
                item => item.GetProperty("task").GetString() == "det");
            Assert.False(detectionTask.GetProperty("releaseBoundaryHeld").GetBoolean());
            Assert.False(detectionTask.GetProperty("sourceTreeRealModelEvidenceReady").GetBoolean());
            Assert.False(detectionTask.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
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

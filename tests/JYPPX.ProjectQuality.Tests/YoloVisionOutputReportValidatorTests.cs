using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class YoloVisionOutputReportValidatorTests
{
    [Fact]
    public void YoloVisionOutputExamplesCoverAllTasksAndKeepProofBoundary()
    {
        string examplesDirectory = Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "examples");
        string[] expectedFiles =
        {
            "yolovision-output-det.example.json",
            "yolovision-output-cls.example.json",
            "yolovision-output-seg.example.json",
            "yolovision-output-obb.example.json",
            "yolovision-output-pose.example.json",
            "yolovision-output-sem.example.json"
        };

        HashSet<string> tasks = new HashSet<string>(StringComparer.Ordinal);
        foreach (string fileName in expectedFiles)
        {
            string path = Path.Combine(examplesDirectory, fileName);
            Assert.True(File.Exists(path), path);

            string text = File.ReadAllText(path);
            using JsonDocument document = JsonDocument.Parse(text);
            JsonElement root = document.RootElement;

            Assert.Equal("yolovision-output.v1", root.GetProperty("schemaVersion").GetString());
            tasks.Add(root.GetProperty("task").GetString()!);
            Assert.NotEmpty(root.GetProperty("outputs").EnumerateArray());
            Assert.NotEmpty(root.GetProperty("predictions").EnumerateArray());
            Assert.False(root.GetProperty("boundary").GetProperty("isRuntimeProof").GetBoolean());
            Assert.Contains("not runtime proof", root.GetProperty("boundary").GetProperty("evidenceKind").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Contains(root.GetProperty("boundary").GetProperty("forbiddenSubstitutes").EnumerateArray(), static item => item.GetString() == "TensorRtExec report");
            Assert.Contains(root.GetProperty("boundary").GetProperty("forbiddenSubstitutes").EnumerateArray(), static item => item.GetString() == "OnnxToEngine report");
            Assert.DoesNotContain("YoloDet", text, StringComparison.OrdinalIgnoreCase);
        }

        Assert.Equal(new[] { "cls", "det", "obb", "pose", "seg", "sem" }, tasks.Order(StringComparer.Ordinal).ToArray());
    }

    [Fact]
    public void YoloVisionOutputValidatorScriptChecksExamplesAndForbiddenPromotion()
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionOutputReport.ps1");
        Assert.True(File.Exists(scriptPath), scriptPath);

        string script = File.ReadAllText(scriptPath);
        Assert.Contains("yolovision-output-report-validation", script, StringComparison.Ordinal);
        Assert.Contains("yolovision-output-report-ready", script, StringComparison.Ordinal);
        Assert.Contains("canPromoteRealModelRuntime = $false", script, StringComparison.Ordinal);
        Assert.Contains("canPromotePackageConsumerRuntime = $false", script, StringComparison.Ordinal);
        Assert.Contains("boundary-not-runtime-proof", script, StringComparison.Ordinal);
        Assert.Contains("task-predictions-match", script, StringComparison.Ordinal);
        Assert.Contains("labels-sha256-well-formed", script, StringComparison.Ordinal);
        Assert.Contains("input-source-kind-supported", script, StringComparison.Ordinal);
        Assert.Contains("input-image-sha256-well-formed", script, StringComparison.Ordinal);
        Assert.Contains("input-preprocessed-tensor-sha256-well-formed", script, StringComparison.Ordinal);
        Assert.Contains("output-value-sha256-well-formed", script, StringComparison.Ordinal);
        Assert.Contains("no-yolodet", script, StringComparison.Ordinal);
        Assert.Contains("samples/YoloVision/examples/yolovision-output-det.example.json", script, StringComparison.Ordinal);
        Assert.Contains("samples/YoloVision/examples/yolovision-output-sem.example.json", script, StringComparison.Ordinal);

        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "README.md"));
        Assert.Contains("samples/YoloVision/examples", readme, StringComparison.Ordinal);
        Assert.Contains("Test-YoloVisionOutputReport.ps1 -Strict", readme, StringComparison.Ordinal);
        Assert.Contains("boundary.isRuntimeProof=false", readme, StringComparison.Ordinal);
        Assert.Contains("valueSha256", readme, StringComparison.Ordinal);
        Assert.Contains("--image", readme, StringComparison.Ordinal);
        Assert.Contains("--preprocessed-output", readme, StringComparison.Ordinal);
        Assert.Contains("--preprocess-only", readme, StringComparison.Ordinal);
        Assert.Contains("--visualization", readme, StringComparison.Ordinal);
    }

    [Fact]
    public void YoloVisionOutputValidatorRejectsInconsistentSegmentationPixelTotals()
    {
        string directory = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "test-temp",
            "yolovision-invalid-seg-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            string inputPath = Path.Combine(directory, "invalid-seg.json");
            string outputPath = Path.Combine(directory, "validation.json");
            JsonNode root = JsonNode.Parse(File.ReadAllText(Path.Combine(
                RepositoryPaths.Root,
                "samples",
                "YoloVision",
                "examples",
                "yolovision-output-seg.example.json")))!;
            root["predictions"]![0]!["maskTotalPixelCount"] = 123;
            File.WriteAllText(inputPath, root.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));

            using Process process = new();
            process.StartInfo.FileName = "pwsh";
            process.StartInfo.ArgumentList.Add("-NoProfile");
            process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
            process.StartInfo.ArgumentList.Add("Bypass");
            process.StartInfo.ArgumentList.Add("-File");
            process.StartInfo.ArgumentList.Add(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionOutputReport.ps1"));
            process.StartInfo.ArgumentList.Add("-InputPath");
            process.StartInfo.ArgumentList.Add(inputPath);
            process.StartInfo.ArgumentList.Add("-OutputPath");
            process.StartInfo.ArgumentList.Add(outputPath);
            process.StartInfo.ArgumentList.Add("-Strict");
            process.StartInfo.WorkingDirectory = RepositoryPaths.Root;
            process.StartInfo.RedirectStandardOutput = true;
            process.StartInfo.RedirectStandardError = true;
            process.StartInfo.UseShellExecute = false;

            process.Start();
            string output = process.StandardOutput.ReadToEnd() + process.StandardError.ReadToEnd();
            process.WaitForExit();

            Assert.NotEqual(0, process.ExitCode);
            Assert.Contains("ValidationState=invalid", output, StringComparison.Ordinal);
            Assert.True(File.Exists(outputPath));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}

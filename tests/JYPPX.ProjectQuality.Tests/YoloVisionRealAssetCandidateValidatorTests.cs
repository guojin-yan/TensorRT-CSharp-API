using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class YoloVisionRealAssetCandidateValidatorTests
{
    [Fact]
    public void ValidatorAcceptsTemplatesButDoesNotPromoteProof()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionRealAssetCandidate.ps1"), "-Strict");

        string validationPath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "yolovision",
            "yolovision-real-asset-candidate-validation.json");
        Assert.True(File.Exists(validationPath));

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(validationPath));
        JsonElement root = document.RootElement;

        Assert.Equal("yolovision-real-asset-candidate-validation", root.GetProperty("recordKind").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.Equal("owner-action-required", root.GetProperty("validationState").GetString());
        Assert.Equal(0, root.GetProperty("failedBlockerCount").GetInt32());

        JsonElement[] records = root.GetProperty("records").EnumerateArray().ToArray();
        Assert.Equal(6, records.Length);
        Assert.Equal(
            new[] { "cls", "det", "obb", "pose", "seg", "sem" },
            records.Select(static record => record.GetProperty("task").GetString()).OrderBy(static task => task).ToArray());

        foreach (JsonElement record in records)
        {
            string task = record.GetProperty("task").GetString()!;
            Assert.Equal("YoloVision", record.GetProperty("sampleName").GetString());
            Assert.Equal("YOLOv8", record.GetProperty("family").GetString());
            Assert.Equal("owner-action-required", record.GetProperty("runtimeProofState").GetString());
            Assert.Equal("template-only", record.GetProperty("proofClassification").GetString());
            Assert.False(record.GetProperty("canPromoteRealModelRuntime").GetBoolean());
            Assert.False(record.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());

            foreach (string id in new[]
            {
                "sample-name-yolovision",
                "family-yolov8",
                "task-supported",
                "package-consumer-never-promoted",
                "template-cannot-promote-real-model-runtime",
                "expected-evidence-passed-line",
                "required-hash-listed-modelSha256",
                "required-hash-listed-imageSha256",
                "required-hash-listed-preprocessedTensorSha256",
                "required-hash-listed-runLogSha256"
            })
            {
                AssertValidationContains(record, id, passed: true);
            }

            string[] taskChecks = task switch
            {
                "pose" => ["pose-keypoint-count-present", "pose-keypoint-layout-present", "pose-score-field-present"],
                "obb" => ["obb-angle-unit-present", "obb-rotated-box-layout-present", "obb-coordinate-space-present"],
                "cls" => ["cls-topk-present", "cls-score-field-present", "cls-labels-required-present"],
                "sem" => ["sem-map-shape-present", "sem-class-map-layout-present", "sem-palette-required-present"],
                _ => []
            };
            Assert.All(taskChecks, id => AssertValidationContains(record, id, passed: true));
        }
    }

    [Fact]
    public void DocumentationAndScriptCaptureOwnerBackfillBoundaries()
    {
        string script = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionRealAssetCandidate.ps1"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "yolovision-real-asset-owner-backfill-validator.md"));
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));

        foreach (string marker in new[]
        {
            "function Test-Sha256",
            "package-consumer-never-promoted",
            "real-runtime-promotion-requires-all-hashes",
            "task-supported",
            "pose-keypoint-count-present",
            "obb-rotated-box-layout-present",
            "cls-topk-present",
            "sem-map-shape-present",
            "YoloVision Passed=True",
            "performsPublish = $false"
        })
        {
            Assert.Contains(marker, script, StringComparison.Ordinal);
        }

        Assert.Contains("articles/zh-cn/yolovision-real-asset-owner-backfill-validator.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/yolovision-real-asset-owner-backfill-validator.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("owner-action-required", article, StringComparison.Ordinal);
        Assert.Contains("64 位 SHA256", article, StringComparison.Ordinal);
        Assert.Contains("不能晋级 package-consumer-runtime", article, StringComparison.Ordinal);
    }

    private static void AssertValidationContains(JsonElement record, string id, bool passed)
    {
        Assert.Contains(record.GetProperty("validationItems").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == id &&
            item.GetProperty("passed").GetBoolean() == passed);
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
    {
        using Process process = new();
        process.StartInfo.FileName = PowerShellHost.ResolveExecutable();
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

using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class TechnicalArticleCampaignMatrixTests
{
    [Fact]
    public void TechnicalArticleCampaignMatrixPlansBroadChinesePublicationWithoutProofOverclaim()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-TechnicalArticleCampaignMatrix.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-TechnicalArticleCampaignMatrix.ps1"), "-Strict");

        using JsonDocument matrixDocument = ReadFinalReleaseJson("technical-article-campaign-matrix.json");
        JsonElement matrix = matrixDocument.RootElement;

        Assert.Equal("technical-article-campaign-matrix", matrix.GetProperty("recordKind").GetString());
        Assert.Equal("publication-campaign-planning-non-proof", matrix.GetProperty("campaignState").GetString());
        Assert.True(matrix.GetProperty("articleCount").GetInt32() >= 30);
        Assert.True(matrix.GetProperty("missingAssetCount").GetInt32() >= 1);
        Assert.True(matrix.GetProperty("requiresRealModelAssets").GetBoolean());
        Assert.False(matrix.GetProperty("performsPublish").GetBoolean());
        Assert.False(matrix.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(matrix.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(matrix.GetProperty("canPromoteRuntimeProof").GetBoolean());

        string[] sourceLinks = matrix.GetProperty("sourceCodeSampleLinks").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("applications/YoloVision", sourceLinks);
        Assert.Contains("applications/OnnxToEngine", sourceLinks);
        Assert.Contains("samples/ComputerVision/01.Classification", sourceLinks);
        Assert.Contains("applications/TensorRtExec", sourceLinks);
        Assert.Contains("src/JYPPX.TensorRtSharp", sourceLinks);

        string[] families = matrix.GetProperty("yoloFamilies").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string family in new[] { "yolov5", "yolov6", "yolov7", "yolov8", "yolov9", "yolov10", "yolov11", "yolov26", "custom" })
        {
            Assert.Contains(family, families);
        }

        string[] tasks = matrix.GetProperty("yoloTasks").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string task in new[] { "det", "cls", "seg", "obb", "pose", "sem" })
        {
            Assert.Contains(task, tasks);
        }

        JsonElement[] tracks = matrix.GetProperty("articleTracks").EnumerateArray().ToArray();
        Assert.True(tracks.Length >= 6);
        Assert.Contains(tracks, item => item.GetProperty("id").GetString() == "yolovision");
        Assert.Contains(tracks, item => item.GetProperty("id").GetString() == "onnx-tensorrtexec");
        Assert.Contains(tracks, item => item.GetProperty("id").GetString() == "proof-release");

        JsonElement[] articles = matrix.GetProperty("articles").EnumerateArray().ToArray();
        Assert.True(articles.Length >= 30);
        Assert.Contains(articles, item => item.GetProperty("title").GetString()!.Contains("项目总览", StringComparison.Ordinal));
        Assert.Contains(articles, item => item.GetProperty("title").GetString()!.Contains("ABI", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(articles, item => item.GetProperty("title").GetString()!.Contains("CUDA / TensorRT", StringComparison.Ordinal));
        Assert.Contains(articles, item => item.GetProperty("title").GetString()!.Contains("NuGet", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(articles, item => item.GetProperty("title").GetString()!.Contains("Windows", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(articles, item => item.GetProperty("title").GetString()!.Contains("Linux", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(articles, item => item.GetProperty("title").GetString()!.Contains("OnnxToEngine", StringComparison.Ordinal));
        Assert.Contains(articles, item => item.GetProperty("title").GetString()!.Contains("TensorRtExec CLI", StringComparison.Ordinal));
        Assert.Contains(articles, item => item.GetProperty("title").GetString()!.Contains("WinForms", StringComparison.Ordinal));
        Assert.Contains(articles, item => item.GetProperty("title").GetString()!.Contains("YoloVision", StringComparison.Ordinal));
        Assert.Contains(articles, item => item.GetProperty("title").GetString()!.Contains("YOLOv8", StringComparison.Ordinal));
        Assert.Contains(articles, item => item.GetProperty("title").GetString()!.Contains("YOLOv26/custom", StringComparison.Ordinal));
        Assert.Contains(articles, item => item.GetProperty("title").GetString()!.Contains("语义分割", StringComparison.Ordinal));
        Assert.Contains(articles, item => item.GetProperty("title").GetString()!.Contains("OBB", StringComparison.Ordinal));
        Assert.Contains(articles, item => item.GetProperty("title").GetString()!.Contains("Pose", StringComparison.Ordinal));
        Assert.Contains(articles, item => item.GetProperty("title").GetString()!.Contains("INT8", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(articles, item => item.GetProperty("title").GetString()!.Contains("Plugin Registry", StringComparison.Ordinal));
        Assert.Contains(articles, item => item.GetProperty("title").GetString()!.Contains("Engine Inspector", StringComparison.Ordinal));
        Assert.Contains(articles, item => item.GetProperty("title").GetString()!.Contains("Package Consumer Runtime Proof", StringComparison.Ordinal));
        Assert.Contains(articles, item => item.GetProperty("title").GetString()!.Contains("Post-Publish Verification", StringComparison.Ordinal));
        Assert.Contains(articles, item => item.GetProperty("title").GetString()!.Contains("常见问题", StringComparison.Ordinal));
        Assert.Contains(articles, item => item.GetProperty("title").GetString()!.Contains("性能调优", StringComparison.Ordinal));
        Assert.Contains(articles, item => item.GetProperty("title").GetString()!.Contains("多平台部署", StringComparison.Ordinal));

        Assert.All(articles, item =>
        {
            Assert.False(item.GetProperty("performsPublish").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("targetAudience").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("articleType").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("modelAcquisitionPlaceholder").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("readiness").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("proofBoundary").GetString()));
            Assert.NotEmpty(item.GetProperty("screenshotsOrImagesNeeded").EnumerateArray());
        });

        string matrixText = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "technical-article-campaign-matrix.json"));
        Assert.DoesNotContain("YoloDet", matrixText, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("dotnet nuget push", matrixText, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("canPublishPublicly=true", matrixText, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("canCloseReleaseIssue=true", matrixText, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("template/dry-run/build-only", matrix.GetProperty("proofBoundary").GetString(), StringComparison.OrdinalIgnoreCase);

        using JsonDocument validationDocument = ReadFinalReleaseJson("technical-article-campaign-matrix-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("technical-article-campaign-matrix-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("technical-article-campaign-matrix-ready", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
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

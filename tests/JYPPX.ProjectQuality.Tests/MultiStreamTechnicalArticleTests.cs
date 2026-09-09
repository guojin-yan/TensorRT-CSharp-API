using System.Security.Cryptography;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class MultiStreamTechnicalArticleTests
{
    [Fact]
    public void RuntimeEvidenceMatchesTrackedSourceAndScreenshot()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "cuda-multistream-article-runtime-evidence.json");
        Assert.True(File.Exists(evidencePath));

        using JsonDocument evidence = JsonDocument.Parse(File.ReadAllText(evidencePath));
        JsonElement root = evidence.RootElement;
        JsonElement assets = root.GetProperty("assets");
        JsonElement validation = root.GetProperty("runtimeValidation");
        JsonElement boundary = root.GetProperty("proofBoundary");
        JsonElement maintenance = root.GetProperty("currentMaintenanceValidation");

        Assert.Equal("cuda-multistream-technical-article-runtime-evidence", root.GetProperty("recordKind").GetString());
        Assert.False(root.GetProperty("workload").GetProperty("modelOrOnnxRequired").GetBoolean());
        Assert.False(root.GetProperty("workload").GetProperty("tensorRtEngineRequired").GetBoolean());
        Assert.True(validation.GetProperty("streamAReadbackPassed").GetBoolean());
        Assert.True(validation.GetProperty("streamBReadbackPassed").GetBoolean());
        Assert.True(validation.GetProperty("independentStreamsPassed").GetBoolean());
        Assert.True(validation.GetProperty("crossStreamWaitPassed").GetBoolean());
        Assert.Equal(0, validation.GetProperty("processExitCode").GetInt32());
        Assert.False(boundary.GetProperty("performsPublish").GetBoolean());
        Assert.False(boundary.GetProperty("uploadsVendorRuntime").GetBoolean());

        string sourcePath = Path.Combine(RepositoryPaths.Root, "samples", "Performance", "01.MultiStream", "Program.cs");
        string screenshotPath = Path.Combine(
            RepositoryPaths.Root,
            assets.GetProperty("runtimeScreenshotPath").GetString()!.Replace('/', Path.DirectorySeparatorChar));
        Assert.Equal(maintenance.GetProperty("currentSourceSha256").GetString(), ComputeSha256(sourcePath));
        Assert.False(maintenance.GetProperty("gpuRuntimeScenarioRerun").GetBoolean());
        Assert.True(maintenance.GetProperty("historicalRuntimeEvidenceRetained").GetBoolean());
        Assert.Equal(assets.GetProperty("runtimeScreenshotSha256").GetString(), ComputeSha256(screenshotPath));

        string sampleReadme = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "samples", "Performance", "01.MultiStream",
            "README.md"));
        Assert.Contains("cuda-stream-event-multistream-tutorial.md", sampleReadme, StringComparison.Ordinal);
    }

    private static string ComputeSha256(string path)
    {
        return Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
    }
}

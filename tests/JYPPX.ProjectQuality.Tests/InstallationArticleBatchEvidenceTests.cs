using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class InstallationArticleBatchEvidenceTests
{
    private static readonly IReadOnlyDictionary<string, string> ExpectedDecisions =
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["INS-001"] = "ready",
            ["INS-002"] = "review",
            ["INS-003"] = "ready",
            ["INS-004"] = "review"
        };

    [Fact]
    public void EvidencePinsExactContainerRuntimeAndSuccessfulGpuConsumer()
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(EvidencePath()));
        JsonElement root = document.RootElement;

        Assert.Equal("installation-article-batch-evidence", root.GetProperty("recordKind").GetString());
        Assert.Matches("^[a-f0-9]{40}$", root.GetProperty("baseCommit").GetString()!);

        JsonElement environment = root.GetProperty("containerEnvironment");
        Assert.Equal("Ubuntu 24.04.2 LTS", environment.GetProperty("os").GetString());
        Assert.Equal("x86_64", environment.GetProperty("architecture").GetString());
        Assert.Matches("^sha256:[a-f0-9]{64}$", environment.GetProperty("imageDigest").GetString()!);
        Assert.Equal("NVIDIA GeForce RTX 3060 Laptop GPU", environment.GetProperty("gpu").GetProperty("name").GetString());

        JsonElement runtime = root.GetProperty("runtimeCombination");
        Assert.Equal("linux-x64-ubuntu24.04-trt10.11-cuda12.9-cudnn9.22", runtime.GetProperty("runtimeKey").GetString());
        Assert.Equal("10.11.0.33", runtime.GetProperty("tensorRtVersion").GetString());
        Assert.Equal("12.9", runtime.GetProperty("cudaToolkitVersion").GetString());
        Assert.Equal("9.22.0.52-1", runtime.GetProperty("cudnnVersion").GetString());
        Assert.True(runtime.GetProperty("inputValidationPassed").GetBoolean());
        Assert.True(runtime.GetProperty("exactVersionMatch").GetBoolean());

        JsonElement officialSmoke = root.GetProperty("officialTensorRtSmoke");
        Assert.True(officialSmoke.GetProperty("passed").GetBoolean());
        Assert.Equal(0, officialSmoke.GetProperty("exitCode").GetInt32());
        Assert.True(officialSmoke.GetProperty("engineSizeBytes").GetInt64() > 0);

        JsonElement repository = root.GetProperty("repositoryValidation");
        JsonElement bridge = repository.GetProperty("nativeBridge");
        Assert.True(bridge.GetProperty("buildPassed").GetBoolean());
        Assert.Equal(0, bridge.GetProperty("unresolvedLddDependencyCount").GetInt32());
        Assert.Matches("^[a-f0-9]{64}$", bridge.GetProperty("sha256").GetString()!);

        JsonElement consumer = repository.GetProperty("minimalPackageRuntimeConsumer");
        Assert.True(consumer.GetProperty("consumerOutsideRepository").GetBoolean());
        Assert.Equal(2, consumer.GetProperty("packageReferenceCount").GetInt32());
        Assert.True(consumer.GetProperty("usesPackageReferenceOnly").GetBoolean());
        Assert.False(consumer.GetProperty("usesProjectReference").GetBoolean());
        Assert.False(consumer.GetProperty("usesDirectAssemblyReference").GetBoolean());
        Assert.True(consumer.GetProperty("readyForEnqueue").GetBoolean());
        Assert.True(consumer.GetProperty("enqueueCompleted").GetBoolean());
        Assert.True(consumer.GetProperty("streamSynchronized").GetBoolean());
        Assert.True(consumer.GetProperty("identityOutputMatch").GetBoolean());
        Assert.True(consumer.GetProperty("passed").GetBoolean());
        Assert.Equal(0, consumer.GetProperty("exitCode").GetInt32());
    }

    [Fact]
    public void EvidenceRetainsCallbackCrashAndPlatformBoundaries()
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(EvidencePath()));
        JsonElement root = document.RootElement;
        JsonElement failedConsumer = root.GetProperty("repositoryValidation")
            .GetProperty("callbackRichPackageRuntimeConsumer");

        Assert.Equal(139, failedConsumer.GetProperty("exitCode").GetInt32());
        Assert.Equal("SIGSEGV", failedConsumer.GetProperty("signal").GetString());
        Assert.Equal("populate_callback_state_snapshot", failedConsumer.GetProperty("nativeFailureSymbol").GetString());
        Assert.True(failedConsumer.GetProperty("bindingReadinessObserved").GetBoolean());
        Assert.False(failedConsumer.GetProperty("enqueueCompleted").GetBoolean());
        Assert.False(failedConsumer.GetProperty("runtimeExecutionProof").GetBoolean());
        Assert.False(failedConsumer.GetProperty("reportedProofBoundaryTextAccepted").GetBoolean());
        Assert.Matches("^[a-f0-9]{64}$", failedConsumer.GetProperty("gdbBacktraceSha256").GetString()!);

        JsonElement boundary = root.GetProperty("proofBoundary");
        Assert.True(boundary.GetProperty("linuxContainerGpuRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("bareMetalLinuxRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("wslGpuRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("gpuCiRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("callbackStateSnapshotProof").GetBoolean());
        Assert.False(boundary.GetProperty("publicPackageProof").GetBoolean());
        Assert.False(boundary.GetProperty("postPublishProof").GetBoolean());
        Assert.False(boundary.GetProperty("performsPublish").GetBoolean());
        Assert.False(boundary.GetProperty("releaseClosureProof").GetBoolean());

        JsonElement audit = root.GetProperty("environmentAudit");
        Assert.Equal(0, audit.GetProperty("wsl").GetProperty("registeredUbuntuDistributionCount").GetInt32());
        Assert.Equal(0, audit.GetProperty("gpuCi").GetProperty("repositoryRunnerTotalCount").GetInt32());

        JsonElement quality = root.GetProperty("qualityValidation");
        Assert.Equal("passed-with-user-untracked-file-isolation", quality.GetProperty("status").GetString());
        Assert.Equal(2233, quality.GetProperty("mainWorkingTreeFullSuite").GetProperty("total").GetInt32());
        Assert.Equal(1, quality.GetProperty("mainWorkingTreeFullSuite").GetProperty("failed").GetInt32());
        Assert.Equal(1, quality.GetProperty("isolatedFailedTestRerun").GetProperty("passed").GetInt32());
        Assert.Equal(0, quality.GetProperty("isolatedFailedTestRerun").GetProperty("failed").GetInt32());
        Assert.Equal("passed", quality.GetProperty("effectiveControlledSourceResult").GetString());
        Assert.True(quality.GetProperty("docfx").GetProperty("buildPassed").GetBoolean());
    }

    [Fact]
    public void CanonicalIndexHeadersAndDocfxMatchEvidenceDecisions()
    {
        using JsonDocument evidence = JsonDocument.Parse(File.ReadAllText(EvidencePath()));
        JsonElement decisions = evidence.RootElement.GetProperty("articleDecisions");

        using JsonDocument index = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root, "docs", "articles", "zh-cn", "article-index.json")));
        Dictionary<string, JsonElement> articles = index.RootElement.GetProperty("articles")
            .EnumerateArray()
            .ToDictionary(article => article.GetProperty("id").GetString()!, article => article);

        foreach ((string id, string expectedStatus) in ExpectedDecisions)
        {
            Assert.Equal(expectedStatus, decisions.GetProperty(id).GetString());
            Assert.Equal(expectedStatus, articles[id].GetProperty("status").GetString());

            string relativePath = articles[id].GetProperty("sourcePath").GetString()!;
            string article = File.ReadAllText(Path.Combine(
                RepositoryPaths.Root,
                relativePath.Replace('/', Path.DirectorySeparatorChar)));
            Assert.Contains(
                $"文章编号：{id}；适用版本：4.0.0；当前状态：{expectedStatus}。",
                article,
                StringComparison.Ordinal);
        }

        string docfx = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "docfx.json"));
        Assert.Contains(
            "articles/zh-cn/05-installation/installation-runtime-evidence-20260814.json",
            docfx,
            StringComparison.Ordinal);
    }

    private static string EvidencePath() => Path.Combine(
        RepositoryPaths.Root,
        "docs",
        "articles",
        "zh-cn",
        "05-installation",
        "installation-runtime-evidence-20260814.json");
}

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
        JsonElement wsl = audit.GetProperty("wsl");
        Assert.Equal(0, wsl.GetProperty("registeredUbuntuDistributionCount").GetInt32());
        Assert.False(wsl.GetProperty("executionStatusReport").GetProperty("isLinux").GetBoolean());
        Assert.False(wsl.GetProperty("executionStatusReport").GetProperty("githubActions").GetBoolean());
        Assert.Equal("blocked", wsl.GetProperty("executionStatusReport").GetProperty("status").GetString());
        Assert.Equal(2, wsl.GetProperty("executionStatusReport").GetProperty("blockerCount").GetInt32());
        Assert.Matches("^[a-f0-9]{64}$", wsl.GetProperty("executionStatusReport").GetProperty("sha256").GetString()!);

        JsonElement wslReport = wsl.GetProperty("wslRuntimeReport");
        Assert.Equal("blocked", wslReport.GetProperty("status").GetString());
        Assert.False(wslReport.GetProperty("environmentReady").GetBoolean());
        Assert.False(wslReport.GetProperty("runtimeConsumerProofAccepted").GetBoolean());
        Assert.Equal(1, wslReport.GetProperty("blockerCount").GetInt32());
        Assert.Matches("^[a-f0-9]{64}$", wslReport.GetProperty("sha256").GetString()!);
        Assert.Matches("^[a-f0-9]{64}$", wslReport.GetProperty("markdownSha256").GetString()!);

        JsonElement gpuCi = audit.GetProperty("gpuCi");
        Assert.Equal(0, gpuCi.GetProperty("repositoryRunnerTotalCount").GetInt32());
        JsonElement runnerReport = gpuCi.GetProperty("runnerAvailabilityReport");
        Assert.True(runnerReport.GetProperty("querySucceeded").GetBoolean());
        Assert.Equal(0, runnerReport.GetProperty("runnerCount").GetInt32());
        Assert.Equal(0, runnerReport.GetProperty("matchingRunnerCount").GetInt32());
        Assert.Equal(0, runnerReport.GetProperty("onlineMatchingRunnerCount").GetInt32());
        Assert.Equal(5, runnerReport.GetProperty("requiredLabelSet").GetArrayLength());
        Assert.Matches("^[a-f0-9]{64}$", runnerReport.GetProperty("sha256").GetString()!);

        JsonElement dedicatedWorkflow = gpuCi.GetProperty("dedicatedWorkflow");
        Assert.Equal(".github/workflows/runtime-linux-gpu-smoke.yml", dedicatedWorkflow.GetProperty("path").GetString());
        Assert.Equal(5, dedicatedWorkflow.GetProperty("requiredRunsOn").GetArrayLength());
        Assert.True(dedicatedWorkflow.GetProperty("workflowAvailable").GetBoolean());
        Assert.False(dedicatedWorkflow.GetProperty("workflowExecuted").GetBoolean());

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

    [Fact]
    public void ReviewArticlesExposeTargetSpecificRuntimeEvidenceEntrypoints()
    {
        string minimalConsumerScript = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Test-MinimalLinuxBridgePackageRuntimeConsumer.ps1"));
        string minimalConsumerProgram = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "templates",
            "MinimalLinuxBridgePackageRuntimeConsumer",
            "Program.cs"));
        string wslExporter = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Export-WslRuntimeEvidence.ps1"));
        string gpuWorkflow = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            ".github",
            "workflows",
            "runtime-linux-gpu-smoke.yml"));

        Assert.Contains("minimal-linux-bridge-package-runtime-consumer", minimalConsumerScript, StringComparison.Ordinal);
        Assert.Contains("packageReferenceCount = 2", minimalConsumerScript, StringComparison.Ordinal);
        Assert.Contains("-RequiredVersion $bridgePackage.version", minimalConsumerScript, StringComparison.Ordinal);
        Assert.Contains("usesPackageReferenceOnly = $true", minimalConsumerScript, StringComparison.Ordinal);
        Assert.Contains("usesProjectReference = $false", minimalConsumerScript, StringComparison.Ordinal);
        Assert.Contains("runtimeExecutionProof = $runtimeExecutionProof", minimalConsumerScript, StringComparison.Ordinal);
        Assert.Contains("hostIsContainer", minimalConsumerScript, StringComparison.Ordinal);
        Assert.Contains("wslKernelDetected", minimalConsumerScript, StringComparison.Ordinal);
        Assert.Contains("GITHUB_WORKFLOW", minimalConsumerScript, StringComparison.Ordinal);
        Assert.Contains("runtime-linux-gpu-smoke", minimalConsumerScript, StringComparison.Ordinal);
        Assert.Contains("canPromoteWslRuntimeProof", minimalConsumerScript, StringComparison.Ordinal);
        Assert.Contains("canPromoteGpuCiRuntimeProof", minimalConsumerScript, StringComparison.Ordinal);
        Assert.Contains("ReadyForEnqueue=", minimalConsumerProgram, StringComparison.Ordinal);
        Assert.Contains("EnqueueCompleted=True", minimalConsumerProgram, StringComparison.Ordinal);
        Assert.Contains("StreamSynchronized=", minimalConsumerProgram, StringComparison.Ordinal);
        Assert.Contains("IdentityOutputMatch=", minimalConsumerProgram, StringComparison.Ordinal);

        Assert.Contains("recordKind = \"wsl-runtime-evidence\"", wslExporter, StringComparison.Ordinal);
        Assert.Contains("No independent Ubuntu WSL distribution is registered.", wslExporter, StringComparison.Ordinal);
        Assert.Contains("Docker Desktop's internal WSL distribution cannot be used", wslExporter, StringComparison.Ordinal);
        Assert.Contains("runtimeConsumerProofAccepted", wslExporter, StringComparison.Ordinal);
        Assert.Contains("wsl-gpu-runtime-proof-candidate", wslExporter, StringComparison.Ordinal);

        Assert.Contains("runs-on: [self-hosted, linux, x64, ubuntu-24.04, gpu]", gpuWorkflow, StringComparison.Ordinal);
        Assert.Contains("Validate GPU runner contract", gpuWorkflow, StringComparison.Ordinal);
        Assert.Contains("Test-MinimalLinuxBridgePackageRuntimeConsumer.ps1", gpuWorkflow, StringComparison.Ordinal);
        Assert.Contains("sha256-inventory.json", gpuWorkflow, StringComparison.Ordinal);
        Assert.Contains("if: ${{ always() }}", gpuWorkflow, StringComparison.Ordinal);

        using JsonDocument evidence = JsonDocument.Parse(File.ReadAllText(EvidencePath()));
        JsonElement entrypoints = evidence.RootElement.GetProperty("completionEntrypoints");
        Assert.True(entrypoints.GetProperty("minimalLinuxBridgeConsumer").GetProperty("packageReferenceOnly").GetBoolean());
        Assert.Equal(5, entrypoints.GetProperty("minimalLinuxBridgeConsumer").GetProperty("requiredRuntimeMarkers").GetArrayLength());
        Assert.False(entrypoints.GetProperty("minimalLinuxBridgeConsumer").GetProperty("executedInWsl").GetBoolean());
        Assert.False(entrypoints.GetProperty("minimalLinuxBridgeConsumer").GetProperty("executedInGpuCi").GetBoolean());
        JsonElement containerRegression = entrypoints.GetProperty("containerRegression");
        Assert.True(containerRegression.GetProperty("runtimeExecutionProof").GetBoolean());
        Assert.Equal("local-package-linux-container-gpu-runtime-proof", containerRegression.GetProperty("proofClassification").GetString());
        Assert.True(containerRegression.GetProperty("containerDetected").GetBoolean());
        Assert.True(containerRegression.GetProperty("wslKernelDetected").GetBoolean());
        Assert.False(containerRegression.GetProperty("wslHostAccepted").GetBoolean());
        Assert.False(containerRegression.GetProperty("canPromoteWslRuntimeProof").GetBoolean());
        Assert.False(containerRegression.GetProperty("canPromoteGpuCiRuntimeProof").GetBoolean());
        Assert.True(containerRegression.GetProperty("readyForEnqueue").GetBoolean());
        Assert.True(containerRegression.GetProperty("enqueueCompleted").GetBoolean());
        Assert.True(containerRegression.GetProperty("streamSynchronized").GetBoolean());
        Assert.True(containerRegression.GetProperty("identityOutputMatch").GetBoolean());
        Assert.Matches("^[a-f0-9]{64}$", containerRegression.GetProperty("sha256").GetString()!);
        Assert.True(entrypoints.GetProperty("wslAudit").GetProperty("rejectsDockerDesktopAsUbuntuProof").GetBoolean());
        Assert.True(entrypoints.GetProperty("gpuCi").GetProperty("routesToRequiredSelfHostedGpuLabels").GetBoolean());

        string wslArticle = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "05-installation",
            "wsl",
            "ins-002-wsl-gpu-passthrough-runtime-validation.md"));
        string gpuCiArticle = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "05-installation",
            "ci",
            "ins-004-gpu-ci-runner-validation.md"));
        Assert.Contains("Export-WslRuntimeEvidence.ps1", wslArticle, StringComparison.Ordinal);
        Assert.Contains("Test-MinimalLinuxBridgePackageRuntimeConsumer.ps1", wslArticle, StringComparison.Ordinal);
        Assert.Contains("runtime-linux-gpu-smoke.yml", gpuCiArticle, StringComparison.Ordinal);
        Assert.Contains("self-hosted,linux,x64,ubuntu-24.04,gpu", gpuCiArticle, StringComparison.Ordinal);
    }

    private static string EvidencePath() => Path.Combine(
        RepositoryPaths.Root,
        "docs",
        "articles",
        "zh-cn",
        "05-installation",
        "installation-runtime-evidence-20260814.json");
}

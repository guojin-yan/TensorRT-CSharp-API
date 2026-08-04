using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ProgressMonitorLocalPackageConsumerTests
{
    [Fact]
    public void ConsumerUsesOnlyPackageReferencesAndExercisesRealProgressAndCancellation()
    {
        string template = ReadSource(
            "samples",
            "ProgressMonitor.PackageConsumer",
            "ProgressMonitor.PackageConsumer.csproj.template");
        string program = ReadSource("samples", "ProgressMonitor.PackageConsumer", "Program.cs");

        Assert.Equal(2, Regex.Matches(template, "<PackageReference ").Count);
        Assert.DoesNotContain("ProjectReference", template, StringComparison.Ordinal);
        Assert.DoesNotContain("HintPath", template, StringComparison.Ordinal);
        Assert.Contains("__MANAGED_PACKAGE_ID__", template, StringComparison.Ordinal);
        Assert.Contains("__BRIDGE_PACKAGE_ID__", template, StringComparison.Ordinal);

        foreach (string required in new[]
        {
            "using JYPPX.TensorRtSharp;",
            "TensorRtEnvironmentProbe.GetCurrent()",
            "TensorRtProgressMonitor positiveMonitor",
            "config.SetProgressMonitor(positiveMonitor)",
            "builder.BuildSerializedNetwork(network, config)",
            "config.ClearProgressMonitor()",
            "ConcurrentDictionary<string, byte>",
            "Interlocked.Increment",
            "TensorRtProgressMonitorEventKind.PhaseStart",
            "TensorRtProgressMonitorEventKind.StepComplete",
            "TensorRtProgressMonitorEventKind.PhaseFinish",
            "return false;",
            "negativeBuildFailed = true",
            "ProgressMonitorRealRuntime=Passed",
            "ProgressMonitorPackageConsumer Passed=True"
        })
        {
            Assert.Contains(required, program, StringComparison.Ordinal);
        }

        Assert.DoesNotContain("JYPPX.SampleSupport", program, StringComparison.Ordinal);
    }

    [Fact]
    public void SharedHarnessSelectsProgressMonitorAndEnforcesPackageIsolation()
    {
        string wrapper = ReadSource("eng", "Test-ProgressMonitorLocalPackageConsumer.ps1");
        string harness = ReadSource("eng", "Test-CallbackOwnerLocalPackageConsumer.ps1");

        Assert.Contains("-Scenario ProgressMonitor @PSBoundParameters", wrapper, StringComparison.Ordinal);
        foreach (string required in new[]
        {
            "ProgressMonitor.PackageConsumer",
            "ProgressMonitor.PackageConsumer.csproj.template",
            "--progress-monitor-runtime-smoke-only",
            "ProgressMonitorRealRuntime=Passed",
            "TensorRtProgressMonitorEventKind",
            "SetProgressMonitor",
            "ClearProgressMonitor",
            "<clear />",
            "JYPPX_NATIVE_BRIDGE_PATH",
            "JYPPX_ENABLE_DEVELOPMENT_PROBING",
            "consumerBridgeMatchesPackage",
            "vendorRuntimeBinaryCountInPackages",
            "vendorRuntimeBinaryCountInConsumerOutput",
            "$negativeBuildFailed",
            "$negativeFailureCount -ne 0",
            "PublicPackageProof=False PerformsPublish=False"
        })
        {
            Assert.Contains(required, harness, StringComparison.Ordinal);
        }

        string combined = wrapper + harness;
        Assert.DoesNotContain("dotnet nuget push", combined, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("gh release create", combined, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("git tag", combined, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void NativeProgressFailureFlagsAreAtomicForConcurrentTensorRtCallbacks()
    {
        foreach (string line in new[] { "v10", "v11" })
        {
            string source = ReadSource("native", "src", "tensorrt", line, "api.cpp");
            int ownerStart = source.IndexOf("class ManagedProgressMonitor final", StringComparison.Ordinal);
            Assert.True(ownerStart >= 0);
            int ownerEnd = source.IndexOf("class ManagedProfiler final", ownerStart, StringComparison.Ordinal);
            Assert.True(ownerEnd > ownerStart);
            string owner = source[ownerStart..ownerEnd];
            Assert.Contains("std::atomic<bool> last_callback_failed_{false};", owner, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void EvidenceMatchesPackagesArtifactsRuntimeAndCancellationCase()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "progress-monitor-local-package-consumer-tensorrt10.11-evidence.json");
        string evidenceText = File.ReadAllText(evidencePath);
        using JsonDocument evidence = JsonDocument.Parse(evidenceText);
        JsonElement root = evidence.RootElement;

        Assert.Equal("local-package-consumer-tensorrt-progress-monitor-runtime", root.GetProperty("evidenceKind").GetString());
        Assert.Equal("passed-local-package-consumer-runtime", root.GetProperty("validationState").GetString());
        Assert.Equal("10.11.0", root.GetProperty("tensorRtVersion").GetString());
        Assert.Equal("12.9", root.GetProperty("cudaToolkitVersion").GetString());
        Assert.Equal(2, root.GetProperty("packages").GetArrayLength());

        JsonElement model = root.GetProperty("model");
        Assert.Equal("not-applicable", model.GetProperty("acquisition").GetString());
        Assert.Equal("not-applicable", model.GetProperty("conversion").GetString());
        Assert.False(model.GetProperty("externalModelRequired").GetBoolean());
        Assert.False(model.GetProperty("outerModelsDirectoryUsed").GetBoolean());

        JsonElement isolation = root.GetProperty("isolation");
        Assert.True(isolation.GetProperty("workspaceOutsideRepository").GetBoolean());
        Assert.True(isolation.GetProperty("packageReferenceOnly").GetBoolean());
        Assert.False(isolation.GetProperty("projectReference").GetBoolean());
        Assert.False(isolation.GetProperty("sourceTreeBinary").GetBoolean());
        Assert.True(isolation.GetProperty("consumerBridgeMatchesPackage").GetBoolean());
        Assert.Equal(0, isolation.GetProperty("vendorRuntimeBinaryCountInPackages").GetInt32());
        Assert.Equal(0, isolation.GetProperty("vendorRuntimeBinaryCountInConsumerOutput").GetInt32());

        JsonElement positive = root.GetProperty("positive");
        Assert.True(positive.GetProperty("passed").GetBoolean());
        Assert.True(positive.GetProperty("attachedDuringBuild").GetBoolean());
        Assert.True(positive.GetProperty("invocationCount").GetUInt64() > 0UL);
        Assert.True(positive.GetProperty("phaseStartCount").GetUInt64() > 0UL);
        Assert.True(positive.GetProperty("stepCompleteCount").GetUInt64() > 0UL);
        Assert.True(positive.GetProperty("phaseFinishCount").GetUInt64() > 0UL);
        Assert.True(positive.GetProperty("distinctPhaseCount").GetUInt64() > 0UL);
        Assert.Equal(0UL, positive.GetProperty("failureCount").GetUInt64());
        Assert.True(positive.GetProperty("metadataCopied").GetBoolean());
        Assert.True(positive.GetProperty("detachVerified").GetBoolean());
        Assert.True(positive.GetProperty("threadSafeHandlerState").GetBoolean());
        Assert.True(positive.GetProperty("nativeFailureFlagAtomic").GetBoolean());
        Assert.True(positive.GetProperty("realCallbackRuntime").GetBoolean());

        JsonElement negative = root.GetProperty("cancellationNegative");
        Assert.True(negative.GetProperty("passed").GetBoolean());
        Assert.True(negative.GetProperty("cancellationRequested").GetBoolean());
        Assert.True(negative.GetProperty("buildFailed").GetBoolean());
        Assert.True(negative.GetProperty("invocationCount").GetUInt64() > 0UL);
        Assert.Equal(0UL, negative.GetProperty("failureCount").GetUInt64());
        Assert.True(negative.GetProperty("detachVerified").GetBoolean());

        foreach (JsonProperty artifact in root.GetProperty("artifacts").EnumerateObject())
        {
            string relativePath = artifact.Value.GetProperty("path").GetString()!;
            string expectedHash = artifact.Value.GetProperty("sha256").GetString()!;
            string path = Path.Combine(RepositoryPaths.Root, relativePath.Replace('/', Path.DirectorySeparatorChar));
            Assert.True(File.Exists(path), $"Missing ProgressMonitor package-consumer artifact: {relativePath}");
            Assert.Equal(artifact.Value.GetProperty("length").GetInt64(), new FileInfo(path).Length);
            string actualHash = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
            Assert.Equal(expectedHash, actualHash);
        }

        JsonElement boundary = root.GetProperty("boundary");
        Assert.True(boundary.GetProperty("isLocalPackageConsumerRuntimeEvidence").GetBoolean());
        Assert.False(boundary.GetProperty("isPublicPackageConsumerProof").GetBoolean());
        Assert.False(boundary.GetProperty("isReleaseProof").GetBoolean());
        Assert.False(boundary.GetProperty("performsPublish").GetBoolean());
        Assert.DoesNotMatch(@"[A-Za-z]:\\", evidenceText);
    }

    [Fact]
    public void TutorialIsCompletePathFreeAndUsesTheRealRuntimeScreenshot()
    {
        string article = ReadSource(
            "docs",
            "articles",
            "zh-cn",
            "progress-monitor-local-package-consumer-tutorial.md");
        string toc = ReadSource("docs", "toc.yml");

        foreach (string required in new[]
        {
            "## 1. 项目与功能背景",
            "## 2. 依赖与包职责",
            "## 3. 模型获取与转换说明",
            "官方获取方式：不适用",
            "ONNX 转换方式：不适用",
            "图像识别结果：不适用",
            "## 5. 仓库外消费者如何隔离",
            "## 7. 正例：线程安全地接收构建进度",
            "## 8. 负例：从 StepComplete 主动取消",
            "## 9. 执行完整验证",
            "## 10. 本机执行结果",
            "progress-monitor-local-package-consumer-terminal.png",
            "RuntimeEnvironment TRT=10.11.0 CUDA=12.9",
            "Callbacks=28409 Start=4768 Step=18873 Finish=4768",
            "CancellationCase=Passed Requested=True BuildFailed=True Callbacks=5",
            "不创建 tag、不创建 GitHub Release、不发布新包"
        })
        {
            Assert.Contains(required, article, StringComparison.Ordinal);
        }

        Assert.Contains("progress-monitor-local-package-consumer-tutorial.md", toc, StringComparison.Ordinal);
        Assert.DoesNotMatch(@"[A-Za-z]:\\", article);
        Assert.DoesNotContain("GitSpace", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("Program Files", article, StringComparison.OrdinalIgnoreCase);
    }

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray()));
    }
}

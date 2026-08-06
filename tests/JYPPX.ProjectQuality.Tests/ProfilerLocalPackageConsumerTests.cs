using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ProfilerLocalPackageConsumerTests
{
    [Fact]
    public void ConsumerUsesOnlyPackagesAndExercisesImmediateDeferredAndExceptionPaths()
    {
        string template = ReadSource("tests", "fixtures", "package-consumers", "Profiler.PackageConsumer", "Profiler.PackageConsumer.csproj.template");
        string program = ReadSource("tests", "fixtures", "package-consumers", "Profiler.PackageConsumer", "Program.cs");

        Assert.Equal(2, Regex.Matches(template, "<PackageReference ").Count);
        Assert.DoesNotContain("ProjectReference", template, StringComparison.Ordinal);
        Assert.DoesNotContain("HintPath", template, StringComparison.Ordinal);
        foreach (string required in new[]
        {
            "using JYPPX.CudaSharp;",
            "using JYPPX.TensorRtSharp;",
            "TensorRtEnvironmentProbe.GetCurrent()",
            "TensorRtProfiler immediateProfiler",
            "context.SetProfiler(immediateProfiler)",
            "context.EnqueueEmitsProfile = true",
            "context.EnqueueEmitsProfile = false",
            "context.ReportToProfiler()",
            "context.EnqueueAsync(stream)",
            "context.ClearProfiler()",
            "ConcurrentDictionary<string, byte>",
            "float.IsFinite(milliseconds)",
            "controlled profiler handler failure",
            "negativeProfiler.CallbackFailureCount > 0",
            "ProfilerRealRuntime=Passed",
            "ProfilerPackageConsumer Passed=True"
        })
        {
            Assert.Contains(required, program, StringComparison.Ordinal);
        }

        Assert.DoesNotContain("JYPPX.SampleSupport", program, StringComparison.Ordinal);
    }

    [Fact]
    public void SharedHarnessSelectsProfilerAndEnforcesIsolationAndRuntimeInvariants()
    {
        string wrapper = ReadSource("eng", "Test-ProfilerLocalPackageConsumer.ps1");
        string harness = ReadSource("eng", "Test-CallbackOwnerLocalPackageConsumer.ps1");
        Assert.Contains("-Scenario Profiler @PSBoundParameters", wrapper, StringComparison.Ordinal);

        foreach (string required in new[]
        {
            "Profiler.PackageConsumer",
            "Profiler.PackageConsumer.csproj.template",
            "--profiler-runtime-smoke-only",
            "ProfilerRealRuntime=Passed",
            "TensorRtProfilerHandler",
            "ReportToProfiler",
            "EnqueueEmitsProfile",
            "<clear />",
            "JYPPX_NATIVE_BRIDGE_PATH",
            "JYPPX_ENABLE_DEVELOPMENT_PROBING",
            "consumerBridgeMatchesPackage",
            "$deferredBeforeReportCount -ne 0",
            "$negativeFailureCount -eq 0",
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
    public void EvidenceMatchesRuntimeArtifactsAndBoundary()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "profiler-local-package-consumer-tensorrt10.11-evidence.json");
        string evidenceText = File.ReadAllText(evidencePath);
        using JsonDocument document = JsonDocument.Parse(evidenceText);
        JsonElement root = document.RootElement;

        Assert.Equal("local-package-consumer-tensorrt-profiler-runtime", root.GetProperty("evidenceKind").GetString());
        Assert.Equal("passed-local-package-consumer-runtime", root.GetProperty("validationState").GetString());
        Assert.Equal("10.11.0", root.GetProperty("tensorRtVersion").GetString());
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

        JsonElement immediate = root.GetProperty("immediate");
        Assert.True(immediate.GetProperty("passed").GetBoolean());
        Assert.True(immediate.GetProperty("enqueueEmitsProfile").GetBoolean());
        Assert.True(immediate.GetProperty("invocationCount").GetUInt64() > 0UL);
        Assert.True(immediate.GetProperty("distinctLayerCount").GetUInt64() > 0UL);
        Assert.Equal(0UL, immediate.GetProperty("failureCount").GetUInt64());
        Assert.True(immediate.GetProperty("metadataCopied").GetBoolean());
        Assert.True(immediate.GetProperty("detachVerified").GetBoolean());

        JsonElement deferred = root.GetProperty("deferred");
        Assert.True(deferred.GetProperty("passed").GetBoolean());
        Assert.False(deferred.GetProperty("enqueueEmitsProfile").GetBoolean());
        Assert.Equal(0UL, deferred.GetProperty("beforeReportCount").GetUInt64());
        Assert.True(deferred.GetProperty("reportToProfilerReturned").GetBoolean());
        Assert.True(deferred.GetProperty("invocationCount").GetUInt64() > 0UL);
        Assert.Equal(0UL, deferred.GetProperty("failureCount").GetUInt64());
        Assert.True(deferred.GetProperty("realCallbackRuntime").GetBoolean());

        JsonElement negative = root.GetProperty("handlerExceptionNegative");
        Assert.True(negative.GetProperty("passed").GetBoolean());
        Assert.True(negative.GetProperty("invocationCount").GetUInt64() > 0UL);
        Assert.True(negative.GetProperty("failureCount").GetUInt64() > 0UL);
        Assert.Equal("InvalidOperationException", negative.GetProperty("lastExceptionType").GetString());
        Assert.True(negative.GetProperty("detachVerified").GetBoolean());

        foreach (JsonProperty artifact in root.GetProperty("artifacts").EnumerateObject())
        {
            string relativePath = artifact.Value.GetProperty("path").GetString()!;
            string path = Path.Combine(RepositoryPaths.Root, relativePath.Replace('/', Path.DirectorySeparatorChar));
            Assert.True(File.Exists(path), $"Missing profiler artifact: {relativePath}");
            Assert.Equal(artifact.Value.GetProperty("length").GetInt64(), new FileInfo(path).Length);
            string hash = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
            Assert.Equal(artifact.Value.GetProperty("sha256").GetString(), hash);
        }

        JsonElement boundary = root.GetProperty("boundary");
        Assert.True(boundary.GetProperty("isLocalPackageConsumerRuntimeEvidence").GetBoolean());
        Assert.False(boundary.GetProperty("isPublicPackageConsumerProof").GetBoolean());
        Assert.False(boundary.GetProperty("isReleaseProof").GetBoolean());
        Assert.False(boundary.GetProperty("performsPublish").GetBoolean());
        Assert.DoesNotMatch(@"[A-Za-z]:\\", evidenceText);
    }

    [Fact]
    public void TutorialIsCompletePathFreeAndUsesTheRuntimeScreenshot()
    {
        string article = ReadSource("docs", "articles", "zh-cn", "profiler-local-package-consumer-tutorial.md");
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
            "## 7. 正例一：enqueue 即时上报",
            "## 8. 正例二与负例",
            "## 9. 执行完整验证",
            "## 10. 本机执行结果",
            "profiler-local-package-consumer-terminal.png",
            "ImmediateCallbacks=3 Layers=3 Failures=0",
            "DeferredBeforeReport=0 Reported=True Callbacks=3 Layers=3",
            "ExceptionCase=Passed EnqueueFailed=False Callbacks=3 Failures=3",
            "不创建 tag、不创建 GitHub Release、不发布新包"
        })
        {
            Assert.Contains(required, article, StringComparison.Ordinal);
        }

        Assert.Contains("profiler-local-package-consumer-tutorial.md", toc, StringComparison.Ordinal);
        Assert.DoesNotMatch(@"[A-Za-z]:\\", article);
        Assert.DoesNotContain("GitSpace", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("Program Files", article, StringComparison.OrdinalIgnoreCase);
    }

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray()));
    }
}

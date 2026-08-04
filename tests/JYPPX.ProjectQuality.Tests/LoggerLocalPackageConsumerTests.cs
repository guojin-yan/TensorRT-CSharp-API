using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class LoggerLocalPackageConsumerTests
{
    [Fact]
    public void ConsumerUsesOnlyPackagesAndReceivesRealTensorRtMessages()
    {
        string template = ReadSource("samples", "Logger.PackageConsumer", "Logger.PackageConsumer.csproj.template");
        string program = ReadSource("samples", "Logger.PackageConsumer", "Program.cs");

        Assert.Equal(2, Regex.Matches(template, "<PackageReference ").Count);
        Assert.DoesNotContain("ProjectReference", template, StringComparison.Ordinal);
        Assert.DoesNotContain("HintPath", template, StringComparison.Ordinal);
        foreach (string required in new[]
        {
            "using JYPPX.CudaSharp;",
            "using JYPPX.TensorRtSharp;",
            "TensorRtEnvironmentProbe.GetCurrent()",
            "new TensorRtLogger(",
            "TensorRtLogSeverity.Verbose",
            "builder.BuildSerializedNetwork(network, config)",
            "runtime.Deserialize(plan)",
            "context.EnqueueAsync(stream)",
            "logger.Dispose();",
            "postDisposeCallbacks",
            "controlled logger handler failure",
            "ConcurrentDictionary<TensorRtLogSeverity, byte>",
            "Interlocked.Increment",
            "LoggerRealRuntime=Passed",
            "SyntheticDiagnosticUsed=False",
            "LoggerPackageConsumer Passed=True"
        })
        {
            Assert.Contains(required, program, StringComparison.Ordinal);
        }

        Assert.DoesNotContain("EmitDiagnostic", program, StringComparison.Ordinal);
        Assert.DoesNotContain("JYPPX.SampleSupport", program, StringComparison.Ordinal);
    }

    [Fact]
    public void SharedHarnessSelectsLoggerAndEnforcesIsolationAndRuntimeInvariants()
    {
        string wrapper = ReadSource("eng", "Test-LoggerLocalPackageConsumer.ps1");
        string harness = ReadSource("eng", "Test-CallbackOwnerLocalPackageConsumer.ps1");
        Assert.Contains("-Scenario Logger @PSBoundParameters", wrapper, StringComparison.Ordinal);

        foreach (string required in new[]
        {
            "Logger.PackageConsumer",
            "Logger.PackageConsumer.csproj.template",
            "--logger-runtime-smoke-only",
            "LoggerRealRuntime=Passed",
            "TensorRtLogHandler",
            "CallbackInvocationCount",
            "LastCallbackException",
            "<clear />",
            "JYPPX_NATIVE_BRIDGE_PATH",
            "JYPPX_ENABLE_DEVELOPMENT_PROBING",
            "consumerBridgeMatchesPackage",
            "$beforeOwnerCount -ne 0",
            "$negativeFailureCount -ne $negativeInvocationCount",
            "$syntheticDiagnosticUsed",
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
    public void NativeMonitoringFailureFlagsAndTensorRt11SnapshotAreThreadSafe()
    {
        foreach (string line in new[] { "v8", "v10", "v11" })
        {
            string source = ReadSource("native", "src", "tensorrt", line, "api.cpp");
            int loggerStart = source.IndexOf("class ManagedLogger final", StringComparison.Ordinal);
            Assert.True(loggerStart >= 0);
            string loggerEndMarker = line == "v8" ? "class ManagedProfiler final" : "class ManagedProgressMonitor final";
            int loggerEnd = source.IndexOf(loggerEndMarker, loggerStart, StringComparison.Ordinal);
            Assert.True(loggerEnd > loggerStart);
            string logger = source[loggerStart..loggerEnd];
            Assert.Contains("std::atomic<bool> last_callback_failed_{false};", logger, StringComparison.Ordinal);
            Assert.Contains("last_callback_failed_.load(std::memory_order_relaxed)", logger, StringComparison.Ordinal);

            int profilerStart = source.IndexOf("class ManagedProfiler final", StringComparison.Ordinal);
            int profilerEnd = source.IndexOf("struct LayerReferencePayload", profilerStart, StringComparison.Ordinal);
            Assert.True(profilerStart >= 0 && profilerEnd > profilerStart);
            string profiler = source[profilerStart..profilerEnd];
            Assert.Contains("std::atomic<bool> last_callback_failed_{false};", profiler, StringComparison.Ordinal);
            Assert.Contains("last_callback_failed_.load(std::memory_order_relaxed)", profiler, StringComparison.Ordinal);
        }

        string tensorRt11 = ReadSource("native", "src", "tensorrt", "v11", "api.cpp");
        Assert.Contains("mutable std::mutex diagnostics_mutex_;", tensorRt11, StringComparison.Ordinal);
        Assert.Contains("void record_message(", tensorRt11, StringComparison.Ordinal);
        Assert.Contains("void copy_runtime_create_diagnostic(", tensorRt11, StringComparison.Ordinal);
        Assert.Contains("std::lock_guard<std::mutex> lock(diagnostics_mutex_);", tensorRt11, StringComparison.Ordinal);
        Assert.DoesNotContain("logger.last_message()", tensorRt11, StringComparison.Ordinal);
    }

    [Fact]
    public void EvidenceMatchesRuntimeArtifactsPackagesAndBoundary()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "logger-local-package-consumer-tensorrt10.11-evidence.json");
        string evidenceText = File.ReadAllText(evidencePath);
        using JsonDocument document = JsonDocument.Parse(evidenceText);
        JsonElement root = document.RootElement;

        Assert.Equal("local-package-consumer-tensorrt-logger-runtime", root.GetProperty("evidenceKind").GetString());
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

        JsonElement positive = root.GetProperty("positive");
        Assert.True(positive.GetProperty("passed").GetBoolean());
        Assert.Equal(0UL, positive.GetProperty("beforeOwnerCount").GetUInt64());
        Assert.True(positive.GetProperty("afterBuildCount").GetUInt64() > 0UL);
        Assert.True(positive.GetProperty("invocationCount").GetUInt64() >= positive.GetProperty("afterBuildCount").GetUInt64());
        Assert.True(positive.GetProperty("distinctSeverityCount").GetUInt64() > 0UL);
        Assert.Equal(0UL, positive.GetProperty("failureCount").GetUInt64());
        Assert.True(positive.GetProperty("metadataCopied").GetBoolean());
        Assert.True(positive.GetProperty("builderAttached").GetBoolean());
        Assert.True(positive.GetProperty("builderDetached").GetBoolean());
        Assert.True(positive.GetProperty("runtimeAttached").GetBoolean());
        Assert.True(positive.GetProperty("runtimeDetached").GetBoolean());
        Assert.True(positive.GetProperty("realCallbackRuntime").GetBoolean());
        Assert.False(positive.GetProperty("syntheticDiagnosticUsed").GetBoolean());

        JsonElement lifecycle = root.GetProperty("deferredDispose");
        Assert.True(lifecycle.GetProperty("passed").GetBoolean());
        Assert.True(lifecycle.GetProperty("attachedAfterDispose").GetBoolean());
        Assert.True(lifecycle.GetProperty("postDisposeCallbacks").GetBoolean());
        Assert.True(lifecycle.GetProperty("detached").GetBoolean());
        Assert.True(lifecycle.GetProperty("rejectsNewBorrower").GetBoolean());

        JsonElement negative = root.GetProperty("handlerExceptionNegative");
        Assert.True(negative.GetProperty("passed").GetBoolean());
        Assert.True(negative.GetProperty("invocationCount").GetUInt64() > 0UL);
        Assert.Equal(negative.GetProperty("invocationCount").GetUInt64(), negative.GetProperty("failureCount").GetUInt64());
        Assert.Equal("InvalidOperationException", negative.GetProperty("lastExceptionType").GetString());
        Assert.True(negative.GetProperty("detachVerified").GetBoolean());

        foreach (JsonProperty artifact in root.GetProperty("artifacts").EnumerateObject())
        {
            string relativePath = artifact.Value.GetProperty("path").GetString()!;
            string path = Path.Combine(RepositoryPaths.Root, relativePath.Replace('/', Path.DirectorySeparatorChar));
            Assert.True(File.Exists(path), $"Missing logger artifact: {relativePath}");
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
        string article = ReadSource("docs", "articles", "zh-cn", "logger-local-package-consumer-tutorial.md");
        string toc = ReadSource("docs", "toc.yml");
        foreach (string required in new[]
        {
            "## 1. 项目与功能背景",
            "## 2. 依赖与包职责",
            "## 3. 模型获取与转换说明",
            "官方获取方式：不适用",
            "ONNX 转换方式：不适用",
            "图像识别结果：不适用",
            "## 4. 并发与 ABI 安全修正",
            "## 6. 仓库外消费者如何隔离",
            "## 8. 生命周期负例：Dispose 不提前释放 borrowed logger",
            "## 9. 异常负例：handler 异常不跨 ABI",
            "## 11. 本机执行结果",
            "logger-local-package-consumer-terminal.png",
            "PositiveCallbacks=305 Severities=2 Failures=0",
            "ExceptionCase=Passed OperationFailed=False Callbacks=299 Failures=299",
            "SyntheticDiagnosticUsed=False",
            "不创建 tag、不创建 GitHub Release、不发布新包"
        })
        {
            Assert.Contains(required, article, StringComparison.Ordinal);
        }

        Assert.Contains("logger-local-package-consumer-tutorial.md", toc, StringComparison.Ordinal);
        Assert.DoesNotMatch(@"[A-Za-z]:\\", article);
        Assert.DoesNotContain("GitSpace", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("Program Files", article, StringComparison.OrdinalIgnoreCase);
    }

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray()));
    }
}

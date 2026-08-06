using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DebugListenerLocalPackageConsumerTests
{
    [Fact]
    public void ConsumerUsesOnlyPackageReferencesAndExercisesAcceptedAndRejectedCallbacks()
    {
        string template = ReadSource(
            "tests", "fixtures", "package-consumers",
            "DebugListener.PackageConsumer",
            "DebugListener.PackageConsumer.csproj.template");
        string program = ReadSource("tests", "fixtures", "package-consumers", "DebugListener.PackageConsumer", "Program.cs");

        Assert.Equal(2, Regex.Matches(template, "<PackageReference ").Count);
        Assert.DoesNotContain("ProjectReference", template, StringComparison.Ordinal);
        Assert.DoesNotContain("HintPath", template, StringComparison.Ordinal);
        Assert.Contains("__MANAGED_PACKAGE_ID__", template, StringComparison.Ordinal);
        Assert.Contains("__BRIDGE_PACKAGE_ID__", template, StringComparison.Ordinal);

        foreach (string required in new[]
        {
            "using JYPPX.CudaSharp;",
            "using JYPPX.TensorRtSharp;",
            "TensorRtEnvironmentProbe.GetCurrent()",
            "network.MarkDebugTensor(output)",
            "TensorRtDebugListenerCallbackOwner",
            "SetDebugListener(positiveOwner)",
            "SetTensorDebugState(\"debug_output\", true)",
            "ClearDebugListener()",
            "copiedMetadata.MetadataCopied",
            "positiveAttached.BorrowedPointerExposed",
            "negativeOwner = new(line, _ => false)",
            "negativeAttached.FailureCount > 0",
            "DebugListenerRealRuntime=Passed",
            "DebugListenerPackageConsumer Passed=True"
        })
        {
            Assert.Contains(required, program, StringComparison.Ordinal);
        }

        Assert.DoesNotContain("JYPPX.SampleSupport", program, StringComparison.Ordinal);
    }

    [Fact]
    public void SharedHarnessSelectsDebugListenerAndEnforcesPackageIsolation()
    {
        string wrapper = ReadSource("eng", "Test-DebugListenerLocalPackageConsumer.ps1");
        string harness = ReadSource("eng", "Test-CallbackOwnerLocalPackageConsumer.ps1");

        Assert.Contains("-Scenario DebugListener @PSBoundParameters", wrapper, StringComparison.Ordinal);
        foreach (string required in new[]
        {
            "DebugListener.PackageConsumer",
            "DebugListener.PackageConsumer.csproj.template",
            "--debug-listener-runtime-smoke-only",
            "DebugListenerRealRuntime=Passed",
            "TensorRtDebugListenerCallbackOwner",
            "SetTensorDebugState",
            "<clear />",
            "JYPPX_NATIVE_BRIDGE_PATH",
            "JYPPX_ENABLE_DEVELOPMENT_PROBING",
            "consumerBridgeMatchesPackage",
            "vendorRuntimeBinaryCountInPackages",
            "vendorRuntimeBinaryCountInConsumerOutput",
            "$negativeFailureCount -eq 0",
            "$inFlightCallbackCount -ne 0",
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
    public void EvidenceMatchesPackagesArtifactsRuntimeAndRejectedCase()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "debug-listener-local-package-consumer-tensorrt10.11-evidence.json");
        string evidenceText = File.ReadAllText(evidencePath);
        using JsonDocument evidence = JsonDocument.Parse(evidenceText);
        JsonElement root = evidence.RootElement;

        Assert.Equal("local-package-consumer-tensorrt-debug-listener-runtime", root.GetProperty("evidenceKind").GetString());
        Assert.Equal("passed-local-package-consumer-runtime", root.GetProperty("validationState").GetString());
        Assert.Equal("10.11.0", root.GetProperty("tensorRtVersion").GetString());
        Assert.Equal("12.9", root.GetProperty("cudaToolkitVersion").GetString());
        Assert.Equal(2, root.GetProperty("packages").GetArrayLength());

        JsonElement model = root.GetProperty("model");
        Assert.Equal("not-applicable", model.GetProperty("acquisition").GetString());
        Assert.Equal("not-applicable", model.GetProperty("conversion").GetString());
        Assert.False(model.GetProperty("externalModelRequired").GetBoolean());

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
        Assert.True(positive.GetProperty("nativeVTableInstalled").GetBoolean());
        Assert.True(positive.GetProperty("processDebugTensorInvoked").GetBoolean());
        Assert.Equal(1UL, positive.GetProperty("invocationCount").GetUInt64());
        Assert.Equal(0UL, positive.GetProperty("failureCount").GetUInt64());
        Assert.Equal(0UL, positive.GetProperty("inFlightCallbackCount").GetUInt64());
        Assert.Equal("debug_output", positive.GetProperty("tensorName").GetString());
        Assert.Equal(new[] { 1, 4 }, positive.GetProperty("shape").EnumerateArray().Select(value => value.GetInt32()));
        Assert.True(positive.GetProperty("metadataCopied").GetBoolean());
        Assert.False(positive.GetProperty("borrowedPointerExposed").GetBoolean());
        Assert.Equal(1UL, positive.GetProperty("detachCount").GetUInt64());
        Assert.True(positive.GetProperty("realCallbackRuntime").GetBoolean());

        JsonElement negative = root.GetProperty("rejectionNegative");
        Assert.True(negative.GetProperty("passed").GetBoolean());
        Assert.True(negative.GetProperty("callbackRejected").GetBoolean());
        Assert.False(negative.GetProperty("enqueueFailed").GetBoolean());
        Assert.Equal(1UL, negative.GetProperty("invocationCount").GetUInt64());
        Assert.Equal(1UL, negative.GetProperty("failureCount").GetUInt64());

        foreach (JsonProperty artifact in root.GetProperty("artifacts").EnumerateObject())
        {
            string relativePath = artifact.Value.GetProperty("path").GetString()!;
            string expectedHash = artifact.Value.GetProperty("sha256").GetString()!;
            string path = Path.Combine(RepositoryPaths.Root, relativePath.Replace('/', Path.DirectorySeparatorChar));
            Assert.True(File.Exists(path), $"Missing DebugListener package-consumer artifact: {relativePath}");
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
            "debug-listener-local-package-consumer-tutorial.md");
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
            "## 7. 正例：接收复制后的调试元数据",
            "## 8. 负例：handler 返回 false",
            "## 9. 执行完整验证",
            "## 10. 本机执行结果",
            "debug-listener-local-package-consumer-terminal.png",
            "RuntimeEnvironment TRT=10.11.0 CUDA=12.9",
            "Callbacks=1 Failures=0 InFlight=0",
            "Tensor=debug_output Shape=[1,4] MetadataCopied=True",
            "RejectionCase=Passed EnqueueFailed=False Callbacks=1 Failures=1",
            "不创建 tag、不创建 GitHub Release、不发布新包"
        })
        {
            Assert.Contains(required, article, StringComparison.Ordinal);
        }

        Assert.Contains("debug-listener-local-package-consumer-tutorial.md", toc, StringComparison.Ordinal);
        Assert.DoesNotMatch(@"[A-Za-z]:\\", article);
        Assert.DoesNotContain("GitSpace", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("Program Files", article, StringComparison.OrdinalIgnoreCase);
    }

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray()));
    }
}

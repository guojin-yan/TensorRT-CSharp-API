using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class GpuAllocatorLocalPackageConsumerTests
{
    [Fact]
    public void ConsumerTemplateUsesOnlyManagedAndBridgePackages()
    {
        string template = ReadSource(
            "tests", "fixtures", "package-consumers",
            "GpuAllocator.PackageConsumer",
            "GpuAllocator.PackageConsumer.csproj.template");
        string program = ReadSource("tests", "fixtures", "package-consumers", "GpuAllocator.PackageConsumer", "Program.cs");

        Assert.Equal(2, Regex.Matches(template, "<PackageReference ").Count);
        Assert.DoesNotContain("ProjectReference", template, StringComparison.Ordinal);
        Assert.DoesNotContain("HintPath", template, StringComparison.Ordinal);
        Assert.Contains("__MANAGED_PACKAGE_ID__", template, StringComparison.Ordinal);
        Assert.Contains("__BRIDGE_PACKAGE_ID__", template, StringComparison.Ordinal);

        Assert.Contains("using JYPPX.TensorRtSharp;", program, StringComparison.Ordinal);
        Assert.Contains("TensorRtGpuAllocatorCallbackOwner", program, StringComparison.Ordinal);
        Assert.Contains("runtime.SetGpuAllocator(runtimeOwner)", program, StringComparison.Ordinal);
        Assert.Contains("builder.SetGpuAllocator(builderOwner)", program, StringComparison.Ordinal);
        Assert.Contains("TensorRtEnvironmentProbe.GetCurrent()", program, StringComparison.Ordinal);
        Assert.Contains("RuntimeEnvironment", program, StringComparison.Ordinal);
        Assert.Contains("builderReleased.LiveAllocationCount == 0UL", program, StringComparison.Ordinal);
        Assert.Contains("RejectedCount == 0UL", program, StringComparison.Ordinal);
        Assert.Contains("CallbackFailureCount == 0UL", program, StringComparison.Ordinal);
        Assert.Contains("GpuAllocatorPackageConsumer Passed=True", program, StringComparison.Ordinal);
        Assert.DoesNotContain("JYPPX.SampleSupport", program, StringComparison.Ordinal);
    }

    [Fact]
    public void ValidationScriptEnforcesExternalPackageIsolationAndNeverPublishes()
    {
        string wrapper = ReadSource("eng", "Test-GpuAllocatorLocalPackageConsumer.ps1");
        string script = ReadSource("eng", "Test-CallbackOwnerLocalPackageConsumer.ps1");

        foreach (string required in new[]
        {
            "GpuAllocator.PackageConsumer",
            "GpuAllocator.PackageConsumer.csproj.template",
            "<clear />",
            "JYPPX_NATIVE_BRIDGE_PATH",
            "JYPPX_ENABLE_DEVELOPMENT_PROBING",
            "vendorRuntimeBinaryCountInPackages",
            "vendorRuntimeBinaryCountInConsumerOutput",
            "consumerBridgeMatchesPackage",
            "GpuAllocatorRealRuntime=Passed",
            "PublicPackageProof=False PerformsPublish=False"
        })
        {
            Assert.Contains(required, script, StringComparison.Ordinal);
        }

        Assert.Contains("-Scenario GpuAllocator @PSBoundParameters", wrapper, StringComparison.Ordinal);
        Assert.DoesNotContain("CallbackAllocatorSafeControlsSmokeRunner\\Program.cs", script, StringComparison.Ordinal);
        string combined = wrapper + script;
        Assert.DoesNotContain("dotnet nuget push", combined, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("gh release create", combined, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("git tag", combined, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void RuntimeEvidenceMatchesPackagesArtifactsAndFailClosedResults()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "gpu-allocator-local-package-consumer-tensorrt10.11-evidence.json");
        string evidenceText = File.ReadAllText(evidencePath);
        using JsonDocument evidence = JsonDocument.Parse(evidenceText);
        JsonElement root = evidence.RootElement;

        Assert.Equal("local-package-consumer-tensorrt-gpu-allocator-runtime", root.GetProperty("evidenceKind").GetString());
        Assert.Equal("passed-local-package-consumer-runtime", root.GetProperty("validationState").GetString());
        Assert.Equal("10.11.0", root.GetProperty("tensorRtVersion").GetString());
        Assert.Equal("12.9", root.GetProperty("cudaToolkitVersion").GetString());
        Assert.Equal(2, root.GetProperty("packages").GetArrayLength());

        JsonElement isolation = root.GetProperty("isolation");
        Assert.True(isolation.GetProperty("workspaceOutsideRepository").GetBoolean());
        Assert.True(isolation.GetProperty("packageReferenceOnly").GetBoolean());
        Assert.False(isolation.GetProperty("projectReference").GetBoolean());
        Assert.False(isolation.GetProperty("sourceTreeBinary").GetBoolean());
        Assert.True(isolation.GetProperty("consumerBridgeMatchesPackage").GetBoolean());
        Assert.Equal(0, isolation.GetProperty("vendorRuntimeBinaryCountInPackages").GetInt32());
        Assert.Equal(0, isolation.GetProperty("vendorRuntimeBinaryCountInConsumerOutput").GetInt32());

        JsonElement builder = root.GetProperty("builderPositive");
        Assert.True(builder.GetProperty("passed").GetBoolean());
        Assert.Equal(8UL, builder.GetProperty("invocationCount").GetUInt64());
        Assert.Equal(3UL, builder.GetProperty("allocateCount").GetUInt64());
        Assert.Equal(2UL, builder.GetProperty("reallocateCount").GetUInt64());
        Assert.Equal(3UL, builder.GetProperty("deallocateCount").GetUInt64());
        Assert.Equal(0UL, builder.GetProperty("liveAllocationCountAfterEngineDispose").GetUInt64());
        Assert.False(builder.GetProperty("nativePointerExposed").GetBoolean());
        Assert.True(builder.GetProperty("realCallbackRuntime").GetBoolean());

        Assert.True(root.GetProperty("rejectionNegative").GetProperty("buildFailed").GetBoolean());
        Assert.Equal(1UL, root.GetProperty("rejectionNegative").GetProperty("rejectedCount").GetUInt64());
        Assert.True(root.GetProperty("exceptionNegative").GetProperty("buildFailed").GetBoolean());
        Assert.Equal(1UL, root.GetProperty("exceptionNegative").GetProperty("callbackFailureCount").GetUInt64());

        foreach (JsonProperty artifact in root.GetProperty("artifacts").EnumerateObject())
        {
            string relativePath = artifact.Value.GetProperty("path").GetString()!;
            string expectedHash = artifact.Value.GetProperty("sha256").GetString()!;
            string path = Path.Combine(RepositoryPaths.Root, relativePath.Replace('/', Path.DirectorySeparatorChar));
            Assert.True(File.Exists(path), $"Missing GPU allocator package-consumer artifact: {relativePath}");
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
    public void TutorialIsACompletePathFreeTechnicalArticleWithRuntimeScreenshot()
    {
        string article = ReadSource(
            "docs",
            "articles",
            "zh-cn",
            "gpu-allocator-local-package-consumer-tutorial.md");
        string toc = ReadSource("docs", "toc.yml");

        foreach (string required in new[]
        {
            "## 1. 项目与验证目标",
            "## 2. 依赖与包职责",
            "## 3. 模型获取与转换说明",
            "官方获取方式：不适用",
            "ONNX 转换方式：不适用",
            "## 4. 安装公开包",
            "## 5. 独立消费者项目",
            "## 6. 创建网络并挂载 allocator",
            "## 8. 两个受控负例",
            "## 9. 执行完整验证",
            "## 10. 本机执行结果",
            "gpu-allocator-local-package-consumer-terminal.png",
            "RuntimeEnvironment TRT=10.11.0 CUDA=12.9",
            "BuilderCallbacks=8",
            "RejectedCount=1",
            "CallbackFailures=1",
            "不创建 tag、不创建 GitHub Release、不发布新包"
        })
        {
            Assert.Contains(required, article, StringComparison.Ordinal);
        }

        Assert.Contains("gpu-allocator-local-package-consumer-tutorial.md", toc, StringComparison.Ordinal);
        Assert.DoesNotMatch(@"[A-Za-z]:\\", article);
        Assert.DoesNotContain("GitSpace", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("Program Files", article, StringComparison.OrdinalIgnoreCase);
    }

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray()));
    }
}

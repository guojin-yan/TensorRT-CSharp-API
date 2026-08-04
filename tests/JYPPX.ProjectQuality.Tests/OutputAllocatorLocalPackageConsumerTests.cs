using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OutputAllocatorLocalPackageConsumerTests
{
    [Fact]
    public void ConsumerUsesOnlyPackageReferencesAndExercisesPositiveAndRejectedAllocation()
    {
        string template = ReadSource(
            "samples",
            "OutputAllocator.PackageConsumer",
            "OutputAllocator.PackageConsumer.csproj.template");
        string program = ReadSource("samples", "OutputAllocator.PackageConsumer", "Program.cs");

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
            "TensorRtOutputAllocatorCallbackOwner",
            "SetOutputAllocator(\"allocator_output\", positiveOwner)",
            "ClearOutputAllocator(\"allocator_output\")",
            "positiveDetached.LiveAllocationCount == 0UL",
            "request.Kind != TensorRtOutputAllocatorCallbackKind.ReallocateOutput",
            "negativeAttached.AllocationCount == 0UL",
            "negativeAttached.FailureCount > 0UL",
            "OutputAllocatorRealRuntime=Passed",
            "OutputAllocatorPackageConsumer Passed=True"
        })
        {
            Assert.Contains(required, program, StringComparison.Ordinal);
        }

        Assert.DoesNotContain("JYPPX.SampleSupport", program, StringComparison.Ordinal);
    }

    [Fact]
    public void SharedHarnessSelectsOutputScenarioAndEnforcesPackageIsolation()
    {
        string wrapper = ReadSource("eng", "Test-OutputAllocatorLocalPackageConsumer.ps1");
        string harness = ReadSource("eng", "Test-CallbackOwnerLocalPackageConsumer.ps1");

        Assert.Contains("-Scenario OutputAllocator @PSBoundParameters", wrapper, StringComparison.Ordinal);
        foreach (string required in new[]
        {
            "OutputAllocator.PackageConsumer",
            "OutputAllocator.PackageConsumer.csproj.template",
            "--output-allocator-runtime-smoke-only",
            "OutputAllocatorRealRuntime=Passed",
            "<clear />",
            "JYPPX_NATIVE_BRIDGE_PATH",
            "JYPPX_ENABLE_DEVELOPMENT_PROBING",
            "consumerBridgeMatchesPackage",
            "vendorRuntimeBinaryCountInPackages",
            "vendorRuntimeBinaryCountInConsumerOutput",
            "$negativeAllocationCount -ne 0",
            "$liveAllocationCount -ne 0",
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
            "output-allocator-local-package-consumer-tensorrt10.11-evidence.json");
        string evidenceText = File.ReadAllText(evidencePath);
        using JsonDocument evidence = JsonDocument.Parse(evidenceText);
        JsonElement root = evidence.RootElement;

        Assert.Equal("local-package-consumer-tensorrt-output-allocator-runtime", root.GetProperty("evidenceKind").GetString());
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

        JsonElement positive = root.GetProperty("positive");
        Assert.True(positive.GetProperty("passed").GetBoolean());
        Assert.Equal(2UL, positive.GetProperty("invocationCount").GetUInt64());
        Assert.Equal(1UL, positive.GetProperty("notifyShapeCount").GetUInt64());
        Assert.Equal(1UL, positive.GetProperty("reallocateOutputCount").GetUInt64());
        Assert.Equal(1UL, positive.GetProperty("allocationCount").GetUInt64());
        Assert.Equal(1UL, positive.GetProperty("releaseCount").GetUInt64());
        Assert.Equal(0UL, positive.GetProperty("liveAllocationCountAfterDetach").GetUInt64());
        Assert.False(positive.GetProperty("nativePointerExposed").GetBoolean());
        Assert.True(positive.GetProperty("realCallbackRuntime").GetBoolean());

        JsonElement negative = root.GetProperty("rejectionNegative");
        Assert.True(negative.GetProperty("passed").GetBoolean());
        Assert.True(negative.GetProperty("enqueueFailed").GetBoolean());
        Assert.Equal(0UL, negative.GetProperty("allocationCount").GetUInt64());
        Assert.Equal(1UL, negative.GetProperty("failureCount").GetUInt64());
        Assert.Equal(0UL, negative.GetProperty("liveAllocationCountAfterDetach").GetUInt64());

        foreach (JsonProperty artifact in root.GetProperty("artifacts").EnumerateObject())
        {
            string relativePath = artifact.Value.GetProperty("path").GetString()!;
            string expectedHash = artifact.Value.GetProperty("sha256").GetString()!;
            string path = Path.Combine(RepositoryPaths.Root, relativePath.Replace('/', Path.DirectorySeparatorChar));
            Assert.True(File.Exists(path), $"Missing output allocator package-consumer artifact: {relativePath}");
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
            "output-allocator-local-package-consumer-tutorial.md");
        string toc = ReadSource("docs", "toc.yml");

        foreach (string required in new[]
        {
            "## 1. 项目与功能背景",
            "## 2. 依赖与包职责",
            "## 3. 模型获取与转换说明",
            "官方获取方式：不适用",
            "ONNX 转换方式：不适用",
            "可视化结果：不适用",
            "## 5. 仓库外消费者如何隔离",
            "## 7. 正例：接受动态输出分配",
            "## 8. 负例：拒绝重分配",
            "## 9. 执行完整验证",
            "## 10. 本机执行结果",
            "output-allocator-local-package-consumer-terminal.png",
            "RuntimeEnvironment TRT=10.11.0 CUDA=12.9",
            "Callbacks=2 NotifyShape=1 Reallocate=1",
            "Allocations=1 Releases=1 LiveAllocations=0",
            "RejectionCase=Passed EnqueueFailed=True Allocations=0 Failures=1",
            "不创建 tag、不创建 GitHub Release、不发布新包"
        })
        {
            Assert.Contains(required, article, StringComparison.Ordinal);
        }

        Assert.Contains("output-allocator-local-package-consumer-tutorial.md", toc, StringComparison.Ordinal);
        Assert.DoesNotMatch(@"[A-Za-z]:\\", article);
        Assert.DoesNotContain("GitSpace", article, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("Program Files", article, StringComparison.OrdinalIgnoreCase);
    }

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray()));
    }
}

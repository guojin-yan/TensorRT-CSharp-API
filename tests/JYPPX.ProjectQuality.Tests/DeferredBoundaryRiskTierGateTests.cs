using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class DeferredBoundaryRiskTierGateTests
{
    [Fact]
    public void DeferredRiskTierGuideDocumentsReleaseProofBoundaryAndUnsafeGroups()
    {
        string guide = ReadSource("docs", "articles", "zh-cn", "deferred-boundary-risk-tier-gate.md");
        string manualGroups = ReadSource("docs", "articles", "zh-cn", "deferred-manual-design-groups.md");
        string index = ReadSource("docs", "index.md");
        string toc = ReadSource("docs", "toc.yml");

        foreach (string marker in new[]
        {
            "deferred-boundary-risk-tier-gate",
            "manifest-source-match-not-release-proof",
            "no-public-raw-pointer",
            "package-consumer-smoke-required",
            "algorithm-selector-borrowed-pointer",
            "A-tier copied value",
            "B-tier safe alternative",
            "C-tier design-gate-required",
            "D-tier keep-deferred",
            "IAlgorithm::getTimingMSec",
            "IAlgorithmContext::getName",
            "IAlgorithmVariant::getTactic",
            "Plugin V2/V3 callback trampoline",
            "IGpuAllocator::allocate/free/deallocate/reallocate",
            "IOutputAllocator::notifyShape/reallocateOutput",
            "IExecutionContext::execute/executeV2/enqueueV2/INoCopy",
        })
        {
            Assert.Contains(marker, guide, StringComparison.Ordinal);
        }

        Assert.Contains("deferred-boundary-risk-tier-gate.md", manualGroups);
        Assert.Contains("Deferred Boundary Risk Tier Gate", index);
        Assert.Contains("articles/zh-cn/deferred-boundary-risk-tier-gate.md", index);
        Assert.Contains("Deferred Boundary Risk Tier Gate", toc);
        Assert.Contains("articles/zh-cn/deferred-boundary-risk-tier-gate.md", toc);
    }

    [Fact]
    public void RemainingHighRiskDeferredGroupsStayExplicitDeferredUntilSafeHandlesExist()
    {
        string comparison = ReadSource("artifacts", "interface-coverage", "tensorrt-interface-comparison.csv");
        string allManifests = string.Join(
            Environment.NewLine,
            Directory.GetFiles(
                    Path.Combine(RepositoryPaths.Root, "native", "manifests", "tensorrt"),
                    "*.manifest.json",
                    SearchOption.AllDirectories)
                .Select(File.ReadAllText));
        string allHeaders = string.Join(
            Environment.NewLine,
            new[]
            {
                ReadSource("native", "include", "jyppx", "tensorrt", "trt8.h"),
                ReadSource("native", "include", "jyppx", "tensorrt", "trt10.h"),
                ReadSource("native", "include", "jyppx", "tensorrt", "trt11.h"),
            });

        foreach (string marker in new[]
        {
            "\"IAlgorithm\",\"getTimingMSec\",\"IAlgorithm::getTimingMSec\",\"other\",\"deferred-only\"",
            "\"IAlgorithmContext\",\"getName\",\"IAlgorithmContext::getName\",\"other\",\"deferred-only\"",
            "\"IAlgorithmIOInfo\",\"getStrides\",\"IAlgorithmIOInfo::getStrides\",\"other\",\"deferred-only\"",
            "\"IAlgorithmVariant\",\"getTactic\",\"IAlgorithmVariant::getTactic\",\"other\",\"deferred-only\"",
            "\"IPluginV2DynamicExt\",\"enqueue\",\"IPluginV2DynamicExt::enqueue\",\"plugin\",\"deferred-only\"",
            "\"IOutputAllocator\",\"reallocateOutput\",\"IOutputAllocator::reallocateOutput\",\"other\",\"deferred-only\"",
            "\"IGpuAllocator\",\"allocate\",\"IGpuAllocator::allocate\",\"other\",\"deferred-only\"",
            "\"IGpuAsyncAllocator\",\"allocateAsync\",\"IGpuAsyncAllocator::allocateAsync\",\"other\",\"deferred-only\"",
        })
        {
            Assert.Contains(marker, comparison, StringComparison.Ordinal);
        }

        foreach (string manifestId in new[]
        {
            "trt8-algorithm-get-timing-msec-deferred",
            "trt8-algorithm-context-get-name-deferred",
            "trt10-algorithm-get-timing-m-sec-deferred",
            "trt10-algorithm-context-get-name-deferred",
            "trt8-plugin-v2-dynamic-ext-enqueue-deferred",
            "trt10-plugin-v2-dynamic-ext-enqueue-deferred",
            "trt11-plugin-v2-dynamic-ext-enqueue-deferred",
            "trt10-gpu-allocator-allocate-deferred",
            "trt11-gpu-async-allocator-allocate-async-deferred",
            "trt11-output-allocator-reallocate-output-deferred",
        })
        {
            Assert.Contains(manifestId, allManifests, StringComparison.Ordinal);
        }

        Assert.Contains("jyppx_trt8_algorithm_get_timing_msec_deferred", allHeaders);
        Assert.Contains("jyppx_trt10_algorithm_get_timing_m_sec_deferred", allHeaders);
        Assert.DoesNotContain("JYPPX_TensorRtAlgorithm*", allHeaders);
        Assert.DoesNotContain("JYPPX_TensorRtAlgorithmContext*", allHeaders);
        Assert.DoesNotContain("JYPPX_TensorRtAlgorithmIOInfo*", allHeaders);
        Assert.DoesNotContain("JYPPX_TensorRtAlgorithmVariant*", allHeaders);
    }

    [Fact]
    public void CurrentSafeWrappersStayPointerFreeAndDoNotClaimRuntimeProof()
    {
        string allocatorOwner = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "MemoryAllocation", "TensorRtAllocatorCallbackOwner.cs");
        string outputAllocatorOwner = ReadSource("src", "JYPPX.TensorRtSharp", "Callbacks", "MemoryAllocation", "TensorRtOutputAllocatorCallbackOwner.cs");
        string callbackSnapshot = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContextCallbackStateSnapshot.cs");
        string engineSnapshot = ReadSource("src", "JYPPX.TensorRtSharp", "Engine", "TensorRtEngine.Trt11BoundaryControls.cs");
        string contextSnapshot = ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.Trt11BoundaryControls.cs");
        string combined = allocatorOwner + outputAllocatorOwner + callbackSnapshot + engineSnapshot + contextSnapshot;

        Assert.Contains("public TensorRtAllocatorNativeDryRunResult RunNativeDryRunDiagnostic", allocatorOwner);
        Assert.Contains("public TensorRtAllocatorOwnerStateDryRunResult RunNativeStateLedgerDryRunDiagnostic", allocatorOwner);
        Assert.Contains("public TensorRtExecutionContextCallbackStateSnapshot GetCallbackStateSnapshot", ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.Trt11RuntimeDiagnostics.cs"));
        Assert.Contains("public TensorRtExecutionContextCallbackStateSnapshot ClearCallbackState", ReadSource("src", "JYPPX.TensorRtSharp", "Execution", "TensorRtExecutionContext.Trt11RuntimeDiagnostics.cs"));
        Assert.Contains("public bool RealCallbackRuntime => false", outputAllocatorOwner);
        Assert.Contains("public bool IsRealCallbackRuntimeProof => false", outputAllocatorOwner);
        Assert.Contains("public bool DevicePointerExposed", allocatorOwner);
        Assert.Contains("public bool BorrowedPointerEscaped", allocatorOwner);

        Assert.DoesNotContain("public IntPtr", combined);
        Assert.DoesNotContain("public nint", combined);
        Assert.DoesNotContain("public unsafe", combined);
        Assert.DoesNotContain("SafeTensorRtObjectHandle Handle", combined);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return RepositorySourceReader.Read(path);
    }
}

using System.Linq;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class RuntimeAliasAndCuda13SearchUpliftTests
{
    [Fact]
    public void RuntimeAliasesPreferExactImplementedEntriesBeforeGenericHeuristics()
    {
        string script = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");

        Assert.Contains("\"IRuntime::deserializeCudaEngine\" = @(\"runtime-deserialize-engine\")", script);
        Assert.Contains("\"IExecutionContext::enqueueV3\" = @(\"execution-context-enqueue-async\")", script);
        Assert.Contains("\"IRuntime::deserializeCudaEngine\" = @(\"id:*runtime-deserialize-engine\")", script);
        Assert.Contains("\"IExecutionContext::enqueueV3\" = @(\"id:*execution-context-enqueue-async\")", script);

        int priorityStart = script.IndexOf("if ($interfaceKey -in @(", StringComparison.Ordinal);
        int priorityEnd = script.IndexOf(")) {", priorityStart, StringComparison.Ordinal);
        Assert.True(priorityStart >= 0 && priorityEnd > priorityStart);
        string priorityBlock = script.Substring(priorityStart, priorityEnd - priorityStart);
        Assert.Contains("\"IRuntime::deserializeCudaEngine\"", priorityBlock);
        Assert.Contains("\"IExecutionContext::enqueueV3\"", priorityBlock);
    }

    [Fact]
    public void CoverageKeepsCallbackVariantsDeferredWhileDirectPathsAreImplemented()
    {
        string[] rows = File.ReadAllLines(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "interface-coverage",
            "tensorrt-interface-comparison.csv"));

        Assert.Contains(rows, row => row.Contains("\"IRuntime\",\"deserializeCudaEngine\",\"IRuntime::deserializeCudaEngine\",\"runtime-serialization\",\"implemented\"", StringComparison.Ordinal));
        Assert.Contains(rows, row => row.Contains("\"IExecutionContext\",\"enqueueV3\",\"IExecutionContext::enqueueV3\",\"engine-context\",\"implemented\"", StringComparison.Ordinal));
        Assert.Contains(rows, row => row.Contains("\"IRuntime\",\"deserializeCudaEngineV2\",\"IRuntime::deserializeCudaEngineV2\",\"runtime-serialization\",\"deferred-only\"", StringComparison.Ordinal));
        Assert.Contains(rows, row => row.Contains("\"IExecutionContext\",\"enqueueV2\",\"IExecutionContext::enqueueV2\",\"engine-context\",\"deferred-only\"", StringComparison.Ordinal));
    }

    [Fact]
    public void Cuda13RuntimeSearchPrefersNestedX64Directory()
    {
        string consumer = ReadSource("eng", "Test-BridgePackageRuntimeConsumer.ps1");
        string lifecycle = ReadSource("eng", "Invoke-WindowsLifecycleSmoke.ps1");
        string rootCause = ReadSource("eng", "Export-Trt11RuntimeSmokeRootCauseReport.ps1");

        Assert.Contains("Join-Path $Roots.CudaRoot \"bin\\x64\"", consumer);
        Assert.Contains("Join-Path $env:JYPPX_CUDA_ROOT \"bin\\x64\"", lifecycle);
        Assert.True(
            consumer.IndexOf("Join-Path $Roots.CudaRoot \"bin\\x64\"", StringComparison.Ordinal) <
            consumer.IndexOf("Join-Path $Roots.CudaRoot \"bin\"", StringComparison.Ordinal));
        Assert.Contains("CUDA error 35", rootCause);
        Assert.Contains("cuda-driver-insufficient-for-runtime", rootCause);
        Assert.Contains("cudaPreflightDriverInsufficient", rootCause);
    }

    [Fact]
    public void RuntimeDeserializationPublicSurfaceRemainsPointerFree()
    {
        string precheck = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtRuntimeDeserializationBoundaryPrecheck.cs");
        string runtime = ReadSource("src", "JYPPX.TensorRtSharp", "TensorRtRuntime.cs");

        Assert.Contains("DirectDeserializeCudaEngineRowsDeferred => false", precheck);
        Assert.Contains("DirectDeserializeCudaEngineRowsImplemented => SafeDeserializeBridgeReady", precheck);
        Assert.Contains("public TensorRtEngine Deserialize(byte[] serializedEngine)", runtime);
        Assert.Contains("public TensorRtEngine Deserialize(Stream serializedEngineStream)", runtime);
        Assert.DoesNotContain("public IntPtr", precheck + runtime);
        Assert.DoesNotContain("public nint", precheck + runtime);
        Assert.DoesNotContain("public SafeHandle", precheck + runtime);
    }

    private static string ReadSource(params string[] pathParts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
    }
}

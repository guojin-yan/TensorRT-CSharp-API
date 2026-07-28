using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaPrimaryExecutionContextOwnerSafeUpliftTests
{
    private static readonly string[] OfficialFunctions =
    {
        "cudaDeviceGetExecutionCtx",
        "cudaExecutionCtxGetDevice",
        "cudaExecutionCtxGetId",
        "cudaExecutionCtxSynchronize",
        "cudaExecutionCtxStreamCreate",
        "cudaExecutionCtxRecordEvent",
        "cudaExecutionCtxWaitEvent",
    };

    [Fact]
    public void ManifestNativeAndGeneratorDefineNonDestroyingPrimaryOwner()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-fifty-second-batch-primary-execution-context.manifest.json");
        string native = ReadSource("native", "src", "cuda", "modules", "deployment", "primary_execution_context.inc");
        string objectHeader = ReadSource("native", "src", "cuda", "object.hpp");
        string generator = ReadSource("tools", "JYPPX.BindingGenerator", "Program.cs");

        foreach (string id in new[]
        {
            "cuda-device-get-primary-execution-context-owner-safe",
            "cuda-primary-execution-context-wrapper-release-safe",
            "cuda-execution-context-get-device-copied-scalar-safe",
            "cuda-execution-context-get-id-copied-scalar-safe",
            "cuda-execution-context-synchronize-owner-safe",
            "cuda-execution-context-create-stream-owner-safe",
            "cuda-execution-context-record-event-owner-safe",
            "cuda-execution-context-wait-event-owner-safe",
        })
        {
            Assert.Contains(id, manifest, StringComparison.Ordinal);
        }

        Assert.Contains("CUDART_VERSION >= 13000", native, StringComparison.Ordinal);
        Assert.Contains("cudaDeviceGetExecutionCtx", native, StringComparison.Ordinal);
        Assert.Contains("cudaExecutionCtxGetDevice", native, StringComparison.Ordinal);
        Assert.Contains("cudaExecutionCtxGetId", native, StringComparison.Ordinal);
        Assert.Contains("cudaExecutionCtxSynchronize", native, StringComparison.Ordinal);
        Assert.Contains("cudaExecutionCtxStreamCreate", native, StringComparison.Ordinal);
        Assert.Contains("cudaExecutionCtxRecordEvent", native, StringComparison.Ordinal);
        Assert.Contains("cudaExecutionCtxWaitEvent", native, StringComparison.Ordinal);
        Assert.DoesNotContain("cudaExecutionCtxDestroy(", native, StringComparison.Ordinal);
        Assert.Contains("context->is_primary = true", native, StringComparison.Ordinal);
        Assert.Contains("context->base.magic = 0", native, StringComparison.Ordinal);
        Assert.Contains("native-exception-caught", native, StringComparison.Ordinal);
        Assert.Contains("windows-seh-caught", native, StringComparison.Ordinal);
        Assert.Contains("ExecutionContext = 13", objectHeader, StringComparison.Ordinal);
        Assert.Contains("JYPPX_CudaExecutionContext**\" => moduleSpecific ? \"out SafeCudaExecutionContextHandle", generator, StringComparison.Ordinal);
        Assert.Contains("JYPPX_CudaExecutionContext*\" => moduleSpecific ? \"SafeCudaExecutionContextHandle", generator, StringComparison.Ordinal);
    }

    [Fact]
    public void ManagedSurfaceIsOwnerSafeAndPointerFree()
    {
        string context = ReadSource("src", "JYPPX.CudaSharp", "Devices", "CudaPrimaryExecutionContext.cs");
        string device = ReadSource("src", "JYPPX.CudaSharp", "Devices", "CudaDevice.cs");
        string handle = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Handles", "SafeCudaExecutionContextHandle.cs");
        string interop = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "NativeCudaApi.ExecutionContext.cs");
        string publicSurface = context + device;

        Assert.Contains("public sealed class CudaPrimaryExecutionContext : IDisposable", context, StringComparison.Ordinal);
        Assert.Contains("public bool IsPrimary => true", context, StringComparison.Ordinal);
        Assert.Contains("public int DeviceOrdinal", context, StringComparison.Ordinal);
        Assert.Contains("public ulong Id", context, StringComparison.Ordinal);
        Assert.Contains("public CudaStream CreateStream", context, StringComparison.Ordinal);
        Assert.Contains("public void RecordEvent(CudaEvent cudaEvent)", context, StringComparison.Ordinal);
        Assert.Contains("public void WaitEvent(CudaEvent cudaEvent)", context, StringComparison.Ordinal);
        Assert.Contains("public void Synchronize()", context, StringComparison.Ordinal);
        Assert.Contains("GetPrimaryExecutionContext(int ordinal)", device, StringComparison.Ordinal);
        Assert.Contains("jyppx_cuda_primary_execution_context_release_wrapper_safe(handle)", handle, StringComparison.Ordinal);
        Assert.Contains("SafeCudaExecutionContextHandle", interop, StringComparison.Ordinal);

        Assert.DoesNotContain("public IntPtr", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public SafeHandle", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public UIntPtr", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("cudaExecutionContext_t", publicSurface, StringComparison.Ordinal);
    }

    [Fact]
    public void CoveragePrioritizesSevenRealAliasesAndRetainsDeferredHistory()
    {
        string coverage = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        string deviceDeferred = ReadSource("native", "manifests", "cuda", "cuda-thirty-fifth-batch-stream-device-boundaries.manifest.json");
        string otherDeferred = ReadSource("native", "manifests", "cuda", "cuda-thirty-eighth-batch-other-boundaries.manifest.json");

        foreach ((string function, string realId, string deferredId) in new[]
        {
            ("cudaDeviceGetExecutionCtx", "cuda-device-get-primary-execution-context-owner-safe", "cuda-device-get-execution-ctx-deferred"),
            ("cudaExecutionCtxGetDevice", "cuda-execution-context-get-device-copied-scalar-safe", "cuda-execution-ctx-get-device-deferred"),
            ("cudaExecutionCtxGetId", "cuda-execution-context-get-id-copied-scalar-safe", "cuda-execution-ctx-get-id-deferred"),
            ("cudaExecutionCtxSynchronize", "cuda-execution-context-synchronize-owner-safe", "cuda-execution-ctx-synchronize-deferred"),
            ("cudaExecutionCtxStreamCreate", "cuda-execution-context-create-stream-owner-safe", "cuda-execution-ctx-stream-create-deferred"),
            ("cudaExecutionCtxRecordEvent", "cuda-execution-context-record-event-owner-safe", "cuda-execution-ctx-record-event-deferred"),
            ("cudaExecutionCtxWaitEvent", "cuda-execution-context-wait-event-owner-safe", "cuda-execution-ctx-wait-event-deferred"),
        })
        {
            Assert.Contains($"\"{function}\" = @(\"id:{realId}\")", coverage, StringComparison.Ordinal);
            Assert.Contains($"\"{function}\" = @(\"id:{deferredId}\")", coverage, StringComparison.Ordinal);
            Assert.Contains(deferredId, deviceDeferred + otherDeferred, StringComparison.Ordinal);
        }

        Assert.Contains("\"cudaExecutionCtxDestroy\" = @(\"execution-ctx-destroy-deferred\")", coverage, StringComparison.Ordinal);
        Assert.DoesNotContain("cuda-execution-context-destroy", coverage, StringComparison.Ordinal);
    }

    [Fact]
    public void CoverageRowsAreImplementedWithDeferredHistoryOnCuda132()
    {
        string comparison = ReadSource("artifacts", "interface-coverage", "cuda-runtime-interface-comparison.csv");
        string[] rows = comparison.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (string function in OfficialFunctions)
        {
            Assert.Contains(rows, row =>
                row.Contains("\"13.2\"", StringComparison.Ordinal) &&
                row.Contains($"\"{function}\"", StringComparison.Ordinal) &&
                row.Contains("\"implemented-with-deferred-history\"", StringComparison.Ordinal));
        }
    }

    [Fact]
    public void SmokeConsumerAndCandidateReviewPreserveProofBoundary()
    {
        string smoke = ReadSource("smoke", "CudaSmokeRunner", "Program.cs");
        string consumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");
        string review = ReadSource("artifacts", "interface-coverage", "cuda-primary-execution-context-owner-candidate-review.md");

        Assert.Contains("CudaPrimaryExecutionContext {ProbeCudaPrimaryExecutionContext", smoke, StringComparison.Ordinal);
        Assert.Contains("VersionGuard=NotSupported", smoke, StringComparison.Ordinal);
        Assert.Contains("context.RecordEvent(cudaEvent)", smoke, StringComparison.Ordinal);
        Assert.Contains("context.WaitEvent(cudaEvent)", smoke, StringComparison.Ordinal);
        Assert.Contains("context.Synchronize()", smoke, StringComparison.Ordinal);
        Assert.Contains("nameof(CudaPrimaryExecutionContext)", consumer, StringComparison.Ordinal);
        Assert.Contains("nameof(CudaPrimaryExecutionContext.CreateStream)", consumer, StringComparison.Ordinal);
        Assert.Contains("`cudaExecutionCtxDestroy` | 对 `cudaDeviceGetExecutionCtx` 返回的主上下文调用属于未定义行为", review, StringComparison.Ordinal);
        Assert.Contains("`cudaGraphNodeGetParams` | tagged union 中包含 driver-owned pointers", review, StringComparison.Ordinal);
        Assert.Contains("旧 deferred manifest 全部保留", review, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] pathParts) =>
        File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
}

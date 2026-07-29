using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaIpcExportTokenUpliftTests
{
    [Fact]
    public void ManifestAndNativeImplementExportOnlyCopiedTokens()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-sixtieth-batch-ipc-export-tokens.manifest.json");
        string native = ReadSource("native", "src", "cuda", "modules", "deployment", "ipc_export_tokens.inc");
        string objects = ReadSource("native", "src", "cuda", "object.hpp");
        string allocations = ReadSource("native", "src", "cuda", "api.cpp") +
            ReadSource("native", "src", "cuda", "modules", "memory", "async_memory_pool.inc");

        Assert.Contains("cuda-ipc-get-event-handle-copied-export-token-safe", manifest, StringComparison.Ordinal);
        Assert.Contains("cuda-ipc-get-mem-handle-copied-export-token-safe", manifest, StringComparison.Ordinal);
        Assert.Contains("cudaIpcGetEventHandle", native, StringComparison.Ordinal);
        Assert.Contains("cudaIpcGetMemHandle", native, StringComparison.Ordinal);
        Assert.Contains("static_assert(sizeof(cudaIpcEventHandle_t) == kCudaIpcExportTokenSize", native, StringComparison.Ordinal);
        Assert.Contains("static_assert(sizeof(cudaIpcMemHandle_t) == kCudaIpcExportTokenSize", native, StringComparison.Ordinal);
        Assert.Contains("std::memcpy(output_buffer, &token, sizeof(token))", native, StringComparison.Ordinal);
        Assert.DoesNotContain("cudaIpcOpenEventHandle", native, StringComparison.Ordinal);
        Assert.DoesNotContain("cudaIpcOpenMemHandle", native, StringComparison.Ordinal);
        Assert.DoesNotContain("cudaIpcCloseMemHandle", native, StringComparison.Ordinal);
        Assert.Contains("bool is_ipc_exportable", objects, StringComparison.Ordinal);
        Assert.Contains("memory->is_ipc_exportable = true", allocations, StringComparison.Ordinal);
        Assert.Contains("memory->is_ipc_exportable = false", allocations, StringComparison.Ordinal);
        Assert.Contains("if (!memory_object->is_ipc_exportable)", native, StringComparison.Ordinal);
        Assert.Contains("synchronous cudaMalloc allocation", native, StringComparison.Ordinal);
    }

    [Fact]
    public void ManagedSurfaceIsImmutableAndPointerFree()
    {
        string token = ReadSource("src", "JYPPX.CudaSharp", "IPC", "CudaIpcExportToken.cs");
        string cudaEvent = ReadSource("src", "JYPPX.CudaSharp", "Events", "CudaEvent.cs");
        string memory = ReadSource(
            "src", "JYPPX.CudaSharp", "Memory", "CudaMemory.Ipc.cs");
        string interop = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "IPC", "NativeCudaApi.IpcExports.cs");
        string publicSurface = token + cudaEvent + memory;

        Assert.Contains("public sealed class CudaIpcExportToken", token, StringComparison.Ordinal);
        Assert.Contains("private readonly byte[] _bytes", token, StringComparison.Ordinal);
        Assert.Contains("public byte[] ToArray() => (byte[])_bytes.Clone()", token, StringComparison.Ordinal);
        Assert.Contains("public CudaIpcExportToken ExportIpcToken()", cudaEvent, StringComparison.Ordinal);
        Assert.Contains("public CudaIpcExportToken ExportIpcToken()", memory, StringComparison.Ordinal);
        Assert.Contains("Array.Empty<byte>()", interop, StringComparison.Ordinal);
        Assert.Contains("ValidateCopiedIpcTokenSize", interop, StringComparison.Ordinal);

        Assert.DoesNotContain("public IntPtr", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public UIntPtr", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public SafeHandle", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("DevicePointer", token, StringComparison.Ordinal);
    }

    [Fact]
    public void CoverageUsesRealAliasesAndRetainsImportOwnershipBoundaries()
    {
        string coverage = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        string deferred = ReadSource("native", "manifests", "cuda", "cuda-thirty-eighth-batch-other-boundaries.manifest.json");
        string audit = ReadSource("artifacts", "interface-coverage", "cuda-ipc-export-token-candidate-audit.md");

        Assert.Contains("\"cudaIpcGetEventHandle\" = @(\"id:cuda-ipc-get-event-handle-copied-export-token-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaIpcGetMemHandle\" = @(\"id:cuda-ipc-get-mem-handle-copied-export-token-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaIpcGetEventHandle\" = @(\"id:cuda-ipc-get-event-handle-deferred\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaIpcGetMemHandle\" = @(\"id:cuda-ipc-get-mem-handle-deferred\")", coverage, StringComparison.Ordinal);
        Assert.Contains("cuda-ipc-open-event-handle-deferred", deferred, StringComparison.Ordinal);
        Assert.Contains("cuda-ipc-open-mem-handle-deferred", deferred, StringComparison.Ordinal);
        Assert.Contains("cuda-ipc-close-mem-handle-deferred", deferred, StringComparison.Ordinal);
        Assert.Contains("source event or memory allocation alive", audit, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Historical deferred records remain present", audit, StringComparison.Ordinal);
    }

    [Fact]
    public void SmokeExercisesTokensWithoutOpeningImportedResources()
    {
        string smoke = ReadSource("smoke", "CudaSmokeRunner", "Program.cs");

        Assert.Contains("CudaIpcExportTokens {ProbeCudaIpcExportTokens()}", smoke, StringComparison.Ordinal);
        Assert.Contains("CudaEventCreationFlags.Interprocess | CudaEventCreationFlags.DisableTiming", smoke, StringComparison.Ordinal);
        Assert.Contains("eventToken.Length != 64", smoke, StringComparison.Ordinal);
        Assert.Contains("memoryToken.Length != 64", smoke, StringComparison.Ordinal);
        Assert.Contains("DefaultEventRejected=True", smoke, StringComparison.Ordinal);
        Assert.Contains("ManagedMemoryRejected=True", smoke, StringComparison.Ordinal);
        Assert.DoesNotContain("IpcOpen", smoke, StringComparison.Ordinal);
        Assert.DoesNotContain("IpcClose", smoke, StringComparison.Ordinal);
    }

    [Fact]
    public void DocumentationStatesOwnerLifetimeAndImportBoundary()
    {
        string english = ReadSource("docs", "articles", "en", "cuda-ipc-export-token.md");
        string chinese = ReadSource("docs", "articles", "zh-cn", "cuda-ipc-export-token.md");
        string toc = ReadSource("docs", "toc.yml");

        Assert.Contains("Interprocess | CudaEventCreationFlags.DisableTiming", english, StringComparison.Ordinal);
        Assert.Contains("source owner alive", english, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("cudaIpcOpenEventHandle", english, StringComparison.Ordinal);
        Assert.Contains("cudaIpcOpenMemHandle", chinese, StringComparison.Ordinal);
        Assert.Contains("不打印 token", chinese, StringComparison.Ordinal);
        Assert.Contains("articles/en/cuda-ipc-export-token.md", toc, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/cuda-ipc-export-token.md", toc, StringComparison.Ordinal);
    }

    [Fact]
    public void PackageConsumerCompilesThePublicTokenSurface()
    {
        string consumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");

        Assert.Contains("Func<CudaEvent, CudaIpcExportToken> exportEventIpcToken", consumer, StringComparison.Ordinal);
        Assert.Contains("Func<CudaMemory, CudaIpcExportToken> exportMemoryIpcToken", consumer, StringComparison.Ordinal);
        Assert.Contains("nameof(CudaEvent.TryExportIpcToken)", consumer, StringComparison.Ordinal);
        Assert.Contains("nameof(CudaMemory.TryExportIpcToken)", consumer, StringComparison.Ordinal);
        Assert.Contains("nameof(CudaIpcExportToken.ToArray)", consumer, StringComparison.Ordinal);
        Assert.Contains("nameof(CudaIpcExportTokenKind.Memory)", consumer, StringComparison.Ordinal);
    }

    [Fact]
    public void LocalRuntimeEvidenceCannotClaimCrossProcessOrPackageProof()
    {
        string evidence = ReadSource(
            "artifacts",
            "interface-coverage",
            "cuda-ipc-export-token-local-runtime-evidence.json");

        Assert.Contains("\"isLocalRuntimeEvidence\": true", evidence, StringComparison.Ordinal);
        Assert.Contains("\"isCrossProcessRuntimeProof\": false", evidence, StringComparison.Ordinal);
        Assert.Contains("\"isPackageConsumerRuntimeProof\": false", evidence, StringComparison.Ordinal);
        Assert.Contains("\"canPromoteRuntimeProof\": false", evidence, StringComparison.Ordinal);
        Assert.Contains("\"canPublishPublicly\": false", evidence, StringComparison.Ordinal);
        Assert.Contains("\"tokenContentsRecorded\": false", evidence, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] pathParts) =>
        File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
}

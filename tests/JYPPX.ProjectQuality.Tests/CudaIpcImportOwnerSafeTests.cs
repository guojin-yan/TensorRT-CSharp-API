using JYPPX.CudaSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaIpcImportOwnerSafeTests
{
    [Fact]
    public void TransportTokensRequireExactlySixtyFourBytesAndCopyInput()
    {
        Assert.Throws<ArgumentException>(() =>
            CudaIpcExportToken.FromBytes(CudaIpcExportTokenKind.Event, new byte[63]));
        Assert.Throws<ArgumentException>(() =>
            CudaIpcExportToken.FromBytes(CudaIpcExportTokenKind.Memory, new byte[65]));

        byte[] transported = new byte[64];
        transported[0] = 0x2A;
        CudaIpcExportToken token = CudaIpcExportToken.FromBytes(CudaIpcExportTokenKind.Event, transported);
        transported[0] = 0;

        Assert.Equal(64, token.Length);
        Assert.Equal(0x2A, token.ToArray()[0]);
    }

    [Fact]
    public void ManifestAndNativeUseDedicatedOpenAndCloseOwners()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-sixty-second-batch-ipc-import-owner-safe.manifest.json");
        string native = ReadSource("native", "src", "cuda", "modules", "deployment", "ipc_import_owner.inc");
        string objects = ReadSource("native", "src", "cuda", "object.hpp");
        string api = ReadSource("native", "src", "cuda", "api.cpp");
        string asyncMemory = ReadSource("native", "src", "cuda", "modules", "memory", "async_memory_pool.inc");

        Assert.Contains("cuda-ipc-open-event-handle-owner-safe", manifest, StringComparison.Ordinal);
        Assert.Contains("cuda-ipc-open-mem-handle-owner-safe", manifest, StringComparison.Ordinal);
        Assert.Contains("cuda-ipc-close-mem-handle-owner-safe", manifest, StringComparison.Ordinal);
        Assert.Contains("cudaIpcOpenEventHandle", native, StringComparison.Ordinal);
        Assert.Contains("cudaIpcOpenMemHandle", native, StringComparison.Ordinal);
        Assert.Contains("cudaIpcCloseMemHandle", native, StringComparison.Ordinal);
        Assert.Contains("cudaIpcMemLazyEnablePeerAccess", native, StringComparison.Ordinal);
        Assert.Contains("MemoryReleaseMode::IpcClose", native, StringComparison.Ordinal);
        Assert.Contains("enum class MemoryReleaseMode", objects, StringComparison.Ordinal);
        Assert.Contains("Imported CUDA IPC memory must be released with cudaIpcCloseMemHandle", api, StringComparison.Ordinal);
        Assert.Contains("Imported CUDA IPC memory cannot be released asynchronously", asyncMemory, StringComparison.Ordinal);
    }

    [Fact]
    public void ManagedImportSurfaceIsPointerFreeAndCarriesExactSize()
    {
        string token = ReadSource("src", "JYPPX.CudaSharp", "CudaIpcExportToken.cs");
        string memory = ReadSource("src", "JYPPX.CudaSharp", "CudaMemory.cs");
        string cudaEvent = ReadSource("src", "JYPPX.CudaSharp", "CudaEvent.cs");
        string interop = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "NativeCudaApi.IpcImports.cs");
        string safeHandle = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Handles", "SafeCudaMemoryHandle.cs");
        string publicSurface = token + memory + cudaEvent;

        Assert.Contains("public sealed class CudaIpcMemoryExportDescriptor", token, StringComparison.Ordinal);
        Assert.Contains("public int SizeInBytes", token, StringComparison.Ordinal);
        Assert.Contains("public static CudaIpcExportToken FromBytes", token, StringComparison.Ordinal);
        Assert.Contains("public CudaIpcMemoryExportDescriptor ExportIpcDescriptor()", memory, StringComparison.Ordinal);
        Assert.Contains("public static CudaMemory ImportIpcDescriptor", memory, StringComparison.Ordinal);
        Assert.Contains("public static CudaEvent ImportIpcToken", cudaEvent, StringComparison.Ordinal);
        Assert.Contains("public bool IsIpcImported", memory, StringComparison.Ordinal);
        Assert.Contains("public bool IsIpcImported", cudaEvent, StringComparison.Ordinal);
        Assert.Contains("memoryHandle.MarkIpcImported()", interop, StringComparison.Ordinal);
        Assert.Contains("jyppx_cuda_ipc_close_imported_memory_safe(handle)", safeHandle, StringComparison.Ordinal);
        Assert.DoesNotContain("public IntPtr", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public UIntPtr", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public SafeHandle", publicSurface, StringComparison.Ordinal);
    }

    [Fact]
    public void CoveragePromotesRealAliasesAndRetainsDeferredHistory()
    {
        string coverage = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        string deferred = ReadSource("native", "manifests", "cuda", "cuda-thirty-eighth-batch-other-boundaries.manifest.json");

        Assert.Contains("\"cudaIpcOpenEventHandle\" = @(\"id:cuda-ipc-open-event-handle-owner-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaIpcOpenMemHandle\" = @(\"id:cuda-ipc-open-mem-handle-owner-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaIpcCloseMemHandle\" = @(\"id:cuda-ipc-close-mem-handle-owner-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaIpcOpenEventHandle\" = @(\"id:cuda-ipc-open-event-handle-deferred\")", coverage, StringComparison.Ordinal);
        Assert.Contains("cuda-ipc-open-event-handle-deferred", deferred, StringComparison.Ordinal);
        Assert.Contains("cuda-ipc-open-mem-handle-deferred", deferred, StringComparison.Ordinal);
        Assert.Contains("cuda-ipc-close-mem-handle-deferred", deferred, StringComparison.Ordinal);
    }

    [Fact]
    public void SmokeUsesARealChildProcessWithoutPrintingTokens()
    {
        string project = ReadSource("smoke", "CudaIpcImportSmokeRunner", "CudaIpcImportSmokeRunner.csproj");
        string program = ReadSource("smoke", "CudaIpcImportSmokeRunner", "Program.cs");
        string solution = ReadSource("TensorRtSharp.sln");

        Assert.Contains("JYPPX.CudaSharp.csproj", project, StringComparison.Ordinal);
        Assert.Contains("Process.Start(startInfo)", program, StringComparison.Ordinal);
        Assert.Contains("--child", program, StringComparison.Ordinal);
        Assert.Contains("CudaEvent.ImportIpcToken", program, StringComparison.Ordinal);
        Assert.Contains("CudaMemory.ImportIpcDescriptor", program, StringComparison.Ordinal);
        Assert.Contains("importedEvent.Synchronize()", program, StringComparison.Ordinal);
        Assert.Contains("importedMemory.Fill(ImporterValue)", program, StringComparison.Ordinal);
        Assert.Contains("CrossProcess=True", program, StringComparison.Ordinal);
        Assert.Contains("TokenContentsPrinted=False", program, StringComparison.Ordinal);
        Assert.Contains("RedirectStandardInput = true", program, StringComparison.Ordinal);
        Assert.Contains("sourceMemory.ExportIpcDescriptor()", program, StringComparison.Ordinal);
        Assert.True(
            program.IndexOf("sourceMemory.ExportIpcDescriptor()", StringComparison.Ordinal) <
            program.IndexOf("sourceMemory.FillAsync", StringComparison.Ordinal));
        Assert.DoesNotContain("ToHexString", program, StringComparison.Ordinal);
        Assert.DoesNotContain("--event-token", program, StringComparison.Ordinal);
        Assert.DoesNotContain("--memory-token", program, StringComparison.Ordinal);
        Assert.Contains("CudaIpcImportSmokeRunner.csproj", solution, StringComparison.Ordinal);
    }

    [Fact]
    public void PackageConsumerCompilesImportAndDescriptorSurface()
    {
        string consumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");

        Assert.Contains("Func<CudaMemory, CudaIpcMemoryExportDescriptor> exportIpcMemoryDescriptor", consumer, StringComparison.Ordinal);
        Assert.Contains("CudaIpcExportToken.FromBytes", consumer, StringComparison.Ordinal);
        Assert.Contains("CudaEvent.ImportIpcToken", consumer, StringComparison.Ordinal);
        Assert.Contains("CudaMemory.ImportIpcDescriptor", consumer, StringComparison.Ordinal);
        Assert.Contains("nameof(CudaMemory.IsIpcImported)", consumer, StringComparison.Ordinal);
    }

    [Fact]
    public void CandidateAuditKeepsVendorAndPublicationBoundariesExplicit()
    {
        string json = ReadSource("artifacts", "interface-coverage", "cuda-ipc-import-owner-safe-candidate-audit.json");
        string markdown = ReadSource("artifacts", "interface-coverage", "cuda-ipc-import-owner-safe-candidate-audit.md");

        Assert.Contains("\"allSelectedSymbolsPresentInHeadersAndImportLibraries\": true", json, StringComparison.Ordinal);
        Assert.Contains("\"allInstalledRuntimeDllExportsPresent\": true", json, StringComparison.Ordinal);
        Assert.Contains("\"historicalDeferredRecordsRetained\": true", json, StringComparison.Ordinal);
        Assert.Contains("\"isPackageConsumerRuntimeProof\": false", json, StringComparison.Ordinal);
        Assert.Contains("\"canPublishPublicly\": false", json, StringComparison.Ordinal);
        Assert.Contains("passed-real-cross-process-smoke", json, StringComparison.Ordinal);
        Assert.Contains("cuda-ipc-import-cross-process-runtime-evidence.json", json, StringComparison.Ordinal);
        Assert.Contains("13.2", markdown, StringComparison.Ordinal);
        Assert.Contains("DLL not separately installed", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void RuntimeEvidenceProvesCrossProcessOnlyAfterTheRealRunnerPasses()
    {
        string evidence = ReadSource(
            "artifacts",
            "interface-coverage",
            "cuda-ipc-import-cross-process-runtime-evidence.json");

        Assert.Contains("\"isCrossProcessRuntimeProof\": true", evidence, StringComparison.Ordinal);
        Assert.Contains("\"eventImported\": true", evidence, StringComparison.Ordinal);
        Assert.Contains("\"memoryImported\": true", evidence, StringComparison.Ordinal);
        Assert.Contains("\"importedCloseSucceeded\": true", evidence, StringComparison.Ordinal);
        Assert.Contains("\"sourceOwnerAlive\": true", evidence, StringComparison.Ordinal);
        Assert.Contains("\"tokenContentsRecorded\": false", evidence, StringComparison.Ordinal);
        Assert.Contains("\"isPackageConsumerRuntimeProof\": false", evidence, StringComparison.Ordinal);
        Assert.Contains("\"canPublishPublicly\": false", evidence, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] pathParts) =>
        File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
}

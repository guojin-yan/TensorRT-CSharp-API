using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaGraphMemoryAllocationOwnerSafeUpliftTests
{
    [Fact]
    public void ManifestAndNativeImplementGraphOwnedAllocationAndFreeNodes()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-sixty-first-batch-graph-memory-allocation-owner-safe.manifest.json");
        string native = ReadSource("native", "src", "cuda", "modules", "graph", "memory_allocation_owner.inc");
        string graphLifecycle = ReadSource("native", "src", "cuda", "modules", "graph", "stream_capture_graph.inc");
        string diagnostics = ReadSource("native", "src", "cuda", "modules", "graph", "owner_scoped_diagnostics.inc");

        Assert.Contains("cuda-graph-add-mem-alloc-node-owner-safe", manifest, StringComparison.Ordinal);
        Assert.Contains("cuda-graph-add-mem-free-node-owner-safe", manifest, StringComparison.Ordinal);
        Assert.Contains("CUDART_VERSION >= 11040", manifest, StringComparison.Ordinal);
        Assert.Contains("cudaGraphAddMemAllocNode", native, StringComparison.Ordinal);
        Assert.Contains("cudaGraphAddMemFreeNode", native, StringComparison.Ordinal);
        Assert.Contains("params.dptr", native, StringComparison.Ordinal);
        Assert.Contains("active_memory_allocations", native, StringComparison.Ordinal);
        Assert.Contains("active_memory_allocations != 0", graphLifecycle, StringComparison.Ordinal);
        Assert.Contains("graph-memory-allocation-owner-active", graphLifecycle, StringComparison.Ordinal);
        Assert.Contains("conditional-owner-active", graphLifecycle, StringComparison.Ordinal);
        Assert.Contains("cudaGraphNodeTypeMemAlloc", diagnostics, StringComparison.Ordinal);
        Assert.Contains("cudaGraphNodeTypeMemFree", diagnostics, StringComparison.Ordinal);
    }

    [Fact]
    public void ManagedSurfaceIsTypedGraphBoundAndPointerFree()
    {
        string allocation = ReadSource("src", "JYPPX.CudaSharp", "CudaGraphMemoryAllocation.cs");
        string graph = ReadSource("src", "JYPPX.CudaSharp", "CudaGraph.cs");
        string publicSurface = allocation + graph;

        Assert.Contains("public sealed class CudaGraphMemoryAllocation : IDisposable", allocation, StringComparison.Ordinal);
        Assert.Contains("private readonly CudaGraph _owner", allocation, StringComparison.Ordinal);
        Assert.Contains("ReferenceEquals(owner, _owner)", allocation, StringComparison.Ordinal);
        Assert.Contains("public CudaGraphMemoryAllocation AddMemoryAllocationNode", graph, StringComparison.Ordinal);
        Assert.Contains("public CudaGraphNode AddMemoryFreeNode", graph, StringComparison.Ordinal);
        Assert.Contains("public CudaGraphNode AddDeviceToHostMemcpyNode", graph, StringComparison.Ordinal);
        Assert.Contains("_activeMemoryAllocationOwners", graph, StringComparison.Ordinal);

        Assert.DoesNotContain("public IntPtr", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public UIntPtr", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("public SafeHandle", publicSurface, StringComparison.Ordinal);
        Assert.DoesNotContain("DevicePointer", allocation, StringComparison.Ordinal);
    }

    [Fact]
    public void OwnerGuardsRejectCrossGraphSecondFreeAndEarlyGraphDispose()
    {
        string allocation = ReadSource("src", "JYPPX.CudaSharp", "CudaGraphMemoryAllocation.cs");
        string graph = ReadSource("src", "JYPPX.CudaSharp", "CudaGraph.cs");
        string native = ReadSource("native", "src", "cuda", "modules", "graph", "memory_allocation_owner.inc");

        Assert.Contains("belongs to a different CUDA graph", allocation, StringComparison.Ordinal);
        Assert.Contains("A memory-free node has already been added", allocation, StringComparison.Ordinal);
        Assert.Contains("cannot be disposed while a graph memory-allocation wrapper is active", graph, StringComparison.Ordinal);
        Assert.Contains("allocation-owner-mismatch", native, StringComparison.Ordinal);
        Assert.Contains("allocation-already-freed", native, StringComparison.Ordinal);
        Assert.Contains("allocation_object->free_node_added = true", native, StringComparison.Ordinal);
    }

    [Fact]
    public void CoverageUsesRealAliasesAndRetainsDeferredHistory()
    {
        string coverage = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");
        string deferred = ReadSource("native", "manifests", "cuda", "cuda-twenty-third-batch-deferred-coverage.manifest.json");

        Assert.Contains("\"cudaGraphAddMemAllocNode\" = @(\"id:cuda-graph-add-mem-alloc-node-owner-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaGraphAddMemFreeNode\" = @(\"id:cuda-graph-add-mem-free-node-owner-safe\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaGraphAddMemAllocNode\" = @(\"id:cuda-cuda-graph-add-mem-alloc-node-deferred\")", coverage, StringComparison.Ordinal);
        Assert.Contains("\"cudaGraphAddMemFreeNode\" = @(\"id:cuda-cuda-graph-add-mem-free-node-deferred\")", coverage, StringComparison.Ordinal);
        Assert.Contains("cuda-cuda-graph-add-mem-alloc-node-deferred", deferred, StringComparison.Ordinal);
        Assert.Contains("cuda-cuda-graph-add-mem-free-node-deferred", deferred, StringComparison.Ordinal);
    }

    [Fact]
    public void SmokeExecutesPointerFreeAllocationRoundTripAndNegativePaths()
    {
        string smoke = ReadSource("smoke", "CudaGraphSmokeRunner", "Program.cs");

        Assert.Contains("ProbeGraphMemoryAllocation(stream, ByteCount)", smoke, StringComparison.Ordinal);
        Assert.Contains("graph.AddMemoryAllocationNode(byteCount, CudaDevice.Current)", smoke, StringComparison.Ordinal);
        Assert.Contains("graph.AddMemsetNode(allocation, 0x6B, byteCount)", smoke, StringComparison.Ordinal);
        Assert.Contains("graph.AddDeviceToHostMemcpyNodeAfter(memset, destination, allocation, byteCount)", smoke, StringComparison.Ordinal);
        Assert.Contains("graph.AddMemoryFreeNode(allocation, copy)", smoke, StringComparison.Ordinal);
        Assert.Contains("GraphDisposeRejected={graphDisposeRejected}", smoke, StringComparison.Ordinal);
        Assert.Contains("CrossGraphRejected={crossGraphRejected}", smoke, StringComparison.Ordinal);
        Assert.Contains("SecondFreeRejected={secondFreeRejected}", smoke, StringComparison.Ordinal);
    }

    [Fact]
    public void PackageConsumerCompilesTheTypedSurface()
    {
        string consumer = ReadSource("eng", "Test-BridgePackageConsumer.ps1");

        Assert.Contains("Func<CudaGraph, CudaGraphMemoryAllocation> addGraphMemoryAllocationNode", consumer, StringComparison.Ordinal);
        Assert.Contains("Func<CudaGraph, CudaGraphMemoryAllocation, CudaGraphNode> addGraphMemoryFreeNode", consumer, StringComparison.Ordinal);
        Assert.Contains("nameof(CudaGraph.AddMemoryAllocationNode)", consumer, StringComparison.Ordinal);
        Assert.Contains("nameof(CudaGraph.AddMemoryFreeNode)", consumer, StringComparison.Ordinal);
        Assert.Contains("nameof(CudaGraphMemoryAllocation.IsFreeNodeAdded)", consumer, StringComparison.Ordinal);
    }

    [Fact]
    public void EvidenceBoundaryCannotClaimPackageOrPublicProof()
    {
        string evidence = ReadSource("artifacts", "interface-coverage", "cuda-graph-memory-allocation-local-runtime-evidence.json");

        Assert.Contains("\"isLocalRuntimeEvidence\": true", evidence, StringComparison.Ordinal);
        Assert.Contains("\"isPackageConsumerRuntimeProof\": false", evidence, StringComparison.Ordinal);
        Assert.Contains("\"canPromoteRuntimeProof\": false", evidence, StringComparison.Ordinal);
        Assert.Contains("\"canPublishPublicly\": false", evidence, StringComparison.Ordinal);
        Assert.Contains("\"devicePointerRecorded\": false", evidence, StringComparison.Ordinal);
    }

    private static string ReadSource(params string[] pathParts) =>
        File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
}

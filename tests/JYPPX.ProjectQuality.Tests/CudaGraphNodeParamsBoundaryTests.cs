using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaGraphNodeParamsBoundaryTests
{
    [Fact]
    public void CudaGraphMemsetNodeParamsApisUseTypedDescriptorsAndManagedMemoryOwners()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-forty-fourth-batch-graph-node-params.manifest.json");
        string deferredManifest = ReadSource("native", "manifests", "cuda", "cuda-thirty-seventh-batch-graph-boundaries.manifest.json");
        string header = ReadSource("native", "include", "jyppx", "cuda", "runtime.h");
        string nativeSource = ReadSource("native", "src", "cuda", "modules", "graph", "node_topology.inc");
        string graph = ReadSource("src", "JYPPX.CudaSharp", "CudaGraph.cs");
        string parameters = ReadSource("src", "JYPPX.CudaSharp", "CudaGraphMemsetNodeParameters.cs");
        string interop = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "Graph", "NativeCudaApi.Graph.cs");
        string smoke = ReadSource("smoke", "CudaGraphSmokeRunner", "Program.cs");
        string coverageScript = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");

        Assert.Contains("cuda-graph-memset-node-get-params-safe", manifest);
        Assert.Contains("cuda-graph-memset-node-set-params-safe", manifest);
        Assert.Contains("\"type\": \"JYPPX_CudaGraphMemsetNodeParams*\", \"direction\": \"out\"", manifest);
        Assert.Contains("\"type\": \"JYPPX_CudaMemory*\", \"direction\": \"in\"", manifest);
        Assert.Contains("\"type\": \"uint32_t\", \"direction\": \"in\"", manifest);
        Assert.Contains("\"type\": \"size_t\", \"direction\": \"in\"", manifest);
        Assert.Contains("cuda-graph-memset-node-get-params-deferred", deferredManifest);
        Assert.Contains("cuda-graph-memset-node-set-params-deferred", deferredManifest);

        Assert.Contains("jyppx_cuda_graph_memset_node_get_params_safe(uintptr_t node, JYPPX_CudaGraphMemsetNodeParams* out_params)", header);
        Assert.Contains("jyppx_cuda_graph_memset_node_set_params_safe(uintptr_t node, JYPPX_CudaMemory* destination, uint32_t value, size_t count)", header);
        Assert.Contains("cudaGraphMemsetNodeGetParams", nativeSource);
        Assert.Contains("cudaGraphMemsetNodeSetParams", nativeSource);
        Assert.Contains("validate_memory(destination, \"destination\")", nativeSource);
        Assert.Contains("count > memory_object->size", nativeSource);
        Assert.Contains("params.elementSize = 1;", nativeSource);

        Assert.Contains("public static CudaGraphMemsetNodeParameters GetMemsetNodeParameters", graph);
        Assert.Contains("public static void SetMemsetNodeParameters(CudaGraphNode node, CudaMemory destination, byte value, int count)", graph);
        Assert.Contains("destination.Handle", graph);
        Assert.Contains("SetGraphMemsetNodeParameters", interop);
        Assert.Contains("SafeCudaMemoryHandle destination", interop);
        Assert.Contains("public readonly struct CudaGraphMemsetNodeParameters", parameters);
        Assert.Contains("diagnostic address value", parameters);
        Assert.DoesNotContain("public IntPtr", graph + parameters);
        Assert.DoesNotContain("public nint", graph + parameters);

        Assert.Contains("CudaGraph.SetMemsetNodeParameters", smoke);
        Assert.Contains("Updated=", smoke);
        Assert.Contains("\"cudaGraphMemsetNodeSetParams\" = @(\"graph-memset-node-set-params-deferred\", \"graph-memset-node-set-params-safe\")", coverageScript);
    }

    [Fact]
    public void CudaGraphNodeParamsDescriptorIsPointerFreeAndCoversHighValueNodeKinds()
    {
        string graph = ReadSource("src", "JYPPX.CudaSharp", "CudaGraph.cs");
        string descriptor = ReadSource("src", "JYPPX.CudaSharp", "CudaGraphNodeParamsDescriptor.cs");
        string descriptorKind = ReadSource("src", "JYPPX.CudaSharp", "CudaGraphNodeParamsDescriptorKind.cs");
        string smoke = ReadSource("smoke", "CudaGraphSmokeRunner", "Program.cs");

        Assert.Contains("GetNodeParamsDescriptor(CudaGraphNode node)", graph);
        Assert.Contains("CudaGraphNodeType.Memset => CudaGraphNodeParamsDescriptor.Memset(GetMemsetNodeParameters(node))", graph);
        Assert.Contains("CudaGraphNodeType.Memcpy => CudaGraphNodeParamsDescriptor.Memcpy(GetMemcpyNodeParameters(node))", graph);
        Assert.Contains("CudaGraphNodeType.EventRecord => CudaGraphNodeParamsDescriptor.EventRecord(EventRecordNodeHasEvent(node))", graph);
        Assert.Contains("CudaGraphNodeType.WaitEvent => CudaGraphNodeParamsDescriptor.EventWait(EventWaitNodeHasEvent(node))", graph);
        Assert.Contains("_ => CudaGraphNodeParamsDescriptor.Unsupported(nodeType)", graph);

        Assert.Contains("public readonly struct CudaGraphNodeParamsDescriptor", descriptor);
        Assert.Contains("public CudaGraphNodeParamsDescriptorKind DescriptorKind", descriptor);
        Assert.Contains("public bool HasBorrowedHandleExposure", descriptor);
        Assert.Contains("public bool HasEvent", descriptor);
        Assert.Contains("copied-event-presence-no-borrowed-handle", descriptor);
        Assert.Contains("copied-memset-scalars-no-address", descriptor);
        Assert.Contains("copied-memcpy-scalars-no-address", descriptor);
        Assert.Contains("unsupported-or-deferred-no-borrowed-handle", descriptor);
        Assert.Contains("public enum CudaGraphNodeParamsDescriptorKind", descriptorKind);
        Assert.Contains("EventRecord", descriptorKind);
        Assert.Contains("EventWait", descriptorKind);
        Assert.Contains("Memset", descriptorKind);
        Assert.Contains("Memcpy", descriptorKind);

        Assert.DoesNotContain("public IntPtr", graph + descriptor + descriptorKind);
        Assert.DoesNotContain("public nint", graph + descriptor + descriptorKind);
        Assert.DoesNotContain("CudaEvent Get", graph + descriptor);
        Assert.DoesNotContain("DestinationAddress { get; }", descriptor);
        Assert.DoesNotContain("SourceAddress { get; }", descriptor);

        Assert.Contains("ProbeGraphNodeParamsTypedDescriptors", smoke);
        Assert.Contains("NodeParamsDescriptor=", smoke);
        Assert.Contains("CudaGraph.GetNodeParamsDescriptor", smoke);
        Assert.Contains("HasBorrowedHandleExposure", smoke);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}

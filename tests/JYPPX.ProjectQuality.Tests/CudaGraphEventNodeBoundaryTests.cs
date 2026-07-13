using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaGraphEventNodeBoundaryTests
{
    [Fact]
    public void CudaGraphEventNodeApisAreLiftedWithoutDeletingDeferredRecords()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-forty-sixth-batch-graph-event-nodes.manifest.json");
        string deferredManifest = ReadSource("native", "manifests", "cuda", "cuda-twenty-third-batch-deferred-coverage.manifest.json");
        string header = ReadSource("native", "include", "jyppx", "cuda", "runtime.h");
        string nativeSource = ReadSource("native", "src", "cuda", "modules", "graph", "node_topology.inc");
        string coverageScript = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");

        Assert.Contains("cuda-graph-add-event-record-node-safe", manifest);
        Assert.Contains("cuda-graph-add-event-wait-node-safe", manifest);
        Assert.Contains("cuda-graph-event-record-node-set-event-safe", manifest);
        Assert.Contains("cuda-graph-event-wait-node-set-event-safe", manifest);
        Assert.Contains("cuda-graph-event-record-node-has-event-safe", manifest);
        Assert.Contains("cuda-graph-event-wait-node-has-event-safe", manifest);
        Assert.Contains("cuda-graph-exec-event-record-node-set-event-safe", manifest);
        Assert.Contains("cuda-graph-exec-event-wait-node-set-event-safe", manifest);
        Assert.Contains("\"type\": \"JYPPX_CudaEvent*\", \"direction\": \"in\"", manifest);
        Assert.Contains("\"type\": \"JYPPX_Boolean*\", \"direction\": \"out\", \"managedType\": \"out int\"", manifest);
        Assert.Contains("\"type\": \"uintptr_t*\", \"direction\": \"out\", \"managedType\": \"out UIntPtr\"", manifest);

        Assert.Contains("cuda-cuda-graph-add-event-record-node-deferred", deferredManifest);
        Assert.Contains("cuda-cuda-graph-add-event-wait-node-deferred", deferredManifest);
        Assert.Contains("cuda-cuda-graph-event-record-node-set-event-deferred", deferredManifest);
        Assert.Contains("cuda-cuda-graph-event-wait-node-set-event-deferred", deferredManifest);
        Assert.Contains("cuda-cuda-graph-exec-event-record-node-set-event-deferred", deferredManifest);
        Assert.Contains("cuda-cuda-graph-exec-event-wait-node-set-event-deferred", deferredManifest);

        Assert.Contains("jyppx_cuda_graph_add_event_record_node_safe", header);
        Assert.Contains("jyppx_cuda_graph_add_event_wait_node_safe", header);
        Assert.Contains("jyppx_cuda_graph_event_record_node_set_event_safe", header);
        Assert.Contains("jyppx_cuda_graph_event_wait_node_set_event_safe", header);
        Assert.Contains("jyppx_cuda_graph_event_record_node_has_event_safe", header);
        Assert.Contains("jyppx_cuda_graph_event_wait_node_has_event_safe", header);
        Assert.Contains("jyppx_cuda_graph_exec_event_record_node_set_event_safe", header);
        Assert.Contains("jyppx_cuda_graph_exec_event_wait_node_set_event_safe", header);
        Assert.Contains("cudaGraphAddEventRecordNode", nativeSource);
        Assert.Contains("cudaGraphAddEventWaitNode", nativeSource);
        Assert.Contains("cudaGraphEventRecordNodeSetEvent", nativeSource);
        Assert.Contains("cudaGraphEventWaitNodeSetEvent", nativeSource);
        Assert.Contains("cudaGraphEventRecordNodeGetEvent", nativeSource);
        Assert.Contains("cudaGraphEventWaitNodeGetEvent", nativeSource);
        Assert.Contains("event_handle != nullptr ? JYPPX_TRUE : JYPPX_FALSE", nativeSource);
        Assert.Contains("cudaGraphExecEventRecordNodeSetEvent", nativeSource);
        Assert.Contains("cudaGraphExecEventWaitNodeSetEvent", nativeSource);
        Assert.Contains("validate_event(event_handle, \"event_handle\")", nativeSource);
        Assert.Contains("dependency_node == 0 ? nullptr : &dependency", nativeSource);
        Assert.Contains("CUDART_VERSION >= 11010", nativeSource);

        Assert.Contains("\"cudaGraphAddEventRecordNode\" = @(\"graph-add-event-record-node-deferred\", \"graph-add-event-record-node-safe\")", coverageScript);
        Assert.Contains("\"cudaGraphAddEventWaitNode\" = @(\"graph-add-event-wait-node-deferred\", \"graph-add-event-wait-node-safe\")", coverageScript);
        Assert.Contains("\"cudaGraphEventRecordNodeSetEvent\" = @(\"graph-event-record-node-set-event-deferred\", \"graph-event-record-node-set-event-safe\")", coverageScript);
        Assert.Contains("\"cudaGraphEventWaitNodeSetEvent\" = @(\"graph-event-wait-node-set-event-deferred\", \"graph-event-wait-node-set-event-safe\")", coverageScript);
        Assert.Contains("\"cudaGraphEventRecordNodeGetEvent\" = @(\"graph-event-record-node-get-event-deferred\", \"graph-event-record-node-has-event-safe\")", coverageScript);
        Assert.Contains("\"cudaGraphEventWaitNodeGetEvent\" = @(\"graph-event-wait-node-get-event-deferred\", \"graph-event-wait-node-has-event-safe\")", coverageScript);
        Assert.Contains("\"cudaGraphExecEventRecordNodeSetEvent\" = @(\"graph-exec-event-record-node-set-event-deferred\", \"graph-exec-event-record-node-set-event-safe\")", coverageScript);
        Assert.Contains("\"cudaGraphExecEventWaitNodeSetEvent\" = @(\"graph-exec-event-wait-node-set-event-deferred\", \"graph-exec-event-wait-node-set-event-safe\")", coverageScript);
    }

    [Fact]
    public void ManagedCudaGraphEventNodeApiUsesCudaEventOwners()
    {
        string graph = ReadSource("src", "JYPPX.CudaSharp", "CudaGraph.cs");
        string graphExec = ReadSource("src", "JYPPX.CudaSharp", "CudaGraphExec.cs");
        string interop = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "Graph", "NativeCudaApi.Graph.cs");

        Assert.Contains("AddEventRecordNode(CudaEvent eventHandle)", graph);
        Assert.Contains("AddEventRecordNodeAfter(CudaGraphNode dependencyNode, CudaEvent eventHandle)", graph);
        Assert.Contains("AddEventWaitNode(CudaEvent eventHandle)", graph);
        Assert.Contains("AddEventWaitNodeAfter(CudaGraphNode dependencyNode, CudaEvent eventHandle)", graph);
        Assert.Contains("SetEventRecordNodeEvent(CudaGraphNode node, CudaEvent eventHandle)", graph);
        Assert.Contains("SetEventWaitNodeEvent(CudaGraphNode node, CudaEvent eventHandle)", graph);
        Assert.Contains("EventRecordNodeHasEvent(CudaGraphNode node)", graph);
        Assert.Contains("EventWaitNodeHasEvent(CudaGraphNode node)", graph);
        Assert.Contains("SetEventRecordNodeEvent(CudaGraphNode node, CudaEvent eventHandle)", graphExec);
        Assert.Contains("SetEventWaitNodeEvent(CudaGraphNode node, CudaEvent eventHandle)", graphExec);
        Assert.Contains("eventHandle.Handle", graph + graphExec);
        Assert.Contains("GraphEventRecordNodeHasEvent(CudaGraphNode node)", interop);
        Assert.Contains("GraphEventWaitNodeHasEvent(CudaGraphNode node)", interop);
        Assert.Contains("out int hasEvent", interop);
        Assert.Contains("SafeCudaEventHandle eventHandle", interop);
        Assert.Contains("remain alive while", graph + graphExec);

        Assert.DoesNotContain("public IntPtr", graph + graphExec);
        Assert.DoesNotContain("public nint", graph + graphExec);
        Assert.DoesNotContain("public CudaEvent Get", graph + graphExec);
    }

    [Fact]
    public void CudaGraphSmokeCoversEventNodeMutationPath()
    {
        string program = ReadSource("smoke", "CudaGraphSmokeRunner", "Program.cs");

        Assert.Contains("ProbeGraphEventNodes", program);
        Assert.Contains("AddEventRecordNode", program);
        Assert.Contains("AddEventWaitNodeAfter", program);
        Assert.Contains("SetEventRecordNodeEvent", program);
        Assert.Contains("SetEventWaitNodeEvent", program);
        Assert.Contains("EventRecordNodeHasEvent", program);
        Assert.Contains("EventWaitNodeHasEvent", program);
        Assert.Contains("RecordHasEvent=", program);
        Assert.Contains("WaitHasEvent=", program);
        Assert.Contains("EventNodes=", program);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}

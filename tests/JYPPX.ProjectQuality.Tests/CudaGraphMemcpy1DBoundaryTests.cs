using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaGraphMemcpy1DBoundaryTests
{
    [Fact]
    public void CudaGraphMemcpy1DApisUseManagedOwnersAndCopiedDiagnostics()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-forty-seventh-batch-graph-memcpy-1d-nodes.manifest.json");
        string deferredManifest = ReadSource("native", "manifests", "cuda", "cuda-thirty-seventh-batch-graph-boundaries.manifest.json");
        string officialDeferredManifest = ReadSource("native", "manifests", "cuda", "cuda-twenty-third-batch-deferred-coverage.manifest.json");
        string header = ReadSource("native", "include", "jyppx", "cuda", "runtime.h");
        string nativeSource = ReadSource("native", "src", "cuda", "modules", "graph", "node_topology.inc");
        string graph = ReadSource("src", "JYPPX.CudaSharp", "CudaGraph.cs");
        string graphExec = ReadSource("src", "JYPPX.CudaSharp", "CudaGraphExec.cs");
        string parameters = ReadSource("src", "JYPPX.CudaSharp", "CudaGraphMemcpyNodeParameters.cs");
        string interop = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "Graph", "NativeCudaApi.Graph.cs");
        string smoke = ReadSource("smoke", "CudaGraphSmokeRunner", "Program.cs");
        string coverageScript = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");

        Assert.Contains("cuda-graph-add-memcpy-node-1d-device-to-device-safe", manifest);
        Assert.Contains("cuda-graph-add-memcpy-node-1d-host-to-device-safe", manifest);
        Assert.Contains("cuda-graph-add-memcpy-node-1d-device-to-host-safe", manifest);
        Assert.Contains("cuda-graph-memcpy-node-get-params-safe", manifest);
        Assert.Contains("cuda-graph-memcpy-node-set-params-1d-device-to-device-safe", manifest);
        Assert.Contains("cuda-graph-exec-memcpy-node-set-params-1d-device-to-host-safe", manifest);
        Assert.Contains("\"type\": \"JYPPX_CudaMemory*\", \"direction\": \"in\"", manifest);
        Assert.Contains("\"type\": \"JYPPX_CudaPinnedMemory*\", \"direction\": \"in\"", manifest);
        Assert.Contains("\"type\": \"JYPPX_CudaGraphMemcpyNodeParams*\", \"direction\": \"out\"", manifest);
        Assert.Contains("cuda-graph-memcpy-node-set-params-1d-deferred", deferredManifest);
        Assert.Contains("cuda-cuda-graph-add-memcpy-node1-d-deferred", officialDeferredManifest);
        Assert.Contains("cuda-cuda-graph-exec-memcpy-node-set-params1-d-deferred", officialDeferredManifest);

        Assert.Contains("jyppx_cuda_graph_add_memcpy_node_1d_device_to_device_safe", header);
        Assert.Contains("jyppx_cuda_graph_memcpy_node_get_params_safe(uintptr_t node, JYPPX_CudaGraphMemcpyNodeParams* out_params)", header);
        Assert.Contains("jyppx_cuda_graph_exec_memcpy_node_set_params_1d_host_to_device_safe", header);
        Assert.Contains("cudaGraphAddMemcpyNode1D", nativeSource);
        Assert.Contains("cudaGraphMemcpyNodeGetParams", nativeSource);
        Assert.Contains("cudaGraphMemcpyNodeSetParams1D", nativeSource);
        Assert.Contains("cudaGraphExecMemcpyNodeSetParams1D", nativeSource);
        Assert.Contains("validate_pinned_memory(source, \"source\")", nativeSource);
        Assert.Contains("count > destination_size", nativeSource);
        Assert.Contains("count > source_size", nativeSource);

        Assert.Contains("public CudaGraphNode AddHostToDeviceMemcpyNode", graph);
        Assert.Contains("public CudaGraphNode AddDeviceToHostMemcpyNodeAfter", graph);
        Assert.Contains("public static CudaGraphMemcpyNodeParameters GetMemcpyNodeParameters", graph);
        Assert.Contains("public static void SetDeviceToDeviceMemcpyNodeParameters", graph);
        Assert.Contains("public void SetHostToDeviceMemcpyNodeParameters", graphExec);
        Assert.Contains("public void SetDeviceToHostMemcpyNodeParameters", graphExec);
        Assert.Contains("ValidateMemcpyCount(count, destination.SizeInBytes, source.SizeInBytes", graph + graphExec);
        Assert.Contains("AddGraphMemcpyNode1DHostToDevice", interop);
        Assert.Contains("SetGraphExecMemcpyNodeParametersDeviceToHost", interop);
        Assert.Contains("public readonly struct CudaGraphMemcpyNodeParameters", parameters);
        Assert.Contains("diagnostic numeric values", parameters);
        Assert.Contains("public enum CudaMemcpyKind", ReadSource("src", "JYPPX.CudaSharp", "CudaMemcpyKind.cs"));
        Assert.DoesNotContain("public IntPtr", graph + graphExec + parameters);
        Assert.DoesNotContain("public nint", graph + graphExec + parameters);

        Assert.Contains("ProbeGraphMemcpy1D", smoke);
        Assert.Contains("AddHostToDeviceMemcpyNode", smoke);
        Assert.Contains("SetDeviceToHostMemcpyNodeParameters", smoke);
        Assert.Contains("\"cudaGraphAddMemcpyNode1D\" = @(\"graph-add-memcpy-node1-d-deferred\", \"graph-add-memcpy-node-1d-device-to-device-safe\", \"graph-add-memcpy-node-1d-host-to-device-safe\", \"graph-add-memcpy-node-1d-device-to-host-safe\")", coverageScript);
        Assert.Contains("\"cudaGraphMemcpyNodeGetParams\" = @(\"graph-memcpy-node-get-params-deferred\", \"graph-memcpy-node-get-params-safe\")", coverageScript);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}

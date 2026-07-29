using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaGraphKernelAttributeBoundaryTests
{
    [Fact]
    public void CudaGraphKernelNodeAttributeApisUseScalarDescriptorAndVersionGuards()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-forty-eighth-batch-graph-kernel-node-attributes.manifest.json");
        string deferredManifest = ReadSource("native", "manifests", "cuda", "cuda-thirty-seventh-batch-graph-boundaries.manifest.json");
        string header = ReadSource("native", "include", "jyppx", "cuda", "runtime.h");
        string types = ReadSource("native", "include", "jyppx", "cuda", "types.h");
        string nativeSource = ReadSource("native", "src", "cuda", "modules", "graph", "node_topology.inc");
        string graph =
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.cs") +
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.NodeInspection.cs") +
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.NodeMutation.cs");
        string value = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphKernelNodeAttributeValue.cs");
        string interop = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "Graph", "NativeCudaApi.Graph.cs");
        string nativeStructs = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "NativeStructs.cs");
        string coverageScript = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");

        Assert.Contains("cuda-graph-kernel-node-get-attribute-scalar-safe", manifest);
        Assert.Contains("cuda-graph-kernel-node-set-attribute-scalar-safe", manifest);
        Assert.Contains("\"type\": \"JYPPX_CudaGraphKernelNodeAttributeValue*\", \"direction\": \"out\"", manifest);
        Assert.Contains("\"type\": \"const JYPPX_CudaGraphKernelNodeAttributeValue*\", \"direction\": \"in\"", manifest);
        Assert.Contains("cuda-graph-kernel-node-get-attribute-deferred", deferredManifest);
        Assert.Contains("cuda-graph-kernel-node-set-attribute-deferred", deferredManifest);
        Assert.Contains("cuda-graph-kernel-node-get-params-deferred", deferredManifest);
        Assert.Contains("cuda-graph-kernel-node-set-params-deferred", deferredManifest);

        Assert.Contains("jyppx_cuda_graph_kernel_node_get_attribute_scalar_safe(uintptr_t node, int32_t attribute, JYPPX_CudaGraphKernelNodeAttributeValue* out_value)", header);
        Assert.Contains("jyppx_cuda_graph_kernel_node_set_attribute_scalar_safe(uintptr_t node, const JYPPX_CudaGraphKernelNodeAttributeValue* value)", header);
        Assert.Contains("typedef struct JYPPX_CudaGraphKernelNodeAttributeValue", types);
        Assert.Contains("int32_t attribute;", types);
        Assert.Contains("uint32_t reserved2;", types);

        Assert.Contains("cudaGraphKernelNodeGetAttribute", nativeSource);
        Assert.Contains("cudaGraphKernelNodeSetAttribute", nativeSource);
        Assert.Contains("cudaKernelNodeAttributeCooperative", nativeSource);
        Assert.Contains("cudaKernelNodeAttributePriority", nativeSource);
        Assert.Contains("cudaKernelNodeAttributeClusterDimension", nativeSource);
        Assert.Contains("cudaKernelNodeAttributeClusterSchedulingPolicyPreference", nativeSource);
        Assert.Contains("CUDART_VERSION >= 11080", nativeSource);
        Assert.Contains("CUDA runtime 11.8 or later", nativeSource);
        Assert.Contains("Only cooperative, priority, cluster dimension, and cluster scheduling policy preference", nativeSource);
        Assert.DoesNotContain("cudaKernelNodeAttributeAccessPolicyWindow, &native_value", nativeSource);
        Assert.DoesNotContain("deviceUpdatableKernelNode", nativeSource);
        Assert.DoesNotContain("programmaticEvent", nativeSource);

        Assert.Contains("public enum CudaGraphKernelNodeAttribute", value);
        Assert.Contains("public enum CudaClusterSchedulingPolicyPreference", value);
        Assert.Contains("public readonly struct CudaGraphKernelNodeAttributeValue", value);
        Assert.Contains("public static CudaGraphKernelNodeAttributeValue Cooperative", value);
        Assert.Contains("public static CudaGraphKernelNodeAttributeValue Priority", value);
        Assert.Contains("public static CudaGraphKernelNodeAttributeValue ClusterDimension", value);
        Assert.Contains("public static CudaGraphKernelNodeAttributeValue ClusterSchedulingPolicy", value);
        Assert.Contains("internal NativeCudaGraphKernelNodeAttributeValue ToNative()", value);
        Assert.Contains("internal struct NativeCudaGraphKernelNodeAttributeValue", nativeStructs);
        Assert.Contains("GetGraphKernelNodeAttribute", interop);
        Assert.Contains("SetGraphKernelNodeAttribute", interop);

        Assert.Contains("public static CudaGraphKernelNodeAttributeValue GetKernelNodeAttribute", graph);
        Assert.Contains("public static void SetKernelNodeAttribute", graph);
        Assert.Contains("public static bool GetKernelNodeCooperative", graph);
        Assert.Contains("public static void SetKernelNodeClusterDimension", graph);
        Assert.Contains("ValidateKernelNodeAttribute", graph);
        Assert.DoesNotContain("public IntPtr", value + graph);
        Assert.DoesNotContain("public nint", value + graph);
        Assert.DoesNotContain("cudaKernelNodeAttrValue", value + graph);
        Assert.DoesNotContain("cudaKernelNodeParams", value + graph);

        Assert.Contains("\"cudaGraphKernelNodeGetAttribute\" = @(\"graph-kernel-node-get-attribute-deferred\", \"graph-kernel-node-get-attribute-scalar-safe\")", coverageScript);
        Assert.Contains("\"cudaGraphKernelNodeSetAttribute\" = @(\"graph-kernel-node-set-attribute-deferred\", \"graph-kernel-node-set-attribute-scalar-safe\")", coverageScript);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}

using System;
using System.IO;
using System.Linq;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CudaConditionalGraphUpliftTests
{
    [Fact]
    public void ManifestPromotesConditionalHandlesAndRetainsDeferredHistory()
    {
        string manifest = ReadSource("native", "manifests", "cuda", "cuda-fifty-eighth-batch-conditional-graph-owner-safe.manifest.json");
        string history = ReadSource("native", "manifests", "cuda", "cuda-twenty-third-batch-deferred-coverage.manifest.json");
        string coverage = ReadSource("eng", "Export-InterfaceCoverageMatrix.ps1");

        Assert.Contains("cuda-graph-conditional-handle-create-owner-safe", manifest, StringComparison.Ordinal);
        Assert.Contains("cuda-graph-conditional-handle-create-v2-owner-safe", manifest, StringComparison.Ordinal);
        Assert.Contains("cuda-graph-add-conditional-node-owner-safe", manifest, StringComparison.Ordinal);
        Assert.Contains("CUDART_VERSION >= 12030", manifest, StringComparison.Ordinal);
        Assert.Contains("CUDART_VERSION >= 13000", manifest, StringComparison.Ordinal);
        Assert.Contains("cuda-cuda-graph-conditional-handle-create-deferred", history, StringComparison.Ordinal);
        Assert.Contains("cudaGraphConditionalHandleCreate", coverage, StringComparison.Ordinal);
        Assert.Contains("cuda-graph-conditional-handle-create-owner-safe", coverage, StringComparison.Ordinal);
    }

    [Fact]
    public void NativeBoundaryKeepsConditionalBodyGraphsInternalAndGuardsDisposal()
    {
        string header = ReadSource("native", "include", "jyppx", "cuda", "runtime.h");
        string objectHeader = ReadSource("native", "src", "cuda", "object.hpp");
        string native = ReadSource("native", "src", "cuda", "modules", "graph", "conditional_graph.inc");
        string graphNative = ReadSource("native", "src", "cuda", "modules", "graph", "stream_capture_graph.inc");
        string nodeNative = ReadSource("native", "src", "cuda", "modules", "graph", "owner_scoped_diagnostics.inc");

        Assert.Contains("JYPPX_CudaGraphConditionalHandle", header, StringComparison.Ordinal);
        Assert.Contains("JYPPX_CudaGraphConditionalNode", header, StringComparison.Ordinal);
        Assert.Contains("cudaGraphConditionalHandleCreate", native, StringComparison.Ordinal);
        Assert.Contains("cudaGraphConditionalHandleCreate_v2", native, StringComparison.Ordinal);
        Assert.Contains("cudaGraphAddNode", native, StringComparison.Ordinal);
        Assert.Contains("body_graphs", native, StringComparison.Ordinal);
        Assert.Contains("active_conditional_handles", objectHeader, StringComparison.Ordinal);
        Assert.Contains("active_conditional_nodes", objectHeader, StringComparison.Ordinal);
        Assert.Contains("conditional-owner-active", graphNative, StringComparison.Ordinal);
        Assert.Contains("conditional-node-owner-active", nodeNative, StringComparison.Ordinal);
        Assert.DoesNotContain("JYPPX_CudaGraph** out_body", native, StringComparison.Ordinal);
    }

    [Fact]
    public void ManagedSurfaceUsesOwnerWrappersAndDoesNotExposeNativePointers()
    {
        string graph =
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.cs") +
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.ConditionalHandles.cs") +
            ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraph.GraphComposition.cs");
        string handle = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphConditionalHandle.cs");
        string node = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphConditionalNode.cs");
        string types = ReadSource("src", "JYPPX.CudaSharp", "Graphs", "CudaGraphConditionalTypes.cs");
        string interop = ReadSource("src", "JYPPX.CudaSharp", "Internal", "Interop", "Graph", "NativeCudaApi.ConditionalGraph.cs");
        string smoke = ReadSource("smoke", "CudaGraphSmokeRunner", "Program.cs");

        Assert.Contains("CreateConditionalHandle", graph, StringComparison.Ordinal);
        Assert.Contains("CreateConditionalHandleV2", graph, StringComparison.Ordinal);
        Assert.Contains("AddConditionalNodeAfter", graph, StringComparison.Ordinal);
        Assert.Contains("EnterConditionalOwner", graph, StringComparison.Ordinal);
        Assert.Contains("cannot be disposed while a conditional handle or node wrapper is active", graph, StringComparison.Ordinal);
        Assert.Contains("public sealed class CudaGraphConditionalHandle", handle, StringComparison.Ordinal);
        Assert.Contains("public sealed class CudaGraphConditionalNode", node, StringComparison.Ordinal);
        Assert.Contains("AddEmptyNodeAfter", node, StringComparison.Ordinal);
        Assert.Contains("CudaGraphConditionalNodeType", types, StringComparison.Ordinal);
        Assert.Contains("SafeCudaGraphConditionalNodeHandle", interop, StringComparison.Ordinal);
        Assert.Contains("ProbeConditionalGraph", smoke, StringComparison.Ordinal);
        Assert.Contains("Instantiated=True", smoke, StringComparison.Ordinal);
        Assert.DoesNotContain("public IntPtr", graph + handle + node + types, StringComparison.Ordinal);
        Assert.DoesNotContain("public nint", graph + handle + node + types, StringComparison.Ordinal);
        Assert.DoesNotContain("public UIntPtr", graph + handle + node + types, StringComparison.Ordinal);
        Assert.DoesNotContain("public SafeHandle", graph + handle + node + types, StringComparison.Ordinal);
    }

    [Fact]
    public void AuditRecordsVersionEvidenceAndHostRuntimeBoundary()
    {
        string auditJson = ReadSource("artifacts", "interface-coverage", "cuda-conditional-graph-candidate-audit.json");
        string auditMarkdown = ReadSource("artifacts", "interface-coverage", "cuda-conditional-graph-candidate-audit.md");
        string comparison = ReadSource("artifacts", "interface-coverage", "cuda-runtime-interface-comparison.csv");

        Assert.Contains("cuda-conditional-graph-candidate-audit.v1", auditJson, StringComparison.Ordinal);
        Assert.Contains("12.3", auditJson, StringComparison.Ordinal);
        Assert.Contains("12.9", auditJson, StringComparison.Ordinal);
        Assert.Contains("13.2", auditJson, StringComparison.Ordinal);
        Assert.Contains("cuda13_2Smoke", auditJson, StringComparison.Ordinal);
        Assert.Contains("implemented-with-deferred-history", comparison, StringComparison.Ordinal);
        Assert.Contains("CUDA Conditional Graph Owner-Safe Audit", auditMarkdown, StringComparison.Ordinal);
        Assert.Contains("Child graph handles are never exposed", auditMarkdown, StringComparison.OrdinalIgnoreCase);
    }

    private static string ReadSource(params string[] pathParts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray());
        return File.ReadAllText(path);
    }
}

using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedCudaGraphMemoryFeatureLayoutTests
{
    private const string CudaGraphOriginalNormalizedSha256 =
        "be9ccd67654b6797e7aeab63bf8baddf23a6cd8381eb03e4733f139dc76ab377";
    private const string CudaMemoryOriginalNormalizedSha256 =
        "3036e5c57cac5684c816a317b905580b3aa6ed0960c6f7f460144ab882f24760";

    private static readonly string[] CudaGraphFeatureOrder =
    {
        "ConditionalHandles",
        "GraphComposition",
        "NodeCreation",
        "TopologyDiagnostics",
        "NodeInspection",
        "NodeMutation",
        "NodeRelations",
        "Instantiation"
    };

    private static readonly string[] CudaMemoryFeatureOrder =
    {
        "Ipc",
        "RangeDiagnostics",
        "AsyncAllocation",
        "Fill",
        "PrefetchAdvice",
        "HostTransfers",
        "DeviceTransfers",
        "AsyncFree",
        "ArrayConversion"
    };

    public static TheoryData<string, string[]> CudaGraphFeatureMethods => new()
    {
        {
            "ConditionalHandles",
            new[] { "CreateConditionalHandle", "CreateConditionalHandleV2" }
        },
        {
            "GraphComposition",
            new[]
            {
                "Clone",
                "AddChildGraphNode",
                "AddChildGraphNodeAfter",
                "AddConditionalNode",
                "AddConditionalNodeAfter"
            }
        },
        {
            "NodeCreation",
            new[]
            {
                "AddEmptyNode",
                "AddEmptyNodeAfter",
                "AddMemoryAllocationNode",
                "AddMemoryAllocationNodeAfter",
                "AddMemsetNode",
                "AddMemsetNodeAfter",
                "AddMemsetNode",
                "AddMemsetNodeAfter",
                "AddEventRecordNode",
                "AddEventRecordNodeAfter",
                "AddEventWaitNode",
                "AddEventWaitNodeAfter",
                "AddDeviceToDeviceMemcpyNode",
                "AddDeviceToDeviceMemcpyNodeAfter",
                "AddHostToDeviceMemcpyNode",
                "AddHostToDeviceMemcpyNodeAfter",
                "AddDeviceToHostMemcpyNode",
                "AddDeviceToHostMemcpyNodeAfter",
                "AddDeviceToHostMemcpyNode",
                "AddDeviceToHostMemcpyNodeAfter",
                "AddMemoryFreeNode"
            }
        },
        {
            "TopologyDiagnostics",
            new[]
            {
                "AddDependency",
                "AddDependency",
                "RemoveDependency",
                "RemoveNode",
                "RemoveDependency",
                "GetNode",
                "GetRootNode",
                "GetEdge",
                "GetEdgeWithEdgeDataCount",
                "GetTopologySnapshot",
                "GetNodeSnapshotList",
                "GetRootNodeSnapshotList",
                "GetEdgeSnapshotList",
                "GetDiagnosticSnapshot",
                "GetEdgeWithEdgeData",
                "ExportDebugDot"
            }
        },
        {
            "NodeInspection",
            new[]
            {
                "FindNodeInClone",
                "ContainsNode",
                "GetChildGraphSnapshot",
                "GetNodeType",
                "GetNodeTopologySnapshot",
                "GetNodeDependencySnapshotList",
                "GetNodeDependentSnapshotList",
                "GetMemsetNodeParameters",
                "GetMemcpyNodeParameters",
                "GetNodeParamsDescriptor",
                "GetKernelNodeAttribute",
                "GetKernelNodeParametersSnapshot",
                "GetHostNodeParametersSnapshot",
                "GetMemoryAllocationNodeSnapshot",
                "GetMemoryFreeNodeSnapshot",
                "GetExternalSemaphoreSignalNodeSnapshot",
                "GetExternalSemaphoreWaitNodeSnapshot",
                "CopyKernelNodeAttributes",
                "GetKernelNodeCooperative",
                "GetKernelNodePriority",
                "GetKernelNodeClusterDimension",
                "GetKernelNodeClusterSchedulingPolicy"
            }
        },
        {
            "NodeMutation",
            new[]
            {
                "SetMemsetNodeParameters",
                "SetKernelNodeAttribute",
                "SetKernelNodeCooperative",
                "SetKernelNodePriority",
                "SetKernelNodeClusterDimension",
                "SetKernelNodeClusterSchedulingPolicy",
                "SetDeviceToDeviceMemcpyNodeParameters",
                "SetHostToDeviceMemcpyNodeParameters",
                "SetDeviceToHostMemcpyNodeParameters",
                "SetEventRecordNodeEvent",
                "EventRecordNodeHasEvent",
                "SetEventWaitNodeEvent",
                "EventWaitNodeHasEvent"
            }
        },
        {
            "NodeRelations",
            new[]
            {
                "GetNodeLocalId",
                "GetNodeToolsId",
                "GetDependencyCount",
                "GetDependency",
                "GetDependencyWithEdgeDataCount",
                "GetDependencyWithEdgeData",
                "GetDependentCount",
                "GetDependent",
                "GetDependentWithEdgeDataCount",
                "GetDependentWithEdgeData"
            }
        },
        {
            "Instantiation",
            new[] { "Instantiate", "InstantiateWithParameters", "InstantiateWithParameters" }
        }
    };

    public static TheoryData<string, string[]> CudaMemoryFeatureMethods => new()
    {
        {
            "Ipc",
            new[]
            {
                "ExportIpcToken",
                "ExportIpcDescriptor",
                "ImportIpcDescriptor",
                "TryImportIpcDescriptor",
                "TryExportIpcToken"
            }
        },
        {
            "RangeDiagnostics",
            new[]
            {
                "GetPointerAttributes",
                "GetRangeAttribute",
                "GetRangeAttribute",
                "GetRangeAttributes",
                "GetRangeAttributes",
                "GetRangeAccessedByDevices",
                "GetRangeAccessedByDevices",
                "GetRangeDiagnosticSummary",
                "GetRangeDiagnosticSummary"
            }
        },
        { "AsyncAllocation", new[] { "AllocateAsync", "AllocateFromPoolAsync" } },
        { "Fill", new[] { "Fill", "Fill", "FillAsync", "FillAsync" } },
        {
            "PrefetchAdvice",
            new[] { "PrefetchAsync", "PrefetchAsync", "PrefetchAsync", "Advise", "Advise", "Advise" }
        },
        {
            "HostTransfers",
            new[]
            {
                "CopyFrom",
                "CopyFrom",
                "CopyFromAsync",
                "CopyFromAsync",
                "CopyTo",
                "CopyTo",
                "CopyToAsync",
                "CopyToAsync"
            }
        },
        {
            "DeviceTransfers",
            new[]
            {
                "CopyTo",
                "CopyTo",
                "CopyToAsync",
                "CopyToAsync",
                "CopyToAuto",
                "CopyToAutoAsync",
                "CopyToPeer",
                "CopyToPeerAsync"
            }
        },
        { "AsyncFree", new[] { "FreeAsync" } },
        { "ArrayConversion", new[] { "ToArray", "ToSingleArray" } }
    };

    [Theory]
    [MemberData(nameof(CudaGraphFeatureMethods))]
    public void CudaGraphFeaturePartialsOwnExactPublicMethodSets(
        string feature,
        string[] expectedMethods)
    {
        string source = ReadSource("Graphs", $"CudaGraph.{feature}.cs");
        Assert.Equal(expectedMethods, EnumeratePublicMethodNames(source));
    }

    [Theory]
    [MemberData(nameof(CudaMemoryFeatureMethods))]
    public void CudaMemoryFeaturePartialsOwnExactPublicMethodSets(
        string feature,
        string[] expectedMethods)
    {
        string source = ReadSource("Memory", $"CudaMemory.{feature}.cs");
        Assert.Equal(expectedMethods, EnumeratePublicMethodNames(source));
    }

    [Fact]
    public void CudaGraphCoreRetainsOnlyOwnerLifetimeAndSharedValidation()
    {
        string core = ReadSource("Graphs", "CudaGraph.cs");

        Assert.Contains("public sealed partial class CudaGraph", core, StringComparison.Ordinal);
        Assert.Equal(new[] { "Create", "Dispose" }, EnumeratePublicMethodNames(core));
        Assert.Empty(EnumeratePublicPropertyNames(core));
        Assert.Contains("private int _activeCaptureToGraphSessions;", core, StringComparison.Ordinal);
        Assert.Contains("private int _activeConditionalOwners;", core, StringComparison.Ordinal);
        Assert.Contains("private int _activeMemoryAllocationOwners;", core, StringComparison.Ordinal);
        Assert.Contains("internal static void ValidateDeviceMemory(", core, StringComparison.Ordinal);
        Assert.Contains("internal static void ValidatePinnedMemory(", core, StringComparison.Ordinal);
        Assert.Contains("internal static void ValidateMemcpyCount(", core, StringComparison.Ordinal);
        Assert.Contains("internal static void ValidateMemsetCount(", core, StringComparison.Ordinal);
        Assert.Contains("private void ThrowIfDisposedCore()", core, StringComparison.Ordinal);
    }

    [Fact]
    public void CudaMemoryCoreRetainsOnlyAllocationLifetimeAndSharedValidation()
    {
        string core = ReadSource("Memory", "CudaMemory.cs");

        Assert.Contains("public partial class CudaMemory", core, StringComparison.Ordinal);
        Assert.Equal(new[] { "Dispose" }, EnumeratePublicMethodNames(core));
        Assert.Equal(
            new[] { "SizeInBytes", "IsIpcImported" },
            EnumeratePublicPropertyNames(core));
        Assert.Contains("private void ValidateCount(", core, StringComparison.Ordinal);
        Assert.Contains("protected void ValidateRange(", core, StringComparison.Ordinal);
        Assert.Contains("private static void ValidateScalarRangeAttribute(", core, StringComparison.Ordinal);
        Assert.Contains("protected static void ValidateMemoryAdvice(", core, StringComparison.Ordinal);
        Assert.Contains("private static SafeCudaMemoryHandle AllocateDeviceMemory(", core, StringComparison.Ordinal);
    }

    [Fact]
    public void CudaGraphFeaturePartialsRecomposeTheOriginalSource()
    {
        string core = Normalize(ReadSource("Graphs", "CudaGraph.cs"));
        int tailStart = core.IndexOf(
            "    internal static void ValidateDeviceMemory(",
            StringComparison.Ordinal);
        Assert.True(tailStart >= 0);

        StringBuilder source = new();
        source.Append(core[..tailStart].Replace(
            "public sealed partial class CudaGraph",
            "public sealed class CudaGraph",
            StringComparison.Ordinal));
        foreach (string feature in CudaGraphFeatureOrder)
        {
            source.Append(ReadPartialBody("Graphs", $"CudaGraph.{feature}.cs", "CudaGraph"));
        }

        source.Append(core[tailStart..]);
        Assert.Equal(CudaGraphOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void CudaMemoryFeaturePartialsRecomposeTheOriginalSource()
    {
        string core = Normalize(ReadSource("Memory", "CudaMemory.cs"));
        int disposeStart = core.IndexOf("    public void Dispose(", StringComparison.Ordinal);
        int tailStart = core.LastIndexOf("    /// <summary>", disposeStart, StringComparison.Ordinal);
        Assert.True(disposeStart >= 0 && tailStart >= 0);

        StringBuilder source = new();
        source.Append(core[..tailStart]);
        foreach (string feature in CudaMemoryFeatureOrder)
        {
            source.Append(ReadPartialBody("Memory", $"CudaMemory.{feature}.cs", "CudaMemory"));
        }

        source.Append(core[tailStart..]);
        Assert.Equal(CudaMemoryOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    private static string ReadPartialBody(string module, string fileName, string typeName)
    {
        string source = Normalize(ReadSource(module, fileName));
        int declarationStart = source.IndexOf($"class {typeName}", StringComparison.Ordinal);
        int bodyStart = source.IndexOf('{', declarationStart);
        int bodyEnd = source.LastIndexOf('}');
        Assert.True(declarationStart >= 0 && bodyStart >= 0 && bodyEnd > bodyStart);
        string body = source[(bodyStart + 1)..bodyEnd];
        Assert.StartsWith("\n", body, StringComparison.Ordinal);
        return body[1..];
    }

    private static string[] EnumeratePublicMethodNames(string source)
    {
        return Regex.Matches(
                source,
                @"public\s+(?:static\s+)?[^\s(]+\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")
            .Select(match => match.Groups["name"].Value)
            .ToArray();
    }

    private static string[] EnumeratePublicPropertyNames(string source)
    {
        return Regex.Matches(
                source,
                @"^\s*public\s+(?!static\s)[^\s(]+\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*\{\s*get;",
                RegexOptions.Multiline)
            .Select(match => match.Groups["name"].Value)
            .ToArray();
    }

    private static string ComputeSha256(string value)
    {
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(value)))
            .ToLowerInvariant();
    }

    private static string Normalize(string value)
    {
        return value.Replace("\r\n", "\n", StringComparison.Ordinal)
            .Replace('\r', '\n');
    }

    private static string ReadSource(string module, string fileName)
    {
        return File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.CudaSharp",
            module,
            fileName));
    }
}

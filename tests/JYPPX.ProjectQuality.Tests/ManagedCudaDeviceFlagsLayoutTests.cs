using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedCudaDeviceFlagsLayoutTests
{
    private const string CudaDeviceOriginalNormalizedSha256 =
        "d864d0cf8626bb59b75b7c5a2b30013ab3652afa0f0e2b5d5ee9b57960b7436d";
    private const string CudaFlagsOriginalNormalizedSha256 =
        "ec815281a28b92d8320d644a89ccc54b6e23d901de39bd353cbf22a766fa9d43";

    private static readonly (string Module, string FileName, string TypeName)[] CudaFlagTypeOrder =
    {
        ("Streams", "CudaStreamFlags.cs", "CudaStreamCreationFlags"),
        ("Events", "CudaEventFlags.cs", "CudaEventCreationFlags"),
        ("Memory", "CudaHostMemoryFlags.cs", "CudaPinnedMemoryAllocationFlags"),
        ("Memory", "CudaHostMemoryFlags.cs", "CudaHostRegistrationFlags"),
        ("Memory", "CudaMemoryAdvice.cs", "CudaMemoryAdvice"),
        ("Devices", "CudaDeviceExecutionEnums.cs", "CudaFunctionCachePreference"),
        ("Devices", "CudaDeviceExecutionEnums.cs", "CudaSharedMemoryConfig"),
        ("Devices", "CudaDeviceExecutionEnums.cs", "CudaDeviceRuntimeFlags"),
        ("Devices", "CudaPeerAccessEnums.cs", "CudaDeviceP2PAttribute"),
        ("Devices", "CudaPeerAccessEnums.cs", "CudaAtomicOperation"),
        ("Devices", "CudaPeerAccessEnums.cs", "CudaAtomicCapability"),
        ("Events", "CudaEventFlags.cs", "CudaEventRecordFlags"),
        ("Devices", "CudaDeviceExecutionEnums.cs", "CudaDeviceLimit"),
        ("Streams", "CudaStreamFlags.cs", "CudaStreamCaptureMode"),
        ("Streams", "CudaStreamFlags.cs", "CudaStreamCaptureDependencyMode"),
        ("Memory", "CudaArrayFlags.cs", "CudaChannelFormatKind"),
        ("Memory", "CudaArrayFlags.cs", "CudaArrayCreationFlags"),
        ("Memory", "CudaArrayFlags.cs", "CudaArraySparseFlags"),
        ("Devices", "CudaGpuDirectRdmaEnums.cs", "CudaGpuDirectRdmaWritesTarget"),
        ("Devices", "CudaGpuDirectRdmaEnums.cs", "CudaGpuDirectRdmaWritesScope"),
        ("Streams", "CudaStreamFlags.cs", "CudaStreamCaptureStatus"),
        ("Graphs", "CudaGraphNodeType.cs", "CudaGraphNodeType"),
        ("Memory", "CudaManagedMemoryAttachmentFlags.cs", "CudaManagedMemoryAttachmentFlags"),
        ("Devices", "CudaDeviceExecutionEnums.cs", "CudaDeviceAttribute")
    };

    public static TheoryData<string, string[], string[]> CudaDeviceFeatureMembers => new()
    {
        {
            "Core",
            new[] { "SetCurrent", "Use", "GetInfo", "GetProperties" },
            new[] { "RuntimeVersion", "DriverVersion", "Count", "Current", "CurrentProperties" }
        },
        {
            "GraphResources",
            new[]
            {
                "GetGraphMemoryAttribute",
                "GetGraphMemoryInfo",
                "GetGraphMemorySummary",
                "GetDevResourceSnapshot",
                "CurrentDevResourceSnapshot",
                "TrimGraphMemory",
                "ResetGraphMemoryHighWatermark",
                "ResetGraphMemoryHighWatermarks"
            },
            new[] { "CurrentGraphMemoryInfo", "CurrentGraphMemorySummary" }
        },
        {
            "RuntimeConfiguration",
            new[] { "GetAttribute", "GetLimit", "SetLimit" },
            new[] { "CacheConfig", "SharedMemoryConfig", "RuntimeFlags" }
        },
        {
            "InitializationSelection",
            new[]
            {
                "InitDevice",
                "GetPrimaryExecutionContext",
                "SetValidDevices",
                "GetPciBusId",
                "GetByPciBusId",
                "GetBooleanAttribute",
                "TryGetAttribute",
                "TryGetBooleanAttribute",
                "ChooseDevice"
            },
            Array.Empty<string>()
        },
        {
            "PeerCapabilities",
            new[]
            {
                "CanAccessPeer",
                "GetP2PAttribute",
                "GetHostAtomicCapabilities",
                "GetP2PAtomicCapabilities",
                "EnablePeerAccess",
                "DisablePeerAccess"
            },
            Array.Empty<string>()
        },
        {
            "MemoryPools",
            new[]
            {
                "GetMemoryInfo",
                "TryGetMemoryInfo",
                "GetMemoryPressureSnapshot",
                "GetDefaultMemoryPool",
                "GetCurrentMemoryPool",
                "SetCurrentMemoryPool"
            },
            Array.Empty<string>()
        },
        {
            "CacheRdma",
            new[] { "GetTexture1DLinearMaxWidth", "ResetPersistingL2Cache", "FlushGpuDirectRdmaWrites" },
            Array.Empty<string>()
        },
        {
            "SynchronizationErrors",
            new[]
            {
                "Synchronize",
                "Reset",
                "GetLastErrorCode",
                "PeekAtLastErrorCode",
                "GetErrorName",
                "GetErrorString"
            },
            Array.Empty<string>()
        }
    };

    public static TheoryData<string, string, string[]> CudaFlagTypeFiles => new()
    {
        {
            "Streams",
            "CudaStreamFlags.cs",
            new[]
            {
                "CudaStreamCreationFlags",
                "CudaStreamCaptureMode",
                "CudaStreamCaptureDependencyMode",
                "CudaStreamCaptureStatus"
            }
        },
        {
            "Events",
            "CudaEventFlags.cs",
            new[] { "CudaEventCreationFlags", "CudaEventRecordFlags" }
        },
        {
            "Memory",
            "CudaHostMemoryFlags.cs",
            new[] { "CudaPinnedMemoryAllocationFlags", "CudaHostRegistrationFlags" }
        },
        { "Memory", "CudaMemoryAdvice.cs", new[] { "CudaMemoryAdvice" } },
        {
            "Devices",
            "CudaDeviceExecutionEnums.cs",
            new[]
            {
                "CudaFunctionCachePreference",
                "CudaSharedMemoryConfig",
                "CudaDeviceRuntimeFlags",
                "CudaDeviceLimit",
                "CudaDeviceAttribute"
            }
        },
        {
            "Devices",
            "CudaPeerAccessEnums.cs",
            new[] { "CudaDeviceP2PAttribute", "CudaAtomicOperation", "CudaAtomicCapability" }
        },
        {
            "Memory",
            "CudaArrayFlags.cs",
            new[] { "CudaChannelFormatKind", "CudaArrayCreationFlags", "CudaArraySparseFlags" }
        },
        {
            "Devices",
            "CudaGpuDirectRdmaEnums.cs",
            new[] { "CudaGpuDirectRdmaWritesTarget", "CudaGpuDirectRdmaWritesScope" }
        },
        { "Graphs", "CudaGraphNodeType.cs", new[] { "CudaGraphNodeType" } },
        {
            "Memory",
            "CudaManagedMemoryAttachmentFlags.cs",
            new[] { "CudaManagedMemoryAttachmentFlags" }
        }
    };

    [Theory]
    [MemberData(nameof(CudaDeviceFeatureMembers))]
    public void CudaDeviceFeaturePartialsOwnExactPublicMembers(
        string feature,
        string[] expectedMethods,
        string[] expectedProperties)
    {
        string fileName = feature == "Core" ? "CudaDevice.cs" : $"CudaDevice.{feature}.cs";
        string source = ReadSource("Devices", fileName);

        Assert.Equal(expectedMethods, EnumeratePublicStaticMethodNames(source));
        Assert.Equal(expectedProperties, EnumeratePublicStaticPropertyNames(source));
    }

    [Theory]
    [MemberData(nameof(CudaFlagTypeFiles))]
    public void CudaFlagEnumsLiveInTheirDedicatedModuleFiles(
        string module,
        string fileName,
        string[] expectedTypes)
    {
        string source = ReadSource(module, fileName);
        Assert.Equal(expectedTypes, EnumeratePublicEnumNames(source));
    }

    [Fact]
    public void CudaDeviceFeatureHelpersStayWithTheirOnlyConsumers()
    {
        string core = ReadSource("Devices", "CudaDevice.cs");
        string initialization = ReadSource("Devices", "CudaDevice.InitializationSelection.cs");
        string peer = ReadSource("Devices", "CudaDevice.PeerCapabilities.cs");

        Assert.Contains("public static partial class CudaDevice", core, StringComparison.Ordinal);
        Assert.DoesNotContain("CopyAtomicOperations", core, StringComparison.Ordinal);
        Assert.DoesNotContain("CopyDeviceOrdinals", core, StringComparison.Ordinal);
        Assert.Contains("private static int[] CopyDeviceOrdinals(", initialization, StringComparison.Ordinal);
        Assert.DoesNotContain("CopyAtomicOperations", initialization, StringComparison.Ordinal);
        Assert.Contains("private static CudaAtomicOperation[] CopyAtomicOperations(", peer, StringComparison.Ordinal);
        Assert.DoesNotContain("CopyDeviceOrdinals", peer, StringComparison.Ordinal);
    }

    [Fact]
    public void LegacyCrossModuleCudaFlagsFileNoLongerExists()
    {
        Assert.False(File.Exists(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.CudaSharp",
            "Core",
            "CudaFlags.cs")));
    }

    [Fact]
    public void CudaDeviceFeaturePartialsRecomposeTheOriginalSource()
    {
        string core = Normalize(ReadSource("Devices", "CudaDevice.cs"));
        int coreEnd = core.LastIndexOf('}');
        Assert.True(coreEnd >= 0);

        string graphResources = ReadDevicePartialBody("GraphResources");
        string runtimeConfiguration = ReadDevicePartialBody("RuntimeConfiguration");
        string initialization = ReadDevicePartialBody("InitializationSelection");
        string peer = ReadDevicePartialBody("PeerCapabilities");
        string memoryPools = ReadDevicePartialBody("MemoryPools");
        string cacheRdma = ReadDevicePartialBody("CacheRdma");
        string synchronizationErrors = ReadDevicePartialBody("SynchronizationErrors");

        int chooseDeviceStart = FindDocumentationStart(
            initialization,
            "    public static int ChooseDevice(");
        int ordinalHelperStart = initialization.IndexOf(
            "    private static int[] CopyDeviceOrdinals(",
            StringComparison.Ordinal);
        int peerAccessStart = FindDocumentationStart(
            peer,
            "    public static void EnablePeerAccess(");
        int atomicHelperStart = peer.IndexOf(
            "    private static CudaAtomicOperation[] CopyAtomicOperations(",
            StringComparison.Ordinal);
        Assert.True(
            chooseDeviceStart >= 0 && ordinalHelperStart > chooseDeviceStart &&
            peerAccessStart >= 0 && atomicHelperStart > peerAccessStart);

        StringBuilder source = new();
        source.Append(core[..coreEnd].Replace(
            "public static partial class CudaDevice",
            "public static class CudaDevice",
            StringComparison.Ordinal));
        source.Append(graphResources);
        source.Append(runtimeConfiguration);
        source.Append(initialization[..chooseDeviceStart]);
        source.Append(peer[..peerAccessStart]);
        source.Append(initialization[chooseDeviceStart..ordinalHelperStart]);
        source.Append(peer[peerAccessStart..atomicHelperStart]);
        source.Append(memoryPools);
        source.Append(cacheRdma);
        source.Append(synchronizationErrors);
        source.Append(peer[atomicHelperStart..]);
        source.Append(initialization[ordinalHelperStart..]);
        source.Append('}');
        source.Append('\n');

        Assert.Equal(CudaDeviceOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void CudaFlagModuleFilesRecomposeTheOriginalSource()
    {
        string first = Normalize(ReadSource("Streams", "CudaStreamFlags.cs"));
        int firstTypeStart = first.IndexOf("/// <summary>", StringComparison.Ordinal);
        Assert.True(firstTypeStart >= 0);

        StringBuilder source = new();
        source.Append(first[..firstTypeStart]);
        for (int index = 0; index < CudaFlagTypeOrder.Length; index++)
        {
            (string module, string fileName, string typeName) = CudaFlagTypeOrder[index];
            source.Append(ReadTopLevelEnumSegment(module, fileName, typeName).TrimEnd('\n'));
            source.Append(index + 1 == CudaFlagTypeOrder.Length ? "\n" : "\n\n");
        }

        Assert.Equal(CudaFlagsOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    private static string ReadDevicePartialBody(string feature)
    {
        string source = Normalize(ReadSource("Devices", $"CudaDevice.{feature}.cs"));
        int declarationStart = source.IndexOf("class CudaDevice", StringComparison.Ordinal);
        int bodyStart = source.IndexOf('{', declarationStart);
        int bodyEnd = source.LastIndexOf('}');
        Assert.True(declarationStart >= 0 && bodyStart >= 0 && bodyEnd > bodyStart);
        string body = source[(bodyStart + 1)..bodyEnd];
        Assert.StartsWith("\n", body, StringComparison.Ordinal);
        return body[1..];
    }

    private static int FindDocumentationStart(string source, string memberMarker)
    {
        int memberStart = source.IndexOf(memberMarker, StringComparison.Ordinal);
        Assert.True(memberStart >= 0);
        return source.LastIndexOf("    /// <summary>", memberStart, StringComparison.Ordinal);
    }

    private static string ReadTopLevelEnumSegment(
        string module,
        string fileName,
        string typeName)
    {
        string source = Normalize(ReadSource(module, fileName));
        int declarationStart = source.IndexOf($"public enum {typeName}", StringComparison.Ordinal);
        int segmentStart = source.LastIndexOf("/// <summary>", declarationStart, StringComparison.Ordinal);
        int nextSegmentStart = source.IndexOf("\n/// <summary>", declarationStart, StringComparison.Ordinal);
        int segmentEnd = nextSegmentStart >= 0 ? nextSegmentStart + 1 : source.Length;
        Assert.True(declarationStart >= 0 && segmentStart >= 0 && segmentEnd > segmentStart);
        return source[segmentStart..segmentEnd];
    }

    private static string[] EnumeratePublicStaticMethodNames(string source)
    {
        return Regex.Matches(
                source,
                @"public\s+static\s+[^\s(]+\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")
            .Select(match => match.Groups["name"].Value)
            .ToArray();
    }

    private static string[] EnumeratePublicStaticPropertyNames(string source)
    {
        return Regex.Matches(
                source,
                @"^\s*public\s+static\s+[^\s{(]+\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:=>|\{)",
                RegexOptions.Multiline)
            .Select(match => match.Groups["name"].Value)
            .ToArray();
    }

    private static string[] EnumeratePublicEnumNames(string source)
    {
        return Regex.Matches(
                source,
                @"^public\s+enum\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)",
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

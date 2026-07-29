using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedCudaStreamMemoryPoolLayoutTests
{
    private const string CudaStreamOriginalNormalizedSha256 =
        "8b6117416cd402dc7bbed964431822652be7ef19fe61a94c7e3924b8f780fd52";
    private const string CudaMemoryPoolOriginalNormalizedSha256 =
        "df79524f844e67591ee942abdca0a9a8ce996cbcc32729695dcc7754722f707e";

    private static readonly string[] CudaStreamFeatureOrder =
    {
        "Diagnostics",
        "CaptureDiagnostics",
        "CaptureDependencies",
        "Synchronization",
        "CaptureLifecycle"
    };

    private static readonly string[] CudaMemoryPoolFeatureOrder =
    {
        "Factories",
        "Allocation",
        "Access",
        "Attributes"
    };

    public static TheoryData<string, string[], string[]> CudaStreamFeatureMembers => new()
    {
        {
            "Core",
            new[] { "Dispose" },
            new[] { "Flags", "Priority", "Id", "DeviceOrdinal", "CaptureStatus" }
        },
        {
            "Diagnostics",
            new[] { "GetPriorityRange", "ExchangeThreadCaptureMode", "IsReady" },
            Array.Empty<string>()
        },
        {
            "CaptureDiagnostics",
            new[]
            {
                "GetCaptureInfo",
                "TryGetCaptureInfo",
                "GetCaptureInfoPtzs",
                "GetDevResourceSnapshot",
                "TryGetCaptureInfoPtzs"
            },
            Array.Empty<string>()
        },
        {
            "CaptureDependencies",
            new[]
            {
                "UpdateCaptureDependenciesPtzs",
                "UpdateCaptureDependenciesV2",
                "UpdateCaptureDependencies"
            },
            Array.Empty<string>()
        },
        {
            "Synchronization",
            new[] { "WaitFor", "CopyAttributesFrom", "Synchronize", "MeasureElapsedTime" },
            Array.Empty<string>()
        },
        {
            "CaptureLifecycle",
            new[] { "BeginCapture", "BeginCaptureToGraph", "EndCapture" },
            Array.Empty<string>()
        }
    };

    public static TheoryData<string, string[], string[]> CudaMemoryPoolFeatureMembers => new()
    {
        { "Core", Array.Empty<string>(), new[] { "DeviceOrdinal" } },
        {
            "Factories",
            new[] { "Create", "GetDefault", "GetCurrent", "MakeCurrent" },
            Array.Empty<string>()
        },
        {
            "Allocation",
            new[] { "AllocateAsync", "TrimTo" },
            Array.Empty<string>()
        },
        {
            "Access",
            new[] { "SetAccess", "GetAccess" },
            Array.Empty<string>()
        },
        {
            "Attributes",
            new[] { "GetAttribute", "SetAttribute", "ResetReservedMemoryHigh", "ResetUsedMemoryHigh" },
            new[]
            {
                "ReleaseThresholdBytes",
                "ReservedMemoryCurrentBytes",
                "ReservedMemoryHighBytes",
                "UsedMemoryCurrentBytes",
                "UsedMemoryHighBytes"
            }
        }
    };

    [Theory]
    [MemberData(nameof(CudaStreamFeatureMembers))]
    public void CudaStreamFeaturePartialsOwnExactPublicMembers(
        string feature,
        string[] expectedMethods,
        string[] expectedProperties)
    {
        string fileName = feature == "Core" ? "CudaStream.cs" : $"CudaStream.{feature}.cs";
        string source = ReadSource("Streams", fileName);
        Assert.Equal(expectedMethods, EnumeratePublicMethodNames(source));
        Assert.Equal(expectedProperties, EnumeratePublicPropertyNames(source));
    }

    [Theory]
    [MemberData(nameof(CudaMemoryPoolFeatureMembers))]
    public void CudaMemoryPoolFeaturePartialsOwnExactPublicMembers(
        string feature,
        string[] expectedMethods,
        string[] expectedProperties)
    {
        string fileName = feature == "Core" ? "CudaMemoryPool.cs" : $"CudaMemoryPool.{feature}.cs";
        string source = ReadSource("Memory", fileName);
        Assert.Equal(expectedMethods, EnumeratePublicMethodNames(source));
        Assert.Equal(expectedProperties, EnumeratePublicPropertyNames(source));
    }

    [Fact]
    public void CudaStreamCoreRetainsCaptureOwnerLifetimeAndDisposeGate()
    {
        string core = ReadSource("Streams", "CudaStream.cs");

        Assert.Contains("public sealed partial class CudaStream", core, StringComparison.Ordinal);
        Assert.Contains("private int _activeCaptureToGraphSessions;", core, StringComparison.Ordinal);
        Assert.Contains("private bool _disposed;", core, StringComparison.Ordinal);
        Assert.Contains("internal void EnterCaptureToGraphSession()", core, StringComparison.Ordinal);
        Assert.Contains("internal void ExitCaptureToGraphSession()", core, StringComparison.Ordinal);
        Assert.Contains(
            "cannot be disposed while a stream-to-graph capture session is active",
            core,
            StringComparison.Ordinal);
    }

    [Fact]
    public void MemoryPoolOwnerAndEnumsLiveInDedicatedFiles()
    {
        string core = ReadSource("Memory", "CudaMemoryPool.cs");
        string owned = ReadSource("Memory", "CudaOwnedMemoryPool.cs");
        string enums = ReadSource("Memory", "CudaMemoryPoolEnums.cs");

        Assert.Contains("public readonly partial struct CudaMemoryPool", core, StringComparison.Ordinal);
        Assert.Equal(new[] { "CudaMemoryPool" }, EnumeratePublicTopLevelTypeNames(core));
        Assert.Equal(new[] { "CudaOwnedMemoryPool" }, EnumeratePublicTopLevelTypeNames(owned));
        Assert.Equal(
            new[] { "MakeCurrent", "AllocateAsync", "TrimTo", "Dispose" },
            EnumeratePublicMethodNames(owned));
        Assert.Equal(new[] { "Pool", "DeviceOrdinal" }, EnumeratePublicPropertyNames(owned));
        Assert.Contains("private void ThrowIfDisposed()", owned, StringComparison.Ordinal);
        Assert.Equal(
            new[] { "CudaMemoryPoolAttribute", "CudaMemoryPoolAccessFlags" },
            EnumeratePublicEnumNames(enums));
    }

    [Fact]
    public void CudaStreamFeaturePartialsRecomposeTheOriginalSource()
    {
        string core = Normalize(ReadSource("Streams", "CudaStream.cs"));
        int disposeStart = core.IndexOf("    public void Dispose(", StringComparison.Ordinal);
        int tailStart = core.LastIndexOf("    /// <summary>", disposeStart, StringComparison.Ordinal);
        Assert.True(disposeStart >= 0 && tailStart >= 0);

        StringBuilder source = new();
        source.Append(core[..tailStart].Replace(
            "public sealed partial class CudaStream",
            "public sealed class CudaStream",
            StringComparison.Ordinal));
        foreach (string feature in CudaStreamFeatureOrder)
        {
            source.Append(ReadPartialBody("Streams", $"CudaStream.{feature}.cs", "CudaStream"));
        }

        source.Append(core[tailStart..]);
        Assert.Equal(CudaStreamOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void CudaMemoryPoolFilesRecomposeTheOriginalSource()
    {
        string core = Normalize(ReadSource("Memory", "CudaMemoryPool.cs"));
        int coreEnd = core.LastIndexOf('}');
        Assert.True(coreEnd >= 0);

        StringBuilder source = new();
        source.Append(core[..coreEnd].Replace(
            "public readonly partial struct CudaMemoryPool",
            "public readonly struct CudaMemoryPool",
            StringComparison.Ordinal));
        foreach (string feature in CudaMemoryPoolFeatureOrder)
        {
            source.Append(ReadPartialBody(
                "Memory",
                $"CudaMemoryPool.{feature}.cs",
                "CudaMemoryPool"));
        }

        source.Append('}');
        source.Append("\n\n");
        source.Append(ReadTopLevelSegment("CudaOwnedMemoryPool.cs").TrimEnd('\n'));
        source.Append("\n\n");
        source.Append(ReadTopLevelSegment("CudaMemoryPoolEnums.cs"));
        Assert.Equal(CudaMemoryPoolOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    private static string ReadPartialBody(
        string module,
        string fileName,
        string typeName)
    {
        string source = Normalize(ReadSource(module, fileName));
        int declarationStart = source.IndexOf($"{typeName}", StringComparison.Ordinal);
        int bodyStart = source.IndexOf('{', declarationStart);
        int bodyEnd = source.LastIndexOf('}');
        Assert.True(declarationStart >= 0 && bodyStart >= 0 && bodyEnd > bodyStart);
        string body = source[(bodyStart + 1)..bodyEnd];
        Assert.StartsWith("\n", body, StringComparison.Ordinal);
        return body[1..];
    }

    private static string ReadTopLevelSegment(string fileName)
    {
        string source = Normalize(ReadSource("Memory", fileName));
        int start = source.IndexOf("/// <summary>", StringComparison.Ordinal);
        Assert.True(start >= 0);
        return source[start..];
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
                @"^\s*public\s+(?!static\s)[^\s{(]+\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:=>|\{)",
                RegexOptions.Multiline)
            .Select(match => match.Groups["name"].Value)
            .ToArray();
    }

    private static string[] EnumeratePublicTopLevelTypeNames(string source)
    {
        return Regex.Matches(
                source,
                @"^public\s+(?:(?:readonly\s+partial\s+struct)|(?:sealed\s+class))\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)",
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

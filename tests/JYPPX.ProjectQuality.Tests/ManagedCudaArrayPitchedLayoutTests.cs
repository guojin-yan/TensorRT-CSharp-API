using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedCudaArrayPitchedLayoutTests
{
    private const string CudaPitchedMemoryOriginalNormalizedSha256 =
        "92238c1fb314ca978e09318573ff60c8413f172f7d505d23a6c26e5551147221";
    private const string CudaArrayOriginalNormalizedSha256 =
        "98814300efb40ed16dc9b95482eb875a3af4f4dfb63085167b0e6b637557e2bf";

    private static readonly string[] CudaPitchedMemoryFeatureOrder =
    {
        "Fill",
        "Transfers2D",
        "Transfers3D",
        "ArrayConversion"
    };

    private static readonly string[] CudaArrayFeatureOrder =
    {
        "Diagnostics",
        "Transfers1D",
        "Transfers2D",
        "Transfers3D",
        "ArrayConversion"
    };

    public static TheoryData<string, string, string[], string[]> FeatureMembers => new()
    {
        {
            "CudaPitchedMemory",
            "Fill",
            new[]
            {
                "Fill2D",
                "Fill2D",
                "Fill2DAsync",
                "Fill2DAsync",
                "Fill3D",
                "Fill3D",
                "Fill3DAsync",
                "Fill3DAsync"
            },
            Array.Empty<string>()
        },
        {
            "CudaPitchedMemory",
            "Transfers2D",
            new[]
            {
                "CopyFrom2D",
                "CopyFrom2D",
                "CopyTo2D",
                "CopyTo2D",
                "CopyTo",
                "CopyFrom2DAsync",
                "CopyFrom2DAsync",
                "CopyTo2DAsync",
                "CopyTo2DAsync",
                "CopyToAsync"
            },
            Array.Empty<string>()
        },
        {
            "CudaPitchedMemory",
            "Transfers3D",
            new[]
            {
                "CopyFrom3D",
                "CopyFrom3D",
                "CopyTo3D",
                "CopyTo3D",
                "CopyTo3D",
                "CopyFrom3DAsync",
                "CopyFrom3DAsync",
                "CopyTo3DAsync",
                "CopyTo3DAsync",
                "CopyTo3DAsync"
            },
            Array.Empty<string>()
        },
        {
            "CudaPitchedMemory",
            "ArrayConversion",
            new[] { "ToArray2D", "ToArray3D" },
            Array.Empty<string>()
        },
        {
            "CudaArray",
            "Diagnostics",
            new[]
            {
                "GetMemoryRequirements",
                "TryGetMemoryRequirements",
                "GetSparseProperties",
                "TryGetSparseProperties"
            },
            Array.Empty<string>()
        },
        {
            "CudaArray",
            "Transfers1D",
            new[]
            {
                "CopyFrom",
                "CopyFrom",
                "CopyFrom",
                "CopyTo",
                "CopyTo",
                "CopyTo",
                "CopyTo",
                "CopyTo",
                "CopyFromAsync",
                "CopyFromAsync",
                "CopyToAsync",
                "CopyToAsync"
            },
            Array.Empty<string>()
        },
        {
            "CudaArray",
            "Transfers2D",
            new[]
            {
                "CopyFrom2D",
                "CopyFrom2D",
                "CopyTo2D",
                "CopyTo2D",
                "CopyTo2D",
                "CopyTo2D",
                "CopyFrom2DAsync",
                "CopyFrom2DAsync",
                "CopyTo2DAsync",
                "CopyTo2DAsync"
            },
            Array.Empty<string>()
        },
        {
            "CudaArray",
            "Transfers3D",
            new[]
            {
                "CopyFrom3D",
                "CopyFrom3D",
                "CopyTo3D",
                "CopyTo3D",
                "CopyTo3D",
                "CopyTo3D",
                "CopyFrom3DAsync",
                "CopyFrom3DAsync",
                "CopyTo3DAsync",
                "CopyTo3DAsync",
                "CopyTo3DAsync",
                "CopyTo3DAsync"
            },
            Array.Empty<string>()
        },
        {
            "CudaArray",
            "ArrayConversion",
            new[] { "ToArray2D", "ToArray3D" },
            Array.Empty<string>()
        }
    };

    [Theory]
    [MemberData(nameof(FeatureMembers))]
    public void MemoryOwnerFeaturePartialsOwnExactPublicMembers(
        string typeName,
        string feature,
        string[] expectedMethods,
        string[] expectedProperties)
    {
        string source = ReadSource($"{typeName}.{feature}.cs");
        Assert.Equal(expectedMethods, EnumeratePublicMethodNames(source));
        Assert.Equal(expectedProperties, EnumeratePublicPropertyNames(source));
    }

    [Fact]
    public void CudaPitchedMemoryCoreRetainsOwnerMetadataAndSharedValidation()
    {
        string core = ReadSource("CudaPitchedMemory.cs");

        Assert.Contains("public sealed partial class CudaPitchedMemory", core, StringComparison.Ordinal);
        Assert.Equal(new[] { "Allocate3D", "Dispose" }, EnumeratePublicMethodNames(core));
        Assert.Equal(
            new[] { "WidthInBytes", "Height", "PitchInBytes" },
            EnumeratePublicPropertyNames(core));
        Assert.Contains("private readonly SafeCudaPitchedMemoryHandle _handle;", core, StringComparison.Ordinal);
        Assert.Contains("private void Validate2DRegion(", core, StringComparison.Ordinal);
        Assert.Contains("private void Validate2DExtent(", core, StringComparison.Ordinal);
        Assert.Contains("private void Validate3DExtent(", core, StringComparison.Ordinal);
        Assert.Contains("private static void Validate3DRegion(", core, StringComparison.Ordinal);
        Assert.Contains("private static void ValidateHostPitch(", core, StringComparison.Ordinal);
        Assert.Contains("private static void ValidatePinnedHostBuffer(", core, StringComparison.Ordinal);
    }

    [Fact]
    public void CudaArrayCoreRetainsOwnerMetadataAndSharedValidation()
    {
        string core = ReadSource("CudaArray.cs");

        Assert.Contains("public sealed partial class CudaArray", core, StringComparison.Ordinal);
        Assert.Equal(new[] { "Create3D", "Dispose" }, EnumeratePublicMethodNames(core));
        Assert.Equal(
            new[] { "Descriptor", "Extent", "Flags", "Info", "ChannelDescriptor" },
            EnumeratePublicPropertyNames(core));
        Assert.Contains("private readonly SafeCudaArrayHandle _handle;", core, StringComparison.Ordinal);
        Assert.Contains("private static void ValidateStream(", core, StringComparison.Ordinal);
        Assert.Contains("private static void ValidateByteCount(", core, StringComparison.Ordinal);
        Assert.Contains("private static void Validate2DExtent(", core, StringComparison.Ordinal);
        Assert.Contains("private static void Validate3DExtent(", core, StringComparison.Ordinal);
        Assert.Contains("private static void ValidatePinnedBuffer(", core, StringComparison.Ordinal);
        Assert.Contains("private static void ValidatePinned2D(", core, StringComparison.Ordinal);
        Assert.Contains("private static void ValidatePinned3D(", core, StringComparison.Ordinal);
    }

    [Fact]
    public void CudaPitchedMemoryFeaturePartialsRecomposeTheOriginalSource()
    {
        AssertRecomposesOriginal(
            "CudaPitchedMemory",
            CudaPitchedMemoryFeatureOrder,
            CudaPitchedMemoryOriginalNormalizedSha256);
    }

    [Fact]
    public void CudaArrayFeaturePartialsRecomposeTheOriginalSource()
    {
        AssertRecomposesOriginal(
            "CudaArray",
            CudaArrayFeatureOrder,
            CudaArrayOriginalNormalizedSha256);
    }

    private static void AssertRecomposesOriginal(
        string typeName,
        IEnumerable<string> featureOrder,
        string expectedSha256)
    {
        string core = Normalize(ReadSource($"{typeName}.cs"));
        int disposeStart = core.IndexOf("    public void Dispose(", StringComparison.Ordinal);
        int tailStart = core.LastIndexOf("    /// <summary>", disposeStart, StringComparison.Ordinal);
        Assert.True(disposeStart >= 0 && tailStart >= 0);

        StringBuilder source = new();
        source.Append(core[..tailStart].Replace(
            $"public sealed partial class {typeName}",
            $"public sealed class {typeName}",
            StringComparison.Ordinal));
        foreach (string feature in featureOrder)
        {
            source.Append(ReadPartialBody(typeName, feature));
        }

        source.Append(core[tailStart..]);
        Assert.Equal(expectedSha256, ComputeSha256(source.ToString()));
    }

    private static string ReadPartialBody(string typeName, string feature)
    {
        string source = Normalize(ReadSource($"{typeName}.{feature}.cs"));
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
                @"^\s*public\s+(?!static\s)[^\s{(]+\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:=>|\{)",
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

    private static string ReadSource(string fileName)
    {
        return File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.CudaSharp",
            "Memory",
            fileName));
    }
}

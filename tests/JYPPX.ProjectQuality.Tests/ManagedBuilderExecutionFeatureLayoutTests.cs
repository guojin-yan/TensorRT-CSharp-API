using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedBuilderExecutionFeatureLayoutTests
{
    private const string BuilderConfigOriginalNormalizedSha256 =
        "5366cc26418ae992a11521c43bc149c594bd1836e6c2506e4e83b2b5d5ee1d50";
    private const string ExecutionContextOriginalNormalizedSha256 =
        "c99328818c9900f8b62b0e13129ffd07a733dbff40d702d875d654fe3768d1ab";

    private static readonly string[] BuilderFeatureOrder =
    {
        "Profiles",
        "CompatibilityPresence",
        "Flags",
        "EngineCompatibility",
        "LayerDevices",
        "MemoryPools",
        "ScalarControls",
        "TimingCache"
    };

    public static TheoryData<string, string[]> BuilderFeatureMethods => new()
    {
        {
            "Profiles",
            new[] { "AddOptimizationProfile", "SetProfileStream", "SetCalibrationProfile" }
        },
        { "CompatibilityPresence", Array.Empty<string>() },
        { "Flags", new[] { "SetFlag", "ClearFlag", "GetFlag" } },
        {
            "EngineCompatibility",
            new[]
            {
                "SetEngineCapability",
                "GetEngineCapability",
                "SetPreviewFeature",
                "GetPreviewFeature",
                "SetHardwareCompatibilityLevel",
                "GetHardwareCompatibilityLevel",
                "SetRuntimePlatform",
                "GetRuntimePlatform"
            }
        },
        {
            "LayerDevices",
            new[]
            {
                "SetLayerDeviceType",
                "GetLayerDeviceType",
                "IsLayerDeviceTypeSet",
                "ResetLayerDeviceType"
            }
        },
        { "MemoryPools", new[] { "SetMemoryPoolLimit", "GetMemoryPoolLimit" } },
        {
            "ScalarControls",
            new[]
            {
                "SetOptimizationLevel",
                "GetOptimizationLevel",
                "SetProfilingVerbosity",
                "GetProfilingVerbosity",
                "SetMaxAuxStreams",
                "GetMaxAuxStreams",
                "SetAverageTimingIterations",
                "GetAverageTimingIterations",
                "SetMaxWorkspaceSizeCompatibility",
                "SetMinTimingIterationsCompatibility",
                "SetTacticSources",
                "GetTacticSources"
            }
        },
        { "TimingCache", new[] { "CreateTimingCache", "SetTimingCache" } }
    };

    public static TheoryData<string, string[]> ExecutionFeatureMethods => new()
    {
        {
            "Shapes",
            new[]
            {
                "SetInputShape",
                "SetBindingDimensions",
                "InferShapes",
                "GetTensorShape",
                "GetShapeBinding",
                "SetInputShapeBinding",
                "SetInputShapeBinding",
                "GetTensorShape64",
                "GetTensorShapeDimensionExtent64",
                "GetTensorStrides",
                "GetTensorStrides64",
                "GetTensorStrideDimensionExtent64"
            }
        },
        {
            "TensorAddresses",
            new[]
            {
                "SetTensorAddress",
                "SetInputTensorAddress",
                "SetOutputTensorAddress",
                "IsTensorAddressBound"
            }
        },
        { "DeviceMemory", new[] { "SetDeviceMemory", "SetDeviceMemoryV2", "UpdateDeviceMemorySizeForShapes" } },
        { "Events", new[] { "SetInputConsumedEvent" } },
        { "Enqueue", new[] { "EnqueueAsync" } }
    };

    [Theory]
    [MemberData(nameof(BuilderFeatureMethods))]
    public void TensorRtBuilderConfigFeaturePartialsOwnExactMethodSets(
        string feature,
        string[] expectedMethods)
    {
        string source = ReadSource("Builder", $"TensorRtBuilderConfig.{feature}.cs");
        Assert.Equal(expectedMethods, EnumeratePublicInstanceMethodNames(source));
    }

    [Theory]
    [MemberData(nameof(ExecutionFeatureMethods))]
    public void TensorRtExecutionContextFeaturePartialsOwnExactMethodSets(
        string feature,
        string[] expectedMethods)
    {
        string source = ReadSource("Execution", $"TensorRtExecutionContext.{feature}.cs");
        Assert.Equal(expectedMethods, EnumeratePublicInstanceMethodNames(source));
    }

    [Fact]
    public void TensorRtBuilderConfigPropertiesAndSharedHelpersStayWithTheirOwners()
    {
        string core = ReadSource("Builder", "TensorRtBuilderConfig.cs");
        string profiles = ReadSource("Builder", "TensorRtBuilderConfig.Profiles.cs");
        string presence = ReadSource("Builder", "TensorRtBuilderConfig.CompatibilityPresence.cs");
        string scalar = ReadSource("Builder", "TensorRtBuilderConfig.ScalarControls.cs");

        Assert.Equal(new[] { "Dispose" }, EnumeratePublicInstanceMethodNames(core));
        Assert.Contains("public int OptimizationProfileCount", profiles, StringComparison.Ordinal);
        Assert.Contains("public bool IsProfileStreamSet", profiles, StringComparison.Ordinal);
        Assert.Contains("public bool HasCalibrationProfile", profiles, StringComparison.Ordinal);
        Assert.Contains("public bool HasAlgorithmSelectorCompatibility", presence, StringComparison.Ordinal);
        Assert.Contains("public bool HasInt8CalibratorCompatibility", presence, StringComparison.Ordinal);
        Assert.Contains("public ulong MaxWorkspaceSizeCompatibilityInBytes", scalar, StringComparison.Ordinal);
        Assert.Contains("public int MinTimingIterationsCompatibility", scalar, StringComparison.Ordinal);
        Assert.Contains("private void ValidateLayer(", core, StringComparison.Ordinal);
        Assert.Contains("private void ThrowIfDisposed()", core, StringComparison.Ordinal);
        Assert.Contains("private TensorRtProgressMonitor? DetachProgressMonitor()", core, StringComparison.Ordinal);
    }

    [Fact]
    public void TensorRtExecutionContextCoreRetainsOwnerLifetimeAndSharedCleanup()
    {
        string core = ReadSource("Execution", "TensorRtExecutionContext.cs");

        Assert.Equal(new[] { "Dispose" }, EnumeratePublicInstanceMethodNames(core));
        Assert.Contains("private readonly SafeTensorRtObjectHandle _handle;", core, StringComparison.Ordinal);
        Assert.Contains("private TensorRtProfiler? _profilerKeepAlive;", core, StringComparison.Ordinal);
        Assert.Contains("private TensorRtAuxiliaryStreamHandleLease? _auxiliaryStreamLease;", core, StringComparison.Ordinal);
        Assert.Contains("private void TryClearAuxStreamsForDispose()", core, StringComparison.Ordinal);
        Assert.Contains("private void TryClearProfilerForDispose()", core, StringComparison.Ordinal);
        Assert.Contains("private TensorRtProfiler? DetachProfiler()", core, StringComparison.Ordinal);
    }

    [Fact]
    public void TensorRtBuilderConfigFeaturePartialsRecomposeTheOriginalSource()
    {
        string core = Normalize(ReadSource("Builder", "TensorRtBuilderConfig.cs"));
        (string prefix, string coreBody) = SplitClass(core, "TensorRtBuilderConfig");
        const string disposeMarker =
            "    /// <summary>\n    /// Releases the TensorRT builder-configuration handle.";
        int disposeStart = coreBody.IndexOf(disposeMarker, StringComparison.Ordinal);
        Assert.True(disposeStart >= 0);

        StringBuilder source = new();
        source.Append(prefix);
        source.Append('\n');
        source.Append(coreBody[..disposeStart]);
        foreach (string feature in BuilderFeatureOrder)
        {
            source.Append(ReadPartialBody(
                "Builder",
                $"TensorRtBuilderConfig.{feature}.cs",
                "TensorRtBuilderConfig"));
        }

        source.Append(coreBody[disposeStart..]);
        source.Append('}');
        source.Append('\n');
        Assert.Equal(BuilderConfigOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    [Fact]
    public void TensorRtExecutionContextFeaturePartialsRecomposeTheOriginalSource()
    {
        string core = Normalize(ReadSource("Execution", "TensorRtExecutionContext.cs"));
        (string prefix, string coreBody) = SplitClass(core, "TensorRtExecutionContext");
        const string disposeMarker =
            "    /// <summary>\n    /// Releases the native TensorRT resources held by this object.";
        int disposeStart = coreBody.IndexOf(disposeMarker, StringComparison.Ordinal);
        Assert.True(disposeStart >= 0);

        string shapes = ReadPartialBody(
            "Execution",
            "TensorRtExecutionContext.Shapes.cs",
            "TensorRtExecutionContext");
        const string inferMarker =
            "    /// <summary>\n    /// Runs TensorRT shape inference for the current execution context.";
        const string shapeReadMarker =
            "    /// <summary>\n    /// Gets the Tensor Shape value.";
        int inferStart = shapes.IndexOf(inferMarker, StringComparison.Ordinal);
        int shapeReadStart = shapes.IndexOf(shapeReadMarker, StringComparison.Ordinal);
        Assert.True(inferStart >= 0 && shapeReadStart > inferStart);

        string addresses = ReadPartialBody(
            "Execution",
            "TensorRtExecutionContext.TensorAddresses.cs",
            "TensorRtExecutionContext");
        const string addressReadMarker =
            "    /// <summary>\n    /// Checks whether Tensor Address Bound is true.";
        int addressReadStart = addresses.IndexOf(addressReadMarker, StringComparison.Ordinal);
        Assert.True(addressReadStart >= 0);

        StringBuilder source = new();
        source.Append(prefix);
        source.Append('\n');
        source.Append(coreBody[..disposeStart]);
        source.Append(shapes[..inferStart]);
        source.Append(addresses[..addressReadStart]);
        source.Append(ReadPartialBody(
            "Execution",
            "TensorRtExecutionContext.DeviceMemory.cs",
            "TensorRtExecutionContext"));
        source.Append(shapes[inferStart..shapeReadStart]);
        source.Append(ReadPartialBody(
            "Execution",
            "TensorRtExecutionContext.Events.cs",
            "TensorRtExecutionContext"));
        source.Append(shapes[shapeReadStart..]);
        source.Append(addresses[addressReadStart..]);
        source.Append(ReadPartialBody(
            "Execution",
            "TensorRtExecutionContext.Enqueue.cs",
            "TensorRtExecutionContext"));
        source.Append(coreBody[disposeStart..]);
        source.Append('}');
        source.Append('\n');
        Assert.Equal(ExecutionContextOriginalNormalizedSha256, ComputeSha256(source.ToString()));
    }

    private static (string Prefix, string Body) SplitClass(string source, string typeName)
    {
        int declarationStart = source.IndexOf(
            $"public sealed partial class {typeName}",
            StringComparison.Ordinal);
        int bodyStart = source.IndexOf('{', declarationStart);
        int bodyEnd = source.LastIndexOf('}');
        Assert.True(declarationStart >= 0 && bodyStart >= 0 && bodyEnd > bodyStart);
        string body = source[(bodyStart + 1)..bodyEnd];
        Assert.StartsWith("\n", body, StringComparison.Ordinal);
        return (source[..(bodyStart + 1)], body[1..]);
    }

    private static string ReadPartialBody(string module, string fileName, string typeName)
    {
        return SplitClass(Normalize(ReadSource(module, fileName)), typeName).Body;
    }

    private static string[] EnumeratePublicInstanceMethodNames(string source)
    {
        return Regex.Matches(
                source,
                @"public\s+[^\s(]+\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")
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
            "JYPPX.TensorRtSharp",
            module,
            fileName));
    }
}

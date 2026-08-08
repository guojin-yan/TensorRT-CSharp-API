using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ManagedWrapperFeatureLayoutTests
{
    private const string LayerOriginalNormalizedSha256 =
        "54cdadddd4e8344d6ab1af79c17b927dc839463acddd1729beeb0f78edcdfb5c";
    private const string NetworkOriginalNormalizedSha256 =
        "7f157345724c73c6d12c1193d1dfe64b691bcad809e687d0cb97ea68996479d1";

    private static readonly string[] LayerFeatureOrder =
    {
        "Shuffle",
        "MatrixMultiply",
        "Reduce",
        "SoftMax",
        "Unary",
        "TopK",
        "Gather",
        "ElementWise",
        "Activation",
        "Pooling",
        "Convolution",
        "Scale",
        "Padding",
        "Resize",
        "Concatenation",
        "Slice",
        "Fill"
    };

    private static readonly string[] NetworkFeatureOrder =
    {
        "Identity",
        "Constant",
        "Convolution",
        "Scale",
        "Padding",
        "ElementWise",
        "MatrixMultiply",
        "Shuffle",
        "Reduce",
        "SoftMax",
        "Unary",
        "TopK",
        "Gather",
        "Activation",
        "Pooling",
        "Resize",
        "Concatenation",
        "Slice",
        "Shape",
        "Select",
        "Fill"
    };

    public static TheoryData<string, string[]> LayerFeatureMethods => new()
    {
        {
            "Shuffle",
            new[]
            {
                "SetShuffleReshapeDimensions",
                "GetShuffleReshapeDimensions",
                "SetShuffleFirstTranspose",
                "GetShuffleFirstTranspose",
                "SetShuffleSecondTranspose",
                "GetShuffleSecondTranspose",
                "SetShuffleZeroIsPlaceholder",
                "GetShuffleZeroIsPlaceholder"
            }
        },
        { "MatrixMultiply", new[] { "SetMatrixMultiplyOperation", "GetMatrixMultiplyOperation" } },
        {
            "Reduce",
            new[]
            {
                "GetReduceOperation",
                "SetReduceOperation",
                "GetReduceAxes",
                "SetReduceAxes",
                "GetReduceKeepDimensions",
                "SetReduceKeepDimensions"
            }
        },
        { "SoftMax", new[] { "SetSoftMaxAxes", "GetSoftMaxAxes" } },
        { "Unary", new[] { "GetUnaryOperation", "SetUnaryOperation" } },
        {
            "TopK",
            new[]
            {
                "GetTopKOperation",
                "SetTopKOperation",
                "GetTopKValue",
                "SetTopKValue",
                "GetTopKAxes",
                "SetTopKAxes",
                "GetTopKIndicesType",
                "SetTopKIndicesType"
            }
        },
        { "Gather", new[] { "GetGatherAxis", "SetGatherAxis" } },
        { "ElementWise", new[] { "GetElementWiseOperation", "SetElementWiseOperation" } },
        {
            "Activation",
            new[]
            {
                "GetActivationType",
                "SetActivationType",
                "GetActivationAlpha",
                "SetActivationAlpha",
                "GetActivationBeta",
                "SetActivationBeta"
            }
        },
        {
            "Pooling",
            new[]
            {
                "GetPoolingType",
                "SetPoolingType",
                "SetPoolingWindowSize",
                "GetPoolingWindowSize",
                "SetPoolingStride",
                "GetPoolingStride",
                "SetPoolingPadding",
                "GetPoolingPadding",
                "GetPoolingBlendFactor",
                "SetPoolingBlendFactor",
                "GetPoolingAverageCountExcludesPadding",
                "SetPoolingAverageCountExcludesPadding",
                "GetPoolingPrePadding",
                "SetPoolingPrePadding",
                "GetPoolingPostPadding",
                "SetPoolingPostPadding",
                "GetPoolingPaddingMode",
                "SetPoolingPaddingMode"
            }
        },
        {
            "Convolution",
            new[]
            {
                "GetConvolutionOutputMaps",
                "SetConvolutionOutputMaps",
                "GetConvolutionGroups",
                "SetConvolutionGroups",
                "GetConvolutionStride",
                "SetConvolutionStride",
                "GetConvolutionPadding",
                "SetConvolutionPadding",
                "GetConvolutionPrePadding",
                "SetConvolutionPrePadding",
                "GetConvolutionPostPadding",
                "SetConvolutionPostPadding",
                "GetConvolutionDilation",
                "SetConvolutionDilation",
                "GetConvolutionPaddingMode",
                "SetConvolutionPaddingMode"
            }
        },
        {
            "Scale",
            new[] { "GetScaleMode", "SetScaleMode", "GetScaleChannelAxis", "SetScaleChannelAxis" }
        },
        {
            "Padding",
            new[]
            {
                "GetPaddingPrePadding",
                "SetPaddingPrePadding",
                "GetPaddingPostPadding",
                "SetPaddingPostPadding"
            }
        },
        {
            "Resize",
            new[]
            {
                "SetResizeOutputDimensions",
                "GetResizeOutputDimensions",
                "SetResizeMode",
                "GetResizeMode",
                "GetResizeAlignCorners",
                "SetResizeAlignCorners",
                "GetResizeCoordinateTransformation",
                "SetResizeCoordinateTransformation",
                "GetResizeSelectorForSinglePixel",
                "SetResizeSelectorForSinglePixel",
                "GetResizeNearestRounding",
                "SetResizeNearestRounding",
                "GetResizeCubicCoefficient",
                "SetResizeCubicCoefficient",
                "GetResizeExcludeOutside",
                "SetResizeExcludeOutside",
                "SetResizeScales",
                "GetResizeScales"
            }
        },
        { "Concatenation", new[] { "SetConcatenationAxis", "GetConcatenationAxis" } },
        {
            "Slice",
            new[]
            {
                "SetSliceStart",
                "GetSliceStart",
                "SetSliceSize",
                "GetSliceSize",
                "SetSliceStride",
                "GetSliceStride",
                "SetSliceAxes",
                "GetSliceAxes",
                "SetSliceMode",
                "GetSliceMode"
            }
        },
        {
            "Fill",
            new[]
            {
                "SetFillDimensions",
                "GetFillDimensions",
                "SetFillOperation",
                "GetFillOperation",
                "SetFillAlpha",
                "GetFillAlpha",
                "SetFillBeta",
                "GetFillBeta"
            }
        }
    };

    public static TheoryData<string, string[]> NetworkFeatureMethods => new()
    {
        { "Identity", new[] { "AddIdentity" } },
        { "Constant", new[] { "AddConstant" } },
        { "Convolution", new[] { "AddConvolution" } },
        { "Scale", new[] { "AddScale" } },
        { "Padding", new[] { "AddPadding" } },
        { "ElementWise", new[] { "AddElementWise" } },
        { "MatrixMultiply", new[] { "AddMatrixMultiply" } },
        { "Shuffle", new[] { "AddShuffle" } },
        { "Reduce", new[] { "AddReduce" } },
        { "SoftMax", new[] { "AddSoftMax" } },
        { "Unary", new[] { "AddUnary" } },
        { "TopK", new[] { "AddTopK" } },
        { "Gather", new[] { "AddGather" } },
        { "Activation", new[] { "AddActivation" } },
        { "Pooling", new[] { "AddPooling" } },
        { "Resize", new[] { "AddResize" } },
        { "Concatenation", new[] { "AddConcatenation" } },
        { "Slice", new[] { "AddSlice" } },
        { "Shape", new[] { "AddShape" } },
        { "Select", new[] { "AddSelect" } },
        { "Fill", new[] { "AddFill" } }
    };

    [Theory]
    [MemberData(nameof(LayerFeatureMethods))]
    public void TensorRtLayerFeaturePartialsOwnExactMethodSets(
        string feature,
        string[] expectedMethods)
    {
        string source = ReadWrapperSource(
            "Layers",
            $"TensorRtLayer.{feature}.cs");

        Assert.Equal(expectedMethods, EnumeratePublicInstanceMethodNames(source));
    }

    [Theory]
    [MemberData(nameof(NetworkFeatureMethods))]
    public void TensorRtNetworkFeaturePartialsOwnExactMethodSets(
        string feature,
        string[] expectedMethods)
    {
        string source = ReadWrapperSource(
            "Network",
            $"TensorRtNetworkDefinition.{feature}.cs");

        Assert.Equal(expectedMethods, EnumeratePublicInstanceMethodNames(source));
    }

    [Fact]
    public void TensorRtLayerCoreRetainsLifetimeMetadataAndSharedValidationOnly()
    {
        string source = ReadWrapperSource("Layers", "TensorRtLayer.cs");

        Assert.Equal(
            new[]
            {
                "ResetPrecision",
                "GetInput",
                "GetOutput",
                "SetOutputType",
                "GetOutputType",
                "IsOutputTypeSet",
                "ResetOutputType",
                "Dispose"
            },
            EnumeratePublicInstanceMethodNames(source));
        Assert.Contains("_ownerLease?.Clone()", source, StringComparison.Ordinal);
        Assert.Contains("internal SafeTensorRtObjectHandleLease CloneRequiredOwnerLease()", source, StringComparison.Ordinal);
        Assert.Contains("private void ValidateOutputIndex(int index)", source, StringComparison.Ordinal);
        Assert.Contains("private static void ValidateDims(TensorRtDims dims, string argumentName)", source, StringComparison.Ordinal);
    }

    [Fact]
    public void TensorRtNetworkCoreRetainsOwnershipOutputsAndSharedValidationOnly()
    {
        string source = ReadWrapperSource("Network", "TensorRtNetworkDefinition.cs");

        Assert.Equal(
            new[]
            {
                "GetFlag",
                "AddInput",
                "GetInput",
                "GetInputShape64",
                "GetInputDimensionExtent64",
                "GetOutput",
                "GetOutputShape64",
                "GetOutputDimensionExtent64",
                "GetLayer",
                "MarkOutput",
                "UnmarkOutput",
                "Dispose"
            },
            EnumeratePublicInstanceMethodNames(source));
        Assert.Contains("SafeTensorRtObjectHandleLease.Create(_handle)", source, StringComparison.Ordinal);
        Assert.Contains("private void ValidateInputTensor(TensorRtTensor tensor, string argumentName)", source, StringComparison.Ordinal);
        Assert.Contains("NativeBridgeApi.MarkNetworkOutput(Line, _handle, tensor.Handle)", source, StringComparison.Ordinal);
        Assert.Contains("NativeBridgeApi.UnmarkNetworkOutput(Line, _handle, tensor.Handle)", source, StringComparison.Ordinal);
    }

    [Fact]
    public void TensorRtLayerFeaturePartialsRecomposeTheOriginalSource()
    {
        string recomposed = RecomposeWrapper(
            "Layers",
            "TensorRtLayer",
            LayerFeatureOrder);

        Assert.Equal(LayerOriginalNormalizedSha256, ComputeSha256(recomposed));
    }

    [Fact]
    public void TensorRtNetworkFeaturePartialsRecomposeTheOriginalSource()
    {
        string recomposed = RecomposeWrapper(
            "Network",
            "TensorRtNetworkDefinition",
            NetworkFeatureOrder);

        Assert.Equal(NetworkOriginalNormalizedSha256, ComputeSha256(recomposed));
    }

    private static string RecomposeWrapper(
        string module,
        string typeName,
        IEnumerable<string> featureOrder)
    {
        string core = NormalizeLineEndings(ReadWrapperSource(module, $"{typeName}.cs"));
        int declarationStart = core.IndexOf(
            $"public sealed partial class {typeName}",
            StringComparison.Ordinal);
        int bodyStart = core.IndexOf('{', declarationStart);
        int bodyEnd = core.LastIndexOf('}');
        Assert.True(declarationStart >= 0 && bodyStart >= 0 && bodyEnd > bodyStart);

        string prefix = core[..(bodyStart + 1)];
        string coreBody = RemoveLeadingLineFeed(core[(bodyStart + 1)..bodyEnd]);
        string tailMarker = typeName == "TensorRtNetworkDefinition"
            ? "    /// <summary>\n    /// Marks the Output value."
            : "    /// <summary>\n    /// Releases the native TensorRT resources held by this object.";
        int tailStart = coreBody.IndexOf(tailMarker, StringComparison.Ordinal);
        Assert.True(tailStart >= 0, $"Missing lifetime tail marker in {typeName}.cs");

        StringBuilder source = new();
        source.Append(prefix);
        source.Append('\n');
        source.Append(coreBody[..tailStart]);
        foreach (string feature in featureOrder)
        {
            source.Append(ReadPartialBody(
                module,
                $"{typeName}.{feature}.cs",
                typeName));
        }

        source.Append(coreBody[tailStart..]);
        source.Append('}');
        source.Append('\n');
        return source.ToString();
    }

    private static string ReadPartialBody(
        string module,
        string fileName,
        string typeName)
    {
        string source = NormalizeLineEndings(ReadWrapperSource(module, fileName));
        int declarationStart = source.IndexOf(
            $"public sealed partial class {typeName}",
            StringComparison.Ordinal);
        int bodyStart = source.IndexOf('{', declarationStart);
        int bodyEnd = source.LastIndexOf('}');
        Assert.True(declarationStart >= 0 && bodyStart >= 0 && bodyEnd > bodyStart);
        return RemoveLeadingLineFeed(source[(bodyStart + 1)..bodyEnd]);
    }

    private static string RemoveLeadingLineFeed(string value)
    {
        Assert.StartsWith("\n", value, StringComparison.Ordinal);
        return value[1..];
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
        byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes(value));
        return Convert.ToHexString(digest).ToLowerInvariant();
    }

    private static string NormalizeLineEndings(string value)
    {
        return value.Replace("\r\n", "\n", StringComparison.Ordinal)
            .Replace('\r', '\n');
    }

    private static string ReadWrapperSource(string module, string fileName)
    {
        return File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp",
            module,
            fileName));
    }
}

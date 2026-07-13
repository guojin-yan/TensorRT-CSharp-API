using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class InferenceBindingSizeFallbackTests
{
    [Fact]
    public void Float_tensor_uses_data_type_size_when_optional_format_metadata_is_zero()
    {
        TensorRtEngineTensorBinding tensor = CreateBinding(TensorRtDataType.Float, bytesPerComponent: 0, componentsPerElement: 0);

        int bytes = tensor.EstimateByteSize(new TensorRtDims(new[] { 2, 4 }));

        Assert.Equal(32, bytes);
        Assert.Equal(4, tensor.EffectiveBytesPerComponent);
        Assert.Equal(1, tensor.EffectiveComponentsPerElement);
        Assert.True(tensor.UsesDataTypeSizeFallback);
    }

    [Fact]
    public void Four_bit_tensor_is_not_rounded_up_without_explicit_format_metadata()
    {
        TensorRtEngineTensorBinding tensor = CreateBinding(TensorRtDataType.Int4, bytesPerComponent: 0, componentsPerElement: 0);

        Assert.Throws<NotSupportedException>(() => tensor.EstimateByteSize(new TensorRtDims(new[] { 2, 4 })));
    }

    private static TensorRtEngineTensorBinding CreateBinding(
        TensorRtDataType dataType,
        int bytesPerComponent,
        int componentsPerElement)
    {
        return new TensorRtEngineTensorBinding(
            index: 0,
            name: "input",
            dataType,
            TensorRtIOMode.Input,
            new TensorRtDims(new[] { -1, 4 }),
            TensorRtTensorLocation.Device,
            isShapeInferenceIO: false,
            bytesPerComponent,
            componentsPerElement,
            TensorRtTensorFormat.Linear,
            formatDescription: string.Empty,
            vectorizedDimension: -1,
            profileIndex: 0,
            profileMinShape: new TensorRtDims(new[] { 1, 4 }),
            profileOptShape: new TensorRtDims(new[] { 2, 4 }),
            profileMaxShape: new TensorRtDims(new[] { 4, 4 }),
            diagnostics: Array.Empty<string>());
    }
}

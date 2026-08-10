using System;
using System.Collections.Generic;
using System.Linq;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

/// <summary>
/// Contains copied metadata for one TensorRT engine I/O tensor.
/// </summary>
public sealed class OnnxEngineBindingTensorMetadata
{
    private OnnxEngineBindingTensorMetadata(
        int index,
        string name,
        string ioMode,
        string dataType,
        IReadOnlyList<int> engineShape,
        string location,
        bool isShapeInferenceIO,
        int bytesPerComponent,
        int componentsPerElement,
        int effectiveBytesPerComponent,
        int effectiveComponentsPerElement,
        bool usesDataTypeSizeFallback,
        string format,
        string formatDescription,
        int vectorizedDimension,
        int profileIndex,
        IReadOnlyList<int>? profileMinShape,
        IReadOnlyList<int>? profileOptShape,
        IReadOnlyList<int>? profileMaxShape,
        IReadOnlyList<string> diagnostics)
    {
        Index = index;
        Name = name ?? string.Empty;
        IOMode = ioMode ?? string.Empty;
        DataType = dataType ?? string.Empty;
        EngineShape = engineShape ?? Array.Empty<int>();
        Location = location ?? string.Empty;
        IsShapeInferenceIO = isShapeInferenceIO;
        BytesPerComponent = bytesPerComponent;
        ComponentsPerElement = componentsPerElement;
        EffectiveBytesPerComponent = effectiveBytesPerComponent;
        EffectiveComponentsPerElement = effectiveComponentsPerElement;
        UsesDataTypeSizeFallback = usesDataTypeSizeFallback;
        Format = format ?? string.Empty;
        FormatDescription = formatDescription ?? string.Empty;
        VectorizedDimension = vectorizedDimension;
        ProfileIndex = profileIndex;
        ProfileMinShape = profileMinShape;
        ProfileOptShape = profileOptShape;
        ProfileMaxShape = profileMaxShape;
        Diagnostics = diagnostics ?? Array.Empty<string>();
    }

    public int Index { get; }

    public string Name { get; }

    public string IOMode { get; }

    public string DataType { get; }

    public IReadOnlyList<int> EngineShape { get; }

    public string Location { get; }

    public bool IsShapeInferenceIO { get; }

    public int BytesPerComponent { get; }

    public int ComponentsPerElement { get; }

    public int EffectiveBytesPerComponent { get; }

    public int EffectiveComponentsPerElement { get; }

    public bool UsesDataTypeSizeFallback { get; }

    public string Format { get; }

    public string FormatDescription { get; }

    public int VectorizedDimension { get; }

    public int ProfileIndex { get; }

    public IReadOnlyList<int>? ProfileMinShape { get; }

    public IReadOnlyList<int>? ProfileOptShape { get; }

    public IReadOnlyList<int>? ProfileMaxShape { get; }

    public IReadOnlyList<string> Diagnostics { get; }

    internal static OnnxEngineBindingTensorMetadata FromBinding(TensorRtEngineTensorBinding binding)
    {
        return new OnnxEngineBindingTensorMetadata(
            binding.Index,
            binding.Name,
            binding.IOMode.ToString(),
            binding.DataType.ToString(),
            binding.EngineShape.Values.ToArray(),
            binding.Location.ToString(),
            binding.IsShapeInferenceIO,
            binding.BytesPerComponent,
            binding.ComponentsPerElement,
            binding.EffectiveBytesPerComponent,
            binding.EffectiveComponentsPerElement,
            binding.UsesDataTypeSizeFallback,
            binding.Format.ToString(),
            binding.FormatDescription,
            binding.VectorizedDimension,
            binding.ProfileIndex,
            binding.ProfileMinShape?.Values.ToArray(),
            binding.ProfileOptShape?.Values.ToArray(),
            binding.ProfileMaxShape?.Values.ToArray(),
            binding.Diagnostics?.ToArray() ?? Array.Empty<string>());
    }
}

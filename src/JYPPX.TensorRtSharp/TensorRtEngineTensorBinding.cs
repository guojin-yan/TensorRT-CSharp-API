using System;
using System.Collections.Generic;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Describes one TensorRT engine I/O tensor for deployment binding diagnostics.
/// 描述一个 TensorRT engine I/O tensor 的部署绑定诊断信息。
/// </summary>
public sealed class TensorRtEngineTensorBinding
{
    /// <summary>
    /// Creates a TensorRT engine tensor binding diagnostic snapshot.
    /// 创建 TensorRT engine tensor 绑定诊断快照。
    /// </summary>
    public TensorRtEngineTensorBinding(
        int index,
        string name,
        TensorRtDataType dataType,
        TensorRtIOMode ioMode,
        TensorRtDims engineShape,
        TensorRtTensorLocation location,
        bool isShapeInferenceIO,
        int bytesPerComponent,
        int componentsPerElement,
        TensorRtTensorFormat format,
        string formatDescription,
        int vectorizedDimension,
        int profileIndex,
        TensorRtDims? profileMinShape,
        TensorRtDims? profileOptShape,
        TensorRtDims? profileMaxShape,
        IReadOnlyList<string> diagnostics)
    {
        Index = index;
        Name = name;
        DataType = dataType;
        IOMode = ioMode;
        EngineShape = engineShape;
        Location = location;
        IsShapeInferenceIO = isShapeInferenceIO;
        BytesPerComponent = bytesPerComponent;
        ComponentsPerElement = componentsPerElement;
        Format = format;
        FormatDescription = formatDescription;
        VectorizedDimension = vectorizedDimension;
        ProfileIndex = profileIndex;
        ProfileMinShape = profileMinShape;
        ProfileOptShape = profileOptShape;
        ProfileMaxShape = profileMaxShape;
        Diagnostics = diagnostics;
    }

    /// <summary>
    /// Gets the engine I/O tensor index.
    /// 获取 engine I/O tensor 索引。
    /// </summary>
    public int Index { get; }

    /// <summary>
    /// Gets the TensorRT tensor name.
    /// 获取 TensorRT tensor 名称。
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the TensorRT tensor data type.
    /// 获取 TensorRT tensor 数据类型。
    /// </summary>
    public TensorRtDataType DataType { get; }

    /// <summary>
    /// Gets whether this tensor is an input or output.
    /// 获取该 tensor 是输入还是输出。
    /// </summary>
    public TensorRtIOMode IOMode { get; }

    /// <summary>
    /// Gets the static or profile-dependent shape recorded by the engine.
    /// 获取 engine 记录的静态或 profile 相关 shape。
    /// </summary>
    public TensorRtDims EngineShape { get; }

    /// <summary>
    /// Gets the expected TensorRT memory location for the tensor.
    /// 获取该 tensor 期望使用的 TensorRT 内存位置。
    /// </summary>
    public TensorRtTensorLocation Location { get; }

    /// <summary>
    /// Gets whether this tensor participates in TensorRT shape inference I/O.
    /// 获取该 tensor 是否参与 TensorRT shape inference I/O。
    /// </summary>
    public bool IsShapeInferenceIO { get; }

    /// <summary>
    /// Gets the number of bytes per scalar component for this tensor.
    /// 获取该 tensor 每个标量 component 的字节数。
    /// </summary>
    public int BytesPerComponent { get; }

    /// <summary>
    /// Gets the number of scalar components per tensor element.
    /// 获取该 tensor 每个元素包含的 component 数量。
    /// </summary>
    public int ComponentsPerElement { get; }

    /// <summary>
    /// Gets the usable bytes-per-component value, falling back to the scalar data type when
    /// older TensorRT lines do not expose optional format metadata.
    /// 获取可用的每 component 字节数；旧 TensorRT 版本未提供可选格式元数据时，
    /// 按标量数据类型安全回退。
    /// </summary>
    public int EffectiveBytesPerComponent =>
        BytesPerComponent > 0 ? BytesPerComponent : GetDefaultBytesPerComponent(DataType);

    /// <summary>
    /// Gets the usable components-per-element value. Non-vectorized tensors fall back to one
    /// component when older TensorRT lines return zero for optional format metadata.
    /// 获取可用的每元素 component 数；旧 TensorRT 版本对可选格式元数据返回零时，
    /// 非向量化 tensor 安全回退为一个 component。
    /// </summary>
    public int EffectiveComponentsPerElement => ComponentsPerElement > 0 ? ComponentsPerElement : 1;

    /// <summary>
    /// Gets whether byte-size estimation uses a data-type metadata fallback.
    /// 获取字节数估算是否使用了数据类型元数据回退。
    /// </summary>
    public bool UsesDataTypeSizeFallback => BytesPerComponent <= 0 || ComponentsPerElement <= 0;

    /// <summary>
    /// Gets the TensorRT tensor format enum value.
    /// 获取 TensorRT tensor format 枚举值。
    /// </summary>
    public TensorRtTensorFormat Format { get; }

    /// <summary>
    /// Gets TensorRT's human-readable tensor format description.
    /// 获取 TensorRT 返回的可读 tensor format 描述。
    /// </summary>
    public string FormatDescription { get; }

    /// <summary>
    /// Gets the vectorized dimension reported by TensorRT, or -1 when not vectorized.
    /// 获取 TensorRT 报告的 vectorized dimension；未向量化时通常为 -1。
    /// </summary>
    public int VectorizedDimension { get; }

    /// <summary>
    /// Gets the optimization profile index used for profile-specific metadata.
    /// 获取用于 profile 相关元数据查询的 optimization profile 索引。
    /// </summary>
    public int ProfileIndex { get; }

    /// <summary>
    /// Gets the minimum profile shape when TensorRT exposes it for this tensor.
    /// 获取 TensorRT 能提供时该 tensor 的最小 profile shape。
    /// </summary>
    public TensorRtDims? ProfileMinShape { get; }

    /// <summary>
    /// Gets the optimum profile shape when TensorRT exposes it for this tensor.
    /// 获取 TensorRT 能提供时该 tensor 的最优 profile shape。
    /// </summary>
    public TensorRtDims? ProfileOptShape { get; }

    /// <summary>
    /// Gets the maximum profile shape when TensorRT exposes it for this tensor.
    /// 获取 TensorRT 能提供时该 tensor 的最大 profile shape。
    /// </summary>
    public TensorRtDims? ProfileMaxShape { get; }

    /// <summary>
    /// Gets non-fatal diagnostics collected while querying optional metadata.
    /// 获取查询可选元数据时收集到的非致命诊断信息。
    /// </summary>
    public IReadOnlyList<string> Diagnostics { get; }
    /// <summary>
    /// Estimates the required byte count for a concrete runtime shape.
    /// 根据具体运行时 shape 估算该 tensor 需要的字节数。
    /// </summary>
    /// <param name="runtimeShape">The concrete runtime shape. 具体运行时 shape。</param>
    /// <returns>The estimated byte count. 估算字节数。</returns>
    public int EstimateByteSize(TensorRtDims runtimeShape)
    {
        if (runtimeShape == null)
        {
            throw new ArgumentNullException(nameof(runtimeShape));
        }

        if (runtimeShape.Values.Length == 0)
        {
            throw new ArgumentException("Runtime shape must contain at least one dimension.", nameof(runtimeShape));
        }

        long elementCount = 1;
        foreach (int value in runtimeShape.Values)
        {
            if (value <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(runtimeShape), "Runtime shape must be concrete and positive.");
            }

            elementCount = checked(elementCount * value);
        }

        int bytesPerComponent = EffectiveBytesPerComponent;
        int componentsPerElement = EffectiveComponentsPerElement;
        if (bytesPerComponent <= 0)
        {
            throw new NotSupportedException(
                $"Tensor '{Name}' data type {DataType} does not have an integral byte-size fallback.");
        }

        long bytes = checked(elementCount * bytesPerComponent * componentsPerElement);
        if (bytes <= 0 || bytes > int.MaxValue)
        {
            throw new InvalidOperationException("Estimated tensor byte size exceeds the managed allocation range.");
        }

        return (int)bytes;
    }

    private static int GetDefaultBytesPerComponent(TensorRtDataType dataType)
    {
        return dataType switch
        {
            TensorRtDataType.Float => sizeof(float),
            TensorRtDataType.Half => sizeof(ushort),
            TensorRtDataType.Int8 => sizeof(byte),
            TensorRtDataType.Int32 => sizeof(int),
            TensorRtDataType.Bool => sizeof(byte),
            TensorRtDataType.UInt8 => sizeof(byte),
            TensorRtDataType.Float8 => sizeof(byte),
            TensorRtDataType.BFloat16 => sizeof(ushort),
            TensorRtDataType.Int64 => sizeof(long),
            TensorRtDataType.E8M0 => sizeof(byte),
            TensorRtDataType.Int4 => 0,
            TensorRtDataType.Float4 => 0,
            _ => 0
        };
    }
}

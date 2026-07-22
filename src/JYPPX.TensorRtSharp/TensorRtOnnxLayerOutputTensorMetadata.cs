using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Contains copied metadata for one ONNX parser layer output tensor without retaining the parser-owned tensor pointer.
/// 包含一个 ONNX parser layer 输出张量的复制元数据，不保留 parser 拥有的 tensor 指针。
/// </summary>
public sealed class TensorRtOnnxLayerOutputTensorMetadata
{
    internal TensorRtOnnxLayerOutputTensorMetadata(
        TensorRtApiLine line,
        string layerName,
        long outputIndex,
        string tensorName,
        TensorRtDims64 shape,
        TensorRtDataType dataType,
        TensorRtTensorLocation location,
        TensorRtTensorFormats allowedFormats,
        bool hasDynamicDimension,
        bool isShapeTensor,
        bool isExecutionTensor,
        bool isNetworkInput,
        bool isNetworkOutput)
    {
        Line = line;
        LayerName = layerName;
        OutputIndex = outputIndex;
        TensorName = tensorName;
        Shape = shape;
        DataType = dataType;
        Location = location;
        AllowedFormats = allowedFormats;
        HasDynamicDimension = hasDynamicDimension;
        IsShapeTensor = isShapeTensor;
        IsExecutionTensor = isExecutionTensor;
        IsNetworkInput = isNetworkInput;
        IsNetworkOutput = isNetworkOutput;
    }

    /// <summary>Gets the TensorRT adapter line. 获取 TensorRT 适配版本线。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the queried ONNX layer name. 获取查询使用的 ONNX layer 名称。</summary>
    public string LayerName { get; }

    /// <summary>Gets the zero-based layer output index. 获取从零开始的 layer 输出索引。</summary>
    public long OutputIndex { get; }

    /// <summary>Gets the copied TensorRT tensor name. 获取复制后的 TensorRT tensor 名称。</summary>
    public string TensorName { get; }

    /// <summary>Gets the copied shape with 64-bit extents. 获取使用 64 位 extent 的复制 shape。</summary>
    public TensorRtDims64 Shape { get; }

    /// <summary>Gets the copied tensor data type. 获取复制后的 tensor 数据类型。</summary>
    public TensorRtDataType DataType { get; }

    /// <summary>Gets the copied tensor memory location. 获取复制后的 tensor 内存位置。</summary>
    public TensorRtTensorLocation Location { get; }

    /// <summary>Gets the copied allowed-format bitmask. 获取复制后的 allowed-format 位掩码。</summary>
    public TensorRtTensorFormats AllowedFormats { get; }

    /// <summary>Gets whether the copied shape contains a dynamic extent or unknown rank. 获取复制 shape 是否包含动态 extent 或未知 rank。</summary>
    public bool HasDynamicDimension { get; }

    /// <summary>Gets whether TensorRT classifies the tensor as a shape tensor. 获取 TensorRT 是否将其标记为 shape tensor。</summary>
    public bool IsShapeTensor { get; }

    /// <summary>Gets whether TensorRT classifies the tensor as an execution tensor. 获取 TensorRT 是否将其标记为 execution tensor。</summary>
    public bool IsExecutionTensor { get; }

    /// <summary>Gets whether the tensor is a network input. 获取该 tensor 是否为 network input。</summary>
    public bool IsNetworkInput { get; }

    /// <summary>Gets whether the tensor is a network output. 获取该 tensor 是否为 network output。</summary>
    public bool IsNetworkOutput { get; }

    /// <summary>Gets whether all metadata is copied and pointer-free. 获取元数据是否均已复制且不包含指针。</summary>
    public bool PointerFreeCopiedMetadata => true;

    /// <summary>Gets whether this snapshot retains the native tensor. 获取该 snapshot 是否保留 native tensor。</summary>
    public bool RetainsNativeTensor => false;

    /// <summary>Gets the evidence classification for this snapshot. 获取该 snapshot 的证据分类。</summary>
    public string EvidenceKind => "copied-readonly-diagnostics";

    /// <summary>Gets whether this snapshot is runtime execution proof. 获取该 snapshot 是否为 runtime execution proof。</summary>
    public bool IsRuntimeExecutionProof => false;

    /// <summary>Gets whether this snapshot can promote public release proof. 获取该 snapshot 是否可晋级 public release proof。</summary>
    public bool CanPromoteReleaseProof => false;

    /// <summary>Gets whether this snapshot permits deleting the original deferred record. 获取该 snapshot 是否允许删除原始 deferred 记录。</summary>
    public bool CanDeleteDeferredRecord => false;

    /// <summary>Returns a compact diagnostic representation. 返回紧凑诊断表示。</summary>
    public override string ToString()
    {
        return $"Layer={LayerName}[{OutputIndex}] Tensor={TensorName} Shape={Shape} Type={DataType} " +
               $"Location={Location} Formats={AllowedFormats} Dynamic={HasDynamicDimension} " +
               $"ShapeTensor={IsShapeTensor} ExecutionTensor={IsExecutionTensor} " +
               $"NetworkInput={IsNetworkInput} NetworkOutput={IsNetworkOutput}";
    }
}

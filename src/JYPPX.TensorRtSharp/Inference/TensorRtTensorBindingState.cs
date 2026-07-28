namespace JYPPX.TensorRtSharp;

/// <summary>
/// Describes one engine tensor's binding state inside a TensorRT execution context.
/// 描述 TensorRT execution context 中某个 engine tensor 的绑定状态。
/// </summary>
public sealed class TensorRtTensorBindingState
{
    /// <summary>
    /// Creates a tensor binding state snapshot.
    /// 创建一个 tensor 绑定状态快照。
    /// </summary>
    public TensorRtTensorBindingState(
        int index,
        string name,
        TensorRtDataType dataType,
        TensorRtIOMode ioMode,
        TensorRtDims engineShape,
        TensorRtDims? contextShape,
        TensorRtDims? contextStrides,
        bool isAddressBound,
        long? maxOutputSizeInBytes,
        string? diagnostic)
    {
        Index = index;
        Name = name;
        DataType = dataType;
        IOMode = ioMode;
        EngineShape = engineShape;
        ContextShape = contextShape;
        ContextStrides = contextStrides;
        IsAddressBound = isAddressBound;
        MaxOutputSizeInBytes = maxOutputSizeInBytes;
        Diagnostic = diagnostic;
    }

    /// <summary>
    /// Gets the engine tensor index.
    /// 获取 engine tensor 索引。
    /// </summary>
    public int Index { get; }

    /// <summary>
    /// Gets the engine tensor name.
    /// 获取 engine tensor 名称。
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the TensorRT data type.
    /// 获取 TensorRT 数据类型。
    /// </summary>
    public TensorRtDataType DataType { get; }

    /// <summary>
    /// Gets whether the tensor is an input or output tensor.
    /// 获取该 tensor 是输入还是输出。
    /// </summary>
    public TensorRtIOMode IOMode { get; }

    /// <summary>
    /// Gets the shape recorded by the engine.
    /// 获取 engine 中记录的形状。
    /// </summary>
    public TensorRtDims EngineShape { get; }

    /// <summary>
    /// Gets the currently resolved execution-context shape.
    /// 获取当前 execution context 已解析的形状。
    /// </summary>
    public TensorRtDims? ContextShape { get; }

    /// <summary>
    /// Gets the currently resolved execution-context strides.
    /// 获取当前 execution context 已解析的 strides。
    /// </summary>
    public TensorRtDims? ContextStrides { get; }

    /// <summary>
    /// Gets whether a device address is bound for this tensor.
    /// 获取该 tensor 是否已经绑定设备地址。
    /// </summary>
    public bool IsAddressBound { get; }

    /// <summary>
    /// Gets TensorRT's reported maximum output size in bytes when available.
    /// 获取 TensorRT 可报告时的最大输出字节数。
    /// </summary>
    public long? MaxOutputSizeInBytes { get; }

    /// <summary>
    /// Gets a non-fatal diagnostic collected while reading this tensor state.
    /// 获取读取该 tensor 状态时收集到的非致命诊断信息。
    /// </summary>
    public string? Diagnostic { get; }

    /// <summary>
    /// Gets whether this tensor needs a device address before enqueue.
    /// 获取该 tensor 在 enqueue 前是否需要绑定设备地址。
    /// </summary>
    public bool RequiresAddress => IOMode == TensorRtIOMode.Input || IOMode == TensorRtIOMode.Output;

    /// <summary>
    /// Gets whether the current context shape still contains unresolved dynamic dimensions.
    /// 获取当前 context shape 是否仍包含未解析的动态维度。
    /// </summary>
    public bool HasUnresolvedContextDimension
    {
        get
        {
            if (ContextShape == null)
            {
                return true;
            }

            foreach (int value in ContextShape.Values)
            {
                if (value < 0)
                {
                    return true;
                }
            }

            return false;
        }
    }
}

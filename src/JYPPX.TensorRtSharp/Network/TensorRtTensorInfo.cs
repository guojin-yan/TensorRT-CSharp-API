namespace JYPPX.TensorRtSharp;

/// <summary>
/// Describes one TensorRT input or output tensor.
/// 描述一个 TensorRT 输入或输出 tensor。
/// </summary>
public sealed class TensorRtTensorInfo
{
    /// <summary>
    /// Initializes tensor metadata.
    /// 初始化 tensor 元数据。
    /// </summary>
    /// <param name="index">The zero-based tensor index. 从零开始的 tensor 索引。</param>
    /// <param name="name">The TensorRT tensor name. TensorRT tensor 名称。</param>
    /// <param name="dataType">The TensorRT tensor data type. TensorRT tensor 数据类型。</param>
    /// <param name="ioMode">The tensor input/output mode. tensor 输入输出模式。</param>
    /// <param name="shape">The tensor shape. tensor 形状。</param>
    public TensorRtTensorInfo(int index, string name, TensorRtDataType dataType, TensorRtIOMode ioMode, TensorRtDims shape)
    {
        Index = index;
        Name = name;
        DataType = dataType;
        IOMode = ioMode;
        Shape = shape;
    }

    /// <summary>
    /// Gets the zero-based tensor index.
    /// 获取从零开始的 tensor 索引。
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
    /// Gets the tensor input/output mode.
    /// 获取 tensor 输入输出模式。
    /// </summary>
    public TensorRtIOMode IOMode { get; }

    /// <summary>
    /// Gets the tensor shape.
    /// 获取 tensor 形状。
    /// </summary>
    public TensorRtDims Shape { get; }
}

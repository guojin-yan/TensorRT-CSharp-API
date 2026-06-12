namespace JYPPX.TensorRtSharp;

public sealed class TensorRtTensorInfo
{
    public TensorRtTensorInfo(int index, string name, TensorRtDataType dataType, TensorRtIOMode ioMode, TensorRtDims shape)
    {
        Index = index;
        Name = name;
        DataType = dataType;
        IOMode = ioMode;
        Shape = shape;
    }

    public int Index { get; }

    public string Name { get; }

    public TensorRtDataType DataType { get; }

    public TensorRtIOMode IOMode { get; }

    public TensorRtDims Shape { get; }
}


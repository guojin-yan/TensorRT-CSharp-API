using System;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Contains caller-owned data and metadata copied from a TensorRT 8 Caffe binaryproto blob.
/// 包含从 TensorRT 8 Caffe binaryproto blob 复制的 caller-owned 数据与元数据。
/// </summary>
public sealed class TensorRtCaffeBinaryProtoSnapshot
{
    private readonly byte[] _data;

    internal TensorRtCaffeBinaryProtoSnapshot(
        TensorRtApiLine line,
        string fileName,
        TensorRtDims dimensions,
        TensorRtDataType dataType,
        byte[] data)
    {
        Line = line;
        FileName = fileName;
        Dimensions = dimensions;
        DataType = dataType;
        _data = (byte[])data.Clone();
    }

    /// <summary>Gets the TensorRT adapter line. 获取 TensorRT 适配版本线。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the source file name without its directory. 获取不含目录的源文件名。</summary>
    public string FileName { get; }

    /// <summary>Gets copied binaryproto dimensions. 获取复制后的 binaryproto 维度。</summary>
    public TensorRtDims Dimensions { get; }

    /// <summary>Gets the copied TensorRT data type. 获取复制后的 TensorRT 数据类型。</summary>
    public TensorRtDataType DataType { get; }

    /// <summary>Gets the copied data length in bytes. 获取复制数据的字节长度。</summary>
    public int DataLength => _data.Length;

    /// <summary>Gets a new caller-owned copy of the binaryproto data. 获取 binaryproto 数据的新 caller-owned 副本。</summary>
    public byte[] Data => (byte[])_data.Clone();

    /// <summary>Gets whether the native parser, blob, or data pointer is retained. 获取是否保留 native parser、blob 或数据指针。</summary>
    public bool RetainsNativeObject => false;

    /// <summary>Gets whether all returned state is copied and pointer-free. 获取返回状态是否均已复制且不含指针。</summary>
    public bool PointerFreeCopiedData => true;

    /// <summary>Gets whether process-global protobuf shutdown is called. 获取是否调用进程级 protobuf shutdown。</summary>
    public bool CallsProcessGlobalProtobufShutdown => false;

    /// <summary>Formats the copied metadata. 格式化复制元数据。</summary>
    public override string ToString()
    {
        return $"File={FileName} Shape={Dimensions} Type={DataType} Bytes={DataLength}";
    }
}

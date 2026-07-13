using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtHostMemory
{
    /// <summary>
    /// Gets the TensorRT data type reported for this host-memory block.
    /// 获取该 host-memory 内存块由 TensorRT 报告的数据类型。
    /// </summary>
    /// <remarks>
    /// This metadata endpoint is available for TensorRT 8, TensorRT 10, and TensorRT 11 adapters.
    /// 当前该属性通过 TensorRT 8、TensorRT 10 和 TensorRT 11 适配线接入，只复制标量数据类型，不暴露 host-memory 指针。
    /// </remarks>
    public TensorRtDataType DataType => NativeBridgeApi.GetHostMemoryDataType(Line, _handle);
}

using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtLayer
{
    /// <summary>
    /// Gets the TensorRT 11 attention object associated with an attention input/output boundary layer.
    /// 获取与 TensorRT 11 attention 输入/输出边界层关联的 attention 对象。
    /// </summary>
    /// <returns>A network-owned attention reference wrapper. / 返回由网络持有生命周期的 attention 引用包装器。</returns>
    public TensorRtAttention GetAttentionFromBoundary()
    {
        return new TensorRtAttention(Line, NativeBridgeApi.GetAttentionFromBoundaryLayer(Line, _handle));
    }
}

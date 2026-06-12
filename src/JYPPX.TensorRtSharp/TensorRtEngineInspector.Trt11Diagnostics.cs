using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtEngineInspector
{
    /// <summary>
    /// Gets layer information for a single TensorRT engine layer.
    /// 获取指定 TensorRT engine layer 的诊断信息。
    /// </summary>
    /// <param name="layerIndex">Zero-based layer index. / 从零开始的 layer 索引。</param>
    /// <param name="format">The output format requested from TensorRT. / 向 TensorRT 请求的诊断信息格式。</param>
    /// <returns>A copied UTF-8 string with TensorRT layer information. / 从 TensorRT 复制出的 UTF-8 layer 信息字符串。</returns>
    public string GetLayerInformation(int layerIndex, TensorRtLayerInformationFormat format = TensorRtLayerInformationFormat.Oneline)
    {
        return NativeBridgeApi.GetEngineInspectorLayerInformation(Line, _handle, layerIndex, format);
    }

    /// <summary>
    /// Gets whether this inspector currently uses an execution context as the inspection source.
    /// 获取当前 inspector 是否已经关联 execution context 作为检查来源。
    /// </summary>
    public bool HasExecutionContext => NativeBridgeApi.HasEngineInspectorExecutionContext(Line, _handle);

    /// <summary>
    /// Clears the execution context associated with this inspector.
    /// 清除当前 inspector 关联的 execution context。
    /// </summary>
    public void ClearExecutionContext()
    {
        NativeBridgeApi.ClearEngineInspectorExecutionContext(Line, _handle);
    }

    /// <summary>
    /// Gets whether TensorRT reports an error recorder attached to this inspector.
    /// 获取 TensorRT 是否报告当前 inspector 绑定了 error recorder。
    /// </summary>
    public bool HasErrorRecorder => NativeBridgeApi.HasEngineInspectorErrorRecorder(Line, _handle);

    /// <summary>
    /// Clears the error recorder attached to this engine inspector when the active TensorRT line supports it.
    /// 在当前 TensorRT 版本线支持时，清除绑定到该 engine inspector 的 error recorder。
    /// </summary>
    /// <remarks>
    /// The managed wrapper does not expose user-owned TensorRT error-recorder callbacks yet; this method only clears a recorder TensorRT may report.
    /// 托管层目前尚未向普通用户暴露自定义 TensorRT error-recorder callback；该方法仅清理 TensorRT 报告的 recorder。
    /// </remarks>
    public void ClearErrorRecorder()
    {
        NativeBridgeApi.ClearEngineInspectorErrorRecorder(Line, _handle);
    }
}

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
    /// <remarks>
    /// Supported on TensorRT 8, TensorRT 10, and TensorRT 11 through a safe boolean boundary.
    /// 通过安全 bool 边界支持 TensorRT 8、TensorRT 10 和 TensorRT 11。
    /// </remarks>
    public bool HasErrorRecorder => NativeBridgeApi.HasEngineInspectorErrorRecorder(Line, _handle);

    /// <summary>
    /// Attempts to collect a copied read-only snapshot from this inspector's error recorder.
    /// 尝试从当前 inspector 的 error recorder 采集只读托管快照。
    /// </summary>
    /// <param name="snapshot">The copied snapshot. 已复制到托管内存的快照。</param>
    /// <returns><see langword="true"/> when a recorder is attached. / 当前附加 recorder 时返回 <see langword="true"/>。</returns>
    /// <remarks>
    /// The snapshot does not expose a borrowed recorder pointer or change recorder ownership.
    /// 该快照不暴露 borrowed recorder 指针，也不改变 recorder ownership。
    /// </remarks>
    public bool TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot)
    {
        snapshot = NativeBridgeApi.GetEngineInspectorErrorRecorderSnapshot(Line, _handle);
        return snapshot.HasRecorder;
    }

    /// <summary>
    /// Clears the error recorder attached to this engine inspector.
    /// 清除绑定到该 engine inspector 的 error recorder。
    /// </summary>
    /// <remarks>
    /// Supported on TensorRT 8, TensorRT 10, and TensorRT 11. The managed wrapper does not expose user-owned TensorRT error-recorder callbacks or borrowed recorder pointers.
    /// 支持 TensorRT 8、TensorRT 10 和 TensorRT 11。托管层不暴露用户自定义 TensorRT error-recorder callback，也不暴露 borrowed recorder 指针。
    /// </remarks>
    public void ClearErrorRecorder()
    {
        NativeBridgeApi.ClearEngineInspectorErrorRecorder(Line, _handle);
    }
}

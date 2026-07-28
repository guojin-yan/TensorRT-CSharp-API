using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtEngine
{
    /// <summary>
    /// Gets whether a native TensorRT error recorder is attached to this engine.
    /// 获取当前 engine 是否绑定了 TensorRT 原生 error recorder；支持 TensorRT 8/10/11，不会暴露 recorder 指针或接管其生命周期。
    /// </summary>
    public bool HasErrorRecorder => NativeBridgeApi.HasEngineErrorRecorder(Line, _handle);

    /// <summary>
    /// Clears the native TensorRT error recorder attached to this engine.
    /// 清除当前 engine 上绑定的 TensorRT 原生 error recorder；支持 TensorRT 8/10/11，不会销毁 recorder 或接管其生命周期。
    /// </summary>
    public void ClearErrorRecorder()
    {
        NativeBridgeApi.ClearEngineErrorRecorder(Line, _handle);
    }

    /// <summary>
    /// Attempts to collect a copied read-only snapshot from this engine's TensorRT error recorder.
    /// 尝试从当前 engine 的 TensorRT error recorder 采集只读托管快照。
    /// </summary>
    /// <param name="snapshot">The copied snapshot. 已复制到托管内存的快照。</param>
    /// <returns><c>true</c> when a recorder was attached; otherwise <c>false</c>. 附加了 recorder 时返回 <c>true</c>，否则返回 <c>false</c>。</returns>
    /// <remarks>
    /// This method does not expose, retain, increment, decrement, or destroy the native recorder pointer.
    /// 此方法不会暴露、持有、增加引用、减少引用或销毁原生 recorder 指针；支持 TensorRT 8/10/11。
    /// </remarks>
    public bool TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot)
    {
        snapshot = NativeBridgeApi.GetEngineErrorRecorderSnapshot(Line, _handle);
        return snapshot.HasRecorder;
    }

    /// <summary>
    /// Gets the input tensor aliased by a TensorRT 11 output tensor, when plugin I/O aliasing is used.
    /// 在使用插件 I/O aliasing 时，获取 TensorRT 11 输出 tensor 所别名引用的输入 tensor 名称。
    /// </summary>
    /// <param name="tensorName">The output tensor name to query. / 要查询的输出 tensor 名称。</param>
    /// <returns>
    /// The aliased input tensor name, or an empty string when TensorRT reports no alias.
    /// 被别名引用的输入 tensor 名称；如果 TensorRT 未报告 alias，则返回空字符串。
    /// </returns>
    public string GetAliasedInputTensorName(string tensorName)
    {
        return NativeBridgeApi.GetEngineAliasedInputTensor(Line, _handle, tensorName);
    }
}

using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtEngine
{
    /// <summary>
    /// Gets whether a native TensorRT error recorder is attached to this engine.
    /// 获取当前 engine 是否绑定了 TensorRT 原生 error recorder。
    /// </summary>
    public bool HasErrorRecorder => NativeBridgeApi.HasEngineErrorRecorder(Line, _handle);

    /// <summary>
    /// Clears the native TensorRT error recorder attached to this engine.
    /// 清除当前 engine 上绑定的 TensorRT 原生 error recorder。
    /// </summary>
    public void ClearErrorRecorder()
    {
        NativeBridgeApi.ClearEngineErrorRecorder(Line, _handle);
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

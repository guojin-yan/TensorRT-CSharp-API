using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtExecutionContext
{
    /// <summary>
    /// Gets whether a native TensorRT error recorder is attached to this execution context.
    /// 获取当前 execution context 是否绑定了 TensorRT 原生 error recorder。
    /// </summary>
    public bool HasErrorRecorder => NativeBridgeApi.HasExecutionContextErrorRecorder(Line, _handle);

    /// <summary>
    /// Clears the native TensorRT error recorder attached to this execution context.
    /// 清除当前 execution context 上绑定的 TensorRT 原生 error recorder。
    /// </summary>
    public void ClearErrorRecorder()
    {
        NativeBridgeApi.ClearExecutionContextErrorRecorder(Line, _handle);
    }
}

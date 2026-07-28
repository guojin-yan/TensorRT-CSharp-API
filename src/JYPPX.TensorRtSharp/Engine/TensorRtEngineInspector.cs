using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Managed wrapper around a TensorRT engine inspector.
/// TensorRT engine inspector 的托管封装。
/// </summary>
public sealed partial class TensorRtEngineInspector : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;

    internal TensorRtEngineInspector(TensorRtApiLine line, SafeTensorRtObjectHandle handle)
    {
        Line = line;
        _handle = handle;
    }

    /// <summary>
    /// Gets the TensorRT API line used by this inspector.
    /// 获取当前 inspector 使用的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets human-readable engine information from TensorRT.
    /// 从 TensorRT 获取可读的 engine 信息。
    /// </summary>
    /// <param name="format">The layer information format. layer 信息格式。</param>
    /// <returns>The engine information text. engine 信息文本。</returns>
    public string GetEngineInformation(TensorRtLayerInformationFormat format = TensorRtLayerInformationFormat.Oneline)
    {
        return NativeBridgeApi.GetEngineInformation(Line, _handle, format);
    }

    /// <summary>
    /// Attaches an execution context to the inspector for context-aware reporting.
    /// 为 inspector 附加一个 execution context，以支持带上下文的信息输出。
    /// </summary>
    /// <param name="context">The execution context to attach. 要附加的 execution context。</param>
    public void SetExecutionContext(TensorRtExecutionContext context)
    {
        if (context == null)
        {
            throw new ArgumentNullException(nameof(context));
        }

        if (context.Line != Line)
        {
            throw new ArgumentException("Execution context must belong to the same TensorRT API line as the engine inspector.", nameof(context));
        }

        NativeBridgeApi.SetEngineInspectorExecutionContext(Line, _handle, context.Handle);
    }

    /// <summary>
    /// Releases the TensorRT engine-inspector handle.
    /// 释放 TensorRT engine inspector 句柄。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}

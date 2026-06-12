using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtEngineInspector : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;

    internal TensorRtEngineInspector(TensorRtApiLine line, SafeTensorRtObjectHandle handle)
    {
        Line = line;
        _handle = handle;
    }

    public TensorRtApiLine Line { get; }

    public string GetEngineInformation(TensorRtLayerInformationFormat format = TensorRtLayerInformationFormat.Oneline)
    {
        return NativeBridgeApi.GetEngineInformation(Line, _handle, format);
    }

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

    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}

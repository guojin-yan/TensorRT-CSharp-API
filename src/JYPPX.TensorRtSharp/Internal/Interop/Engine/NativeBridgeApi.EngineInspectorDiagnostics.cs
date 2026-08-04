using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static string GetEngineInspectorLayerInformation(TensorRtApiLine line, SafeTensorRtObjectHandle inspector, int layerIndex, TensorRtLayerInformationFormat format)
    {
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_inspector_get_layer_information(inspector, layerIndex, (int)format, buffer, size, out required),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_inspector_get_layer_information(inspector, layerIndex, (int)format, buffer, size, out required),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_inspector_get_layer_information(inspector, layerIndex, (int)format, buffer, size, out required),
                _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
            },
            "Engine inspector layer information is too large for the managed buffer.");
    }

    public static bool HasEngineInspectorExecutionContext(TensorRtApiLine line, SafeTensorRtObjectHandle inspector)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(HasEngineInspectorExecutionContext));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_engine_inspector_has_execution_context(inspector, out int hasContext);
        NativeStatus.ThrowIfFailed(status);
        return hasContext != 0;
    }

    public static void ClearEngineInspectorExecutionContext(TensorRtApiLine line, SafeTensorRtObjectHandle inspector)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(ClearEngineInspectorExecutionContext));
        NativeStatus.ThrowIfFailed(NativeMethodsTensorRt.jyppx_trt11_engine_inspector_clear_execution_context(inspector));
    }

    public static bool HasEngineInspectorErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle inspector)
    {
        int hasRecorder = 0;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_inspector_has_error_recorder(inspector, out hasRecorder),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_inspector_has_error_recorder(inspector, out hasRecorder),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_inspector_has_error_recorder(inspector, out hasRecorder),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return hasRecorder != 0;
    }

}

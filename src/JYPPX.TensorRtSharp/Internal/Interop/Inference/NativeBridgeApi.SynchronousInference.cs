using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static void ExecuteV2(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_execute_v2_safe(context),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_execute_v2_safe(context),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_execute_v2_safe(context),
            _ => throw UnsupportedSynchronousInferenceLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static void ExecuteLegacy(TensorRtApiLine line, SafeTensorRtObjectHandle context, int batchSize)
    {
        if (line != TensorRtApiLine.TensorRt8)
        {
            throw UnsupportedSynchronousInferenceLine();
        }

        NativeStatus.ThrowIfFailed(
            NativeMethodsTensorRt.jyppx_trt8_execution_context_execute_legacy_safe(context, batchSize));
    }

    public static void EnqueueV2(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle context,
        SafeCudaStreamHandle stream)
    {
        if (line != TensorRtApiLine.TensorRt8)
        {
            throw UnsupportedSynchronousInferenceLine();
        }

        NativeStatus.ThrowIfFailed(
            NativeMethodsTensorRt.jyppx_trt8_execution_context_enqueue_v2_safe(context, stream));
    }

    private static BridgeProbeException UnsupportedSynchronousInferenceLine()
    {
        return new BridgeProbeException(
            BridgeStatusCode.NotSupported,
            BridgeErrorCategory.TensorRt,
            "The requested synchronous/legacy inference API is not supported by this TensorRT adapter line.");
    }
}

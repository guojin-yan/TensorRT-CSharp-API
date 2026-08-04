using System;
using System.Runtime.InteropServices;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle CreateEngineInspector(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        SafeTensorRtObjectHandle inspector;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_engine_create_inspector(engine, out inspector);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_engine_create_inspector(engine, out inspector);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_engine_create_inspector(engine, out inspector);
                break;
            default:
                throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.");
        }

        NativeStatus.ThrowIfFailed(status);
        return inspector;
    }

    public static void SetEngineInspectorExecutionContext(TensorRtApiLine line, SafeTensorRtObjectHandle inspector, SafeTensorRtObjectHandle context)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_inspector_set_execution_context(inspector, context),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_inspector_set_execution_context(inspector, context),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_inspector_set_execution_context(inspector, context),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static string GetEngineInformation(TensorRtApiLine line, SafeTensorRtObjectHandle inspector, TensorRtLayerInformationFormat format)
    {
        BridgeStatusCode status = GetEngineInformationNative(line, inspector, format, IntPtr.Zero, UIntPtr.Zero, out UIntPtr requiredSize);
        NativeStatus.ThrowIfFailed(status);

        ulong required = requiredSize.ToUInt64();
        if (required == 0)
        {
            return string.Empty;
        }

        if (required > int.MaxValue)
        {
            throw new BridgeProbeException(BridgeStatusCode.BufferTooSmall, BridgeErrorCategory.TensorRt, "Engine inspector output is too large for the managed buffer.");
        }

        IntPtr buffer = Marshal.AllocHGlobal(checked((int)required));
        try
        {
            status = GetEngineInformationNative(line, inspector, format, buffer, requiredSize, out _);
            NativeStatus.ThrowIfFailed(status);
            return Utf8Interop.ReadString(buffer);
        }
        finally
        {
            Marshal.FreeHGlobal(buffer);
        }
    }

    private static BridgeStatusCode GetEngineInformationNative(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle inspector,
        TensorRtLayerInformationFormat format,
        IntPtr outputBuffer,
        UIntPtr outputBufferSize,
        out UIntPtr requiredSize)
    {
        return line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_inspector_get_engine_information(inspector, (int)format, outputBuffer, outputBufferSize, out requiredSize),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_inspector_get_engine_information(inspector, (int)format, outputBuffer, outputBufferSize, out requiredSize),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_inspector_get_engine_information(inspector, (int)format, outputBuffer, outputBufferSize, out requiredSize),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };
    }

}

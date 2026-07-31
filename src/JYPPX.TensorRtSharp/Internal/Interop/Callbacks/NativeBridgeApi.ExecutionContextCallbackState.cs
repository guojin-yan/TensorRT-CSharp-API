using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static NativeTensorRtExecutionContextCallbackStateInfo GetExecutionContextCallbackStateSnapshot(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle context,
        string tensorName)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);

        BridgeStatusCode status = InvokeGetExecutionContextCallbackStateSnapshot(
            line,
            context,
            tensorNameUtf8.Pointer,
            out NativeTensorRtExecutionContextCallbackStateInfo info);

        NativeStatus.ThrowIfFailed(status);
        return info;
    }

    public static bool TryGetExecutionContextCallbackStateSnapshot(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle context,
        string tensorName,
        out NativeTensorRtExecutionContextCallbackStateInfo info,
        out string diagnostic)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);

        BridgeStatusCode status = InvokeGetExecutionContextCallbackStateSnapshot(
            line,
            context,
            tensorNameUtf8.Pointer,
            out info);
        diagnostic = status == BridgeStatusCode.Ok
            ? BridgeInfoMapper.ReadFixedUtf8(info.LastDiagnostic)
            : GetLastErrorMessageOrFallback($"Execution-context callback state snapshot failed with status '{status}'.");
        return status == BridgeStatusCode.Ok && info.LastStatus == (int)BridgeStatusCode.Ok;
    }

    public static NativeTensorRtExecutionContextCallbackStateInfo ClearExecutionContextCallbackState(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle context,
        string tensorName)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);

        NativeTensorRtExecutionContextCallbackStateInfo info;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_clear_callback_state(context, tensorNameUtf8.Pointer, out info),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_clear_callback_state(context, tensorNameUtf8.Pointer, out info),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_clear_callback_state(context, tensorNameUtf8.Pointer, out info),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return info;
    }

    private static BridgeStatusCode InvokeGetExecutionContextCallbackStateSnapshot(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle context,
        nint tensorName,
        out NativeTensorRtExecutionContextCallbackStateInfo info)
    {
        return line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_get_callback_state_snapshot(context, tensorName, out info),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_get_callback_state_snapshot(context, tensorName, out info),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_get_callback_state_snapshot(context, tensorName, out info),
            _ => throw UnsupportedLine()
        };
    }
}

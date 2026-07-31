using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle CreateDebugListenerOwner(
        TensorRtApiLine line,
        TensorRtDebugListenerNativeCallback callback,
        IntPtr userState)
    {
        SafeTensorRtObjectHandle owner;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_debug_listener_owner_create(callback, userState, out owner),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_debug_listener_owner_create(callback, userState, out owner),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(
                BridgeStatusCode.NotSupported,
                BridgeErrorCategory.TensorRt,
                "TensorRT debug listener callback owners require TensorRT 10 or TensorRT 11."),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return owner;
    }

    public static bool AttachDebugListenerOwner(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle owner,
        SafeTensorRtObjectHandle context)
    {
        int attached;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_debug_listener_owner_attach(owner, context, out attached),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_debug_listener_owner_attach(owner, context, out attached),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(
                BridgeStatusCode.NotSupported,
                BridgeErrorCategory.TensorRt,
                "TensorRT debug listener callback owners require TensorRT 10 or TensorRT 11."),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return attached != 0;
    }

    public static bool DetachDebugListenerOwner(TensorRtApiLine line, SafeTensorRtObjectHandle owner)
    {
        int detached;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_debug_listener_owner_detach(owner, out detached),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_debug_listener_owner_detach(owner, out detached),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(
                BridgeStatusCode.NotSupported,
                BridgeErrorCategory.TensorRt,
                "TensorRT debug listener callback owners require TensorRT 10 or TensorRT 11."),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return detached != 0;
    }

    public static NativeTensorRtDebugListenerOwnerInfo GetDebugListenerOwnerInfo(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle owner)
    {
        NativeTensorRtDebugListenerOwnerInfo info;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_debug_listener_owner_get_info(owner, out info),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_debug_listener_owner_get_info(owner, out info),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(
                BridgeStatusCode.NotSupported,
                BridgeErrorCategory.TensorRt,
                "TensorRT debug listener callback owners require TensorRT 10 or TensorRT 11."),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return info;
    }
}

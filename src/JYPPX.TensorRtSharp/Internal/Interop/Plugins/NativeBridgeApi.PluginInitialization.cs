using System;
using System.Runtime.InteropServices;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static bool InitializeLibNvInferPlugins(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle logger,
        string? libNamespace)
    {
        if (logger == null || logger.IsInvalid)
        {
            throw new ArgumentException("A valid TensorRT logger is required for plugin initialization.", nameof(logger));
        }

        using Utf8Interop.Utf8StringScope namespaceUtf8 = Utf8Interop.ToNativeString(libNamespace);
        int initialized;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_global_init_lib_nvinfer_plugins(logger, namespaceUtf8.Pointer, out initialized),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_global_init_lib_nvinfer_plugins(logger, namespaceUtf8.Pointer, out initialized),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_global_init_lib_nvinfer_plugins(logger, namespaceUtf8.Pointer, out initialized),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return initialized != 0;
    }
}

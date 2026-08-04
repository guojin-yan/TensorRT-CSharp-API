using System;
using System.Runtime.InteropServices;
using JYPPX.TensorRtSharp.Shared;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeMethodsTensorRt
{
    // Manifest-driven TensorRT interop declarations are generated into
    // NativeMethodsTensorRt.Generated.g.cs. Keep only non-manifest or
    // hand-tuned declarations here when a later phase needs them.

    [DllImport(BridgeConstants.NativeBridgeLibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern BridgeStatusCode jyppx_trt11_engine_inspector_get_engine_information(SafeTensorRtObjectHandle inspector, int format, IntPtr output_buffer, UIntPtr output_buffer_size, out UIntPtr out_required_size);

    [DllImport(BridgeConstants.NativeBridgeLibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern BridgeStatusCode jyppx_trt11_network_get_name(SafeTensorRtObjectHandle network, IntPtr output_buffer, UIntPtr output_buffer_size, out UIntPtr out_required_size);

    [DllImport(BridgeConstants.NativeBridgeLibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern BridgeStatusCode jyppx_trt11_runtime_create_diagnostic(SafeTensorRtObjectHandle logger, out NativeTensorRtRuntimeCreateDiagnosticInfo out_info);
}

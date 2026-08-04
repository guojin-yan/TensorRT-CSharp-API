using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static TensorRtGlobalRuntimeVersion GetGlobalRuntimeVersion(TensorRtApiLine line)
    {
        return new TensorRtGlobalRuntimeVersion(
            line,
            GetGlobalInferLibVersion(line),
            GetGlobalInferLibMajorVersion(line),
            GetGlobalInferLibMinorVersion(line),
            GetGlobalInferLibPatchVersion(line),
            GetGlobalInferLibBuildVersion(line),
            GetGlobalOnnxParserVersion(line),
            GlobalHasLogger(line));
    }

    public static int GetGlobalInferLibVersion(TensorRtApiLine line)
    {
        int value;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_global_get_infer_lib_version(out value);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_global_get_infer_lib_version(out value);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_global_get_infer_lib_version(out value);
                break;
            default:
                throw UnsupportedGlobalRuntimeProbeLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    public static int GetGlobalInferLibMajorVersion(TensorRtApiLine line)
    {
        int value;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_global_get_infer_lib_major_version(out value);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_global_get_infer_lib_major_version(out value);
                break;
            default:
                throw UnsupportedGlobalRuntimeProbeLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    public static int GetGlobalInferLibMinorVersion(TensorRtApiLine line)
    {
        int value;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_global_get_infer_lib_minor_version(out value);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_global_get_infer_lib_minor_version(out value);
                break;
            default:
                throw UnsupportedGlobalRuntimeProbeLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    public static int GetGlobalInferLibPatchVersion(TensorRtApiLine line)
    {
        int value;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_global_get_infer_lib_patch_version(out value);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_global_get_infer_lib_patch_version(out value);
                break;
            default:
                throw UnsupportedGlobalRuntimeProbeLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    public static int GetGlobalInferLibBuildVersion(TensorRtApiLine line)
    {
        int value;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_global_get_infer_lib_build_version(out value);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_global_get_infer_lib_build_version(out value);
                break;
            default:
                throw UnsupportedGlobalRuntimeProbeLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    public static bool GlobalHasLogger(TensorRtApiLine line)
    {
        int hasLogger;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_global_has_logger(out hasLogger);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_global_has_logger(out hasLogger);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_global_has_logger(out hasLogger);
                break;
            default:
                throw UnsupportedGlobalRuntimeProbeLine();
        }

        NativeStatus.ThrowIfFailed(status);
        return hasLogger != 0;
    }

}

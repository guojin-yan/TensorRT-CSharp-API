using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle CreateRuntime(TensorRtApiLine line, SafeTensorRtObjectHandle logger)
    {
        TensorRtLineBindings bindings = GetBindings(line);
        BridgeStatusCode status = bindings.RuntimeCreate(logger, out SafeTensorRtObjectHandle runtime);
        NativeStatus.ThrowIfFailed(status);
        return runtime;
    }

    public static TensorRtRuntimeCreateDiagnosticSnapshot GetRuntimeCreateDiagnostic(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle logger)
    {
        if (line != TensorRtApiLine.TensorRt11)
        {
            return new TensorRtRuntimeCreateDiagnosticSnapshot(
                line,
                diagnosticAvailable: false,
                attempted: false,
                loggerHandlePresent: logger != null && !logger.IsInvalid,
                loggerPayloadPresent: false,
                createInferRuntimeReturnedNonNull: false,
                createInferRuntimeReturnedNull: false,
                lastStatus: BridgeStatusCode.NotSupported,
                tensorRtAvailable: false,
                expectedMajor: 11,
                bridgeBuiltMajor: 0,
                detectedVersion: string.Empty,
                loggerCallbackAvailable: false,
                loggerMessageCount: 0,
                lastLoggerSeverity: 0,
                lastLoggerMessage: string.Empty,
                createRuntimePhase: "not-supported",
                nativeDetail: "TRT11 runtime create diagnostic is not implemented for this TensorRT API line.",
                diagnostic: "TRT11 runtime create diagnostic is only available for the TensorRT 11 adapter.");
        }

        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_runtime_create_diagnostic(logger, out NativeTensorRtRuntimeCreateDiagnosticInfo info);
        NativeStatus.ThrowIfFailed(status);
        return BridgeInfoMapper.ToManaged(info);
    }

}

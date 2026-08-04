using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Queries the current bridge state without requiring TensorRT inference to succeed.
/// 在不要求 TensorRT 推理成功的前提下查询当前 bridge 状态。
/// </summary>
public static partial class TensorRtEnvironmentProbe
{
    /// <summary>
    /// Tries to create a TensorRT logger for one API line.
    /// 尝试为一个 API line 创建 TensorRT logger。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <param name="message">A success or failure diagnostic message. 成功或失败诊断消息。</param>
    /// <returns><see langword="true"/> when logger creation succeeds. logger 创建成功时返回 <see langword="true"/>。</returns>
    public static bool TryCreateLogger(TensorRtApiLine line, out string message)
    {
        NativeBridgeLoader.EnsureInitialized();

        try
        {
            using SafeTensorRtObjectHandle logger = NativeBridgeApi.CreateLogger(line);
            if (!logger.IsInvalid)
            {
                message = "Logger handle created successfully.";
                return true;
            }

            message = NativeBridgeApi.GetLastErrorMessageOrFallback("Logger creation failed.");
            return false;
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            message = FormatProbeException("Logger creation", exception);
            return false;
        }
    }

    /// <summary>
    /// Tries to create a TensorRT runtime for one API line.
    /// 尝试为一个 API line 创建 TensorRT runtime。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <param name="message">A success or failure diagnostic message. 成功或失败诊断消息。</param>
    /// <returns><see langword="true"/> when runtime creation succeeds. runtime 创建成功时返回 <see langword="true"/>。</returns>
    public static bool TryCreateRuntime(TensorRtApiLine line, out string message)
    {
        NativeBridgeLoader.EnsureInitialized();

        try
        {
            using SafeTensorRtObjectHandle logger = NativeBridgeApi.CreateLogger(line);
            return NativeBridgeApi.TryCreateRuntime(line, logger, out message);
        }
        catch (BridgeProbeException exception)
        {
            message = exception.Message;
            return false;
        }
        catch (SEHException exception)
        {
            message = $"Runtime creation raised SEHException: {exception.Message}";
            return false;
        }
        catch (AccessViolationException exception)
        {
            message = $"Runtime creation raised AccessViolationException: {exception.Message}";
            return false;
        }
    }

    /// <summary>
    /// Gets a copied no-throw diagnostic snapshot for TensorRT runtime creation.
    /// 获取 TensorRT runtime 创建的复制型 no-throw 诊断快照。
    /// </summary>
    /// <param name="line">The TensorRT API line to diagnose. 要诊断的 TensorRT API line。</param>
    /// <returns>A pointer-free runtime creation diagnostic snapshot. 无指针 runtime 创建诊断快照。</returns>
    /// <remarks>
    /// The native diagnostic entry is currently implemented for TensorRT 11. Other lines return a managed not-supported snapshot.
    /// native 诊断入口目前仅实现 TensorRT 11；其他版本线返回托管 not-supported 快照。
    /// </remarks>
    public static TensorRtRuntimeCreateDiagnosticSnapshot GetRuntimeCreateDiagnostic(TensorRtApiLine line)
    {
        NativeBridgeLoader.EnsureInitialized();

        try
        {
            using SafeTensorRtObjectHandle logger = NativeBridgeApi.CreateLogger(line);
            return NativeBridgeApi.GetRuntimeCreateDiagnostic(line, logger);
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            return new TensorRtRuntimeCreateDiagnosticSnapshot(
                line,
                diagnosticAvailable: false,
                attempted: false,
                loggerHandlePresent: false,
                loggerPayloadPresent: false,
                createInferRuntimeReturnedNonNull: false,
                createInferRuntimeReturnedNull: false,
                lastStatus: exception is BridgeProbeException bridgeException ? bridgeException.StatusCode : BridgeStatusCode.RuntimeError,
                tensorRtAvailable: false,
                expectedMajor: line == TensorRtApiLine.TensorRt11 ? 11 : 0,
                bridgeBuiltMajor: 0,
                detectedVersion: string.Empty,
                loggerCallbackAvailable: false,
                loggerMessageCount: 0,
                lastLoggerSeverity: 0,
                lastLoggerMessage: string.Empty,
                createRuntimePhase: "managed-probe-exception",
                nativeDetail: "Managed probe caught an exception before a native diagnostic snapshot was available.",
                diagnostic: FormatProbeException("Runtime create diagnostic", exception));
        }
    }

    /// <summary>
    /// Tries to create a TensorRT builder for one API line.
    /// 尝试为一个 API line 创建 TensorRT builder。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <param name="message">A success or failure diagnostic message. 成功或失败诊断消息。</param>
    /// <returns><see langword="true"/> when builder creation succeeds. builder 创建成功时返回 <see langword="true"/>。</returns>
    public static bool TryCreateBuilder(TensorRtApiLine line, out string message)
    {
        NativeBridgeLoader.EnsureInitialized();
        try
        {
            return NativeBridgeApi.TryCreateBuilder(line, out message);
        }
        catch (BridgeProbeException exception)
        {
            message = exception.Message;
            return false;
        }
        catch (SEHException exception)
        {
            message = $"Builder creation raised SEHException: {exception.Message}";
            return false;
        }
        catch (AccessViolationException exception)
        {
            message = $"Builder creation raised AccessViolationException: {exception.Message}";
            return false;
        }
    }

}

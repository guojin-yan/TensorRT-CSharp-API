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
    /// Initializes and registers TensorRT built-in plugins for the logger's API line.
    /// 使用 logger 所属版本线初始化并注册 TensorRT 内置 plugin。
    /// </summary>
    /// <param name="logger">The logger used synchronously by vendor plugin initialization. vendor 初始化期间同步使用的 logger。</param>
    /// <param name="libNamespace">Optional namespace for the built-in plugin registrations. 内置 plugin 注册使用的可选 namespace。</param>
    /// <returns><see langword="true"/> when the vendor reports successful initialization. vendor 报告初始化成功时返回 <see langword="true"/>。</returns>
    /// <remarks>
    /// This is an explicit process-global registration operation. It does not create, return, or own plugin objects,
    /// and the logger is borrowed only for the synchronous vendor call.
    /// 这是显式的进程级注册操作；不会创建、返回或接管 plugin 对象，logger 只在同步 vendor 调用期间被借用。
    /// </remarks>
    public static bool InitializeBuiltInPlugins(TensorRtLogger logger, string? libNamespace = null)
    {
        if (logger == null)
        {
            throw new ArgumentNullException(nameof(logger));
        }

        NativeBridgeLoader.EnsureInitialized();
        return NativeBridgeApi.InitializeLibNvInferPlugins(logger.Line, logger.Handle, libNamespace);
    }

    /// <summary>
    /// Tries to initialize TensorRT built-in plugins and returns a bounded diagnostic.
    /// 尝试初始化 TensorRT 内置 plugin，并返回受控诊断。
    /// </summary>
    public static bool TryInitializeBuiltInPlugins(
        TensorRtLogger logger,
        string? libNamespace,
        out bool initialized,
        out string diagnostic)
    {
        try
        {
            initialized = InitializeBuiltInPlugins(logger, libNamespace);
            diagnostic = initialized ? "OK" : "TensorRT vendor reported plugin initialization failure.";
            return initialized;
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            initialized = false;
            diagnostic = FormatProbeException("TensorRT built-in plugin initialization", exception);
            return false;
        }
    }

}

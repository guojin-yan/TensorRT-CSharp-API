using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtLogger
{
    /// <summary>
    /// Gets copied TensorRT versioned-interface metadata for this logger on TensorRT 11.
    /// 获取 TensorRT 11 logger 的 versioned-interface 元数据副本。
    /// </summary>
    /// <remarks>
    /// TensorRT 8 and TensorRT 10 loggers are not versioned interfaces in the NVIDIA headers used by this bridge.
    /// TensorRT 8 和 TensorRT 10 的 logger 在当前 bridge 使用的 NVIDIA 头文件中不是 versioned interface。
    /// </remarks>
    public TensorRtInterfaceInfo InterfaceInfo
    {
        get
        {
            ThrowIfDisposed();
            return NativeBridgeApi.GetLoggerInterfaceInfo(Line, _handle);
        }
    }

    /// <summary>
    /// Gets copied TensorRT versioned-interface API language metadata for this logger on TensorRT 11.
    /// 获取 TensorRT 11 logger 的 versioned-interface API language 元数据副本。
    /// </summary>
    /// <remarks>
    /// The native bridge returns the scalar enum value and does not expose the logger pointer. TensorRT 8 and TensorRT 10 loggers
    /// are not versioned interfaces in the NVIDIA headers used by this bridge.
    /// native bridge 只返回标量 enum 值，不暴露 logger 指针。TensorRT 8 和 TensorRT 10 的 logger 在当前 bridge 使用的
    /// NVIDIA 头文件中不是 versioned interface。
    /// </remarks>
    public TensorRtApiLanguage ApiLanguage
    {
        get
        {
            ThrowIfDisposed();
            return NativeBridgeApi.GetLoggerApiLanguage(Line, _handle);
        }
    }

    /// <summary>
    /// Tries to get copied TensorRT versioned-interface metadata for this logger without exposing the native pointer.
    /// 尝试获取当前 logger 的 TensorRT versioned-interface 元数据副本；不会暴露 native 指针。
    /// </summary>
    /// <param name="interfaceInfo">The copied interface metadata when the query succeeds. 查询成功时复制出的 interface 元数据。</param>
    /// <returns><see langword="true"/> when interface metadata is available for this TensorRT line. 当前 TensorRT 版本线支持该元数据时返回 <see langword="true"/>。</returns>
    public bool TryGetInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo)
    {
        return TryGetInterfaceInfo(out interfaceInfo, out _);
    }

    /// <summary>
    /// Tries to get copied TensorRT versioned-interface metadata for this logger without exposing the native pointer.
    /// 尝试获取当前 logger 的 TensorRT versioned-interface 元数据副本；不会暴露 native 指针。
    /// </summary>
    /// <param name="interfaceInfo">The copied interface metadata when the query succeeds. 查询成功时复制出的 interface 元数据。</param>
    /// <param name="diagnostic">A short diagnostic string describing success or the reason for failure. 描述成功或失败原因的简短诊断。</param>
    /// <returns><see langword="true"/> when interface metadata is available for this TensorRT line. 当前 TensorRT 版本线支持该元数据时返回 <see langword="true"/>。</returns>
    /// <remarks>
    /// TensorRT 8 and TensorRT 10 loggers are not versioned interfaces in the NVIDIA headers used by this bridge, so this method returns <see langword="false"/> for those lines.
    /// TensorRT 8 和 TensorRT 10 的 logger 在当前 bridge 使用的 NVIDIA 头文件中不是 versioned interface，因此这些版本线会返回 <see langword="false"/>。
    /// </remarks>
    public bool TryGetInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo, out string diagnostic)
    {
        ThrowIfDisposed();
        try
        {
            interfaceInfo = NativeBridgeApi.GetLoggerInterfaceInfo(Line, _handle);
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            interfaceInfo = new TensorRtInterfaceInfo(string.Empty, 0, 0);
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Tries to get copied TensorRT API language metadata for this logger without exposing the native pointer.
    /// 尝试获取当前 logger 的 TensorRT API language 元数据副本；不会暴露 native 指针。
    /// </summary>
    /// <param name="apiLanguage">The copied API language when the query succeeds. 查询成功时复制出的 API language。</param>
    /// <returns><see langword="true"/> when API language metadata is available for this TensorRT line. 当前 TensorRT 版本线支持该元数据时返回 <see langword="true"/>。</returns>
    public bool TryGetApiLanguage(out TensorRtApiLanguage apiLanguage)
    {
        return TryGetApiLanguage(out apiLanguage, out _);
    }

    /// <summary>
    /// Tries to get copied TensorRT API language metadata for this logger without exposing the native pointer.
    /// 尝试获取当前 logger 的 TensorRT API language 元数据副本；不会暴露 native 指针。
    /// </summary>
    /// <param name="apiLanguage">The copied API language when the query succeeds. 查询成功时复制出的 API language。</param>
    /// <param name="diagnostic">A short diagnostic string describing success or the reason for failure. 描述成功或失败原因的简短诊断。</param>
    /// <returns><see langword="true"/> when API language metadata is available for this TensorRT line. 当前 TensorRT 版本线支持该元数据时返回 <see langword="true"/>。</returns>
    public bool TryGetApiLanguage(out TensorRtApiLanguage apiLanguage, out string diagnostic)
    {
        ThrowIfDisposed();
        try
        {
            apiLanguage = NativeBridgeApi.GetLoggerApiLanguage(Line, _handle);
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            apiLanguage = TensorRtApiLanguage.Unknown;
            diagnostic = exception.Message;
            return false;
        }
    }
}

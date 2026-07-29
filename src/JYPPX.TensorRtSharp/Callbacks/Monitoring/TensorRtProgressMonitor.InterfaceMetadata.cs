using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtProgressMonitor
{
    /// <summary>
    /// Gets copied TensorRT versioned-interface metadata for this progress monitor.
    /// 获取当前 progress monitor 的 TensorRT versioned-interface 元数据副本。
    /// </summary>
    /// <remarks>
    /// TensorRT 10 and TensorRT 11 expose progress monitors as versioned interfaces.
    /// TensorRT 10 和 TensorRT 11 将 progress monitor 暴露为 versioned interface。
    /// </remarks>
    public TensorRtInterfaceInfo InterfaceInfo
    {
        get
        {
            ThrowIfDisposed();
            return NativeBridgeApi.GetProgressMonitorInterfaceInfo(Line, _handle);
        }
    }

    /// <summary>
    /// Gets copied TensorRT versioned-interface API language metadata for this progress monitor.
    /// 获取当前 progress monitor 的 TensorRT versioned-interface API language 元数据副本。
    /// </summary>
    /// <remarks>
    /// TensorRT 10 and TensorRT 11 expose progress monitors as versioned interfaces. The native bridge returns the scalar enum
    /// value and does not expose the monitor pointer.
    /// TensorRT 10 和 TensorRT 11 将 progress monitor 暴露为 versioned interface。native bridge 只返回标量 enum 值，
    /// 不暴露 monitor 指针。
    /// </remarks>
    public TensorRtApiLanguage ApiLanguage
    {
        get
        {
            ThrowIfDisposed();
            return NativeBridgeApi.GetProgressMonitorApiLanguage(Line, _handle);
        }
    }

    /// <summary>
    /// Tries to get copied TensorRT versioned-interface metadata for this progress monitor without exposing the native pointer.
    /// 尝试获取当前 progress monitor 的 TensorRT versioned-interface 元数据副本；不会暴露 native 指针。
    /// </summary>
    /// <param name="interfaceInfo">The copied interface metadata when the query succeeds. 查询成功时复制出的 interface 元数据。</param>
    /// <returns><see langword="true"/> when interface metadata is available for this TensorRT line. 当前 TensorRT 版本线支持该元数据时返回 <see langword="true"/>。</returns>
    public bool TryGetInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo)
    {
        return TryGetInterfaceInfo(out interfaceInfo, out _);
    }

    /// <summary>
    /// Tries to get copied TensorRT versioned-interface metadata for this progress monitor without exposing the native pointer.
    /// 尝试获取当前 progress monitor 的 TensorRT versioned-interface 元数据副本；不会暴露 native 指针。
    /// </summary>
    /// <param name="interfaceInfo">The copied interface metadata when the query succeeds. 查询成功时复制出的 interface 元数据。</param>
    /// <param name="diagnostic">A short diagnostic string describing success or the reason for failure. 描述成功或失败原因的简短诊断。</param>
    /// <returns><see langword="true"/> when interface metadata is available for this TensorRT line. 当前 TensorRT 版本线支持该元数据时返回 <see langword="true"/>。</returns>
    /// <remarks>
    /// TensorRT 10 and TensorRT 11 expose progress monitors as versioned interfaces. Unsupported or unavailable adapters return <see langword="false"/> with a diagnostic.
    /// TensorRT 10 和 TensorRT 11 将 progress monitor 暴露为 versioned interface；不支持或不可用的 adapter 会返回 <see langword="false"/> 并给出诊断。
    /// </remarks>
    public bool TryGetInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo, out string diagnostic)
    {
        ThrowIfDisposed();
        try
        {
            interfaceInfo = NativeBridgeApi.GetProgressMonitorInterfaceInfo(Line, _handle);
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
    /// Tries to get copied TensorRT API language metadata for this progress monitor without exposing the native pointer.
    /// 尝试获取当前 progress monitor 的 TensorRT API language 元数据副本；不会暴露 native 指针。
    /// </summary>
    /// <param name="apiLanguage">The copied API language when the query succeeds. 查询成功时复制出的 API language。</param>
    /// <returns><see langword="true"/> when API language metadata is available for this TensorRT line. 当前 TensorRT 版本线支持该元数据时返回 <see langword="true"/>。</returns>
    public bool TryGetApiLanguage(out TensorRtApiLanguage apiLanguage)
    {
        return TryGetApiLanguage(out apiLanguage, out _);
    }

    /// <summary>
    /// Tries to get copied TensorRT API language metadata for this progress monitor without exposing the native pointer.
    /// 尝试获取当前 progress monitor 的 TensorRT API language 元数据副本；不会暴露 native 指针。
    /// </summary>
    /// <param name="apiLanguage">The copied API language when the query succeeds. 查询成功时复制出的 API language。</param>
    /// <param name="diagnostic">A short diagnostic string describing success or the reason for failure. 描述成功或失败原因的简短诊断。</param>
    /// <returns><see langword="true"/> when API language metadata is available for this TensorRT line. 当前 TensorRT 版本线支持该元数据时返回 <see langword="true"/>。</returns>
    public bool TryGetApiLanguage(out TensorRtApiLanguage apiLanguage, out string diagnostic)
    {
        ThrowIfDisposed();
        try
        {
            apiLanguage = NativeBridgeApi.GetProgressMonitorApiLanguage(Line, _handle);
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

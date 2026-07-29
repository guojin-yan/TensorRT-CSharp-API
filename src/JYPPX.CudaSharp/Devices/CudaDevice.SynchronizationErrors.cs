using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public static partial class CudaDevice
{
    /// <summary>
    /// Synchronizes the current CUDA device.
    /// 同步当前 CUDA 设备。
    /// </summary>
    public static void Synchronize()
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.SynchronizeDevice();
    }

    /// <summary>
    /// Resets the current CUDA device.
    /// 重置当前 CUDA 设备。
    /// </summary>
    public static void Reset()
    {
        NativeBridgeLoader.EnsureInitialized();
        NativeCudaApi.ResetDevice();
    }

    /// <summary>
    /// Gets and clears the last CUDA error code.
    /// 获取并清除最近一次 CUDA 错误码。
    /// </summary>
    /// <returns>The last CUDA error code. 最近一次 CUDA 错误码。</returns>
    public static int GetLastErrorCode()
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetLastErrorCode();
    }

    /// <summary>
    /// Gets the last CUDA error code without clearing it.
    /// 获取最近一次 CUDA 错误码，但不清除。
    /// </summary>
    /// <returns>The last CUDA error code. 最近一次 CUDA 错误码。</returns>
    public static int PeekAtLastErrorCode()
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.PeekAtLastErrorCode();
    }

    /// <summary>
    /// Gets the symbolic CUDA error name for an error code.
    /// 获取某个错误码对应的 CUDA 符号名。
    /// </summary>
    /// <param name="errorCode">The CUDA error code. CUDA 错误码。</param>
    /// <returns>The CUDA symbolic error name. CUDA 符号错误名。</returns>
    public static string GetErrorName(int errorCode)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetErrorName(errorCode);
    }

    /// <summary>
    /// Gets the human-readable CUDA error string for an error code.
    /// 获取某个错误码对应的 CUDA 可读错误描述。
    /// </summary>
    /// <param name="errorCode">The CUDA error code. CUDA 错误码。</param>
    /// <returns>The CUDA error description. CUDA 错误描述。</returns>
    public static string GetErrorString(int errorCode)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeCudaApi.GetErrorString(errorCode);
    }

}

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using JYPPX.Shared.Interop;
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
    /// Tries to run the minimal TensorRT 10 build chain.
    /// 尝试运行最小 TensorRT 10 构建链路。
    /// </summary>
    /// <param name="message">A success or failure diagnostic message. 成功或失败诊断消息。</param>
    /// <returns><see langword="true"/> when the minimal build chain succeeds. 最小构建链路成功时返回 <see langword="true"/>。</returns>
    public static bool TryRunTensorRt10MinimalBuildChain(out string message)
    {
        NativeBridgeLoader.EnsureInitialized();
        try
        {
            return NativeBridgeApi.TryRunTrt10MinimalBuildChain(out message);
        }
        catch (BridgeProbeException exception)
        {
            message = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Tries to build a TensorRT 10 serialized network without running inference.
    /// 尝试构建一个不执行推理的 TensorRT 10 serialized network。
    /// </summary>
    /// <param name="message">A success or failure diagnostic message. 成功或失败诊断消息。</param>
    /// <returns><see langword="true"/> when serialized-network build succeeds. serialized network 构建成功时返回 <see langword="true"/>。</returns>
    public static bool TryBuildTensorRt10SerializedNetworkOnly(out string message)
    {
        NativeBridgeLoader.EnsureInitialized();
        try
        {
            return NativeBridgeApi.TryBuildTrt10SerializedNetworkOnly(out message);
        }
        catch (BridgeProbeException exception)
        {
            message = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Tries to run the minimal TensorRT 8 build chain.
    /// 尝试运行最小 TensorRT 8 构建链路。
    /// </summary>
    /// <param name="message">A success or failure diagnostic message. 成功或失败诊断消息。</param>
    /// <returns><see langword="true"/> when the minimal build chain succeeds. 最小构建链路成功时返回 <see langword="true"/>。</returns>
    public static bool TryRunTensorRt8MinimalBuildChain(out string message)
    {
        NativeBridgeLoader.EnsureInitialized();
        try
        {
            return NativeBridgeApi.TryRunTrt8MinimalBuildChain(out message);
        }
        catch (BridgeProbeException exception)
        {
            message = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Tries to build a TensorRT 8 serialized network without running inference.
    /// 尝试构建一个不执行推理的 TensorRT 8 serialized network。
    /// </summary>
    /// <param name="message">A success or failure diagnostic message. 成功或失败诊断消息。</param>
    /// <returns><see langword="true"/> when serialized-network build succeeds. serialized network 构建成功时返回 <see langword="true"/>。</returns>
    public static bool TryBuildTensorRt8SerializedNetworkOnly(out string message)
    {
        NativeBridgeLoader.EnsureInitialized();
        try
        {
            return NativeBridgeApi.TryBuildTrt8SerializedNetworkOnly(out message);
        }
        catch (BridgeProbeException exception)
        {
            message = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Tries to run the minimal TensorRT 11 build chain.
    /// 尝试运行最小 TensorRT 11 构建链路。
    /// </summary>
    /// <param name="message">A success or failure diagnostic message. 成功或失败诊断消息。</param>
    /// <returns><see langword="true"/> when the minimal build chain succeeds. 最小构建链路成功时返回 <see langword="true"/>。</returns>
    public static bool TryRunTensorRt11MinimalBuildChain(out string message)
    {
        NativeBridgeLoader.EnsureInitialized();
        try
        {
            return NativeBridgeApi.TryRunTrt11MinimalBuildChain(out message);
        }
        catch (BridgeProbeException exception)
        {
            message = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Tries to build a TensorRT 11 serialized network without running inference.
    /// 尝试构建一个不执行推理的 TensorRT 11 serialized network。
    /// </summary>
    /// <param name="message">A success or failure diagnostic message. 成功或失败诊断消息。</param>
    /// <returns><see langword="true"/> when serialized-network build succeeds. serialized network 构建成功时返回 <see langword="true"/>。</returns>
    public static bool TryBuildTensorRt11SerializedNetworkOnly(out string message)
    {
        NativeBridgeLoader.EnsureInitialized();
        try
        {
            return NativeBridgeApi.TryBuildTrt11SerializedNetworkOnly(out message);
        }
        catch (BridgeProbeException exception)
        {
            message = exception.Message;
            return false;
        }
    }

}

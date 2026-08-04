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
    /// Gets the current high-level bridge environment snapshot.
    /// 获取当前高层 bridge 环境快照。
    /// </summary>
    /// <returns>The current bridge environment snapshot. 当前 bridge 环境快照。</returns>
    public static TensorRtEnvironmentSnapshot GetCurrent()
    {
        NativeBridgeLoader.EnsureInitialized();

        var buildInfo = BridgeInfoMapper.ToManaged(NativeBridgeApi.GetBuildInfo());
        var runtimeInfo = BridgeInfoMapper.ToManaged(NativeBridgeApi.GetRuntimeInfo());
        var capabilityInfo = BridgeInfoMapper.ToManaged(NativeBridgeApi.GetCapabilityInfo());
        var trt8 = GetAdapterInfoOrFallback(TensorRtApiLine.TensorRt8, buildInfo);
        var trt10 = GetAdapterInfoOrFallback(TensorRtApiLine.TensorRt10, buildInfo);
        var trt11 = GetAdapterInfoOrFallback(TensorRtApiLine.TensorRt11, buildInfo);

        return new TensorRtEnvironmentSnapshot(buildInfo, runtimeInfo, capabilityInfo, trt8, trt10, trt11);
    }

    /// <summary>
    /// Gets TensorRT global runtime version details without creating a runtime.
    /// 无需创建 runtime 即可获取 TensorRT 全局 runtime 版本信息。
    /// </summary>
    /// <param name="line">The TensorRT API line to query. 要查询的 TensorRT API line。</param>
    /// <returns>A global runtime version snapshot. 全局 runtime 版本快照。</returns>
    public static TensorRtGlobalRuntimeVersion GetGlobalRuntimeVersion(TensorRtApiLine line)
    {
        NativeBridgeLoader.EnsureInitialized();
        return NativeBridgeApi.GetGlobalRuntimeVersion(line);
    }

    /// <summary>
    /// Tries to get TensorRT global runtime version details without throwing for unsupported lines.
    /// 尝试获取 TensorRT 全局 runtime 版本信息；不支持时不抛出异常。
    /// </summary>
    /// <param name="line">The TensorRT API line to query. 要查询的 TensorRT API line。</param>
    /// <param name="version">The version snapshot when the query succeeds. 查询成功时的版本快照。</param>
    /// <param name="diagnostic">A diagnostic string describing success or failure. 描述成功或失败原因的诊断字符串。</param>
    /// <returns><see langword="true"/> when the version was collected successfully. 成功采集版本信息时返回 <see langword="true"/>。</returns>
    public static bool TryGetGlobalRuntimeVersion(TensorRtApiLine line, out TensorRtGlobalRuntimeVersion? version, out string diagnostic)
    {
        try
        {
            version = GetGlobalRuntimeVersion(line);
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            version = null;
            diagnostic = FormatProbeException("Global runtime version query", exception);
            return false;
        }
    }

}

using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtExecutionContext
{
    /// <summary>
    /// Gets the native address value currently bound to a named TensorRT tensor for diagnostics.
    /// 获取当前绑定到指定 TensorRT tensor 的原生地址数值，仅用于诊断。
    /// </summary>
    /// <param name="tensorName">The input or output tensor name. / 输入或输出 tensor 名称。</param>
    /// <returns>
    /// The address value reported by TensorRT, or 0 when no address is bound.
    /// TensorRT 报告的地址数值；未绑定地址时返回 0。
    /// </returns>
    /// <remarks>
    /// This value is intentionally exposed as an integer diagnostic value instead of a user-owned pointer.
    /// 该值特意以整数诊断值形式暴露，而不是用户可拥有或解引用的指针。
    /// </remarks>
    public ulong GetTensorAddressValue(string tensorName)
    {
        return NativeBridgeApi.GetExecutionContextTensorAddressValue(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets the native address value currently bound to a named TensorRT output tensor for diagnostics.
    /// 获取当前绑定到指定 TensorRT 输出 tensor 的原生地址数值，仅用于诊断。
    /// </summary>
    /// <param name="tensorName">The output tensor name. / 输出 tensor 名称。</param>
    /// <returns>The address value reported by TensorRT, or 0 when no address is bound. / TensorRT 报告的地址数值；未绑定时返回 0。</returns>
    public ulong GetOutputTensorAddressValue(string tensorName)
    {
        return NativeBridgeApi.GetExecutionContextOutputTensorAddressValue(Line, _handle, tensorName);
    }

    /// <summary>
    /// Clears the output allocator attached to a named output tensor.
    /// 清除绑定到指定输出 tensor 的 output allocator。
    /// </summary>
    /// <param name="tensorName">The output tensor name. / 输出 tensor 名称。</param>
    /// <returns><see langword="true"/> when TensorRT accepts the clear operation. / TensorRT 接受清理操作时返回 <see langword="true"/>。</returns>
    public bool ClearOutputAllocator(string tensorName)
    {
        return NativeBridgeApi.ClearExecutionContextOutputAllocator(Line, _handle, tensorName);
    }

    /// <summary>
    /// Clears the temporary-storage allocator attached to this execution context.
    /// 清除绑定到当前 execution context 的 temporary-storage allocator。
    /// </summary>
    /// <returns><see langword="true"/> when TensorRT accepts the clear operation. / TensorRT 接受清理操作时返回 <see langword="true"/>。</returns>
    public bool ClearTemporaryStorageAllocator()
    {
        return NativeBridgeApi.ClearExecutionContextTemporaryStorageAllocator(Line, _handle);
    }

    /// <summary>
    /// Clears the debug listener attached to this TensorRT 11 execution context.
    /// 清除绑定到当前 TensorRT 11 execution context 的 debug listener。
    /// </summary>
    /// <returns><see langword="true"/> when TensorRT accepts the clear operation. / TensorRT 接受清理操作时返回 <see langword="true"/>。</returns>
    public bool ClearDebugListener()
    {
        return NativeBridgeApi.ClearExecutionContextDebugListener(Line, _handle);
    }

    /// <summary>
    /// Gets whether this TensorRT 11 execution context has a debug listener attached.
    /// 获取当前 TensorRT 11 execution context 是否绑定了 debug listener。
    /// </summary>
    public bool HasDebugListener => NativeBridgeApi.HasExecutionContextDebugListener(Line, _handle);

    /// <summary>
    /// Clears the profiler attached to this TensorRT 11 execution context.
    /// 清除绑定到当前 TensorRT 11 execution context 的 profiler。
    /// </summary>
    public void ClearProfiler()
    {
        NativeBridgeApi.ClearExecutionContextProfiler(Line, _handle);
    }

    /// <summary>
    /// Gets whether this TensorRT 11 execution context has a profiler attached.
    /// 获取当前 TensorRT 11 execution context 是否绑定了 profiler。
    /// </summary>
    public bool HasProfiler => NativeBridgeApi.HasExecutionContextProfiler(Line, _handle);

    /// <summary>
    /// Gets whether this TensorRT 11 execution context has an associated runtime config object.
    /// 获取当前 TensorRT 11 execution context 是否有关联的 runtime config 对象。
    /// </summary>
    public bool HasRuntimeConfig => NativeBridgeApi.HasExecutionContextRuntimeConfig(Line, _handle);

    /// <summary>
    /// Sets the NVTX verbosity used by this TensorRT 11 execution context.
    /// 设置当前 TensorRT 11 execution context 使用的 NVTX 详细程度。
    /// </summary>
    /// <param name="verbosity">The desired NVTX verbosity. / 期望的 NVTX 详细程度。</param>
    /// <returns><see langword="true"/> when TensorRT accepts the value. / TensorRT 接受该值时返回 <see langword="true"/>。</returns>
    public bool SetNvtxVerbosity(TensorRtProfilingVerbosity verbosity)
    {
        return NativeBridgeApi.SetExecutionContextNvtxVerbosity(Line, _handle, verbosity);
    }

    /// <summary>
    /// Gets the NVTX verbosity currently used by this TensorRT 11 execution context.
    /// 获取当前 TensorRT 11 execution context 使用的 NVTX 详细程度。
    /// </summary>
    public TensorRtProfilingVerbosity GetNvtxVerbosity()
    {
        return NativeBridgeApi.GetExecutionContextNvtxVerbosity(Line, _handle);
    }

    /// <summary>
    /// Clears user-provided auxiliary streams so TensorRT may use its default auxiliary-stream behavior.
    /// 清除用户提供的 auxiliary stream，让 TensorRT 回到默认 auxiliary-stream 行为。
    /// </summary>
    public void ClearAuxStreams()
    {
        NativeBridgeApi.ClearExecutionContextAuxStreams(Line, _handle);
    }

    /// <summary>
    /// Enables or disables debug state for TensorRT 11 unfused debug tensors.
    /// 启用或禁用 TensorRT 11 未融合 debug tensor 的 debug state。
    /// </summary>
    /// <param name="enabled">Whether unfused tensor debug state should be enabled. / 是否启用未融合 tensor debug state。</param>
    /// <returns><see langword="true"/> when TensorRT accepts the setting. / TensorRT 接受该设置时返回 <see langword="true"/>。</returns>
    public bool SetUnfusedTensorsDebugState(bool enabled)
    {
        return NativeBridgeApi.SetExecutionContextUnfusedTensorsDebugState(Line, _handle, enabled);
    }

    /// <summary>
    /// Gets debug state for TensorRT 11 unfused debug tensors.
    /// 获取 TensorRT 11 未融合 debug tensor 的 debug state。
    /// </summary>
    public bool GetUnfusedTensorsDebugState()
    {
        return NativeBridgeApi.GetExecutionContextUnfusedTensorsDebugState(Line, _handle);
    }
}

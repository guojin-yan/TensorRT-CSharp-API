using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtBuilder
{
    /// <summary>
    /// Gets TensorRT's maximum DLA batch size for legacy DLA checks.
    /// 获取 TensorRT 用于传统 DLA 检查的最大 DLA batch size。
    /// </summary>
    /// <remarks>
    /// TensorRT reports that this value does not apply to dynamic shapes. Use it as a compatibility diagnostic, not as a dynamic-shape capacity limit.
    /// TensorRT 官方说明该值不适用于 dynamic shape；请把它作为兼容性诊断信息，而不是动态 shape 的容量上限。
    /// </remarks>
    public int MaxDlaBatchSize => NativeBridgeApi.GetBuilderMaxDlaBatchSize(Line, _handle);

    /// <summary>
    /// Gets TensorRT 8's legacy explicit builder max batch size.
    /// 获取 TensorRT 8 legacy builder max batch size；TensorRT 10/11 已移除此 builder 级查询。
    /// </summary>
    /// <remarks>
    /// Prefer explicit batch networks and optimization profiles for modern TensorRT builds. This property is a compatibility diagnostic only.
    /// 现代 TensorRT 构建请优先使用 explicit batch network 与 optimization profile；该属性仅作为兼容性诊断。
    /// </remarks>
    public int MaxBatchSizeCompatibility => NativeBridgeApi.GetBuilderMaxBatchSizeCompatibility(Line, _handle);

    /// <summary>
    /// Sets the deprecated TensorRT 8 implicit-batch maximum.
    /// 设置已弃用的 TensorRT 8 implicit-batch 最大值。
    /// </summary>
    /// <remarks>
    /// Modern TensorRT code should use explicit-batch networks and optimization profiles. TensorRT 10 and 11 report this control as unsupported.
    /// 现代 TensorRT 代码应使用 explicit-batch network 与 optimization profile；TensorRT 10/11 会将此控制报告为不支持。
    /// </remarks>
    /// <param name="maxBatchSize">A positive legacy maximum batch size. 大于零的 legacy 最大 batch size。</param>
    public void SetMaxBatchSizeCompatibility(int maxBatchSize)
    {
        NativeBridgeApi.SetBuilderMaxBatchSizeCompatibility(Line, _handle, maxBatchSize);
    }

    /// <summary>
    /// Gets TensorRT's current builder worker-thread limit.
    /// 获取 TensorRT 当前 builder 工作线程上限。
    /// </summary>
    public int MaxThreads => NativeBridgeApi.GetBuilderMaxThreads(Line, _handle);

    /// <summary>
    /// Gets whether a native TensorRT error recorder is currently attached to this builder.
    /// 获取当前 builder 是否绑定了 TensorRT 原生 error recorder；不会暴露 recorder 指针或接管其生命周期。
    /// </summary>
    public bool HasErrorRecorder => NativeBridgeApi.HasBuilderErrorRecorder(Line, _handle);

    /// <summary>
    /// Attempts to collect a copied read-only snapshot from this builder's error recorder.
    /// 尝试从当前 builder 的 error recorder 采集只读托管快照。
    /// </summary>
    /// <param name="snapshot">The copied snapshot. 已复制到托管内存的快照。</param>
    /// <returns><see langword="true"/> when a recorder is attached. / 当前附加 recorder 时返回 <see langword="true"/>。</returns>
    /// <remarks>
    /// The snapshot copies counts, overflow state, interface metadata when available, and error descriptions. It never exposes or retains an <c>IErrorRecorder*</c>.
    /// 该快照复制数量、溢出状态、可用时的 interface 元数据和错误描述；不会暴露或持有 <c>IErrorRecorder*</c>。
    /// </remarks>
    public bool TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot)
    {
        snapshot = NativeBridgeApi.GetBuilderErrorRecorderSnapshot(Line, _handle);
        return snapshot.HasRecorder;
    }

    /// <summary>
    /// Gets whether this builder still has a native TensorRT logger attached.
    /// 获取当前 builder 是否仍附加 TensorRT 原生 logger；不会暴露 logger 指针或接管其生命周期。
    /// </summary>
    public bool HasLogger => NativeBridgeApi.HasBuilderLogger(Line, _handle);

    /// <summary>
    /// Sets TensorRT's builder worker-thread limit.
    /// 设置 TensorRT 的 builder 工作线程上限。
    /// </summary>
    /// <param name="maxThreads">The desired worker-thread limit. / 期望的工作线程上限。</param>
    /// <returns><see langword="true"/> when TensorRT accepted the setting. / TensorRT 接受该设置时返回 <see langword="true"/>。</returns>
    public bool SetMaxThreads(int maxThreads)
    {
        if (maxThreads < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(maxThreads), "Builder maxThreads must be greater than or equal to 1.");
        }

        return NativeBridgeApi.SetBuilderMaxThreads(Line, _handle, maxThreads);
    }

    /// <summary>
    /// Clears the custom GPU allocator on this builder.
    /// 清除当前 builder 上绑定的自定义 GPU allocator；适用于 TensorRT 8/10/11，不会调用用户 allocator 的 free/deallocate 回调。
    /// </summary>
    /// <remarks>
    /// The bridge currently exposes clearing only. Installing a managed allocator is intentionally left out until callback ownership is designed.
    /// 当前桥接层只暴露清除操作；托管 allocator 回调涉及生命周期和线程边界，暂不在普通高层 API 中暴露。
    /// </remarks>
    public void ClearGpuAllocator()
    {
        NativeBridgeApi.ClearBuilderGpuAllocator(Line, _handle);
    }

    /// <summary>
    /// Clears the native TensorRT error recorder attached to this builder.
    /// 清除当前 builder 上绑定的 TensorRT 原生 error recorder；不会销毁 recorder 或接管其生命周期。
    /// </summary>
    public void ClearErrorRecorder()
    {
        NativeBridgeApi.ClearBuilderErrorRecorder(Line, _handle);
    }

    /// <summary>
    /// Resets TensorRT 11 builder state back to its default configuration.
    /// 将 TensorRT 11 builder 状态重置为默认配置。
    /// </summary>
    public void Reset()
    {
        NativeBridgeApi.ResetBuilder(Line, _handle);
    }

    /// <summary>
    /// Checks whether TensorRT can build the supplied network with the supplied builder config.
    /// 检查 TensorRT 是否支持使用指定 builder config 构建给定 network。
    /// </summary>
    /// <param name="network">The network definition to validate. / 要验证的 network definition。</param>
    /// <param name="config">The builder config used for validation. / 用于验证的 builder config。</param>
    /// <returns><see langword="true"/> when TensorRT reports the network as supported. / TensorRT 报告该 network 可支持时返回 <see langword="true"/>。</returns>
    public bool IsNetworkSupported(TensorRtNetworkDefinition network, TensorRtBuilderConfig config)
    {
        if (network == null)
        {
            throw new ArgumentNullException(nameof(network));
        }

        if (config == null)
        {
            throw new ArgumentNullException(nameof(config));
        }

        if (network.Line != Line || config.Line != Line)
        {
            throw new ArgumentException("Network and config must belong to the same TensorRT API line as the builder.");
        }

        return NativeBridgeApi.IsNetworkSupported(Line, _handle, network.Handle, config.Handle);
    }
}

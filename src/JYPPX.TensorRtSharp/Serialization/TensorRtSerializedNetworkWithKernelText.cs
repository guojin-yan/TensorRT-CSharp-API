using System;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Owns a serialized TensorRT engine plan and the optional kernel text emitted by TensorRT 11.
/// 持有 TensorRT 11 生成的序列化 engine plan，以及可选的 kernel text 输出。
/// </summary>
/// <remarks>
/// Dispose this object after saving or consuming both memory blocks. The contained <see cref="TensorRtHostMemory"/>
/// instances are owned by this result object.
/// 在保存或使用完两个内存块后请释放该对象；其中包含的 <see cref="TensorRtHostMemory"/> 实例由本结果对象拥有。
/// </remarks>
public sealed class TensorRtSerializedNetworkWithKernelText : IDisposable
{
    internal TensorRtSerializedNetworkWithKernelText(TensorRtHostMemory plan, TensorRtHostMemory? kernelText)
    {
        Plan = plan ?? throw new ArgumentNullException(nameof(plan));
        KernelText = kernelText;
    }

    /// <summary>
    /// Gets the serialized TensorRT engine plan.
    /// 获取序列化后的 TensorRT engine plan。
    /// </summary>
    public TensorRtHostMemory Plan { get; }

    /// <summary>
    /// Gets optional kernel source text emitted by TensorRT, when the selected build mode produces it.
    /// 获取 TensorRT 在当前构建模式下可能生成的 kernel 源码文本；未生成时为 <see langword="null"/>。
    /// </summary>
    public TensorRtHostMemory? KernelText { get; }

    /// <summary>
    /// Releases the plan and optional kernel text buffers.
    /// 释放 plan 与可选 kernel text 缓冲区。
    /// </summary>
    public void Dispose()
    {
        KernelText?.Dispose();
        Plan.Dispose();
        GC.SuppressFinalize(this);
    }
}

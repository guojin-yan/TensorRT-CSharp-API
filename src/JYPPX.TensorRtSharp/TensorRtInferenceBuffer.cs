using JYPPX.CudaSharp;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Describes one managed inference buffer bound to a TensorRT engine tensor.
/// 描述一个绑定到 TensorRT engine tensor 的托管推理缓冲区。
/// </summary>
public sealed class TensorRtInferenceBuffer
{
    internal TensorRtInferenceBuffer(
        TensorRtEngineTensorBinding tensor,
        CudaMemory memory,
        TensorRtDims? runtimeShape,
        int sizeInBytes,
        bool ownsMemory)
    {
        Tensor = tensor;
        Memory = memory;
        RuntimeShape = runtimeShape;
        SizeInBytes = sizeInBytes;
        OwnsMemory = ownsMemory;
    }

    /// <summary>
    /// Gets the TensorRT tensor metadata used to create or attach this buffer.
    /// 获取创建或附加该缓冲区时使用的 TensorRT tensor 元数据。
    /// </summary>
    public TensorRtEngineTensorBinding Tensor { get; }

    /// <summary>
    /// Gets the managed CUDA memory allocation.
    /// 获取托管 CUDA 设备内存分配。
    /// </summary>
    public CudaMemory Memory { get; }

    /// <summary>
    /// Gets the runtime shape used to estimate the buffer size.
    /// 获取用于估算缓冲区大小的运行时 shape。
    /// </summary>
    public TensorRtDims? RuntimeShape { get; internal set; }

    /// <summary>
    /// Gets the byte count expected by this binding.
    /// 获取该绑定预期使用的字节数。
    /// </summary>
    public int SizeInBytes { get; internal set; }

    /// <summary>
    /// Gets whether this object owns and will dispose the CUDA memory.
    /// 获取该对象是否拥有并会释放对应 CUDA 内存。
    /// </summary>
    public bool OwnsMemory { get; }

    /// <summary>
    /// Gets whether the buffer has been bound to the TensorRT execution context.
    /// 获取缓冲区是否已经绑定到 TensorRT execution context。
    /// </summary>
    public bool IsBound { get; internal set; }

    /// <summary>
    /// Gets a short human-readable summary for diagnostics.
    /// 获取用于诊断的简短可读摘要。
    /// </summary>
    /// <returns>A deployment binding summary. 部署绑定摘要。</returns>
    public override string ToString()
    {
        string shapeText = RuntimeShape == null ? "unknown" : RuntimeShape.ToString();
        return $"{Tensor.Name} {Tensor.IOMode} {Tensor.DataType} shape={shapeText} bytes={SizeInBytes} bound={IsBound}";
    }
}

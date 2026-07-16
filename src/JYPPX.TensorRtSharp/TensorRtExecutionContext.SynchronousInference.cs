using System;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtExecutionContext
{
    /// <summary>
    /// Executes an explicit-batch network synchronously using the tensor addresses already bound to this context.
    /// 使用已绑定到当前 context 的 tensor 地址同步执行 explicit-batch 网络。
    /// </summary>
    /// <remarks>
    /// The native bridge copies bound addresses into a temporary TensorRT binding array and does not expose device pointers.
    /// native bridge 将已绑定地址复制到临时 TensorRT binding 数组，不向托管层暴露 device pointer。
    /// </remarks>
    public void ExecuteV2()
    {
        NativeBridgeApi.ExecuteV2(Line, _handle);
    }

    /// <summary>
    /// Executes a TensorRT 8 implicit-batch network synchronously.
    /// 同步执行 TensorRT 8 implicit-batch 网络。
    /// </summary>
    /// <param name="batchSize">The positive legacy batch size. 正数 legacy batch size。</param>
    public void ExecuteLegacy(int batchSize)
    {
        if (batchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Legacy execution batch size must be positive.");
        }

        NativeBridgeApi.ExecuteLegacy(Line, _handle, batchSize);
    }

    internal void EnqueueV2(CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        NativeBridgeApi.EnqueueV2(Line, _handle, stream.Handle);
    }
}

using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public sealed partial class CudaStream
{
    /// <summary>
    /// Inserts a wait on a CUDA event into this stream.
    /// 在当前 stream 中插入对 CUDA event 的等待。
    /// </summary>
    /// <param name="cudaEvent">The event to wait on. 要等待的 event。</param>
    public void WaitFor(CudaEvent cudaEvent)
    {
        if (cudaEvent == null)
        {
            throw new ArgumentNullException(nameof(cudaEvent));
        }

        NativeCudaApi.WaitStreamForEvent(_handle, cudaEvent.Handle);
    }

    /// <summary>
    /// Enqueues a bridge-owned delay in this CUDA stream.
    /// 在当前 CUDA stream 中排入由桥接层持有的延迟。
    /// </summary>
    /// <remarks>
    /// The delay is ordered with other stream work and does not expose a managed callback or borrowed callback state.
    /// 延迟与其他 stream 工作保持顺序，且不会暴露托管回调或借用的回调状态。
    /// </remarks>
    /// <param name="milliseconds">The non-negative delay in milliseconds. 非负延迟毫秒数。</param>
    public void EnqueueDelay(int milliseconds)
    {
        if (milliseconds < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(milliseconds), "Delay must be non-negative.");
        }

        NativeCudaApi.EnqueueStreamDelay(_handle, checked((uint)milliseconds));
    }

    /// <summary>
    /// Copies stream attributes from another CUDA stream into this stream.
    /// 将另一个 CUDA stream 的属性复制到当前 stream。
    /// </summary>
    /// <param name="source">The source stream whose attributes should be copied. 要复制属性的源 stream。</param>
    public void CopyAttributesFrom(CudaStream source)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        NativeCudaApi.CopyStreamAttributes(_handle, source.Handle);
    }

    /// <summary>
    /// Blocks the calling thread until this stream completes.
    /// 阻塞调用线程，直到当前 stream 完成。
    /// </summary>
    public void Synchronize()
    {
        NativeCudaApi.SynchronizeStream(_handle);
    }

    /// <summary>
    /// Measures elapsed GPU time for work submitted to this stream.
    /// 测量提交到当前 stream 的工作所消耗的 GPU 时间。
    /// </summary>
    /// <param name="submitWork">The delegate that submits CUDA work to this stream. 向当前 stream 提交 CUDA 工作的委托。</param>
    /// <returns>The elapsed GPU time in milliseconds. GPU 经过时间，单位为毫秒。</returns>
    public float MeasureElapsedTime(Action<CudaStream> submitWork)
    {
        if (submitWork == null)
        {
            throw new ArgumentNullException(nameof(submitWork));
        }

        using CudaEvent start = new CudaEvent();
        using CudaEvent stop = new CudaEvent();
        start.Record(this);
        submitWork(this);
        stop.Record(this);
        stop.Synchronize();
        return stop.ElapsedTimeSince(start);
    }

}

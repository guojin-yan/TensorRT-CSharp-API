using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public partial class CudaMemory
{
    /// <summary>
    /// Asynchronously prefetches part of the allocation to a target device.
    /// 异步将分配的一部分预取到目标设备。
    /// </summary>
    /// <param name="offset">The byte offset within this allocation. 当前分配内的字节偏移。</param>
    /// <param name="count">The number of bytes to prefetch. 要预取的字节数。</param>
    /// <param name="destinationDevice">The target CUDA device ordinal. 目标 CUDA 设备序号。</param>
    /// <param name="stream">The CUDA stream that orders the prefetch. 用于排序预取操作的 CUDA stream。</param>
    public void PrefetchAsync(int offset, int count, int destinationDevice, CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        ValidateRange(offset, count, nameof(offset), nameof(count));
        NativeCudaApi.PrefetchMemoryRangeAsync(_handle, offset, count, destinationDevice, stream.Handle);
    }

    /// <summary>
    /// Asynchronously prefetches the first bytes of the allocation to a target device.
    /// 异步将分配起始处的一部分预取到目标设备。
    /// </summary>
    /// <param name="count">The number of bytes to prefetch. 要预取的字节数。</param>
    /// <param name="destinationDevice">The target CUDA device ordinal. 目标 CUDA 设备序号。</param>
    /// <param name="stream">The CUDA stream that orders the prefetch. 用于排序预取操作的 CUDA stream。</param>
    public void PrefetchAsync(int count, int destinationDevice, CudaStream stream)
    {
        PrefetchAsync(0, count, destinationDevice, stream);
    }

    /// <summary>
    /// Asynchronously prefetches the entire allocation to a target CUDA device.
    /// 异步将整个分配预取到目标 CUDA 设备。
    /// </summary>
    /// <param name="destinationDevice">The target CUDA device ordinal. 目标 CUDA 设备序号。</param>
    /// <param name="stream">The CUDA stream that orders the prefetch. 用于排序预取操作的 CUDA stream。</param>
    public void PrefetchAsync(int destinationDevice, CudaStream stream)
    {
        PrefetchAsync(SizeInBytes, destinationDevice, stream);
    }

    /// <summary>
    /// Applies CUDA memory advice to part of the allocation.
    /// 对分配的一部分应用 CUDA memory advice。
    /// </summary>
    /// <param name="offset">The byte offset within this allocation. 当前分配内的字节偏移。</param>
    /// <param name="count">The number of bytes covered by the advice. advice 覆盖的字节数。</param>
    /// <param name="advice">The CUDA memory advice. CUDA 内存建议。</param>
    /// <param name="device">The device ordinal associated with the advice. 与 advice 关联的设备序号。</param>
    public void Advise(int offset, int count, CudaMemoryAdvice advice, int device)
    {
        ValidateRange(offset, count, nameof(offset), nameof(count));
        ValidateMemoryAdvice(advice, nameof(advice));
        NativeCudaApi.AdviseMemoryRange(_handle, offset, count, (int)advice, device);
    }

    /// <summary>
    /// Applies CUDA memory advice to the first bytes of the allocation.
    /// 对分配起始处的一部分应用 CUDA memory advice。
    /// </summary>
    /// <param name="count">The number of bytes covered by the advice. advice 覆盖的字节数。</param>
    /// <param name="advice">The CUDA memory advice. CUDA 内存建议。</param>
    /// <param name="device">The device ordinal associated with the advice. 与 advice 关联的设备序号。</param>
    public void Advise(int count, CudaMemoryAdvice advice, int device)
    {
        Advise(0, count, advice, device);
    }

    /// <summary>
    /// Applies a CUDA memory advice to the entire allocation.
    /// 对整个分配应用 CUDA memory advice。
    /// </summary>
    /// <param name="advice">The CUDA memory advice. CUDA 内存建议。</param>
    /// <param name="device">The device ordinal associated with the advice. 与 advice 关联的设备序号。</param>
    public void Advise(CudaMemoryAdvice advice, int device)
    {
        Advise(SizeInBytes, advice, device);
    }

}

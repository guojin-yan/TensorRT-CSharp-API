using System;
using System.Collections.Generic;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a managed TensorRT Tensor Rt Execution Context wrapper.
/// 表示托管 TensorRT Tensor Rt Execution Context 包装器。
/// </summary>
public sealed partial class TensorRtExecutionContext
{
    /// <summary>
    /// Sets caller-owned device memory and retains its native owner until this context is disposed.
    /// 设置调用方拥有的 device memory，并在当前 context 释放前保持其 native owner 有效。
    /// </summary>
    /// <param name="memory">The CUDA memory borrowed by TensorRT. TensorRT 借用的 CUDA 内存。</param>
    /// <remarks>
    /// TensorRT 8 uses <c>setDeviceMemory</c>; TensorRT 10 and 11 use the size-aware
    /// <c>setDeviceMemoryV2</c> implementation behind the stable compatibility method.
    /// TensorRT 8 使用 <c>setDeviceMemory</c>；TensorRT 10/11 在稳定兼容方法后使用带 size 的
    /// <c>setDeviceMemoryV2</c> 实现。
    /// </remarks>
    public void SetDeviceMemory(CudaMemory memory)
    {
        BindDeviceMemory(memory, useExplicitV2EntryPoint: false);
    }

    /// <summary>
    /// Sets caller-owned device memory through the explicit TensorRT 10/11 <c>setDeviceMemoryV2</c> bridge.
    /// 通过显式 TensorRT 10/11 <c>setDeviceMemoryV2</c> bridge 设置调用方拥有的 device memory。
    /// </summary>
    /// <param name="memory">The CUDA memory borrowed by TensorRT. TensorRT 借用的 CUDA 内存。</param>
    /// <remarks>
    /// The context retains a SafeHandle lease. Disposing <paramref name="memory"/> after this call does not free the
    /// native allocation while the context or any queued inference may still use it. TensorRT 8 callers should use
    /// <see cref="SetDeviceMemory"/>.
    /// Context 会保留 SafeHandle lease；调用后释放 <paramref name="memory"/> 不会在 context 或已排队推理仍可能使用时
    /// 提前释放 native allocation。TensorRT 8 调用方应使用 <see cref="SetDeviceMemory"/>。
    /// </remarks>
    public void SetDeviceMemoryV2(CudaMemory memory)
    {
        BindDeviceMemory(memory, useExplicitV2EntryPoint: true);
    }

    /// <summary>Gets whether caller-owned device memory is currently bound. 获取当前是否绑定了调用方拥有的 device memory。</summary>
    public bool HasBoundDeviceMemory
    {
        get
        {
            lock (_deviceMemoryLeaseLock)
            {
                return _deviceMemoryLease != null;
            }
        }
    }

    /// <summary>Gets the size of the currently bound device-memory lease. 获取当前绑定的 device-memory lease 大小。</summary>
    public int BoundDeviceMemorySizeInBytes
    {
        get
        {
            lock (_deviceMemoryLeaseLock)
            {
                return _boundDeviceMemorySizeInBytes;
            }
        }
    }

    /// <summary>
    /// Gets the number of current and retired memory leases retained until context disposal.
    /// 获取在 context 释放前保留的当前及历史 memory lease 数量。
    /// </summary>
    public int RetainedDeviceMemoryLeaseCount
    {
        get
        {
            lock (_deviceMemoryLeaseLock)
            {
                return _retiredDeviceMemoryLeases.Count + (_deviceMemoryLease == null ? 0 : 1);
            }
        }
    }

    /// <summary>
    /// Updates the Device Memory Size For Shapes value.
    /// 更新 Device Memory Size For Shapes 值。
    /// </summary>
    public ulong UpdateDeviceMemorySizeForShapes()
    {
        return NativeBridgeApi.UpdateExecutionContextDeviceMemorySizeForShapes(Line, _handle);
    }

    private void BindDeviceMemory(CudaMemory memory, bool useExplicitV2EntryPoint)
    {
        if (memory == null)
        {
            throw new ArgumentNullException(nameof(memory));
        }

        TensorRtDeviceMemoryHandleLease? pendingLease =
            TensorRtDeviceMemoryHandleLease.Create(memory.Handle, memory.SizeInBytes);
        try
        {
            lock (_deviceMemoryLeaseLock)
            {
                ThrowIfDeviceMemoryContextDisposed();
                if (useExplicitV2EntryPoint)
                {
                    NativeBridgeApi.SetExecutionContextDeviceMemoryV2(Line, _handle, pendingLease.Handle);
                }
                else
                {
                    NativeBridgeApi.SetExecutionContextDeviceMemory(Line, _handle, pendingLease.Handle);
                }

                TensorRtDeviceMemoryHandleLease? previousLease = _deviceMemoryLease;
                _deviceMemoryLease = pendingLease;
                _boundDeviceMemorySizeInBytes = pendingLease.SizeInBytes;
                pendingLease = null;
                if (previousLease != null)
                {
                    // A prior asynchronous inference may still use the previous block.
                    _retiredDeviceMemoryLeases.Add(previousLease);
                }
            }
        }
        finally
        {
            pendingLease?.Dispose();
        }
    }

    private void ClearDeviceMemoryCore()
    {
        lock (_deviceMemoryLeaseLock)
        {
            ThrowIfDeviceMemoryContextDisposed();
            NativeBridgeApi.ClearExecutionContextDeviceMemory(Line, _handle);
            if (_deviceMemoryLease != null)
            {
                // Clearing the native binding does not prove earlier queued work has completed.
                _retiredDeviceMemoryLeases.Add(_deviceMemoryLease);
                _deviceMemoryLease = null;
            }

            _boundDeviceMemorySizeInBytes = 0;
        }
    }

    private void ThrowIfDeviceMemoryContextDisposed()
    {
        if (_deviceMemoryContextDisposed || _handle.IsClosed || _handle.IsInvalid)
        {
            throw new ObjectDisposedException(nameof(TensorRtExecutionContext));
        }
    }

}

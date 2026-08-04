using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtExecutionContext
{
    /// <summary>
    /// Attaches an owner-safe managed output allocator to a named output tensor.
    /// 将 owner-safe 托管 output allocator 绑定到指定输出 tensor。
    /// </summary>
    /// <param name="tensorName">The output tensor name. 输出 tensor 名称。</param>
    /// <param name="owner">The callback owner borrowed until clear or context disposal. 借用到 clear 或 context dispose 为止的 callback owner。</param>
    public void SetOutputAllocator(string tensorName, TensorRtOutputAllocatorCallbackOwner owner)
    {
        if (string.IsNullOrWhiteSpace(tensorName))
        {
            throw new ArgumentException("Output tensor name must not be empty.", nameof(tensorName));
        }

        if (owner == null)
        {
            throw new ArgumentNullException(nameof(owner));
        }

        if (TensorRtOutputAllocatorCallbackOwner.IsExecutingRuntimeCallbackOnCurrentThread)
        {
            throw new InvalidOperationException("An output allocator cannot be replaced from inside its own callback.");
        }

        lock (_outputAllocatorLeaseLock)
        {
            if (_outputAllocatorContextDisposed)
            {
                throw new ObjectDisposedException(nameof(TensorRtExecutionContext));
            }

            if (_outputAllocatorKeepAlive.TryGetValue(tensorName, out TensorRtOutputAllocatorCallbackOwner? previous))
            {
                if (ReferenceEquals(previous, owner))
                {
                    return;
                }

                NativeBridgeApi.DetachOutputAllocatorOwner(Line, previous.NativeHandle);
                _outputAllocatorKeepAlive.Remove(tensorName);
                previous.DetachBorrower();
            }

            owner.ThrowIfDisposed();
            owner.AttachBorrower(Line);
            try
            {
                if (!NativeBridgeApi.AttachOutputAllocatorOwner(Line, owner.NativeHandle, _handle, tensorName))
                {
                    throw new InvalidOperationException("TensorRT did not accept the output allocator callback owner.");
                }

                _outputAllocatorKeepAlive.Add(tensorName, owner);
            }
            catch
            {
                owner.DetachBorrower();
                throw;
            }
        }
    }

    /// <summary>
    /// Gets whether this wrapper owns a managed output allocator borrow for a tensor.
    /// 获取当前 wrapper 是否为指定 tensor 持有 managed output allocator 借用。
    /// </summary>
    /// <param name="tensorName">The output tensor name. 输出 tensor 名称。</param>
    /// <returns>Whether an owner is retained. 是否持有 owner。</returns>
    public bool HasManagedOutputAllocator(string tensorName)
    {
        if (string.IsNullOrWhiteSpace(tensorName))
        {
            throw new ArgumentException("Output tensor name must not be empty.", nameof(tensorName));
        }

        lock (_outputAllocatorLeaseLock)
        {
            return _outputAllocatorKeepAlive.ContainsKey(tensorName);
        }
    }
}

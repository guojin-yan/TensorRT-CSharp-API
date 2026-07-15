using System;
using System.Threading;

namespace JYPPX.TensorRtSharp.Internal.Handles;

/// <summary>
/// Keeps a TensorRT SafeHandle alive while borrowed child wrappers exist.
/// 在借用的子包装器存续期间保持 TensorRT SafeHandle 有效。
/// </summary>
internal sealed class SafeTensorRtObjectHandleLease : IDisposable
{
    private SharedState? _state;

    private SafeTensorRtObjectHandleLease(SharedState state)
    {
        _state = state;
    }

    public static SafeTensorRtObjectHandleLease Create(SafeTensorRtObjectHandle owner)
    {
        if (owner == null)
        {
            throw new ArgumentNullException(nameof(owner));
        }

        bool addedRef = false;
        try
        {
            owner.DangerousAddRef(ref addedRef);
            if (!addedRef)
            {
                throw new ObjectDisposedException(nameof(owner));
            }

            return new SafeTensorRtObjectHandleLease(new SharedState(owner));
        }
        catch
        {
            if (addedRef)
            {
                owner.DangerousRelease();
            }

            throw;
        }
    }

    public SafeTensorRtObjectHandleLease Clone()
    {
        SharedState state = Volatile.Read(ref _state)
            ?? throw new ObjectDisposedException(nameof(SafeTensorRtObjectHandleLease));
        state.AddReference();
        return new SafeTensorRtObjectHandleLease(state);
    }

    public void Dispose()
    {
        DisposeCore();
        GC.SuppressFinalize(this);
    }

    ~SafeTensorRtObjectHandleLease()
    {
        DisposeCore();
    }

    private void DisposeCore()
    {
        SharedState? state = Interlocked.Exchange(ref _state, null);
        state?.ReleaseReference();
    }

    private sealed class SharedState
    {
        private readonly SafeTensorRtObjectHandle _owner;
        private int _referenceCount = 1;

        public SharedState(SafeTensorRtObjectHandle owner)
        {
            _owner = owner;
        }

        public void AddReference()
        {
            int current = Volatile.Read(ref _referenceCount);
            while (current != 0)
            {
                int updated = checked(current + 1);
                int observed = Interlocked.CompareExchange(ref _referenceCount, updated, current);
                if (observed == current)
                {
                    return;
                }

                current = observed;
            }

            throw new ObjectDisposedException(nameof(SafeTensorRtObjectHandleLease));
        }

        public void ReleaseReference()
        {
            if (Interlocked.Decrement(ref _referenceCount) == 0)
            {
                _owner.DangerousRelease();
            }
        }
    }
}

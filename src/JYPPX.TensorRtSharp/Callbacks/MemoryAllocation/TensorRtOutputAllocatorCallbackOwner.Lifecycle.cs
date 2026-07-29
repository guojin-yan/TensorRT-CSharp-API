using System;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtOutputAllocatorCallbackOwner
{
    /// <summary>
    /// Releases managed keep-alive handles owned by the design gate.
    /// 释放该设计门禁持有的托管 keep-alive 句柄。
    /// </summary>
    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
        }

        _runtimeGate.Dispose();
        _nativeLedgerOwner.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        lock (_gate)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(TensorRtOutputAllocatorCallbackOwner));
            }
        }
    }
}

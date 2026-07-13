using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace JYPPX.TensorRtSharp.Internal;

internal sealed class TensorRtPinnedInitializerSet : IDisposable
{
    private readonly Dictionary<string, PinnedInitializer> _initializers = new Dictionary<string, PinnedInitializer>(StringComparer.Ordinal);
    private bool _disposed;

    public bool LoadOrReplace(string name, byte[] data, Func<IntPtr, UIntPtr, bool> nativeLoad)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(TensorRtPinnedInitializerSet));
        }

        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("ONNX initializer name must not be null or empty.", nameof(name));
        }

        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        if (data.Length == 0)
        {
            throw new ArgumentException("ONNX initializer data must not be empty.", nameof(data));
        }

        if (nativeLoad == null)
        {
            throw new ArgumentNullException(nameof(nativeLoad));
        }

        PinnedInitializer next = PinnedInitializer.Create(data);
        try
        {
            bool loaded = nativeLoad(next.Pointer, new UIntPtr((ulong)next.Length));
            if (!loaded)
            {
                next.Dispose();
                return false;
            }

            if (_initializers.ContainsKey(name))
            {
                PinnedInitializer previous = _initializers[name];
                previous.Dispose();
            }

            _initializers[name] = next;
            return true;
        }
        catch
        {
            next.Dispose();
            throw;
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        foreach (PinnedInitializer initializer in _initializers.Values)
        {
            initializer.Dispose();
        }

        _initializers.Clear();
    }

    private sealed class PinnedInitializer : IDisposable
    {
        private GCHandle _handle;
        private bool _disposed;

        private PinnedInitializer(byte[] data)
        {
            Data = data;
            _handle = GCHandle.Alloc(Data, GCHandleType.Pinned);
        }

        public byte[] Data { get; }

        public int Length => Data.Length;

        public IntPtr Pointer => _handle.AddrOfPinnedObject();

        public static PinnedInitializer Create(byte[] source)
        {
            byte[] owned = new byte[source.Length];
            Buffer.BlockCopy(source, 0, owned, 0, source.Length);
            return new PinnedInitializer(owned);
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            if (_handle.IsAllocated)
            {
                _handle.Free();
            }
        }
    }
}

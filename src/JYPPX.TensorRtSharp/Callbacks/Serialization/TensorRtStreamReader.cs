using System;
using System.IO;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>Owns an immutable native TensorRT IStreamReaderV2 data source.</summary>
/// <remarks>
/// Input bytes are copied into native immutable storage. TensorRT callback destinations and CUDA stream values never
/// cross the public managed boundary. Dispose is deferred while a deserialize call or returned engine retains the owner.
/// </remarks>
public sealed class TensorRtStreamReader : IDisposable
{
    private readonly object _gate = new object();
    private readonly SafeTensorRtObjectHandle _nativeHandle = new SafeTensorRtObjectHandle();
    private bool _disposeRequested;
    private bool _resourcesReleased;
    private int _borrowerCount;
    private int _activeDeserializeCount;

    /// <summary>Creates a native IStreamReaderV2 owner from serialized engine bytes.</summary>
    /// <param name="line">TensorRT 10 or TensorRT 11.</param>
    /// <param name="serializedEngine">Serialized engine bytes copied into native immutable storage.</param>
    public TensorRtStreamReader(TensorRtApiLine line, byte[] serializedEngine)
    {
        if (line != TensorRtApiLine.TensorRt10 && line != TensorRtApiLine.TensorRt11)
        {
            throw new NotSupportedException("IStreamReaderV2 requires TensorRT 10 or TensorRT 11.");
        }
        if (serializedEngine == null)
        {
            throw new ArgumentNullException(nameof(serializedEngine));
        }
        if (serializedEngine.Length == 0)
        {
            throw new ArgumentException("Serialized engine data must not be empty.", nameof(serializedEngine));
        }

        NativeBridgeLoader.EnsureInitialized();
        Line = line;
        Length = (ulong)serializedEngine.Length;
        _nativeHandle = NativeBridgeApi.CreateStreamReaderV2Owner(line, serializedEngine);
    }

    /// <summary>Creates a native IStreamReaderV2 owner by copying a readable managed stream.</summary>
    /// <param name="line">TensorRT 10 or TensorRT 11.</param>
    /// <param name="serializedEngineStream">Readable stream copied from its current position to the end.</param>
    public TensorRtStreamReader(TensorRtApiLine line, Stream serializedEngineStream)
        : this(line, CopyReadableStream(serializedEngineStream))
    {
    }

    /// <summary>Releases the owner if it was abandoned without an explicit Dispose call.</summary>
    ~TensorRtStreamReader()
    {
        Dispose();
    }

    /// <summary>Gets the TensorRT API line.</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the immutable native source length in bytes.</summary>
    public ulong Length { get; }

    /// <summary>Gets whether Dispose has been requested.</summary>
    public bool IsDisposed
    {
        get { lock (_gate) { return _disposeRequested; } }
    }

    /// <summary>Gets a copied, pointer-free native callback snapshot.</summary>
    public TensorRtStreamReaderRuntimeSnapshot GetRuntimeSnapshot()
    {
        lock (_gate)
        {
            if (_resourcesReleased)
            {
                throw new ObjectDisposedException(nameof(TensorRtStreamReader));
            }

            NativeTensorRtStreamReaderOwnerInfo info = NativeBridgeApi.GetStreamReaderV2OwnerInfo(Line, _nativeHandle);
            TensorRtStreamSeekPosition seekPosition = Enum.IsDefined(typeof(TensorRtStreamSeekPosition), info.LastSeekPosition)
                ? (TensorRtStreamSeekPosition)info.LastSeekPosition
                : TensorRtStreamSeekPosition.Unknown;
            return new TensorRtStreamReaderRuntimeSnapshot(
                (TensorRtApiLine)info.Line,
                info.OwnerId,
                info.Length,
                info.Position,
                info.DeserializeAttemptCount,
                info.SuccessfulDeserializeCount,
                info.FailedDeserializeCount,
                info.ReadCount,
                info.SeekCount,
                info.HostReadCount,
                info.DeviceReadCount,
                info.BytesRead,
                info.RequestedBytes,
                info.FailureCount,
                info.InFlightCallbackCount,
                info.MaxInFlightCallbackCount,
                (BridgeStatusCode)info.LastStatus,
                seekPosition,
                info.LastHadCudaStream != 0,
                info.LastReadToDevice != 0,
                info.LastOperationSucceeded != 0,
                info.IsDeserializing != 0,
                BridgeInfoMapper.ReadFixedUtf8(info.LastDiagnostic));
        }
    }

    /// <summary>Requests release after all active deserialize and engine borrowers are gone.</summary>
    public void Dispose()
    {
        bool releaseNow;
        lock (_gate)
        {
            if (_disposeRequested)
            {
                return;
            }
            _disposeRequested = true;
            releaseNow = _borrowerCount == 0;
        }

        if (releaseNow)
        {
            ReleaseResources();
        }
        GC.SuppressFinalize(this);
    }

    internal SafeTensorRtObjectHandle NativeHandle => _nativeHandle;

    internal void BeginDeserializeBorrower(TensorRtApiLine expectedLine)
    {
        lock (_gate)
        {
            if (_disposeRequested || _resourcesReleased)
            {
                throw new ObjectDisposedException(nameof(TensorRtStreamReader));
            }
            if (expectedLine != Line)
            {
                throw new ArgumentException("Stream reader and runtime must use the same TensorRT API line.");
            }
            if (_activeDeserializeCount != 0)
            {
                throw new InvalidOperationException("A stream reader owner cannot service concurrent deserialize calls.");
            }

            checked
            {
                _borrowerCount++;
            }
            _activeDeserializeCount = 1;
        }
    }

    internal void RetainEngineBorrower(TensorRtApiLine expectedLine)
    {
        lock (_gate)
        {
            if (_resourcesReleased || expectedLine != Line || _activeDeserializeCount != 1)
            {
                throw new InvalidOperationException("The stream reader can be retained only by an engine created by its active deserialize call.");
            }
            checked
            {
                _borrowerCount++;
            }
        }
    }

    internal void EndDeserializeBorrower()
    {
        bool releaseNow;
        lock (_gate)
        {
            if (_activeDeserializeCount != 1 || _borrowerCount <= 0)
            {
                throw new InvalidOperationException("Stream reader deserialize borrower ledger underflow.");
            }
            _activeDeserializeCount = 0;
            _borrowerCount--;
            releaseNow = _disposeRequested && _borrowerCount == 0;
        }

        if (releaseNow)
        {
            ReleaseResources();
        }
    }

    internal void ReleaseEngineBorrower()
    {
        bool releaseNow;
        lock (_gate)
        {
            if (_borrowerCount <= 0)
            {
                throw new InvalidOperationException("Stream reader engine borrower ledger underflow.");
            }
            _borrowerCount--;
            releaseNow = _disposeRequested && _borrowerCount == 0;
        }

        if (releaseNow)
        {
            ReleaseResources();
        }
    }

    private void ReleaseResources()
    {
        lock (_gate)
        {
            if (_resourcesReleased)
            {
                return;
            }
            _resourcesReleased = true;
        }
        _nativeHandle.Dispose();
    }

    private static byte[] CopyReadableStream(Stream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }
        if (!stream.CanRead)
        {
            throw new ArgumentException("Serialized engine stream must be readable.", nameof(stream));
        }

        using MemoryStream copy = new MemoryStream();
        stream.CopyTo(copy);
        byte[] data = copy.ToArray();
        if (data.Length == 0)
        {
            throw new ArgumentException("Serialized engine stream must not be empty.", nameof(stream));
        }
        return data;
    }
}

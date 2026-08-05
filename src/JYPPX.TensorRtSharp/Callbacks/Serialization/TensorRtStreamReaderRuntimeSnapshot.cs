using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>Copied, pointer-free diagnostics for a native TensorRT IStreamReaderV2 owner.</summary>
public sealed class TensorRtStreamReaderRuntimeSnapshot
{
    internal TensorRtStreamReaderRuntimeSnapshot(
        TensorRtApiLine line,
        ulong ownerId,
        ulong length,
        ulong position,
        ulong deserializeAttemptCount,
        ulong successfulDeserializeCount,
        ulong failedDeserializeCount,
        ulong readCount,
        ulong seekCount,
        ulong hostReadCount,
        ulong deviceReadCount,
        ulong bytesRead,
        ulong requestedBytes,
        ulong failureCount,
        ulong inFlightCallbackCount,
        ulong maxInFlightCallbackCount,
        BridgeStatusCode lastStatus,
        TensorRtStreamSeekPosition lastSeekPosition,
        bool lastHadCudaStream,
        bool lastReadToDevice,
        bool lastOperationSucceeded,
        bool isDeserializing,
        string lastDiagnostic)
    {
        Line = line;
        OwnerId = ownerId;
        Length = length;
        Position = position;
        DeserializeAttemptCount = deserializeAttemptCount;
        SuccessfulDeserializeCount = successfulDeserializeCount;
        FailedDeserializeCount = failedDeserializeCount;
        ReadCount = readCount;
        SeekCount = seekCount;
        HostReadCount = hostReadCount;
        DeviceReadCount = deviceReadCount;
        BytesRead = bytesRead;
        RequestedBytes = requestedBytes;
        FailureCount = failureCount;
        InFlightCallbackCount = inFlightCallbackCount;
        MaxInFlightCallbackCount = maxInFlightCallbackCount;
        LastStatus = lastStatus;
        LastSeekPosition = lastSeekPosition;
        LastHadCudaStream = lastHadCudaStream;
        LastReadToDevice = lastReadToDevice;
        LastOperationSucceeded = lastOperationSucceeded;
        IsDeserializing = isDeserializing;
        LastDiagnostic = lastDiagnostic;
    }

    /// <summary>Gets the TensorRT API line.</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the native owner identity copied as a numeric value.</summary>
    public ulong OwnerId { get; }

    /// <summary>Gets the immutable source length in bytes.</summary>
    public ulong Length { get; }

    /// <summary>Gets the current native reader position.</summary>
    public ulong Position { get; }

    /// <summary>Gets the number of real TensorRT deserialize attempts.</summary>
    public ulong DeserializeAttemptCount { get; }

    /// <summary>Gets the number of successful real TensorRT deserializations.</summary>
    public ulong SuccessfulDeserializeCount { get; }

    /// <summary>Gets the number of failed real TensorRT deserializations.</summary>
    public ulong FailedDeserializeCount { get; }

    /// <summary>Gets the number of IStreamReaderV2 read callbacks.</summary>
    public ulong ReadCount { get; }

    /// <summary>Gets the number of IStreamReaderV2 seek callbacks.</summary>
    public ulong SeekCount { get; }

    /// <summary>Gets the number of reads copied to host destinations.</summary>
    public ulong HostReadCount { get; }

    /// <summary>Gets the number of reads copied to device or managed destinations.</summary>
    public ulong DeviceReadCount { get; }

    /// <summary>Gets the cumulative bytes returned to TensorRT.</summary>
    public ulong BytesRead { get; }

    /// <summary>Gets the cumulative bytes requested by TensorRT.</summary>
    public ulong RequestedBytes { get; }

    /// <summary>Gets the cumulative native callback and deserialize failure count.</summary>
    public ulong FailureCount { get; }

    /// <summary>Gets the current in-flight callback count.</summary>
    public ulong InFlightCallbackCount { get; }

    /// <summary>Gets the maximum concurrent callback count.</summary>
    public ulong MaxInFlightCallbackCount { get; }

    /// <summary>Gets the copied bridge status for the most recent operation.</summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>Gets the origin of the most recent seek.</summary>
    public TensorRtStreamSeekPosition LastSeekPosition { get; }

    /// <summary>Gets whether the most recent read supplied a CUDA stream.</summary>
    public bool LastHadCudaStream { get; }

    /// <summary>Gets whether the most recent read destination was device or managed memory.</summary>
    public bool LastReadToDevice { get; }

    /// <summary>Gets whether the most recent operation succeeded.</summary>
    public bool LastOperationSucceeded { get; }

    /// <summary>Gets whether a native deserialize call is currently active.</summary>
    public bool IsDeserializing { get; }

    /// <summary>Gets a copied diagnostic string without native pointers.</summary>
    public string LastDiagnostic { get; }
}

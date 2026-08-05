using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>Copied, pointer-free diagnostics for a native TensorRT IStreamReaderV2 owner. 原生 TensorRT IStreamReaderV2 所有者的复制型无指针诊断信息。</summary>
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

    /// <summary>Gets the TensorRT API line. 获取 TensorRT API 版本线。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the native owner identity copied as a numeric value. 获取以数值复制的原生所有者标识。</summary>
    public ulong OwnerId { get; }

    /// <summary>Gets the immutable source length in bytes. 获取不可变数据源的字节长度。</summary>
    public ulong Length { get; }

    /// <summary>Gets the current native reader position. 获取原生 reader 的当前位置。</summary>
    public ulong Position { get; }

    /// <summary>Gets the number of real TensorRT deserialize attempts. 获取真实 TensorRT 反序列化尝试次数。</summary>
    public ulong DeserializeAttemptCount { get; }

    /// <summary>Gets the number of successful real TensorRT deserializations. 获取真实 TensorRT 反序列化成功次数。</summary>
    public ulong SuccessfulDeserializeCount { get; }

    /// <summary>Gets the number of failed real TensorRT deserializations. 获取真实 TensorRT 反序列化失败次数。</summary>
    public ulong FailedDeserializeCount { get; }

    /// <summary>Gets the number of IStreamReaderV2 read callbacks. 获取 IStreamReaderV2 读取回调次数。</summary>
    public ulong ReadCount { get; }

    /// <summary>Gets the number of IStreamReaderV2 seek callbacks. 获取 IStreamReaderV2 定位回调次数。</summary>
    public ulong SeekCount { get; }

    /// <summary>Gets the number of reads copied to host destinations. 获取复制到主机目标的读取次数。</summary>
    public ulong HostReadCount { get; }

    /// <summary>Gets the number of reads copied to device or managed destinations. 获取复制到设备或托管目标的读取次数。</summary>
    public ulong DeviceReadCount { get; }

    /// <summary>Gets the cumulative bytes returned to TensorRT. 获取累计返回给 TensorRT 的字节数。</summary>
    public ulong BytesRead { get; }

    /// <summary>Gets the cumulative bytes requested by TensorRT. 获取 TensorRT 累计请求的字节数。</summary>
    public ulong RequestedBytes { get; }

    /// <summary>Gets the cumulative native callback and deserialize failure count. 获取原生回调与反序列化的累计失败次数。</summary>
    public ulong FailureCount { get; }

    /// <summary>Gets the current in-flight callback count. 获取当前正在执行的回调数。</summary>
    public ulong InFlightCallbackCount { get; }

    /// <summary>Gets the maximum concurrent callback count. 获取最大并发回调数。</summary>
    public ulong MaxInFlightCallbackCount { get; }

    /// <summary>Gets the copied bridge status for the most recent operation. 获取最近一次操作复制后的 bridge 状态。</summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>Gets the origin of the most recent seek. 获取最近一次定位的基准位置。</summary>
    public TensorRtStreamSeekPosition LastSeekPosition { get; }

    /// <summary>Gets whether the most recent read supplied a CUDA stream. 获取最近一次读取是否提供了 CUDA stream。</summary>
    public bool LastHadCudaStream { get; }

    /// <summary>Gets whether the most recent read destination was device or managed memory. 获取最近一次读取目标是否为设备或托管内存。</summary>
    public bool LastReadToDevice { get; }

    /// <summary>Gets whether the most recent operation succeeded. 获取最近一次操作是否成功。</summary>
    public bool LastOperationSucceeded { get; }

    /// <summary>Gets whether a native deserialize call is currently active. 获取当前是否存在活动的原生反序列化调用。</summary>
    public bool IsDeserializing { get; }

    /// <summary>Gets a copied diagnostic string without native pointers. 获取不含原生指针的复制型诊断字符串。</summary>
    public string LastDiagnostic { get; }
}

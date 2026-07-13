using System;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Describes one diagnostic request for the future TensorRT debug-listener callback owner.
/// 描述未来 TensorRT debug-listener callback owner 的一次诊断请求。
/// </summary>
/// <remarks>
/// This request contains copied metadata only. It does not carry a TensorRT tensor pointer, tensor buffer pointer, CUDA
/// stream handle, or debug tensor data ownership.
/// 该请求只包含复制出的元数据；不携带 TensorRT tensor pointer、tensor buffer pointer、CUDA stream handle 或 debug tensor
/// 数据所有权。
/// </remarks>
public readonly struct TensorRtDebugListenerCallbackRequest
{
    private const int MaxShapeRank = 8;
    private readonly long[] _shapeDimensions;

    /// <summary>
    /// Creates a debug-listener owner diagnostic request.
    /// 创建 debug-listener owner 诊断请求。
    /// </summary>
    /// <param name="tensorName">The copied debug tensor name. 复制出的 debug tensor 名称。</param>
    /// <param name="dataType">The copied debug tensor data type. 复制出的 debug tensor 数据类型。</param>
    /// <param name="location">The copied debug tensor location. 复制出的 debug tensor 位置。</param>
    /// <param name="shapeDimensions">The copied debug tensor shape dimensions. 复制出的 debug tensor shape 维度。</param>
    /// <param name="reason">A diagnostic reason copied from the caller. 调用方提供的诊断原因。</param>
    /// <param name="isInput">Whether the copied metadata describes an input tensor. 复制出的元数据是否表示输入 tensor。</param>
    /// <param name="isOutput">Whether the copied metadata describes an output tensor. 复制出的元数据是否表示输出 tensor。</param>
    /// <param name="isShapeTensor">Whether the copied metadata describes a shape tensor. 复制出的元数据是否表示 shape tensor。</param>
    /// <param name="isExecutionTensor">Whether the copied metadata describes an execution tensor. 复制出的元数据是否表示 execution tensor。</param>
    public TensorRtDebugListenerCallbackRequest(
        string tensorName,
        TensorRtDataType dataType,
        TensorRtTensorLocation location,
        long[]? shapeDimensions,
        string reason = "",
        bool isInput = false,
        bool isOutput = false,
        bool isShapeTensor = false,
        bool isExecutionTensor = true)
    {
        if (string.IsNullOrWhiteSpace(tensorName))
        {
            throw new ArgumentException("Debug listener tensor name must not be empty.", nameof(tensorName));
        }

        if (!Enum.IsDefined(typeof(TensorRtDataType), dataType))
        {
            throw new ArgumentOutOfRangeException(nameof(dataType), "Debug listener data type must be a known TensorRT data type.");
        }

        if (!Enum.IsDefined(typeof(TensorRtTensorLocation), location))
        {
            throw new ArgumentOutOfRangeException(nameof(location), "Debug listener tensor location must be a known TensorRT tensor location.");
        }

        _shapeDimensions = shapeDimensions == null ? Array.Empty<long>() : (long[])shapeDimensions.Clone();
        if (_shapeDimensions.Length > MaxShapeRank)
        {
            throw new ArgumentOutOfRangeException(nameof(shapeDimensions), "Debug listener diagnostic shape rank must be 8 or less.");
        }

        TensorName = tensorName;
        DataType = dataType;
        Location = location;
        Reason = reason ?? string.Empty;
        IsInput = isInput;
        IsOutput = isOutput;
        IsShapeTensor = isShapeTensor;
        IsExecutionTensor = isExecutionTensor;
    }

    /// <summary>Gets the copied debug tensor name. 获取复制出的 debug tensor 名称。</summary>
    public string TensorName { get; }

    /// <summary>Gets the copied debug tensor data type. 获取复制出的 debug tensor 数据类型。</summary>
    public TensorRtDataType DataType { get; }

    /// <summary>Gets the copied debug tensor location. 获取复制出的 debug tensor 位置。</summary>
    public TensorRtTensorLocation Location { get; }

    /// <summary>Gets the copied debug tensor shape rank. 获取复制出的 debug tensor shape rank。</summary>
    public int ShapeRank => _shapeDimensions.Length;

    /// <summary>Gets the copied debug tensor shape dimensions. 获取复制出的 debug tensor shape 维度。</summary>
    public ReadOnlyCollection<long> ShapeDimensions => Array.AsReadOnly(_shapeDimensions);

    /// <summary>Gets the copied diagnostic reason. 获取复制出的诊断原因。</summary>
    public string Reason { get; }

    /// <summary>Gets whether the copied metadata describes an input tensor. 获取复制出的元数据是否表示输入 tensor。</summary>
    public bool IsInput { get; }

    /// <summary>Gets whether the copied metadata describes an output tensor. 获取复制出的元数据是否表示输出 tensor。</summary>
    public bool IsOutput { get; }

    /// <summary>Gets whether the copied metadata describes a shape tensor. 获取复制出的元数据是否表示 shape tensor。</summary>
    public bool IsShapeTensor { get; }

    /// <summary>Gets whether the copied metadata describes an execution tensor. 获取复制出的元数据是否表示 execution tensor。</summary>
    public bool IsExecutionTensor { get; }

    internal long[] CopyShapeDimensions()
    {
        return (long[])_shapeDimensions.Clone();
    }
}

/// <summary>
/// Reports copied diagnostics for the debug-listener callback owner design gate.
/// 表示 debug-listener callback owner 设计门禁复制出的诊断信息。
/// </summary>
/// <remarks>
/// This snapshot intentionally exposes no native handle, callback owner pointer, borrowed TensorRT tensor pointer, debug
/// tensor data pointer, or tensor buffer ownership. It is not proof that TensorRT has invoked
/// <c>IDebugListener::processDebugTensor</c>.
/// 该快照有意不暴露 native handle、callback owner pointer、borrowed TensorRT tensor pointer、debug tensor data pointer 或
/// tensor buffer ownership。它不证明 TensorRT 已调用 <c>IDebugListener::processDebugTensor</c>。
/// </remarks>
public readonly struct TensorRtDebugListenerCallbackOwnerSnapshot
{
    internal TensorRtDebugListenerCallbackOwnerSnapshot(
        long ownerId,
        string operation,
        TensorRtApiLine line,
        BridgeStatusCode lastStatus,
        string tensorName,
        TensorRtDataType dataType,
        TensorRtTensorLocation location,
        int shapeRank,
        string shapeSummary,
        bool isInput,
        bool isOutput,
        bool isShapeTensor,
        bool isExecutionTensor,
        long invocationCount,
        long processDebugTensorCount,
        long failureCount,
        long inFlightCallbackCount,
        long maxInFlightCallbackCount,
        int activeGateCallCount,
        long releaseHookCount,
        bool callbackStatePinned,
        bool delegatePinned,
        bool disposeRequested,
        string lastDiagnostic,
        string releaseDiagnostic)
    {
        OwnerId = ownerId;
        Operation = operation ?? string.Empty;
        Line = line;
        LastStatus = lastStatus;
        TensorName = tensorName ?? string.Empty;
        DataType = dataType;
        Location = location;
        ShapeRank = shapeRank;
        ShapeSummary = shapeSummary ?? "[]";
        IsInput = isInput;
        IsOutput = isOutput;
        IsShapeTensor = isShapeTensor;
        IsExecutionTensor = isExecutionTensor;
        InvocationCount = invocationCount;
        ProcessDebugTensorCount = processDebugTensorCount;
        FailureCount = failureCount;
        InFlightCallbackCount = inFlightCallbackCount;
        MaxInFlightCallbackCount = maxInFlightCallbackCount;
        ActiveGateCallCount = activeGateCallCount;
        ReleaseHookCount = releaseHookCount;
        CallbackStatePinned = callbackStatePinned;
        DelegatePinned = delegatePinned;
        DisposeRequested = disposeRequested;
        LastDiagnostic = lastDiagnostic ?? string.Empty;
        ReleaseDiagnostic = releaseDiagnostic ?? string.Empty;
    }

    /// <summary>Gets the marker used by readiness to identify this gate. 获取 readiness 用于识别该门禁的 marker。</summary>
    public string EvidenceKind => "debug-listener-callback-owner-design";

    /// <summary>Gets the callback kind represented by this diagnostic snapshot. 获取该诊断快照代表的 callback 类型。</summary>
    public string CallbackKind => "debug-listener-prototype";

    /// <summary>Gets the runtime evidence kind. 获取 runtime 证据类型。</summary>
    public string RuntimeEvidenceKind => "not-present";

    /// <summary>Gets whether this snapshot proves a real TensorRT callback runtime. 获取该快照是否证明真实 TensorRT callback runtime。</summary>
    public bool RealCallbackRuntime => false;

    /// <summary>Gets whether readiness may promote this snapshot as real callback runtime proof. 获取 readiness 是否可将该快照提升为真实 callback runtime proof。</summary>
    public bool IsRealCallbackRuntimeProof => false;

    /// <summary>Gets the copied owner id. 获取复制出的 owner id。</summary>
    public long OwnerId { get; }

    /// <summary>Gets the copied operation label. 获取复制出的操作标签。</summary>
    public string Operation { get; }

    /// <summary>Gets the TensorRT API line used for this diagnostic. 获取该诊断使用的 TensorRT API line。</summary>
    public TensorRtApiLine Line { get; }

    /// <summary>Gets the last copied bridge status. 获取最近一次复制出的 bridge status。</summary>
    public BridgeStatusCode LastStatus { get; }

    /// <summary>Gets the copied debug tensor name. 获取复制出的 debug tensor 名称。</summary>
    public string TensorName { get; }

    /// <summary>Gets the copied debug tensor data type. 获取复制出的 debug tensor 数据类型。</summary>
    public TensorRtDataType DataType { get; }

    /// <summary>Gets the copied debug tensor location. 获取复制出的 debug tensor 位置。</summary>
    public TensorRtTensorLocation Location { get; }

    /// <summary>Gets the copied debug tensor shape rank. 获取复制出的 debug tensor shape rank。</summary>
    public int ShapeRank { get; }

    /// <summary>Gets a compact copied debug tensor shape summary. 获取紧凑的复制 debug tensor shape 摘要。</summary>
    public string ShapeSummary { get; }

    /// <summary>Gets whether the copied metadata describes an input tensor. 获取复制出的元数据是否表示输入 tensor。</summary>
    public bool IsInput { get; }

    /// <summary>Gets whether the copied metadata describes an output tensor. 获取复制出的元数据是否表示输出 tensor。</summary>
    public bool IsOutput { get; }

    /// <summary>Gets whether the copied metadata describes a shape tensor. 获取复制出的元数据是否表示 shape tensor。</summary>
    public bool IsShapeTensor { get; }

    /// <summary>Gets whether the copied metadata describes an execution tensor. 获取复制出的元数据是否表示 execution tensor。</summary>
    public bool IsExecutionTensor { get; }

    /// <summary>Gets the copied invocation count. 获取复制出的调用次数。</summary>
    public long InvocationCount { get; }

    /// <summary>Gets the copied processDebugTensor diagnostic count. 获取复制出的 processDebugTensor 诊断次数。</summary>
    public long ProcessDebugTensorCount { get; }

    /// <summary>Gets the copied failure count. 获取复制出的失败次数。</summary>
    public long FailureCount { get; }

    /// <summary>Gets the copied in-flight callback count. 获取复制出的 in-flight callback 数量。</summary>
    public long InFlightCallbackCount { get; }

    /// <summary>Gets the copied maximum in-flight callback count. 获取复制出的最大 in-flight callback 数量。</summary>
    public long MaxInFlightCallbackCount { get; }

    /// <summary>Gets the copied active gate call count. 获取复制出的 active gate 调用数量。</summary>
    public int ActiveGateCallCount { get; }

    /// <summary>Gets the copied release hook count. 获取复制出的 release hook 次数。</summary>
    public long ReleaseHookCount { get; }

    /// <summary>Gets whether managed callback state remains pinned. 获取托管 callback state 是否仍被 pin 住。</summary>
    public bool CallbackStatePinned { get; }

    /// <summary>Gets whether the managed delegate remains pinned. 获取托管 delegate 是否仍被 pin 住。</summary>
    public bool DelegatePinned { get; }

    /// <summary>Gets whether dispose has been requested. 获取是否已请求释放。</summary>
    public bool DisposeRequested { get; }

    /// <summary>Gets whether this design owner is attached to a TensorRT execution context. 获取该设计 owner 是否已绑定到 TensorRT execution context。</summary>
    public bool IsAttached => false;

    /// <summary>Gets whether debug tensor metadata was copied into this snapshot. 获取 debug tensor metadata 是否已复制到该快照。</summary>
    public bool DebugTensorMetadataCopied => ProcessDebugTensorCount > 0 && !string.IsNullOrEmpty(TensorName);

    /// <summary>Gets whether a debug tensor pointer is exposed by this public API. 获取该 public API 是否暴露 debug tensor pointer。</summary>
    public bool DebugTensorPointerExposed => false;

    /// <summary>Gets whether a debug tensor pointer was produced by this diagnostic gate. 获取该诊断门禁是否产生 debug tensor pointer。</summary>
    public bool DebugTensorPointerProduced => false;

    /// <summary>Gets whether a borrowed debug tensor pointer escaped this public API. 获取 borrowed debug tensor pointer 是否逃逸出该 public API。</summary>
    public bool BorrowedDebugTensorPointerEscaped => false;

    /// <summary>Gets the copied diagnostic message. 获取复制出的诊断消息。</summary>
    public string LastDiagnostic { get; }

    /// <summary>Gets the copied release diagnostic message. 获取复制出的释放诊断消息。</summary>
    public string ReleaseDiagnostic { get; }

    /// <summary>Gets whether the diagnostic completed without failures. 获取诊断是否未出现失败。</summary>
    public bool Succeeded => LastStatus == BridgeStatusCode.Ok && FailureCount == 0 && InFlightCallbackCount == 0;

    /// <summary>Returns a compact diagnostic representation. 返回紧凑的诊断表示。</summary>
    /// <returns>A diagnostic string. 诊断字符串。</returns>
    public override string ToString()
    {
        return $"{EvidenceKind}:owner={OwnerId}:operation={Operation}:status={LastStatus}:process={ProcessDebugTensorCount}:failures={FailureCount}:proof={IsRealCallbackRuntimeProof}";
    }
}

/// <summary>
/// High-level diagnostic owner for the future TensorRT debug-listener callback bridge.
/// 未来 TensorRT debug-listener callback bridge 的高层诊断 owner。
/// </summary>
/// <remarks>
/// This class is a design gate. It validates copied debug tensor metadata, no-throw exception-to-status behavior,
/// in-flight callback accounting, dispose-order diagnostics, and pointer-free public API shape. It does not call
/// TensorRT <c>setDebugListener</c> and does not unlock the <c>IDebugListener::processDebugTensor</c> deferred row.
/// 该类是设计门禁。它验证 debug tensor metadata copy-out、no-throw exception-to-status 行为、in-flight callback 计数、
/// dispose 顺序诊断和不暴露 pointer 的 public API 形状。它不会调用 TensorRT <c>setDebugListener</c>，也不会解除
/// <c>IDebugListener::processDebugTensor</c> 的 deferred row。
/// </remarks>
public sealed class TensorRtDebugListenerCallbackOwner : IDisposable
{
    private const int MaxShapeRank = 8;
    private static long s_nextOwnerId;

    private readonly object _gate = new object();
    private readonly long _ownerId;
    private readonly CallbackState _callbackState = new CallbackState();
    private readonly TensorRtDebugListenerDesignGateCallback _callback;
    private GCHandle _callbackStateHandle;
    private GCHandle _callbackHandle;
    private bool _hasCallbackStateHandle;
    private bool _hasCallbackHandle;
    private bool _disposeRequested;
    private int _activeGateCallCount;

    /// <summary>
    /// Creates a debug-listener callback owner design gate.
    /// 创建 debug-listener callback owner 设计门禁。
    /// </summary>
    public TensorRtDebugListenerCallbackOwner()
    {
        _ownerId = Interlocked.Increment(ref s_nextOwnerId);
        _callback = InvokeDebugListenerDesignGate;
        _callbackStateHandle = GCHandle.Alloc(_callbackState);
        _callbackHandle = GCHandle.Alloc(_callback);
        _hasCallbackStateHandle = true;
        _hasCallbackHandle = true;
    }

    /// <summary>Gets whether dispose has been requested. 获取是否已请求释放。</summary>
    public bool IsDisposed
    {
        get
        {
            lock (_gate)
            {
                return _disposeRequested;
            }
        }
    }

    /// <summary>Gets whether this owner is attached to a TensorRT execution context. 获取该 owner 是否已绑定到 TensorRT execution context。</summary>
    public bool IsAttached => false;

    /// <summary>
    /// Runs the debug-listener owner design diagnostic.
    /// 执行 debug-listener owner 设计诊断。
    /// </summary>
    /// <param name="line">The TensorRT API line represented by this diagnostic. 该诊断代表的 TensorRT API line。</param>
    /// <param name="request">The copied debug tensor diagnostic request. 复制出的 debug tensor 诊断请求。</param>
    /// <returns>A pointer-free copied diagnostic snapshot. 不含 pointer 的复制诊断快照。</returns>
    public TensorRtDebugListenerCallbackOwnerSnapshot RunDesignDiagnostic(
        TensorRtApiLine line,
        TensorRtDebugListenerCallbackRequest request)
    {
        IntPtr callbackState;
        TensorRtDebugListenerDesignGateCallback callback;
        lock (_gate)
        {
            if (_disposeRequested || !_hasCallbackStateHandle)
            {
                throw new ObjectDisposedException(nameof(TensorRtDebugListenerCallbackOwner));
            }

            checked
            {
                _activeGateCallCount++;
            }

            callbackState = GCHandle.ToIntPtr(_callbackStateHandle);
            callback = _callback;
        }

        BridgeStatusCode status;
        try
        {
            long[] shape = request.CopyShapeDimensions();
            using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(request.TensorName);
            using Utf8Interop.Utf8StringScope reasonUtf8 = Utf8Interop.ToNativeString(request.Reason);
            status = callback(
                (int)line,
                tensorNameUtf8.Pointer,
                (int)request.DataType,
                (int)request.Location,
                request.IsInput ? 1 : 0,
                request.IsOutput ? 1 : 0,
                request.IsShapeTensor ? 1 : 0,
                request.IsExecutionTensor ? 1 : 0,
                request.ShapeRank,
                GetDimension(shape, 0),
                GetDimension(shape, 1),
                GetDimension(shape, 2),
                GetDimension(shape, 3),
                GetDimension(shape, 4),
                GetDimension(shape, 5),
                GetDimension(shape, 6),
                GetDimension(shape, 7),
                reasonUtf8.Pointer,
                callbackState);
        }
        finally
        {
            bool releaseNow;
            lock (_gate)
            {
                _activeGateCallCount--;
                releaseNow = _activeGateCallCount == 0 && _disposeRequested;
            }

            if (releaseNow)
            {
                FreeCallbackState();
            }
        }

        return CreateSnapshot(status, "process-debug-tensor", line);
    }

    /// <summary>
    /// Gets a copied snapshot of the current design gate state.
    /// 获取当前设计门禁状态的复制快照。
    /// </summary>
    /// <param name="operation">A copied operation label. 复制出的操作标签。</param>
    /// <returns>A pointer-free copied diagnostic snapshot. 不含 pointer 的复制诊断快照。</returns>
    public TensorRtDebugListenerCallbackOwnerSnapshot GetSnapshot(string operation = "snapshot")
    {
        return CreateSnapshot(_callbackState.LastStatus, operation, _callbackState.LastLine);
    }

    /// <summary>
    /// Releases managed keep-alive handles owned by the design gate.
    /// 释放该设计门禁持有的托管 keep-alive 句柄。
    /// </summary>
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
            releaseNow = _activeGateCallCount == 0;
        }

        if (releaseNow)
        {
            FreeCallbackState();
        }

        GC.SuppressFinalize(this);
    }

    private void FreeCallbackState()
    {
        bool released = false;
        if (_hasCallbackHandle)
        {
            _callbackHandle.Free();
            _hasCallbackHandle = false;
            released = true;
        }

        if (_hasCallbackStateHandle)
        {
            _callbackStateHandle.Free();
            _hasCallbackStateHandle = false;
            released = true;
        }

        if (released)
        {
            _callbackState.RecordReleaseHook("debug-listener-callback-owner-design release hook released managed GCHandle and delegate keep-alive handles after callbacks drained.");
            GC.KeepAlive(_callback);
        }
    }

    private TensorRtDebugListenerCallbackOwnerSnapshot CreateSnapshot(BridgeStatusCode status, string operation, TensorRtApiLine line)
    {
        bool callbackStatePinned;
        bool delegatePinned;
        bool disposeRequested;
        int activeGateCallCount;
        lock (_gate)
        {
            callbackStatePinned = _hasCallbackStateHandle;
            delegatePinned = _hasCallbackHandle;
            disposeRequested = _disposeRequested;
            activeGateCallCount = _activeGateCallCount;
        }

        BridgeStatusCode lastStatus = _callbackState.LastStatus;
        if (lastStatus != status)
        {
            lastStatus = status;
        }

        return new TensorRtDebugListenerCallbackOwnerSnapshot(
            ownerId: _ownerId,
            operation: operation,
            line: line,
            lastStatus: lastStatus,
            tensorName: _callbackState.LastTensorName,
            dataType: _callbackState.LastDataType,
            location: _callbackState.LastLocation,
            shapeRank: _callbackState.LastShapeRank,
            shapeSummary: _callbackState.LastShapeSummary,
            isInput: _callbackState.LastIsInput,
            isOutput: _callbackState.LastIsOutput,
            isShapeTensor: _callbackState.LastIsShapeTensor,
            isExecutionTensor: _callbackState.LastIsExecutionTensor,
            invocationCount: _callbackState.InvocationCount,
            processDebugTensorCount: _callbackState.ProcessDebugTensorCount,
            failureCount: _callbackState.FailureCount,
            inFlightCallbackCount: _callbackState.InFlightCallbackCount,
            maxInFlightCallbackCount: _callbackState.MaxInFlightCallbackCount,
            activeGateCallCount: activeGateCallCount,
            releaseHookCount: _callbackState.ReleaseHookCount,
            callbackStatePinned: callbackStatePinned,
            delegatePinned: delegatePinned,
            disposeRequested: disposeRequested,
            lastDiagnostic: _callbackState.LastDiagnostic,
            releaseDiagnostic: _callbackState.LastReleaseDiagnostic);
    }

    private static BridgeStatusCode InvokeDebugListenerDesignGate(
        int line,
        IntPtr tensorName,
        int dataType,
        int location,
        int isInput,
        int isOutput,
        int isShapeTensor,
        int isExecutionTensor,
        int shapeRank,
        long dim0,
        long dim1,
        long dim2,
        long dim3,
        long dim4,
        long dim5,
        long dim6,
        long dim7,
        IntPtr reason,
        IntPtr userState)
    {
        CallbackState? state = null;
        try
        {
            if (userState == IntPtr.Zero)
            {
                return BridgeStatusCode.InvalidArgument;
            }

            state = GCHandle.FromIntPtr(userState).Target as CallbackState;
            if (state == null)
            {
                return BridgeStatusCode.InvalidState;
            }

            state.EnterCallback();
            state.RecordInvocation();

            TensorRtApiLine apiLine = Enum.IsDefined(typeof(TensorRtApiLine), line)
                ? (TensorRtApiLine)line
                : TensorRtApiLine.TensorRt11;
            string tensor = Utf8Interop.ReadString(tensorName);
            string gateReason = Utf8Interop.ReadString(reason);
            TensorRtDataType copiedDataType = Enum.IsDefined(typeof(TensorRtDataType), dataType)
                ? (TensorRtDataType)dataType
                : TensorRtDataType.Unknown;
            TensorRtTensorLocation copiedLocation = Enum.IsDefined(typeof(TensorRtTensorLocation), location)
                ? (TensorRtTensorLocation)location
                : TensorRtTensorLocation.Device;
            string shapeSummary = FormatShape(shapeRank, dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7);
            state.RecordRequest(
                apiLine,
                tensor,
                copiedDataType,
                copiedLocation,
                shapeRank,
                shapeSummary,
                isInput != 0,
                isOutput != 0,
                isShapeTensor != 0,
                isExecutionTensor != 0);

            if (string.Equals(gateReason, "throw", StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidOperationException("synthetic debug listener callback owner design failure");
            }

            if (string.IsNullOrWhiteSpace(tensor))
            {
                const string diagnostic = "debug-listener-callback-owner-design invalid tensor name; no TensorRT debug listener callback was invoked.";
                state.RecordReturnedFailure(diagnostic, BridgeStatusCode.InvalidArgument);
                return BridgeStatusCode.InvalidArgument;
            }

            if (shapeRank < 0 || shapeRank > MaxShapeRank)
            {
                const string diagnostic = "debug-listener-callback-owner-design invalid shape rank; no TensorRT debug listener callback was invoked.";
                state.RecordReturnedFailure(diagnostic, BridgeStatusCode.InvalidArgument);
                return BridgeStatusCode.InvalidArgument;
            }

            string successDiagnostic =
                "debug-listener-callback-owner-design process-debug-tensor copied tensor=" + tensor +
                " dataType=" + copiedDataType +
                " location=" + copiedLocation +
                " shape=" + shapeSummary +
                " input=" + (isInput != 0).ToString(CultureInfo.InvariantCulture) +
                " output=" + (isOutput != 0).ToString(CultureInfo.InvariantCulture) +
                " shapeTensor=" + (isShapeTensor != 0).ToString(CultureInfo.InvariantCulture) +
                " executionTensor=" + (isExecutionTensor != 0).ToString(CultureInfo.InvariantCulture) +
                " debug-tensor-pointer-exposed=false; no TensorRT debug listener callback was invoked.";
            state.RecordStatus(BridgeStatusCode.Ok, successDiagnostic);
            return BridgeStatusCode.Ok;
        }
        catch (Exception exception)
        {
            string diagnostic = "debug listener callback owner design handler threw " + exception.GetType().Name + ": " + exception.Message;
            state?.RecordFailure(exception, diagnostic, BridgeStatusCode.InvalidState);
            return BridgeStatusCode.InvalidState;
        }
        finally
        {
            state?.ExitCallback();
        }
    }

    private static long GetDimension(long[] dimensions, int index)
    {
        return index >= 0 && index < dimensions.Length ? dimensions[index] : 0L;
    }

    private static string FormatShape(
        int rank,
        long dim0,
        long dim1,
        long dim2,
        long dim3,
        long dim4,
        long dim5,
        long dim6,
        long dim7)
    {
        if (rank <= 0)
        {
            return "[]";
        }

        long[] dimensions = new[] { dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7 };
        int copiedRank = Math.Min(rank, MaxShapeRank);
        string[] values = new string[copiedRank];
        for (int index = 0; index < copiedRank; index++)
        {
            values[index] = dimensions[index].ToString(CultureInfo.InvariantCulture);
        }

        return "[" + string.Join("x", values) + "]";
    }

    private sealed class CallbackState
    {
        private long _invocationCount;
        private long _processDebugTensorCount;
        private long _failureCount;
        private long _inFlightCallbackCount;
        private long _maxInFlightCallbackCount;
        private long _releaseHookCount;
        private int _lastStatus;
        private int _lastLine = (int)TensorRtApiLine.TensorRt11;
        private int _lastDataType = (int)TensorRtDataType.Unknown;
        private int _lastLocation = (int)TensorRtTensorLocation.Device;
        private int _lastShapeRank;
        private int _lastIsInput;
        private int _lastIsOutput;
        private int _lastIsShapeTensor;
        private int _lastIsExecutionTensor;
        private Exception? _lastException;
        private string _lastTensorName = string.Empty;
        private string _lastShapeSummary = "[]";
        private string _lastDiagnostic = string.Empty;
        private string _lastReleaseDiagnostic = string.Empty;

        public long InvocationCount => Interlocked.Read(ref _invocationCount);

        public long ProcessDebugTensorCount => Interlocked.Read(ref _processDebugTensorCount);

        public long FailureCount => Interlocked.Read(ref _failureCount);

        public long InFlightCallbackCount => Interlocked.Read(ref _inFlightCallbackCount);

        public long MaxInFlightCallbackCount => Interlocked.Read(ref _maxInFlightCallbackCount);

        public long ReleaseHookCount => Interlocked.Read(ref _releaseHookCount);

        public BridgeStatusCode LastStatus => (BridgeStatusCode)Volatile.Read(ref _lastStatus);

        public TensorRtApiLine LastLine => (TensorRtApiLine)Volatile.Read(ref _lastLine);

        public TensorRtDataType LastDataType => (TensorRtDataType)Volatile.Read(ref _lastDataType);

        public TensorRtTensorLocation LastLocation => (TensorRtTensorLocation)Volatile.Read(ref _lastLocation);

        public int LastShapeRank => Volatile.Read(ref _lastShapeRank);

        public bool LastIsInput => Volatile.Read(ref _lastIsInput) != 0;

        public bool LastIsOutput => Volatile.Read(ref _lastIsOutput) != 0;

        public bool LastIsShapeTensor => Volatile.Read(ref _lastIsShapeTensor) != 0;

        public bool LastIsExecutionTensor => Volatile.Read(ref _lastIsExecutionTensor) != 0;

        public Exception? LastException => Volatile.Read(ref _lastException);

        public string LastTensorName => Volatile.Read(ref _lastTensorName);

        public string LastShapeSummary => Volatile.Read(ref _lastShapeSummary);

        public string LastDiagnostic => Volatile.Read(ref _lastDiagnostic);

        public string LastReleaseDiagnostic => Volatile.Read(ref _lastReleaseDiagnostic);

        public void EnterCallback()
        {
            long current = Interlocked.Increment(ref _inFlightCallbackCount);
            while (true)
            {
                long observedMax = Interlocked.Read(ref _maxInFlightCallbackCount);
                if (current <= observedMax)
                {
                    return;
                }

                if (Interlocked.CompareExchange(ref _maxInFlightCallbackCount, current, observedMax) == observedMax)
                {
                    return;
                }
            }
        }

        public void ExitCallback()
        {
            Interlocked.Decrement(ref _inFlightCallbackCount);
        }

        public void RecordInvocation()
        {
            Interlocked.Increment(ref _invocationCount);
            Interlocked.Increment(ref _processDebugTensorCount);
        }

        public void RecordRequest(
            TensorRtApiLine line,
            string tensorName,
            TensorRtDataType dataType,
            TensorRtTensorLocation location,
            int shapeRank,
            string shapeSummary,
            bool isInput,
            bool isOutput,
            bool isShapeTensor,
            bool isExecutionTensor)
        {
            Volatile.Write(ref _lastLine, (int)line);
            Volatile.Write(ref _lastTensorName, tensorName ?? string.Empty);
            Volatile.Write(ref _lastDataType, (int)dataType);
            Volatile.Write(ref _lastLocation, (int)location);
            Volatile.Write(ref _lastShapeRank, shapeRank);
            Volatile.Write(ref _lastShapeSummary, shapeSummary ?? "[]");
            Volatile.Write(ref _lastIsInput, isInput ? 1 : 0);
            Volatile.Write(ref _lastIsOutput, isOutput ? 1 : 0);
            Volatile.Write(ref _lastIsShapeTensor, isShapeTensor ? 1 : 0);
            Volatile.Write(ref _lastIsExecutionTensor, isExecutionTensor ? 1 : 0);
        }

        public void RecordStatus(BridgeStatusCode status, string diagnostic)
        {
            Volatile.Write(ref _lastStatus, (int)status);
            Volatile.Write(ref _lastDiagnostic, diagnostic ?? string.Empty);
        }

        public void RecordReturnedFailure(string diagnostic, BridgeStatusCode status)
        {
            RecordStatus(status, diagnostic);
            Interlocked.Increment(ref _failureCount);
        }

        public void RecordFailure(Exception exception, string diagnostic, BridgeStatusCode status)
        {
            Volatile.Write(ref _lastException, exception);
            RecordStatus(status, diagnostic);
            Interlocked.Increment(ref _failureCount);
        }

        public void RecordReleaseHook(string diagnostic)
        {
            Volatile.Write(ref _lastReleaseDiagnostic, diagnostic ?? string.Empty);
            Interlocked.Increment(ref _releaseHookCount);
        }
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate BridgeStatusCode TensorRtDebugListenerDesignGateCallback(
        int line,
        IntPtr tensorName,
        int dataType,
        int location,
        int isInput,
        int isOutput,
        int isShapeTensor,
        int isExecutionTensor,
        int shapeRank,
        long dim0,
        long dim1,
        long dim2,
        long dim3,
        long dim4,
        long dim5,
        long dim6,
        long dim7,
        IntPtr reason,
        IntPtr userState);
}

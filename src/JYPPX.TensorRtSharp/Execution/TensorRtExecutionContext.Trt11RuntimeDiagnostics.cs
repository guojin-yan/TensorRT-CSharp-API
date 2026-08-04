using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtExecutionContext
{
    /// <summary>
    /// Probes the retained execution-context error-buffer compatibility API.
    /// 探测保留的 execution context error-buffer 兼容 API。
    /// </summary>
    /// <param name="errorBuffer">The copied error text, or an empty string when unavailable. / 复制出的错误文本；不可用时为空字符串。</param>
    /// <returns><see langword="false"/> for the currently supported standard TensorRT execution-context types. / 对当前支持的标准 TensorRT execution context 类型返回 <see langword="false"/>。</returns>
    /// <remarks>
    /// The standard <c>nvinfer1::IExecutionContext</c> type used by this bridge does not expose the safe-runtime
    /// <c>getErrorBuffer</c> query in the supported vendor headers. The ABI and managed method remain available only
    /// to return a controlled deferred diagnostic; no borrowed vendor pointer is accessed or exposed.
    /// 本 bridge 使用的标准 <c>nvinfer1::IExecutionContext</c> 类型在已支持 vendor headers 中不提供 safe-runtime
    /// <c>getErrorBuffer</c> 查询。ABI 与托管方法仅保留用于返回受控 deferred 诊断，不访问或暴露 vendor borrowed pointer。
    /// </remarks>
    public bool TryGetErrorBuffer(out string errorBuffer)
    {
        return TryGetErrorBuffer(out errorBuffer, out _);
    }

    /// <summary>
    /// Probes the retained execution-context error-buffer compatibility API and returns its diagnostic.
    /// 探测保留的 execution context error-buffer 兼容 API，并返回诊断信息。
    /// </summary>
    /// <param name="errorBuffer">The copied error text, or an empty string when unavailable. / 复制出的错误文本；不可用时为空字符串。</param>
    /// <param name="diagnostic">A short diagnostic string describing success or failure. / 描述成功或失败原因的简短诊断。</param>
    /// <returns><see langword="false"/> while the vendor query remains deferred. / vendor 查询保持 deferred 时返回 <see langword="false"/>。</returns>
    public bool TryGetErrorBuffer(out string errorBuffer, out string diagnostic)
    {
        try
        {
            errorBuffer = NativeBridgeApi.GetExecutionContextErrorBuffer(Line, _handle);
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            errorBuffer = string.Empty;
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Gets the native address value currently bound to a named TensorRT tensor for diagnostics.
    /// 获取当前绑定到指定 TensorRT tensor 的原生地址数值，仅用于诊断。
    /// </summary>
    /// <param name="tensorName">The input or output tensor name. / 输入或输出 tensor 名称。</param>
    /// <returns>
    /// The address value reported by TensorRT, or 0 when no address is bound.
    /// TensorRT 报告的地址数值；未绑定地址时返回 0。
    /// </returns>
    /// <remarks>
    /// This value is intentionally exposed as an integer diagnostic value instead of a user-owned pointer.
    /// 该值特意以整数诊断值形式暴露，而不是用户可拥有或解引用的指针。
    /// </remarks>
    public ulong GetTensorAddressValue(string tensorName)
    {
        return NativeBridgeApi.GetExecutionContextTensorAddressValue(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets the native address value currently bound to a named TensorRT output tensor for diagnostics.
    /// 获取当前绑定到指定 TensorRT 输出 tensor 的原生地址数值，仅用于诊断。
    /// </summary>
    /// <param name="tensorName">The output tensor name. / 输出 tensor 名称。</param>
    /// <returns>The address value reported by TensorRT, or 0 when no address is bound. / TensorRT 报告的地址数值；未绑定时返回 0。</returns>
    public ulong GetOutputTensorAddressValue(string tensorName)
    {
        return NativeBridgeApi.GetExecutionContextOutputTensorAddressValue(Line, _handle, tensorName);
    }

    /// <summary>
    /// Clears the output allocator attached to a named output tensor.
    /// 清除绑定到指定输出 tensor 的 output allocator；适用于 TensorRT 8/10/11，不会销毁 allocator 或调用用户 reallocate 回调。
    /// </summary>
    /// <param name="tensorName">The output tensor name. / 输出 tensor 名称。</param>
    /// <returns><see langword="true"/> when TensorRT accepts the clear operation. / TensorRT 接受清理操作时返回 <see langword="true"/>。</returns>
    public bool ClearOutputAllocator(string tensorName)
    {
        if (string.IsNullOrWhiteSpace(tensorName))
        {
            throw new ArgumentException("Output tensor name must not be empty.", nameof(tensorName));
        }

        if (TensorRtOutputAllocatorCallbackOwner.IsExecutingRuntimeCallbackOnCurrentThread)
        {
            throw new InvalidOperationException("An output allocator cannot be cleared from inside its own callback.");
        }

        lock (_outputAllocatorLeaseLock)
        {
            if (_outputAllocatorContextDisposed)
            {
                throw new ObjectDisposedException(nameof(TensorRtExecutionContext));
            }

            if (!_outputAllocatorKeepAlive.TryGetValue(tensorName, out TensorRtOutputAllocatorCallbackOwner? owner))
            {
                return NativeBridgeApi.ClearExecutionContextOutputAllocator(Line, _handle, tensorName);
            }

            bool detached = NativeBridgeApi.DetachOutputAllocatorOwner(Line, owner.NativeHandle);
            if (detached)
            {
                _outputAllocatorKeepAlive.Remove(tensorName);
                owner.DetachBorrower();
            }

            return detached;
        }
    }

    /// <summary>
    /// Clears the temporary-storage allocator attached to this execution context.
    /// 清除绑定到当前 execution context 的 temporary-storage allocator；适用于 TensorRT 8/10/11，不会销毁 allocator 或调用用户 deallocate 回调。
    /// </summary>
    /// <returns><see langword="true"/> when TensorRT accepts the clear operation. / TensorRT 接受清理操作时返回 <see langword="true"/>。</returns>
    public bool ClearTemporaryStorageAllocator()
    {
        return NativeBridgeApi.ClearExecutionContextTemporaryStorageAllocator(Line, _handle);
    }

    /// <summary>
    /// Clears the debug listener attached to this TensorRT 10/11 execution context.
    /// 清除绑定到当前 TensorRT 10/11 execution context 的 debug listener；不会销毁 listener 或调用用户回调。
    /// </summary>
    /// <returns><see langword="true"/> when TensorRT accepts the clear operation. / TensorRT 接受清理操作时返回 <see langword="true"/>。</returns>
    public bool ClearDebugListener()
    {
        if (TensorRtDebugListenerCallbackOwner.IsExecutingRuntimeCallbackOnCurrentThread)
        {
            throw new InvalidOperationException("A debug listener cannot be detached from inside its own callback.");
        }

        lock (_debugListenerLeaseLock)
        {
            if (_debugListenerContextDisposed)
            {
                throw new ObjectDisposedException(nameof(TensorRtExecutionContext));
            }

            TensorRtDebugListenerCallbackOwner? listener = _debugListenerKeepAlive;
            if (listener == null)
            {
                return NativeBridgeApi.ClearExecutionContextDebugListener(Line, _handle);
            }

            bool detached = NativeBridgeApi.DetachDebugListenerOwner(Line, listener.NativeHandle);
            if (detached)
            {
                _debugListenerKeepAlive = null;
                listener.DetachBorrower();
            }

            return detached;
        }
    }

    /// <summary>
    /// Gets whether this TensorRT 10/11 execution context has a debug listener attached.
    /// 获取当前 TensorRT 10/11 execution context 是否绑定了 debug listener；不会暴露 listener 指针或接管其生命周期。
    /// </summary>
    public bool HasDebugListener => NativeBridgeApi.HasExecutionContextDebugListener(Line, _handle);

    /// <summary>
    /// Tries to get copied versioned-interface metadata for the output allocator attached to a named output tensor.
    /// 尝试获取指定输出 tensor 已绑定 output allocator 的 versioned-interface 元数据副本。
    /// </summary>
    /// <param name="tensorName">The output tensor name. / 输出 tensor 名称。</param>
    /// <param name="interfaceInfo">The copied interface metadata when the query succeeds. / 查询成功时复制出的 interface 元数据。</param>
    /// <returns><see langword="true"/> when TensorRT reports an output allocator for the tensor and metadata was copied. / TensorRT 报告该 tensor 已绑定 output allocator 且成功复制元数据时返回 <see langword="true"/>。</returns>
    public bool TryGetOutputAllocatorInterfaceInfo(string tensorName, out TensorRtInterfaceInfo interfaceInfo)
    {
        return TryGetOutputAllocatorInterfaceInfo(tensorName, out interfaceInfo, out _);
    }

    /// <summary>
    /// Tries to get copied versioned-interface metadata for the output allocator attached to a named output tensor.
    /// 尝试获取指定输出 tensor 已绑定 output allocator 的 versioned-interface 元数据副本。
    /// </summary>
    /// <param name="tensorName">The output tensor name. / 输出 tensor 名称。</param>
    /// <param name="interfaceInfo">The copied interface metadata when the query succeeds. / 查询成功时复制出的 interface 元数据。</param>
    /// <param name="diagnostic">A short diagnostic string describing success or the reason for failure. / 描述成功或失败原因的简短诊断。</param>
    /// <returns><see langword="true"/> when TensorRT reports an output allocator for the tensor and metadata was copied. / TensorRT 报告该 tensor 已绑定 output allocator 且成功复制元数据时返回 <see langword="true"/>。</returns>
    /// <remarks>
    /// This query copies TensorRT metadata immediately and does not expose, retain, or take ownership of the borrowed allocator pointer.
    /// 该查询会立即复制 TensorRT 元数据，不会暴露、保留或接管 borrowed allocator 指针。
    /// </remarks>
    public bool TryGetOutputAllocatorInterfaceInfo(string tensorName, out TensorRtInterfaceInfo interfaceInfo, out string diagnostic)
    {
        try
        {
            interfaceInfo = NativeBridgeApi.GetExecutionContextOutputAllocatorInterfaceInfo(Line, _handle, tensorName);
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            interfaceInfo = new TensorRtInterfaceInfo(string.Empty, 0, 0);
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Tries to get copied versioned-interface metadata for the temporary-storage allocator attached to this execution context.
    /// 尝试获取当前 execution context 已绑定 temporary-storage allocator 的 versioned-interface 元数据副本。
    /// </summary>
    /// <param name="interfaceInfo">The copied interface metadata when the query succeeds. / 查询成功时复制出的 interface 元数据。</param>
    /// <returns><see langword="true"/> when TensorRT reports a temporary-storage allocator and metadata was copied. / TensorRT 报告已绑定 temporary-storage allocator 且成功复制元数据时返回 <see langword="true"/>。</returns>
    public bool TryGetTemporaryStorageAllocatorInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo)
    {
        return TryGetTemporaryStorageAllocatorInterfaceInfo(out interfaceInfo, out _);
    }

    /// <summary>
    /// Tries to get copied versioned-interface metadata for the temporary-storage allocator attached to this execution context.
    /// 尝试获取当前 execution context 已绑定 temporary-storage allocator 的 versioned-interface 元数据副本。
    /// </summary>
    /// <param name="interfaceInfo">The copied interface metadata when the query succeeds. / 查询成功时复制出的 interface 元数据。</param>
    /// <param name="diagnostic">A short diagnostic string describing success or the reason for failure. / 描述成功或失败原因的简短诊断。</param>
    /// <returns><see langword="true"/> when TensorRT reports a temporary-storage allocator and metadata was copied. / TensorRT 报告已绑定 temporary-storage allocator 且成功复制元数据时返回 <see langword="true"/>。</returns>
    /// <remarks>
    /// This query maps the TensorRT <c>IGpuAllocator::getInterfaceInfo</c> surface through the execution context's borrowed temporary-storage allocator.
    /// It copies metadata only and does not expose or own the allocator pointer.
    /// 该查询通过 execution context 借出的 temporary-storage allocator 覆盖 TensorRT <c>IGpuAllocator::getInterfaceInfo</c> 表面；
    /// 只复制元数据，不暴露或拥有 allocator 指针。
    /// </remarks>
    public bool TryGetTemporaryStorageAllocatorInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo, out string diagnostic)
    {
        try
        {
            interfaceInfo = NativeBridgeApi.GetExecutionContextTemporaryStorageAllocatorInterfaceInfo(Line, _handle);
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            interfaceInfo = new TensorRtInterfaceInfo(string.Empty, 0, 0);
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Tries to get copied versioned-interface metadata for the debug listener attached to this execution context.
    /// 尝试获取当前 execution context 已绑定 debug listener 的 versioned-interface 元数据副本。
    /// </summary>
    /// <param name="interfaceInfo">The copied interface metadata when the query succeeds. / 查询成功时复制出的 interface 元数据。</param>
    /// <returns><see langword="true"/> when TensorRT reports a debug listener and metadata was copied. / TensorRT 报告已绑定 debug listener 且成功复制元数据时返回 <see langword="true"/>。</returns>
    public bool TryGetDebugListenerInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo)
    {
        return TryGetDebugListenerInterfaceInfo(out interfaceInfo, out _);
    }

    /// <summary>
    /// Tries to get copied versioned-interface metadata for the debug listener attached to this execution context.
    /// 尝试获取当前 execution context 已绑定 debug listener 的 versioned-interface 元数据副本。
    /// </summary>
    /// <param name="interfaceInfo">The copied interface metadata when the query succeeds. / 查询成功时复制出的 interface 元数据。</param>
    /// <param name="diagnostic">A short diagnostic string describing success or the reason for failure. / 描述成功或失败原因的简短诊断。</param>
    /// <returns><see langword="true"/> when TensorRT reports a debug listener and metadata was copied. / TensorRT 报告已绑定 debug listener 且成功复制元数据时返回 <see langword="true"/>。</returns>
    /// <remarks>
    /// This query copies TensorRT metadata immediately and does not expose, retain, or take ownership of the borrowed debug-listener pointer.
    /// 该查询会立即复制 TensorRT 元数据，不会暴露、保留或接管 borrowed debug-listener 指针。
    /// </remarks>
    public bool TryGetDebugListenerInterfaceInfo(out TensorRtInterfaceInfo interfaceInfo, out string diagnostic)
    {
        try
        {
            interfaceInfo = NativeBridgeApi.GetExecutionContextDebugListenerInterfaceInfo(Line, _handle);
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            interfaceInfo = new TensorRtInterfaceInfo(string.Empty, 0, 0);
            diagnostic = exception.Message;
            return false;
        }
    }

    /// <summary>
    /// Gets a copied snapshot of output allocator, temporary-storage allocator, and debug listener boundary state.
    /// 获取 output allocator、temporary-storage allocator 与 debug listener 边界状态的复制快照。
    /// </summary>
    /// <param name="outputTensorName">The output tensor name used for output allocator queries. / 用于 output allocator 查询的输出 tensor 名称。</param>
    /// <returns>A copied callback state snapshot. 复制出的回调状态快照。</returns>
    /// <remarks>
    /// This query copies state and interface metadata immediately. It does not expose, retain, or take ownership of
    /// borrowed allocator/debug-listener pointers and it does not invoke output allocator or debug listener callbacks.
    /// 该查询会立即复制状态与 interface 元数据，不会暴露、保留或接管 borrowed allocator/debug-listener 指针，也不会调用
    /// output allocator 或 debug listener 回调。
    /// </remarks>
    public TensorRtExecutionContextCallbackStateSnapshot GetCallbackStateSnapshot(string outputTensorName)
    {
        NativeTensorRtExecutionContextCallbackStateInfo info =
            NativeBridgeApi.GetExecutionContextCallbackStateSnapshot(Line, _handle, outputTensorName);
        return CreateCallbackStateSnapshot(info);
    }

    /// <summary>
    /// Tries to get a copied callback-state snapshot, including partial state when an individual native query fails.
    /// 尝试获取回调状态副本；单个原生查询失败时仍返回已经复制出的 partial state。
    /// </summary>
    /// <param name="outputTensorName">The output tensor name used for output allocator queries. / 用于 output allocator 查询的输出 tensor 名称。</param>
    /// <param name="snapshot">The complete or partial pointer-free snapshot. / 完整或部分可用的 pointer-free 快照。</param>
    /// <returns><see langword="true"/> only when every snapshot phase completed successfully. / 仅当所有快照阶段均成功时返回 <see langword="true"/>。</returns>
    public bool TryGetCallbackStateSnapshot(
        string outputTensorName,
        out TensorRtExecutionContextCallbackStateSnapshot snapshot)
    {
        return TryGetCallbackStateSnapshot(outputTensorName, out snapshot, out _);
    }

    /// <summary>
    /// Tries to get a copied callback-state snapshot, including partial state and a failure diagnostic.
    /// 尝试获取回调状态副本，并在失败时保留 partial state 与诊断信息。
    /// </summary>
    /// <param name="outputTensorName">The output tensor name used for output allocator queries. / 用于 output allocator 查询的输出 tensor 名称。</param>
    /// <param name="snapshot">The complete or partial pointer-free snapshot. / 完整或部分可用的 pointer-free 快照。</param>
    /// <param name="diagnostic">A copied native diagnostic for the first failed phase. / 首个失败阶段的原生诊断副本。</param>
    /// <returns><see langword="true"/> only when every snapshot phase completed successfully. / 仅当所有快照阶段均成功时返回 <see langword="true"/>。</returns>
    /// <remarks>
    /// A <see langword="false"/> result does not invalidate fields collected before or after an optional interface-info
    /// failure. Check <see cref="TensorRtExecutionContextCallbackStateSnapshot.LastOperation"/> and
    /// <see cref="TensorRtExecutionContextCallbackStateSnapshot.LastStatus"/> before consuming partial metadata.
    /// 返回 <see langword="false"/> 不会使可选 interface-info 失败前后已采集的字段失效；使用 partial metadata 前应检查
    /// <see cref="TensorRtExecutionContextCallbackStateSnapshot.LastOperation"/> 与
    /// <see cref="TensorRtExecutionContextCallbackStateSnapshot.LastStatus"/>。
    /// </remarks>
    public bool TryGetCallbackStateSnapshot(
        string outputTensorName,
        out TensorRtExecutionContextCallbackStateSnapshot snapshot,
        out string diagnostic)
    {
        bool complete = NativeBridgeApi.TryGetExecutionContextCallbackStateSnapshot(
            Line,
            _handle,
            outputTensorName,
            out NativeTensorRtExecutionContextCallbackStateInfo info,
            out diagnostic);
        snapshot = CreateCallbackStateSnapshot(info);
        if (string.IsNullOrEmpty(diagnostic))
        {
            diagnostic = snapshot.Diagnostic;
        }

        return complete;
    }

    /// <summary>
    /// Gets a pointer-free runtime diagnostic snapshot for one output tensor.
    /// 获取单个 output tensor 对应的 pointer-free 运行时诊断快照。
    /// </summary>
    /// <param name="outputTensorName">The output tensor name used for output-address and callback-state queries. / 用于输出地址与回调状态查询的 output tensor 名称。</param>
    /// <returns>A copied runtime diagnostic snapshot. 复制出的运行时诊断快照。</returns>
    /// <remarks>
    /// This helper aggregates existing safe read-only queries. It does not expose, retain, or take ownership of borrowed
    /// TensorRT pointers, and address values are emitted only as integer diagnostics.
    /// 该 helper 聚合既有安全只读查询；不会暴露、保留或接管 TensorRT borrowed pointer，地址也仅以整数诊断值返回。
    /// </remarks>
    public TensorRtExecutionContextRuntimeDiagnosticSnapshot GetRuntimeDiagnosticSnapshot(string outputTensorName)
    {
        if (outputTensorName == null)
        {
            throw new ArgumentNullException(nameof(outputTensorName));
        }

        List<string> diagnostics = new List<string>();
        TensorRtExecutionContextCallbackStateSnapshot callbackState =
            TryCollect("CallbackState", diagnostics, () => GetCallbackStateSnapshot(outputTensorName), CreateUnavailableCallbackStateSnapshot(outputTensorName));

        return new TensorRtExecutionContextRuntimeDiagnosticSnapshot(
            line: Line,
            outputTensorName: outputTensorName,
            hasErrorRecorder: TryCollect("HasErrorRecorder", diagnostics, () => HasErrorRecorder, false),
            isInputConsumedEventSet: TryCollect("IsInputConsumedEventSet", diagnostics, () => IsInputConsumedEventSet, false),
            inputConsumedEventAddressValue: TryCollect("InputConsumedEventAddressValue", diagnostics, () => InputConsumedEventAddressValue, 0UL),
            hasOutputAllocator: TryCollect("HasOutputAllocator", diagnostics, () => HasOutputAllocator(outputTensorName), false),
            isOutputTensorAddressSet: TryCollect("IsOutputTensorAddressSet", diagnostics, () => IsOutputTensorAddressSet(outputTensorName), false),
            outputTensorAddressValue: TryCollect("OutputTensorAddressValue", diagnostics, () => GetOutputTensorAddressValue(outputTensorName), 0UL),
            hasTemporaryStorageAllocator: TryCollect("HasTemporaryStorageAllocator", diagnostics, () => HasTemporaryStorageAllocator, false),
            hasDebugListener: TryCollect("HasDebugListener", diagnostics, () => HasDebugListener, false),
            hasManagedProfiler: TryCollect("HasProfiler", diagnostics, () => HasProfiler, false),
            hasNativeProfiler: TryCollect("HasNativeProfiler", diagnostics, () => HasNativeProfiler, false),
            hasRuntimeConfig: TryCollect("HasRuntimeConfig", diagnostics, () => HasRuntimeConfig, false),
            nvtxVerbosity: TryCollect("NvtxVerbosity", diagnostics, GetNvtxVerbosity, TensorRtProfilingVerbosity.LayerNamesOnly),
            unfusedTensorsDebugState: TryCollect("UnfusedTensorsDebugState", diagnostics, GetUnfusedTensorsDebugState, false),
            callbackState: callbackState,
            diagnostics: diagnostics);
    }

    /// <summary>
    /// Gets a high-level pointer-free summary of callback allocator safe controls for one output tensor.
    /// 获取单个 output tensor 的 callback allocator 安全控制高层无指针摘要。
    /// </summary>
    /// <param name="outputTensorName">The output tensor name used for output allocator queries. / 用于 output allocator 查询的输出 tensor 名称。</param>
    /// <returns>A copied safe-control summary. 复制式安全控制摘要。</returns>
    /// <remarks>
    /// This helper aggregates copied metadata only from existing safe read-only queries. Borrowed pointer not
    /// exposed/owned, no callback invocation is attempted, and the result is not runtime proof of TensorRT callback
    /// execution.
    /// 该 helper 只聚合现有安全只读查询复制出的元数据；不会暴露或拥有 borrowed pointer，不会尝试 callback 调用，
    /// 其结果也不是 TensorRT callback 已真实执行的 runtime proof。
    /// </remarks>
    public TensorRtExecutionContextCallbackAllocatorSafeControlSummary GetCallbackAllocatorSafeControlSummary(string outputTensorName)
    {
        if (outputTensorName == null)
        {
            throw new ArgumentNullException(nameof(outputTensorName));
        }

        List<string> diagnostics = new List<string>();
        bool hasOutputAllocator = TryCollect("HasOutputAllocator", diagnostics, () => HasOutputAllocator(outputTensorName), false);
        bool hasTemporaryStorageAllocator = TryCollect("HasTemporaryStorageAllocator", diagnostics, () => HasTemporaryStorageAllocator, false);
        bool hasDebugListener = TryCollect("HasDebugListener", diagnostics, () => HasDebugListener, false);

        bool outputAllocatorInfoAvailable = TryGetOutputAllocatorInterfaceInfo(
            outputTensorName,
            out TensorRtInterfaceInfo outputAllocatorInfo,
            out string outputAllocatorDiagnostic);
        AddCallbackAllocatorSafeControlDiagnostic(
            diagnostics,
            "OutputAllocatorInterfaceInfo",
            outputAllocatorInfoAvailable,
            outputAllocatorDiagnostic);

        bool temporaryStorageAllocatorInfoAvailable = TryGetTemporaryStorageAllocatorInterfaceInfo(
            out TensorRtInterfaceInfo temporaryStorageAllocatorInfo,
            out string temporaryStorageAllocatorDiagnostic);
        AddCallbackAllocatorSafeControlDiagnostic(
            diagnostics,
            "TemporaryStorageAllocatorInterfaceInfo",
            temporaryStorageAllocatorInfoAvailable,
            temporaryStorageAllocatorDiagnostic);

        bool debugListenerInfoAvailable = TryGetDebugListenerInterfaceInfo(
            out TensorRtInterfaceInfo debugListenerInfo,
            out string debugListenerDiagnostic);
        AddCallbackAllocatorSafeControlDiagnostic(
            diagnostics,
            "DebugListenerInterfaceInfo",
            debugListenerInfoAvailable,
            debugListenerDiagnostic);

        TensorRtExecutionContextCallbackStateSnapshot callbackState =
            TryCollect("CallbackState", diagnostics, () => GetCallbackStateSnapshot(outputTensorName), CreateUnavailableCallbackStateSnapshot(outputTensorName));

        return new TensorRtExecutionContextCallbackAllocatorSafeControlSummary(
            line: Line,
            outputTensorName: outputTensorName,
            hasOutputAllocator: hasOutputAllocator,
            hasTemporaryStorageAllocator: hasTemporaryStorageAllocator,
            hasDebugListener: hasDebugListener,
            outputAllocatorInterfaceInfoAvailable: outputAllocatorInfoAvailable,
            temporaryStorageAllocatorInterfaceInfoAvailable: temporaryStorageAllocatorInfoAvailable,
            debugListenerInterfaceInfoAvailable: debugListenerInfoAvailable,
            outputAllocatorInterfaceInfo: outputAllocatorInfo,
            temporaryStorageAllocatorInterfaceInfo: temporaryStorageAllocatorInfo,
            debugListenerInterfaceInfo: debugListenerInfo,
            outputAllocatorDiagnostic: outputAllocatorDiagnostic,
            temporaryStorageAllocatorDiagnostic: temporaryStorageAllocatorDiagnostic,
            debugListenerDiagnostic: debugListenerDiagnostic,
            callbackState: callbackState,
            diagnostics: diagnostics.ToArray());
    }

    /// <summary>
    /// Clears supported callback attachments and returns a copied post-clear callback boundary snapshot.
    /// 清除受支持的回调附加项，并返回清除后的回调边界复制快照。
    /// </summary>
    /// <param name="outputTensorName">The output tensor name used for output allocator clearing. / 用于 output allocator 清理的输出 tensor 名称。</param>
    /// <returns>A copied post-clear callback state snapshot. 清理后的回调状态复制快照。</returns>
    /// <remarks>
    /// This method may call TensorRT clear operations for output allocator, temporary-storage allocator, and TensorRT
    /// 10/11 debug listener. It never destroys the callback objects and never calls
    /// <c>IOutputAllocator::reallocateOutput</c>, <c>IOutputAllocator::notifyShape</c>, or
    /// <c>IDebugListener::processDebugTensor</c>.
    /// 该方法可能调用 TensorRT 的 output allocator、temporary-storage allocator 和 TensorRT 10/11 debug listener 清理操作；
    /// 它不会销毁回调对象，也不会调用 <c>IOutputAllocator::reallocateOutput</c>、<c>IOutputAllocator::notifyShape</c>
    /// 或 <c>IDebugListener::processDebugTensor</c>。
    /// </remarks>
    public TensorRtExecutionContextCallbackStateSnapshot ClearCallbackState(string outputTensorName)
    {
        if (HasManagedDebugListener)
        {
            ClearDebugListener();
        }

        NativeTensorRtExecutionContextCallbackStateInfo info =
            NativeBridgeApi.ClearExecutionContextCallbackState(Line, _handle, outputTensorName);
        return CreateCallbackStateSnapshot(info);
    }

    /// <summary>
    /// Clears the profiler attached to this execution context.
    /// 清除绑定到当前 execution context 的 profiler；不会销毁 profiler 或调用用户回调。
    /// </summary>
    public void ClearProfiler()
    {
        NativeBridgeApi.ClearExecutionContextProfiler(Line, _handle);
        DetachProfiler();
    }

    /// <summary>
    /// Gets whether this execution context has a profiler attached.
    /// 获取当前 execution context 是否绑定了由托管 wrapper 借出的 profiler；不会暴露 profiler 指针或接管其生命周期。
    /// </summary>
    /// <remarks>
    /// This is the managed ownership signal used by <see cref="SetProfiler"/> and <see cref="ClearProfiler"/>.
    /// TensorRT may still report an internal native profiler through <see cref="HasNativeProfiler"/> after the managed
    /// borrow has been detached.
    /// 该属性表示 <see cref="SetProfiler"/> 与 <see cref="ClearProfiler"/> 管理的托管借用状态。即使托管借用已解除，
    /// TensorRT 仍可能通过 <see cref="HasNativeProfiler"/> 报告内部 native profiler。
    /// </remarks>
    public bool HasProfiler => _profilerKeepAlive != null;

    /// <summary>
    /// Gets whether TensorRT currently reports a non-null native profiler pointer.
    /// 获取 TensorRT 当前是否报告非空 native profiler 指针。
    /// </summary>
    /// <remarks>
    /// This diagnostic does not expose the pointer and must not be used as a managed ownership signal.
    /// 该诊断不会暴露指针，且不应作为托管 ownership 信号使用。
    /// </remarks>
    public bool HasNativeProfiler => NativeBridgeApi.HasExecutionContextProfiler(Line, _handle);

    /// <summary>
    /// Gets whether this TensorRT 11 execution context has an associated runtime config object.
    /// 获取当前 TensorRT 11 execution context 是否有关联的 runtime config 对象。
    /// </summary>
    public bool HasRuntimeConfig => NativeBridgeApi.HasExecutionContextRuntimeConfig(Line, _handle);

    /// <summary>
    /// Sets the NVTX verbosity used by this TensorRT execution context.
    /// 设置当前 TensorRT execution context 使用的 NVTX 详细程度。
    /// </summary>
    /// <param name="verbosity">The desired NVTX verbosity. / 期望的 NVTX 详细程度。</param>
    /// <returns><see langword="true"/> when TensorRT accepts the value. / TensorRT 接受该值时返回 <see langword="true"/>。</returns>
    public bool SetNvtxVerbosity(TensorRtProfilingVerbosity verbosity)
    {
        return NativeBridgeApi.SetExecutionContextNvtxVerbosity(Line, _handle, verbosity);
    }

    /// <summary>
    /// Gets the NVTX verbosity currently used by this TensorRT execution context.
    /// 获取当前 TensorRT execution context 使用的 NVTX 详细程度。
    /// </summary>
    public TensorRtProfilingVerbosity GetNvtxVerbosity()
    {
        return NativeBridgeApi.GetExecutionContextNvtxVerbosity(Line, _handle);
    }

    /// <summary>
    /// Clears user-provided auxiliary streams so TensorRT may use its default auxiliary-stream behavior.
    /// 清除用户提供的 auxiliary stream，让 TensorRT 回到默认 auxiliary-stream 行为。
    /// </summary>
    public void ClearAuxStreams()
    {
        lock (_auxiliaryStreamLeaseLock)
        {
            ThrowIfAuxiliaryStreamContextDisposed();
            NativeBridgeApi.ClearExecutionContextAuxStreams(Line, _handle);

            TensorRtAuxiliaryStreamHandleLease? previousLease = _auxiliaryStreamLease;
            _auxiliaryStreamLease = null;
            _auxiliaryStreamAssignedCount = 0;
            _auxiliaryStreamsCleared = true;
            _auxiliaryStreamDiagnostic = "Caller-provided auxiliary CUDA streams are cleared; no managed handle lease is active.";
            previousLease?.Dispose();
        }
    }

    /// <summary>
    /// Enables or disables debug state for TensorRT 11 unfused debug tensors.
    /// 启用或禁用 TensorRT 11 未融合 debug tensor 的 debug state。
    /// </summary>
    /// <param name="enabled">Whether unfused tensor debug state should be enabled. / 是否启用未融合 tensor debug state。</param>
    /// <returns><see langword="true"/> when TensorRT accepts the setting. / TensorRT 接受该设置时返回 <see langword="true"/>。</returns>
    public bool SetUnfusedTensorsDebugState(bool enabled)
    {
        return NativeBridgeApi.SetExecutionContextUnfusedTensorsDebugState(Line, _handle, enabled);
    }

    /// <summary>
    /// Gets debug state for TensorRT 11 unfused debug tensors.
    /// 获取 TensorRT 11 未融合 debug tensor 的 debug state。
    /// </summary>
    public bool GetUnfusedTensorsDebugState()
    {
        return NativeBridgeApi.GetExecutionContextUnfusedTensorsDebugState(Line, _handle);
    }

    private static TensorRtExecutionContextCallbackStateSnapshot CreateCallbackStateSnapshot(
        NativeTensorRtExecutionContextCallbackStateInfo info)
    {
        TensorRtInterfaceInfo outputAllocatorInfo = new TensorRtInterfaceInfo(
            BridgeInfoMapper.ReadFixedUtf8(info.OutputAllocatorInterfaceKind),
            info.OutputAllocatorInterfaceMajor,
            info.OutputAllocatorInterfaceMinor);
        TensorRtInterfaceInfo temporaryStorageAllocatorInfo = new TensorRtInterfaceInfo(
            BridgeInfoMapper.ReadFixedUtf8(info.TemporaryStorageAllocatorInterfaceKind),
            info.TemporaryStorageAllocatorInterfaceMajor,
            info.TemporaryStorageAllocatorInterfaceMinor);
        TensorRtInterfaceInfo debugListenerInfo = new TensorRtInterfaceInfo(
            BridgeInfoMapper.ReadFixedUtf8(info.DebugListenerInterfaceKind),
            info.DebugListenerInterfaceMajor,
            info.DebugListenerInterfaceMinor);

        return new TensorRtExecutionContextCallbackStateSnapshot(
            line: (TensorRtApiLine)info.Line,
            hasOutputAllocator: info.HasOutputAllocator != 0,
            hasTemporaryStorageAllocator: info.HasTemporaryStorageAllocator != 0,
            hasDebugListener: info.HasDebugListener != 0,
            outputAllocatorInterfaceInfoAvailable: info.OutputAllocatorInterfaceInfoAvailable != 0,
            temporaryStorageAllocatorInterfaceInfoAvailable: info.TemporaryStorageAllocatorInterfaceInfoAvailable != 0,
            debugListenerInterfaceInfoAvailable: info.DebugListenerInterfaceInfoAvailable != 0,
            outputAllocatorClearSupported: info.OutputAllocatorClearSupported != 0,
            temporaryStorageAllocatorClearSupported: info.TemporaryStorageAllocatorClearSupported != 0,
            debugListenerClearSupported: info.DebugListenerClearSupported != 0,
            outputAllocatorCleared: info.OutputAllocatorCleared != 0,
            temporaryStorageAllocatorCleared: info.TemporaryStorageAllocatorCleared != 0,
            debugListenerCleared: info.DebugListenerCleared != 0,
            outputAllocatorInterfaceInfo: outputAllocatorInfo,
            temporaryStorageAllocatorInterfaceInfo: temporaryStorageAllocatorInfo,
            debugListenerInterfaceInfo: debugListenerInfo,
            lastStatus: (BridgeStatusCode)info.LastStatus,
            lastOperation: BridgeInfoMapper.ReadFixedUtf8(info.LastOperation),
            diagnostic: BridgeInfoMapper.ReadFixedUtf8(info.LastDiagnostic));
    }

    private TensorRtExecutionContextCallbackStateSnapshot CreateUnavailableCallbackStateSnapshot(string outputTensorName)
    {
        _ = outputTensorName;
        TensorRtInterfaceInfo unavailable = new TensorRtInterfaceInfo(string.Empty, 0, 0);
        return new TensorRtExecutionContextCallbackStateSnapshot(
            line: Line,
            hasOutputAllocator: false,
            hasTemporaryStorageAllocator: false,
            hasDebugListener: false,
            outputAllocatorInterfaceInfoAvailable: false,
            temporaryStorageAllocatorInterfaceInfoAvailable: false,
            debugListenerInterfaceInfoAvailable: false,
            outputAllocatorClearSupported: false,
            temporaryStorageAllocatorClearSupported: false,
            debugListenerClearSupported: false,
            outputAllocatorCleared: false,
            temporaryStorageAllocatorCleared: false,
            debugListenerCleared: false,
            outputAllocatorInterfaceInfo: unavailable,
            temporaryStorageAllocatorInterfaceInfo: unavailable,
            debugListenerInterfaceInfo: unavailable,
            lastStatus: BridgeStatusCode.NotSupported,
            lastOperation: "Unavailable",
            diagnostic: "Callback state snapshot unavailable.");
    }

    private static void AddCallbackAllocatorSafeControlDiagnostic(
        List<string> diagnostics,
        string fieldName,
        bool available,
        string diagnostic)
    {
        if (!available || !string.Equals(diagnostic, "OK", StringComparison.Ordinal))
        {
            diagnostics.Add($"{fieldName}: {diagnostic}");
        }
    }
}

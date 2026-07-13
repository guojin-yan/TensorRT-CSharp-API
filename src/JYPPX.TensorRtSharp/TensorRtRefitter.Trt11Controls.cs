using System;
using System.Collections.Generic;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtRefitter
{
    /// <summary>
    /// Gets or sets the maximum number of worker threads TensorRT may use while refitting.
    /// 获取或设置 TensorRT 在 refit 时可使用的最大工作线程数。适用于 TensorRT 8/10/11。
    /// </summary>
    public int MaxThreads
    {
        get => NativeBridgeApi.GetRefitterMaxThreads(Line, _handle);
        set => NativeBridgeApi.SetRefitterMaxThreads(Line, _handle, value);
    }

    /// <summary>
    /// Gets or sets whether TensorRT validates refit weights before applying them.
    /// 获取或设置 TensorRT 在应用 refit weights 前是否执行权重校验。适用于 TensorRT 10/11。
    /// </summary>
    public bool WeightsValidation
    {
        get => NativeBridgeApi.GetRefitterWeightsValidation(Line, _handle);
        set => NativeBridgeApi.SetRefitterWeightsValidation(Line, _handle, value);
    }

    /// <summary>
    /// Gets whether this refitter currently has a TensorRT error recorder attached.
    /// 获取当前 refitter 是否附加了 TensorRT error recorder；不会暴露 recorder 指针。适用于 TensorRT 8/10/11。
    /// </summary>
    public bool HasErrorRecorder => NativeBridgeApi.HasRefitterErrorRecorder(Line, _handle);

    /// <summary>
    /// Attempts to collect a copied read-only snapshot from the refitter error recorder.
    /// 尝试从 refitter error recorder 采集只读托管快照。
    /// </summary>
    /// <param name="snapshot">The copied snapshot. 已复制到托管内存的快照。</param>
    /// <returns><c>true</c> when a recorder was attached; otherwise <c>false</c>. 附加了 recorder 时返回 <c>true</c>，否则返回 <c>false</c>。</returns>
    /// <remarks>
    /// This method does not expose, retain, increment, decrement, or destroy the native recorder pointer.
    /// 此方法不会暴露、持有、增加引用、减少引用或销毁原生 recorder 指针。适用于 TensorRT 8/10/11。
    /// </remarks>
    public bool TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot)
    {
        snapshot = NativeBridgeApi.GetRefitterErrorRecorderSnapshot(Line, _handle);
        return snapshot.HasRecorder;
    }

    /// <summary>
    /// Gets a copied read-only diagnostic snapshot for this refitter.
    /// 获取当前 refitter 的复制型只读诊断快照。
    /// </summary>
    public TensorRtRefitterDiagnosticSnapshot GetDiagnosticSnapshot()
    {
        List<string> diagnostics = new List<string>();
        bool hasErrorRecorder = TryCollect("HasErrorRecorder", diagnostics, () => HasErrorRecorder, false);
        TensorRtErrorRecorderSnapshot errorRecorder = TryCollect(
            "ErrorRecorderSnapshot",
            diagnostics,
            () =>
            {
                TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot);
                return snapshot;
            },
            new TensorRtErrorRecorderSnapshot(Line, false, 0, false, Array.Empty<TensorRtErrorRecord>()));

        return new TensorRtRefitterDiagnosticSnapshot(
            Line,
            TryCollect("MaxThreads", diagnostics, () => MaxThreads, 0),
            TryCollect("WeightsValidation", diagnostics, () => WeightsValidation, false),
            TryCollect("HasLogger", diagnostics, () => HasLogger, false),
            hasErrorRecorder,
            errorRecorder,
            TryCollect("DynamicRangeTensorCount", diagnostics, () => DynamicRangeTensorCount, 0),
            TryCollect("MissingNamedWeightCount", diagnostics, () => MissingNamedWeightCount, 0),
            TryCollect("AllNamedWeightCount", diagnostics, () => AllNamedWeightCount, 0),
            TryCollect<IReadOnlyList<string>>("DynamicRangeTensorNames", diagnostics, GetDynamicRangeTensorNames, Array.Empty<string>()),
            TryCollect<IReadOnlyList<string>>("MissingNamedWeights", diagnostics, GetMissingNamedWeights, Array.Empty<string>()),
            TryCollect<IReadOnlyList<string>>("AllNamedWeights", diagnostics, GetAllNamedWeights, Array.Empty<string>()),
            diagnostics);
    }

    /// <summary>
    /// Gets whether this refitter has a TensorRT logger associated with it.
    /// 获取当前 refitter 是否关联了 TensorRT logger；只返回布尔值，不跨 ABI 暴露借用的 logger 指针。适用于 TensorRT 8/10。
    /// </summary>
    public bool HasLogger => NativeBridgeApi.HasRefitterLogger(Line, _handle);

    /// <summary>
    /// Gets the number of refittable dynamic-range tensor names reported by TensorRT.
    /// 获取 TensorRT 报告的可 refit dynamic range tensor 名称数量。适用于 TensorRT 8/10；TensorRT 10.1 起该能力由显式量化取代。
    /// </summary>
    public int DynamicRangeTensorCount => NativeBridgeApi.GetRefitterDynamicRangeTensorCount(Line, _handle);

    /// <summary>
    /// Gets the number of missing named weights before the refit can complete.
    /// 获取 refit 完成前仍缺失的 named weights 数量。适用于 TensorRT 8/10，返回的是拷贝后的名称集合计数。
    /// </summary>
    public int MissingNamedWeightCount => NativeBridgeApi.GetRefitterMissingWeightsCount(Line, _handle);

    /// <summary>
    /// Gets the number of all named weights TensorRT reports as refittable.
    /// 获取 TensorRT 报告的全部可 refit named weights 数量。适用于 TensorRT 8/10，返回的是拷贝后的名称集合计数。
    /// </summary>
    public int AllNamedWeightCount => NativeBridgeApi.GetRefitterAllWeightsCount(Line, _handle);

    /// <summary>
    /// Applies configured refit weights asynchronously on the supplied CUDA stream.
    /// 在指定 CUDA stream 上异步应用已配置的 refit weights。适用于 TensorRT 10/11。
    /// </summary>
    /// <param name="stream">The CUDA stream used for the async refit. 用于异步 refit 的 CUDA stream。</param>
    /// <returns><c>true</c> when TensorRT reports the refit was accepted. TensorRT 接受 refit 时返回 <c>true</c>。</returns>
    public bool RefitCudaEngineAsync(CudaStream stream)
    {
        if (stream == null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        return NativeBridgeApi.RefitCudaEngineAsync(Line, _handle, stream.Handle);
    }

    /// <summary>
    /// Updates the dynamic range for a refittable tensor.
    /// 更新可 refit tensor 的 dynamic range。适用于 TensorRT 8/10；不会持有或返回原生字符串指针。
    /// </summary>
    /// <param name="tensorName">The TensorRT tensor name. TensorRT tensor 名称。</param>
    /// <param name="minimum">The minimum dynamic range value. dynamic range 最小值。</param>
    /// <param name="maximum">The maximum dynamic range value. dynamic range 最大值。</param>
    /// <returns><c>true</c> if TensorRT accepted the range. TensorRT 接受该范围时返回 <c>true</c>。</returns>
    public bool SetDynamicRange(string tensorName, float minimum, float maximum)
    {
        return NativeBridgeApi.SetRefitterDynamicRange(Line, _handle, tensorName, minimum, maximum);
    }

    /// <summary>
    /// Gets the minimum dynamic range value for a refittable tensor.
    /// 获取可 refit tensor 的 dynamic range 最小值。适用于 TensorRT 8/10。
    /// </summary>
    /// <param name="tensorName">The TensorRT tensor name. TensorRT tensor 名称。</param>
    /// <returns>The dynamic range minimum. dynamic range 最小值。</returns>
    public float GetDynamicRangeMinimum(string tensorName)
    {
        return NativeBridgeApi.GetRefitterDynamicRangeMinimum(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets the maximum dynamic range value for a refittable tensor.
    /// 获取可 refit tensor 的 dynamic range 最大值。适用于 TensorRT 8/10。
    /// </summary>
    /// <param name="tensorName">The TensorRT tensor name. TensorRT tensor 名称。</param>
    /// <returns>The dynamic range maximum. dynamic range 最大值。</returns>
    public float GetDynamicRangeMaximum(string tensorName)
    {
        return NativeBridgeApi.GetRefitterDynamicRangeMaximum(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets all tensor names that TensorRT reports as having refittable dynamic ranges.
    /// 获取 TensorRT 报告的全部可 refit dynamic range tensor 名称；名称已复制到托管内存。适用于 TensorRT 8/10。
    /// </summary>
    /// <returns>Copied tensor names. 已复制的 tensor 名称集合。</returns>
    public IReadOnlyList<string> GetDynamicRangeTensorNames()
    {
        return ConvertNames(NativeBridgeApi.GetRefitterDynamicRangeTensorEntries(Line, _handle));
    }

    /// <summary>
    /// Sets named weights using a pinned refit weights buffer.
    /// 使用 pinned refit weights buffer 设置 named weights；调用方必须保证 buffer 在 refit 完成前保持有效。适用于 TensorRT 8/10。
    /// </summary>
    /// <param name="weightsName">The refittable weights name. 可 refit 权重名称。</param>
    /// <param name="weights">The pinned refit weights buffer. pinned refit 权重缓冲区。</param>
    /// <returns><c>true</c> if TensorRT accepted the named weights. TensorRT 接受该 named weights 时返回 <c>true</c>。</returns>
    public bool SetNamedWeights(string weightsName, TensorRtRefitWeightsBuffer weights)
    {
        return NativeBridgeApi.SetRefitterNamedWeights(Line, _handle, weightsName, weights);
    }

    /// <summary>
    /// Gets missing named weights before TensorRT can complete refit.
    /// 获取 TensorRT 完成 refit 前仍缺失的 named weights；名称已复制到托管内存。适用于 TensorRT 8/10。
    /// </summary>
    /// <returns>Copied missing named weights. 已复制的缺失 named weights 名称集合。</returns>
    public IReadOnlyList<string> GetMissingNamedWeights()
    {
        return ConvertNames(NativeBridgeApi.GetRefitterMissingWeightsEntries(Line, _handle));
    }

    /// <summary>
    /// Gets all named weights TensorRT reports as refittable.
    /// 获取 TensorRT 报告的全部可 refit named weights；名称已复制到托管内存。适用于 TensorRT 8/10。
    /// </summary>
    /// <returns>Copied refittable named weights. 已复制的可 refit named weights 名称集合。</returns>
    public IReadOnlyList<string> GetAllNamedWeights()
    {
        return ConvertNames(NativeBridgeApi.GetRefitterAllWeightsEntries(Line, _handle));
    }

    /// <summary>
    /// Unsets a named refit weights entry.
    /// 取消设置指定名称的 refit weights。适用于 TensorRT 10/11。
    /// </summary>
    /// <param name="weightsName">The refittable weights name. 可 refit 权重名称。</param>
    /// <returns><c>true</c> when TensorRT accepted the unset operation. TensorRT 接受取消设置操作时返回 <c>true</c>。</returns>
    public bool UnsetNamedWeights(string weightsName)
    {
        return NativeBridgeApi.UnsetRefitterNamedWeights(Line, _handle, weightsName);
    }

    /// <summary>
    /// Gets the TensorRT memory location for a named refit weights entry.
    /// 获取指定 refit weights 的 TensorRT 内存位置。适用于 TensorRT 10/11。
    /// </summary>
    /// <param name="weightsName">The refittable weights name. 可 refit 权重名称。</param>
    /// <returns>The TensorRT tensor location. TensorRT tensor 的内存位置。</returns>
    public TensorRtTensorLocation GetWeightsLocation(string weightsName)
    {
        return NativeBridgeApi.GetRefitterWeightsLocation(Line, _handle, weightsName);
    }

    /// <summary>
    /// Gets metadata for the current named weights supplied to the refitter.
    /// 获取当前已提供给 refitter 的 named weights 元数据；不会暴露原生 weights 指针。适用于 TensorRT 10/11。
    /// </summary>
    /// <param name="weightsName">The refittable weights name. 可 refit 权重名称。</param>
    /// <returns>Weights metadata without native pointer exposure. 不暴露原生指针的 weights 元数据。</returns>
    public TensorRtWeightsInfo GetNamedWeightsInfo(string weightsName)
    {
        return new TensorRtWeightsInfo(NativeBridgeApi.GetRefitterNamedWeightsInfo(Line, _handle, weightsName));
    }

    /// <summary>
    /// Gets TensorRT's expected prototype metadata for a named refit weights entry.
    /// 获取 TensorRT 对指定 refit weights 的期望 prototype 元数据；不会暴露原生 weights 指针。适用于 TensorRT 10/11。
    /// </summary>
    /// <param name="weightsName">The refittable weights name. 可 refit 权重名称。</param>
    /// <returns>Expected weights metadata without native pointer exposure. 不暴露原生指针的期望 weights 元数据。</returns>
    public TensorRtWeightsInfo GetWeightsPrototypeInfo(string weightsName)
    {
        return new TensorRtWeightsInfo(NativeBridgeApi.GetRefitterWeightsPrototypeInfo(Line, _handle, weightsName));
    }

    /// <summary>
    /// Clears the native error recorder pointer if one was attached externally.
    /// 清除外部附加的原生 error recorder 指针；不会接管 recorder 生命周期。适用于 TensorRT 8/10/11。
    /// </summary>
    public void ClearErrorRecorder()
    {
        NativeBridgeApi.ClearRefitterErrorRecorder(Line, _handle);
    }

    private static IReadOnlyList<string> ConvertNames(NativeTensorRtRefitEntryInfo[] nativeEntries)
    {
        if (nativeEntries.Length == 0)
        {
            return Array.Empty<string>();
        }

        string[] names = new string[nativeEntries.Length];
        for (int index = 0; index < nativeEntries.Length; index++)
        {
            names[index] = DecodeFixedUtf8(nativeEntries[index].LayerName);
        }

        return names;
    }

    private static T TryCollect<T>(string fieldName, List<string> diagnostics, Func<T> getter, T fallback)
    {
        try
        {
            return getter();
        }
        catch (Exception ex) when (ex is BridgeProbeException || ex is NotSupportedException || ex is InvalidOperationException)
        {
            diagnostics.Add($"{fieldName}: {ex.Message}");
            return fallback;
        }
    }
}

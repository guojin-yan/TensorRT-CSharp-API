using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Managed wrapper around a TensorRT engine.
/// TensorRT engine 的托管封装。
/// </summary>
public sealed partial class TensorRtEngine
{
    /// <summary>
    /// Returns metadata for every engine I/O tensor.
    /// 返回所有 engine I/O tensor 的元数据。
    /// </summary>
    /// <returns>A read-only list of tensor metadata. 只读 tensor 元数据列表。</returns>
    public IReadOnlyList<TensorRtTensorInfo> GetIOTensors()
    {
        int count = IOTensorCount;
        List<TensorRtTensorInfo> tensors = new List<TensorRtTensorInfo>(count);
        for (int index = 0; index < count; index++)
        {
            tensors.Add(GetIOTensorInfo(index));
        }

        return tensors;
    }

    /// <summary>
    /// Builds a deployment binding diagnostic snapshot for one engine I/O tensor.
    /// 为一个 engine I/O tensor 构建部署绑定诊断快照。
    /// </summary>
    /// <param name="index">The engine I/O tensor index. Engine I/O tensor 索引。</param>
    /// <param name="profileIndex">The optimization profile index used for profile-specific metadata. 用于 profile 相关元数据的 optimization profile 索引。</param>
    /// <returns>A high-level tensor binding snapshot. 高层 tensor 绑定快照。</returns>
    public TensorRtEngineTensorBinding GetTensorBinding(int index, int profileIndex)
    {
        string tensorName = GetIOTensorName(index);
        return GetTensorBinding(tensorName, profileIndex);
    }

    /// <summary>
    /// Builds a deployment binding diagnostic snapshot for one engine tensor.
    /// 为一个 engine tensor 构建部署绑定诊断快照。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. Engine tensor 名称。</param>
    /// <param name="profileIndex">The optimization profile index used for profile-specific metadata. 用于 profile 相关元数据的 optimization profile 索引。</param>
    /// <returns>A high-level tensor binding snapshot. 高层 tensor 绑定快照。</returns>
    public TensorRtEngineTensorBinding GetTensorBinding(string tensorName, int profileIndex)
    {
        if (profileIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(profileIndex), "Profile index must be greater than or equal to zero.");
        }

        List<string> diagnostics = new List<string>();
        TensorRtDims? minShape = TryGetProfileShape(tensorName, profileIndex, TensorRtOptimizationProfileSelector.Min, diagnostics);
        TensorRtDims? optShape = TryGetProfileShape(tensorName, profileIndex, TensorRtOptimizationProfileSelector.Opt, diagnostics);
        TensorRtDims? maxShape = TryGetProfileShape(tensorName, profileIndex, TensorRtOptimizationProfileSelector.Max, diagnostics);

        return new TensorRtEngineTensorBinding(
            GetTensorIndex(tensorName),
            tensorName,
            GetTensorDataType(tensorName),
            GetTensorIOMode(tensorName),
            GetTensorShape(tensorName),
            GetTensorLocation(tensorName),
            IsShapeInferenceIO(tensorName),
            GetTensorBytesPerComponent(tensorName, profileIndex),
            GetTensorComponentsPerElement(tensorName, profileIndex),
            GetTensorFormat(tensorName, profileIndex),
            GetTensorFormatDescription(tensorName, profileIndex),
            GetTensorVectorizedDimension(tensorName, profileIndex),
            profileIndex,
            minShape,
            optShape,
            maxShape,
            diagnostics.Count == 0 ? Array.Empty<string>() : diagnostics);
    }

    /// <summary>
    /// Builds a deployment binding report for all engine I/O tensors.
    /// 为所有 engine I/O tensor 构建部署绑定报告。
    /// </summary>
    /// <param name="profileIndex">The optimization profile index used for profile-specific metadata. 用于 profile 相关元数据的 optimization profile 索引。</param>
    /// <returns>An engine binding report. Engine 绑定报告。</returns>
    public TensorRtEngineBindingReport GetBindingReport(int profileIndex)
    {
        return CreateBindingReport(profileIndex, null, false);
    }

    /// <summary>
    /// Builds a deployment binding report and attaches execution-context readiness.
    /// 构建部署绑定报告，并附加 execution context 就绪状态。
    /// </summary>
    /// <param name="context">The execution context to inspect. 要检查的 execution context。</param>
    /// <param name="profileIndex">The optimization profile index used for profile-specific metadata. 用于 profile 相关元数据的 optimization profile 索引。</param>
    /// <param name="runShapeInference">Whether to run TensorRT shape inference while collecting readiness. 是否在收集就绪状态时执行 TensorRT shape inference。</param>
    /// <returns>An engine binding report with readiness. 带就绪状态的 engine 绑定报告。</returns>
    public TensorRtEngineBindingReport GetBindingReport(TensorRtExecutionContext context, int profileIndex, bool runShapeInference = false)
    {
        if (context == null)
        {
            throw new ArgumentNullException(nameof(context));
        }

        if (context.Line != Line)
        {
            throw new ArgumentException("Execution context and engine must belong to the same TensorRT API line.", nameof(context));
        }

        return CreateBindingReport(profileIndex, context, runShapeInference);
    }

    private TensorRtEngineBindingReport CreateBindingReport(int profileIndex, TensorRtExecutionContext? context, bool runShapeInference)
    {
        if (profileIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(profileIndex), "Profile index must be greater than or equal to zero.");
        }

        int count = IOTensorCount;
        List<TensorRtEngineTensorBinding> tensors = new List<TensorRtEngineTensorBinding>(count);
        for (int index = 0; index < count; index++)
        {
            tensors.Add(GetTensorBinding(index, profileIndex));
        }

        TensorRtExecutionContextReadiness? readiness = context == null ? null : context.GetReadiness(this, runShapeInference);
        string engineName = Line == TensorRtApiLine.TensorRt11 ? "TensorRT11Engine" : Name;
        return new TensorRtEngineBindingReport(engineName, profileIndex, tensors, readiness);
    }

    private TensorRtDims? TryGetProfileShape(string tensorName, int profileIndex, TensorRtOptimizationProfileSelector selector, List<string> diagnostics)
    {
        try
        {
            return GetProfileShape(tensorName, profileIndex, selector);
        }
        catch (Exception ex)
        {
            diagnostics.Add($"{selector}: {ex.Message}");
            return null;
        }
    }

}

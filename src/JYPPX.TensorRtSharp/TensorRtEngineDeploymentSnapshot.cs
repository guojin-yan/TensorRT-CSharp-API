using System;
using System.Collections.Generic;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Captures TensorRT engine deployment metadata that is useful before binding buffers and enqueueing inference.
/// 捕获在绑定缓冲区与提交推理前有价值的 TensorRT engine 部署元数据。
/// </summary>
public sealed class TensorRtEngineDeploymentSnapshot
{
    internal TensorRtEngineDeploymentSnapshot(
        string engineName,
        int profileIndex,
        int ioTensorCount,
        int layerCount,
        int optimizationProfileCount,
        ulong deviceMemorySizeInBytes,
        ulong deviceMemorySizeV2InBytes,
        ulong profileDeviceMemorySizeInBytes,
        ulong profileDeviceMemorySizeV2InBytes,
        int auxiliaryStreamCount,
        TensorRtEngineCapability capability,
        TensorRtTacticSources tacticSources,
        TensorRtProfilingVerbosity profilingVerbosity,
        TensorRtHardwareCompatibilityLevel hardwareCompatibilityLevel,
        bool isRefittable,
        long streamableWeightsSizeInBytes,
        long weightStreamingBudgetV2InBytes,
        long weightStreamingAutomaticBudgetInBytes,
        long weightStreamingScratchMemorySizeInBytes,
        long totalWeightsSizeInBytes,
        long strippedWeightsSizeInBytes,
        IReadOnlyList<TensorRtEngineTensorBinding> tensors,
        IReadOnlyList<string> diagnostics)
    {
        EngineName = engineName;
        ProfileIndex = profileIndex;
        IOTensorCount = ioTensorCount;
        LayerCount = layerCount;
        OptimizationProfileCount = optimizationProfileCount;
        DeviceMemorySizeInBytes = deviceMemorySizeInBytes;
        DeviceMemorySizeV2InBytes = deviceMemorySizeV2InBytes;
        ProfileDeviceMemorySizeInBytes = profileDeviceMemorySizeInBytes;
        ProfileDeviceMemorySizeV2InBytes = profileDeviceMemorySizeV2InBytes;
        AuxiliaryStreamCount = auxiliaryStreamCount;
        Capability = capability;
        TacticSources = tacticSources;
        ProfilingVerbosity = profilingVerbosity;
        HardwareCompatibilityLevel = hardwareCompatibilityLevel;
        IsRefittable = isRefittable;
        StreamableWeightsSizeInBytes = streamableWeightsSizeInBytes;
        WeightStreamingBudgetV2InBytes = weightStreamingBudgetV2InBytes;
        WeightStreamingAutomaticBudgetInBytes = weightStreamingAutomaticBudgetInBytes;
        WeightStreamingScratchMemorySizeInBytes = weightStreamingScratchMemorySizeInBytes;
        TotalWeightsSizeInBytes = totalWeightsSizeInBytes;
        StrippedWeightsSizeInBytes = strippedWeightsSizeInBytes;
        Tensors = tensors;
        Diagnostics = diagnostics;
    }

    /// <summary>
    /// Gets the TensorRT engine name.
    /// 获取 TensorRT engine 名称。
    /// </summary>
    public string EngineName { get; }

    /// <summary>
    /// Gets the optimization profile used when collecting profile-specific fields.
    /// 获取采集 profile 相关字段时使用的 optimization profile 索引。
    /// </summary>
    public int ProfileIndex { get; }

    /// <summary>
    /// Gets the number of named input/output tensors in the engine.
    /// 获取 engine 中命名输入/输出 tensor 的数量。
    /// </summary>
    public int IOTensorCount { get; }

    /// <summary>
    /// Gets the number of layers recorded in the engine.
    /// 获取 engine 中记录的 layer 数量。
    /// </summary>
    public int LayerCount { get; }

    /// <summary>
    /// Gets the number of optimization profiles available in the engine.
    /// 获取 engine 中可用的 optimization profile 数量。
    /// </summary>
    public int OptimizationProfileCount { get; }

    /// <summary>
    /// Gets TensorRT's base device-memory requirement.
    /// 获取 TensorRT 报告的基础 device memory 需求。
    /// </summary>
    public ulong DeviceMemorySizeInBytes { get; }

    /// <summary>
    /// Gets TensorRT 11 V2 device-memory requirement.
    /// 获取 TensorRT 11 V2 device memory 需求。
    /// </summary>
    public ulong DeviceMemorySizeV2InBytes { get; }

    /// <summary>
    /// Gets profile-specific device-memory requirement.
    /// 获取指定 profile 的 device memory 需求。
    /// </summary>
    public ulong ProfileDeviceMemorySizeInBytes { get; }

    /// <summary>
    /// Gets profile-specific V2 device-memory requirement.
    /// 获取指定 profile 的 V2 device memory 需求。
    /// </summary>
    public ulong ProfileDeviceMemorySizeV2InBytes { get; }

    /// <summary>
    /// Gets the auxiliary stream count requested by the engine.
    /// 获取 engine 请求的辅助 CUDA stream 数量。
    /// </summary>
    public int AuxiliaryStreamCount { get; }

    /// <summary>
    /// Gets the engine capability recorded by TensorRT.
    /// 获取 TensorRT 记录的 engine capability。
    /// </summary>
    public TensorRtEngineCapability Capability { get; }

    /// <summary>
    /// Gets tactic-source bit flags used by the engine.
    /// 获取 engine 使用的 tactic source 位标志。
    /// </summary>
    public TensorRtTacticSources TacticSources { get; }

    /// <summary>
    /// Gets profiling verbosity recorded by the engine.
    /// 获取 engine 记录的 profiling verbosity。
    /// </summary>
    public TensorRtProfilingVerbosity ProfilingVerbosity { get; }

    /// <summary>
    /// Gets TensorRT 11 hardware compatibility level recorded by the engine.
    /// 获取 TensorRT 11 engine 记录的硬件兼容级别。
    /// </summary>
    public TensorRtHardwareCompatibilityLevel HardwareCompatibilityLevel { get; }

    /// <summary>
    /// Gets whether TensorRT reports this engine as refittable.
    /// 获取 TensorRT 是否报告该 engine 可 refit。
    /// </summary>
    public bool IsRefittable { get; }

    /// <summary>
    /// Gets streamable weights size reported by TensorRT 11.
    /// 获取 TensorRT 11 报告的可流式加载权重大小。
    /// </summary>
    public long StreamableWeightsSizeInBytes { get; }

    /// <summary>
    /// Gets current weight-streaming budget.
    /// 获取当前权重流式加载预算。
    /// </summary>
    public long WeightStreamingBudgetV2InBytes { get; }

    /// <summary>
    /// Gets TensorRT's automatic weight-streaming budget.
    /// 获取 TensorRT 自动选择的权重流式加载预算。
    /// </summary>
    public long WeightStreamingAutomaticBudgetInBytes { get; }

    /// <summary>
    /// Gets scratch-memory requirement for weight streaming.
    /// 获取权重流式加载所需的临时内存大小。
    /// </summary>
    public long WeightStreamingScratchMemorySizeInBytes { get; }

    /// <summary>
    /// Gets total weights size reported by TensorRT.
    /// 获取 TensorRT 报告的总权重大小。
    /// </summary>
    public long TotalWeightsSizeInBytes { get; }

    /// <summary>
    /// Gets stripped weights size reported by TensorRT.
    /// 获取 TensorRT 报告的 stripped weights 大小。
    /// </summary>
    public long StrippedWeightsSizeInBytes { get; }

    /// <summary>
    /// Gets per-tensor binding metadata collected from the engine.
    /// 获取从 engine 采集的逐 tensor 绑定元数据。
    /// </summary>
    public IReadOnlyList<TensorRtEngineTensorBinding> Tensors { get; }

    /// <summary>
    /// Gets non-fatal diagnostics collected while building the snapshot.
    /// 获取构建快照时收集的非致命诊断信息。
    /// </summary>
    public IReadOnlyList<string> Diagnostics { get; }

    /// <summary>
    /// Creates a compact one-line diagnostic summary.
    /// 创建紧凑的单行诊断摘要。
    /// </summary>
    public override string ToString()
    {
        return $"{EngineName} profile={ProfileIndex} io={IOTensorCount} layers={LayerCount} profiles={OptimizationProfileCount} memory={DeviceMemorySizeV2InBytes} tensors={Tensors.Count} diagnostics={Diagnostics.Count}";
    }
}

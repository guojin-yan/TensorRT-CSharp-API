using System;
using System.Collections.Generic;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Compact pointer-free summary of TensorRT engine deployment metadata.
/// TensorRT engine deployment 元数据的紧凑无指针摘要。
/// </summary>
public sealed class TensorRtEngineDeploymentSummary
{
    internal TensorRtEngineDeploymentSummary(
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
        bool isRefittable,
        long streamableWeightsSizeInBytes,
        long weightStreamingBudgetV2InBytes,
        long weightStreamingAutomaticBudgetInBytes,
        long weightStreamingScratchMemorySizeInBytes,
        long totalWeightsSizeInBytes,
        long strippedWeightsSizeInBytes,
        int copiedTensorCount,
        int copiedProfileTensorValueCount,
        int diagnosticCount)
    {
        EngineName = engineName ?? string.Empty;
        ProfileIndex = profileIndex;
        IOTensorCount = ioTensorCount < 0 ? 0 : ioTensorCount;
        LayerCount = layerCount < 0 ? 0 : layerCount;
        OptimizationProfileCount = optimizationProfileCount < 0 ? 0 : optimizationProfileCount;
        DeviceMemorySizeInBytes = deviceMemorySizeInBytes;
        DeviceMemorySizeV2InBytes = deviceMemorySizeV2InBytes;
        ProfileDeviceMemorySizeInBytes = profileDeviceMemorySizeInBytes;
        ProfileDeviceMemorySizeV2InBytes = profileDeviceMemorySizeV2InBytes;
        AuxiliaryStreamCount = auxiliaryStreamCount < 0 ? 0 : auxiliaryStreamCount;
        IsRefittable = isRefittable;
        StreamableWeightsSizeInBytes = streamableWeightsSizeInBytes < 0 ? 0 : streamableWeightsSizeInBytes;
        WeightStreamingBudgetV2InBytes = weightStreamingBudgetV2InBytes < 0 ? 0 : weightStreamingBudgetV2InBytes;
        WeightStreamingAutomaticBudgetInBytes = weightStreamingAutomaticBudgetInBytes < 0 ? 0 : weightStreamingAutomaticBudgetInBytes;
        WeightStreamingScratchMemorySizeInBytes = weightStreamingScratchMemorySizeInBytes < 0 ? 0 : weightStreamingScratchMemorySizeInBytes;
        TotalWeightsSizeInBytes = totalWeightsSizeInBytes < 0 ? 0 : totalWeightsSizeInBytes;
        StrippedWeightsSizeInBytes = strippedWeightsSizeInBytes < 0 ? 0 : strippedWeightsSizeInBytes;
        CopiedTensorCount = copiedTensorCount < 0 ? 0 : copiedTensorCount;
        CopiedProfileTensorValueCount = copiedProfileTensorValueCount < 0 ? 0 : copiedProfileTensorValueCount;
        DiagnosticCount = diagnosticCount < 0 ? 0 : diagnosticCount;
    }

    /// <summary>Gets the copied engine name. 获取已复制 engine 名称。</summary>
    public string EngineName { get; }

    /// <summary>Gets the optimization profile index used for the source snapshot. 获取源快照使用的 optimization profile 索引。</summary>
    public int ProfileIndex { get; }

    /// <summary>Gets the TensorRT-reported I/O tensor count. 获取 TensorRT 报告的 I/O tensor 数量。</summary>
    public int IOTensorCount { get; }

    /// <summary>Gets the TensorRT-reported layer count. 获取 TensorRT 报告的 layer 数量。</summary>
    public int LayerCount { get; }

    /// <summary>Gets the TensorRT-reported optimization profile count. 获取 TensorRT 报告的 optimization profile 数量。</summary>
    public int OptimizationProfileCount { get; }

    /// <summary>Gets copied base device-memory requirement. 获取已复制基础 device memory 需求。</summary>
    public ulong DeviceMemorySizeInBytes { get; }

    /// <summary>Gets copied V2 device-memory requirement. 获取已复制 V2 device memory 需求。</summary>
    public ulong DeviceMemorySizeV2InBytes { get; }

    /// <summary>Gets copied profile-specific device-memory requirement. 获取已复制 profile-specific device memory 需求。</summary>
    public ulong ProfileDeviceMemorySizeInBytes { get; }

    /// <summary>Gets copied profile-specific V2 device-memory requirement. 获取已复制 profile-specific V2 device memory 需求。</summary>
    public ulong ProfileDeviceMemorySizeV2InBytes { get; }

    /// <summary>Gets copied auxiliary stream count. 获取已复制 auxiliary stream 数量。</summary>
    public int AuxiliaryStreamCount { get; }

    /// <summary>Gets whether TensorRT reported the engine as refittable. 获取 TensorRT 是否报告 engine 可 refit。</summary>
    public bool IsRefittable { get; }

    /// <summary>Gets copied streamable weights size. 获取已复制可流式权重大小。</summary>
    public long StreamableWeightsSizeInBytes { get; }

    /// <summary>Gets copied weight-streaming budget. 获取已复制权重流式加载预算。</summary>
    public long WeightStreamingBudgetV2InBytes { get; }

    /// <summary>Gets copied automatic weight-streaming budget. 获取已复制自动权重流式加载预算。</summary>
    public long WeightStreamingAutomaticBudgetInBytes { get; }

    /// <summary>Gets copied weight-streaming scratch memory size. 获取已复制权重流式加载 scratch memory 大小。</summary>
    public long WeightStreamingScratchMemorySizeInBytes { get; }

    /// <summary>Gets copied total weights size. 获取已复制总权重大小。</summary>
    public long TotalWeightsSizeInBytes { get; }

    /// <summary>Gets copied stripped weights size. 获取已复制 stripped weights 大小。</summary>
    public long StrippedWeightsSizeInBytes { get; }

    /// <summary>Gets copied tensor metadata count. 获取已复制 tensor metadata 数量。</summary>
    public int CopiedTensorCount { get; }

    /// <summary>Gets copied profile tensor-value snapshot count. 获取已复制 profile tensor-value snapshot 数量。</summary>
    public int CopiedProfileTensorValueCount { get; }

    /// <summary>Gets diagnostic count collected while building the snapshot. 获取构建快照时收集的诊断数量。</summary>
    public int DiagnosticCount { get; }

    /// <summary>Gets whether copied tensor metadata covers the reported I/O tensor count. 获取已复制 tensor metadata 是否覆盖报告的 I/O tensor 数量。</summary>
    public bool CopiedTensorCountMatchesReportedIOTensorCount => CopiedTensorCount == IOTensorCount;

    /// <summary>Gets whether this summary is copied and pointer-free. 获取该摘要是否为复制型且无指针逃逸。</summary>
    public bool PointerFreeCopiedSummary => true;

    /// <summary>Gets whether this summary can be promoted as runtime proof. 获取该摘要是否可晋级为 runtime proof。</summary>
    public bool CanPromoteRuntimeProof => false;

    /// <summary>Gets whether deferred records can be deleted because of this summary. 获取是否可因该摘要删除 deferred 记录。</summary>
    public bool CanDeleteDeferredRecord => false;

    /// <summary>Formats this summary for smoke output and logs. 将该摘要格式化为 smoke 输出和日志。</summary>
    public override string ToString()
    {
        return $"{EngineName} profile={ProfileIndex} io={IOTensorCount}/{CopiedTensorCount} layers={LayerCount} profiles={OptimizationProfileCount} memory={DeviceMemorySizeV2InBytes} profileMemory={ProfileDeviceMemorySizeV2InBytes} profileTensorValues={CopiedProfileTensorValueCount} diagnostics={DiagnosticCount} RuntimeProof={CanPromoteRuntimeProof}";
    }
}

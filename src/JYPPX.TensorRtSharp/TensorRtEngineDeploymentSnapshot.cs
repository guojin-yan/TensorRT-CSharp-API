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
        IReadOnlyList<TensorRtEngineProfileTensorValuesSnapshot> profileTensorValues,
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
        ProfileTensorValues = profileTensorValues;
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
    /// Gets copied profile tensor values snapshots collected for tensor/profile diagnostics.
    /// 获取为 tensor/profile 诊断采集的 copied profile tensor values 快照。
    /// </summary>
    public IReadOnlyList<TensorRtEngineProfileTensorValuesSnapshot> ProfileTensorValues { get; }

    /// <summary>
    /// Gets non-fatal diagnostics collected while building the snapshot.
    /// 获取构建快照时收集的非致命诊断信息。
    /// </summary>
    public IReadOnlyList<string> Diagnostics { get; }

    /// <summary>
    /// Converts this copied engine deployment snapshot into a compact pointer-free summary.
    /// 将当前已复制 engine deployment 快照转换为紧凑、无指针逃逸的摘要。
    /// </summary>
    /// <remarks>
    /// This method only reads managed snapshot values. It does not call TensorRT, expose native engine pointers,
    /// or promote local deployment diagnostics to runtime or package-consumer proof.
    /// 该方法只读取托管快照值；不会调用 TensorRT、暴露原生 engine 指针，也不会将本地部署诊断
    /// 晋级为 runtime 或 package-consumer proof。
    /// </remarks>
    public TensorRtEngineDeploymentSummary ToSummary()
    {
        return new TensorRtEngineDeploymentSummary(
            EngineName,
            ProfileIndex,
            IOTensorCount,
            LayerCount,
            OptimizationProfileCount,
            DeviceMemorySizeInBytes,
            DeviceMemorySizeV2InBytes,
            ProfileDeviceMemorySizeInBytes,
            ProfileDeviceMemorySizeV2InBytes,
            AuxiliaryStreamCount,
            IsRefittable,
            StreamableWeightsSizeInBytes,
            WeightStreamingBudgetV2InBytes,
            WeightStreamingAutomaticBudgetInBytes,
            WeightStreamingScratchMemorySizeInBytes,
            TotalWeightsSizeInBytes,
            StrippedWeightsSizeInBytes,
            Tensors.Count,
            ProfileTensorValues.Count,
            Diagnostics.Count);
    }

    /// <summary>
    /// Creates a compact one-line diagnostic summary.
    /// 创建紧凑的单行诊断摘要。
    /// </summary>
    public override string ToString()
    {
        return $"{EngineName} profile={ProfileIndex} io={IOTensorCount} layers={LayerCount} profiles={OptimizationProfileCount} memory={DeviceMemorySizeV2InBytes} tensors={Tensors.Count} profileTensorValues={ProfileTensorValues.Count} diagnostics={Diagnostics.Count}";
    }
}

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

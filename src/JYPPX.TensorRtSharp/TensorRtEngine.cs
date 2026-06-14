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
public sealed partial class TensorRtEngine : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;

    internal TensorRtEngine(TensorRtApiLine line, SafeTensorRtObjectHandle handle)
    {
        Line = line;
        _handle = handle;
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    /// <summary>
    /// Gets the TensorRT API line used by this engine.
    /// 获取当前 engine 使用的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets the number of engine I/O tensors.
    /// 获取 engine I/O tensor 数量。
    /// </summary>
    public int IOTensorCount => NativeBridgeApi.GetEngineIOTensorCount(Line, _handle);

    /// <summary>
    /// Gets the engine name reported by TensorRT.
    /// 获取 TensorRT 报告的 engine 名称。
    /// </summary>
    public string Name => NativeBridgeApi.GetEngineName(Line, _handle);

    /// <summary>
    /// Gets the number of layers in the engine.
    /// 获取 engine 中的 layer 数量。
    /// </summary>
    public int LayerCount => NativeBridgeApi.GetEngineLayerCount(Line, _handle);

    /// <summary>
    /// Gets whether TensorRT reports this engine as refittable.
    /// 获取 TensorRT 是否报告当前 engine 支持 refit。
    /// </summary>
    public bool IsRefittable => NativeBridgeApi.IsEngineRefittable(Line, _handle);

    /// <summary>
    /// Gets the engine device-memory requirement in bytes.
    /// 获取 engine 的设备内存需求，单位为字节。
    /// </summary>
    public ulong DeviceMemorySizeInBytes => NativeBridgeApi.GetEngineDeviceMemorySize(Line, _handle);

    /// <summary>
    /// Gets the TensorRT 10 V2 device-memory requirement for this engine.
    /// 获取当前 engine 的 TensorRT 10 V2 设备内存需求；TensorRT 8 会返回不支持。
    /// </summary>
    public ulong DeviceMemorySizeV2InBytes => NativeBridgeApi.GetEngineDeviceMemorySizeV2(Line, _handle);

    /// <summary>
    /// Gets the number of auxiliary streams requested by the built engine.
    /// 获取已构建 engine 请求的辅助 CUDA stream 数量。
    /// </summary>
    public int AuxiliaryStreamCount => NativeBridgeApi.GetEngineAuxiliaryStreamCount(Line, _handle);

    /// <summary>
    /// Gets the number of optimization profiles in the engine.
    /// 获取 engine 中的 optimization profile 数量。
    /// </summary>
    public int OptimizationProfileCount => NativeBridgeApi.GetEngineOptimizationProfileCount(Line, _handle);

    /// <summary>
    /// Gets the engine capability reported by TensorRT.
    /// 获取 TensorRT 报告的 engine capability。
    /// </summary>
    public TensorRtEngineCapability Capability => NativeBridgeApi.GetEngineCapability(Line, _handle);

    /// <summary>
    /// Gets the tactic-source mask used by the engine.
    /// 获取 engine 使用的 tactic source 掩码。
    /// </summary>
    public TensorRtTacticSources TacticSources => NativeBridgeApi.GetEngineTacticSources(Line, _handle);

    /// <summary>
    /// Gets the engine profiling verbosity.
    /// 获取 engine 的 profiling verbosity。
    /// </summary>
    public TensorRtProfilingVerbosity ProfilingVerbosity => NativeBridgeApi.GetEngineProfilingVerbosity(Line, _handle);

    /// <summary>
    /// Gets the TensorRT 8 compatibility max-batch-size value.
    /// 获取 TensorRT 8 兼容路径中的 max batch size 值。
    /// </summary>
    public int MaxBatchSizeCompatibility => NativeBridgeApi.GetEngineMaxBatchSizeCompatibility(Line, _handle);

    /// <summary>
    /// Gets high-level metadata for one engine I/O tensor.
    /// 获取一个 engine I/O tensor 的高层元数据。
    /// </summary>
    /// <param name="index">The zero-based tensor index. 从零开始的 tensor 索引。</param>
    /// <returns>The tensor metadata. tensor 元数据。</returns>
    public TensorRtTensorInfo GetIOTensorInfo(int index)
    {
        return BridgeInfoMapper.ToManaged(NativeBridgeApi.GetEngineIOTensorInfo(Line, _handle, index));
    }

    /// <summary>
    /// Gets the name of one engine I/O tensor.
    /// 获取一个 engine I/O tensor 的名称。
    /// </summary>
    /// <param name="index">The zero-based tensor index. 从零开始的 tensor 索引。</param>
    /// <returns>The TensorRT tensor name. TensorRT tensor 名称。</returns>
    public string GetIOTensorName(int index)
    {
        if (Line == TensorRtApiLine.TensorRt11)
        {
            return GetIOTensorInfo(index).Name;
        }

        return NativeBridgeApi.GetEngineIOTensorName(Line, _handle, index);
    }

    /// <summary>
    /// Gets the zero-based index for one named engine tensor.
    /// 获取一个已命名 engine tensor 的从零开始索引。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <returns>The zero-based tensor index. 从零开始的 tensor 索引。</returns>
    public int GetTensorIndex(string tensorName)
    {
        if (Line == TensorRtApiLine.TensorRt11)
        {
            if (string.IsNullOrWhiteSpace(tensorName))
            {
                throw new ArgumentException("Tensor name must not be null or empty.", nameof(tensorName));
            }

            int tensorCount = IOTensorCount;
            for (int index = 0; index < tensorCount; index++)
            {
                if (string.Equals(GetIOTensorInfo(index).Name, tensorName, StringComparison.Ordinal))
                {
                    return index;
                }
            }

            throw new TensorRtException(BridgeStatusCode.NotFound, BridgeErrorCategory.TensorRt, $"Tensor '{tensorName}' was not found in this TensorRT 11 engine.");
        }

        return NativeBridgeApi.GetEngineTensorIndex(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets the TensorRT data type for one engine tensor.
    /// 获取一个 engine tensor 的 TensorRT 数据类型。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <returns>The TensorRT tensor data type. TensorRT tensor 数据类型。</returns>
    public TensorRtDataType GetTensorDataType(string tensorName)
    {
        return NativeBridgeApi.GetEngineTensorDataType(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets the shape of one engine tensor.
    /// 获取一个 engine tensor 的形状。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <returns>The tensor shape. tensor 形状。</returns>
    public TensorRtDims GetTensorShape(string tensorName)
    {
        return NativeBridgeApi.GetEngineTensorShape(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets a TensorRT 11 engine tensor shape with 64-bit dimension extents.
    /// 获取 TensorRT 11 引擎张量形状，并保留 64 位维度 extent。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. 引擎张量名称。</param>
    /// <returns>The tensor shape reported by TensorRT. TensorRT 报告的张量形状。</returns>
    public TensorRtDims64 GetTensorShape64(string tensorName)
    {
        return NativeBridgeApi.GetEngineTensorShape64(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets one TensorRT 11 engine tensor dimension extent as a 64-bit value.
    /// 以 64 位整数获取 TensorRT 11 引擎张量的单个维度 extent。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. 引擎张量名称。</param>
    /// <param name="dimensionIndex">The zero-based dimension index. 从零开始的维度索引。</param>
    /// <returns>The dimension extent reported by TensorRT. TensorRT 报告的维度 extent。</returns>
    public long GetTensorDimensionExtent64(string tensorName, int dimensionIndex)
    {
        return NativeBridgeApi.GetEngineTensorDimensionExtent64(Line, _handle, tensorName, dimensionIndex);
    }

    /// <summary>
    /// Gets whether one engine tensor is an input or output tensor.
    /// 获取一个 engine tensor 是输入还是输出。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <returns>The TensorRT I/O mode. TensorRT I/O 模式。</returns>
    public TensorRtIOMode GetTensorIOMode(string tensorName)
    {
        return NativeBridgeApi.GetEngineTensorIOMode(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets the TensorRT tensor location for one engine tensor.
    /// 获取一个 engine tensor 的 TensorRT tensor location。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <returns>The TensorRT tensor location. TensorRT tensor 的内存位置。</returns>
    public TensorRtTensorLocation GetTensorLocation(string tensorName)
    {
        return NativeBridgeApi.GetEngineTensorLocation(Line, _handle, tensorName);
    }

    /// <summary>
    /// Returns whether one engine tensor participates in shape inference.
    /// 返回一个 engine tensor 是否参与 shape inference。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <returns><see langword="true"/> when the tensor is shape-inference I/O. 当该 tensor 是 shape inference I/O 时返回 <see langword="true"/>。</returns>
    public bool IsShapeInferenceIO(string tensorName)
    {
        return NativeBridgeApi.IsEngineShapeInferenceIO(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets the bytes-per-component value for one engine tensor.
    /// 获取一个 engine tensor 的每分量字节数。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <returns>The bytes-per-component value. 每分量字节数。</returns>
    public int GetTensorBytesPerComponent(string tensorName)
    {
        return NativeBridgeApi.GetEngineTensorBytesPerComponent(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets the profile-specific bytes-per-component value for one engine tensor.
    /// 获取一个 engine tensor 在指定 profile 下的每分量字节数。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <param name="profileIndex">The optimization profile index. optimization profile 索引。</param>
    /// <returns>The bytes-per-component value. 每分量字节数。</returns>
    public int GetTensorBytesPerComponent(string tensorName, int profileIndex)
    {
        return NativeBridgeApi.GetEngineTensorBytesPerComponent(Line, _handle, tensorName, profileIndex);
    }

    /// <summary>
    /// Gets the components-per-element value for one engine tensor.
    /// 获取一个 engine tensor 的每元素分量数。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <returns>The components-per-element value. 每元素分量数。</returns>
    public int GetTensorComponentsPerElement(string tensorName)
    {
        return NativeBridgeApi.GetEngineTensorComponentsPerElement(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets the profile-specific components-per-element value for one engine tensor.
    /// 获取一个 engine tensor 在指定 profile 下的每元素分量数。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <param name="profileIndex">The optimization profile index. optimization profile 索引。</param>
    /// <returns>The components-per-element value. 每元素分量数。</returns>
    public int GetTensorComponentsPerElement(string tensorName, int profileIndex)
    {
        return NativeBridgeApi.GetEngineTensorComponentsPerElement(Line, _handle, tensorName, profileIndex);
    }

    /// <summary>
    /// Gets the TensorRT tensor format for one engine tensor.
    /// 获取一个 engine tensor 的 TensorRT tensor format。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <returns>The TensorRT tensor format. TensorRT tensor 格式。</returns>
    public TensorRtTensorFormat GetTensorFormat(string tensorName)
    {
        return NativeBridgeApi.GetEngineTensorFormat(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets the profile-specific TensorRT tensor format for one engine tensor.
    /// 获取一个 engine tensor 在指定 profile 下的 TensorRT tensor format。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <param name="profileIndex">The optimization profile index. optimization profile 索引。</param>
    /// <returns>The TensorRT tensor format. TensorRT tensor 格式。</returns>
    public TensorRtTensorFormat GetTensorFormat(string tensorName, int profileIndex)
    {
        return NativeBridgeApi.GetEngineTensorFormat(Line, _handle, tensorName, profileIndex);
    }

    /// <summary>
    /// Gets TensorRT's human-readable tensor format description.
    /// 获取 TensorRT 返回的可读 tensor format 描述。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. Engine tensor 名称。</param>
    /// <returns>The TensorRT tensor format description. TensorRT tensor format 描述。</returns>
    public string GetTensorFormatDescription(string tensorName)
    {
        return NativeBridgeApi.GetEngineTensorFormatDescription(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets TensorRT's profile-specific human-readable tensor format description.
    /// 获取 TensorRT 针对指定 profile 返回的可读 tensor format 描述。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. Engine tensor 名称。</param>
    /// <param name="profileIndex">The optimization profile index. Optimization profile 索引。</param>
    /// <returns>The TensorRT tensor format description. TensorRT tensor format 描述。</returns>
    public string GetTensorFormatDescription(string tensorName, int profileIndex)
    {
        return NativeBridgeApi.GetEngineTensorFormatDescription(Line, _handle, tensorName, profileIndex);
    }

    /// <summary>
    /// Gets the vectorized dimension index for one engine tensor.
    /// 获取一个 engine tensor 的向量化维度索引。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <returns>The vectorized dimension index, or TensorRT's sentinel value. 向量化维度索引，或 TensorRT 的哨兵值。</returns>
    public int GetTensorVectorizedDimension(string tensorName)
    {
        return NativeBridgeApi.GetEngineTensorVectorizedDimension(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets the profile-specific vectorized dimension index for one engine tensor.
    /// 获取一个 engine tensor 在指定 profile 下的向量化维度索引。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <param name="profileIndex">The optimization profile index. optimization profile 索引。</param>
    /// <returns>The vectorized dimension index, or TensorRT's sentinel value. 向量化维度索引，或 TensorRT 的哨兵值。</returns>
    public int GetTensorVectorizedDimension(string tensorName, int profileIndex)
    {
        return NativeBridgeApi.GetEngineTensorVectorizedDimension(Line, _handle, tensorName, profileIndex);
    }

    /// <summary>
    /// Gets one optimization-profile shape for one engine tensor.
    /// 获取一个 engine tensor 在某个 optimization profile 中的形状。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <param name="profileIndex">The optimization profile index. optimization profile 索引。</param>
    /// <param name="selector">The min/opt/max selector. min/opt/max 选择器。</param>
    /// <returns>The profile shape. profile 形状。</returns>
    public TensorRtDims GetProfileShape(string tensorName, int profileIndex, TensorRtOptimizationProfileSelector selector)
    {
        return NativeBridgeApi.GetEngineProfileShape(Line, _handle, tensorName, profileIndex, selector);
    }

    /// <summary>
    /// Gets one TensorRT 11 engine profile shape with 64-bit dimension extents.
    /// 获取 TensorRT 11 引擎中某个 profile selector 的形状，并保留 64 位维度 extent。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. 引擎张量名称。</param>
    /// <param name="profileIndex">The optimization profile index. 优化 profile 索引。</param>
    /// <param name="selector">The min/opt/max profile selector. min/opt/max profile 选择器。</param>
    /// <returns>The profile shape reported by TensorRT. TensorRT 报告的 profile 形状。</returns>
    public TensorRtDims64 GetProfileShape64(string tensorName, int profileIndex, TensorRtOptimizationProfileSelector selector)
    {
        return NativeBridgeApi.GetEngineProfileShape64(Line, _handle, tensorName, profileIndex, selector);
    }

    /// <summary>
    /// Gets one dimension extent from a TensorRT 11 engine profile shape as a 64-bit value.
    /// 以 64 位整数获取 TensorRT 11 引擎 profile shape 的单个维度 extent。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. 引擎张量名称。</param>
    /// <param name="profileIndex">The optimization profile index. 优化 profile 索引。</param>
    /// <param name="selector">The min/opt/max profile selector. min/opt/max profile 选择器。</param>
    /// <param name="dimensionIndex">The zero-based dimension index. 从零开始的维度索引。</param>
    /// <returns>The profile dimension extent reported by TensorRT. TensorRT 报告的 profile 维度 extent。</returns>
    public long GetProfileShapeDimensionExtent64(string tensorName, int profileIndex, TensorRtOptimizationProfileSelector selector, int dimensionIndex)
    {
        return NativeBridgeApi.GetEngineProfileShapeDimensionExtent64(Line, _handle, tensorName, profileIndex, selector, dimensionIndex);
    }

    /// <summary>
    /// Gets the device-memory requirement for a specific optimization profile.
    /// 获取指定 optimization profile 的设备内存需求。
    /// </summary>
    /// <param name="profileIndex">The optimization profile index. Optimization profile 索引。</param>
    /// <returns>The required bytes reported by TensorRT. TensorRT 报告的字节数。</returns>
    public ulong GetDeviceMemorySizeForProfile(int profileIndex)
    {
        return NativeBridgeApi.GetEngineDeviceMemorySizeForProfile(Line, _handle, profileIndex);
    }

    /// <summary>
    /// Gets the TensorRT 10 V2 device-memory requirement for a specific optimization profile.
    /// 获取指定 optimization profile 的 TensorRT 10 V2 设备内存需求。
    /// </summary>
    /// <param name="profileIndex">The optimization profile index. Optimization profile 索引。</param>
    /// <returns>The required bytes reported by TensorRT. TensorRT 报告的字节数。</returns>
    public ulong GetDeviceMemorySizeForProfileV2(int profileIndex)
    {
        return NativeBridgeApi.GetEngineDeviceMemorySizeForProfileV2(Line, _handle, profileIndex);
    }

    /// <summary>
    /// Gets whether TensorRT marks the named tensor as a debug tensor.
    /// 获取 TensorRT 是否将指定 tensor 标记为 debug tensor。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. Engine tensor 名称。</param>
    /// <returns><c>true</c> when the tensor is a debug tensor. 如果该 tensor 是 debug tensor，则返回 <c>true</c>。</returns>
    public bool IsDebugTensor(string tensorName)
    {
        return NativeBridgeApi.IsEngineDebugTensor(Line, _handle, tensorName);
    }

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

    /// <summary>
    /// Creates a TensorRT execution context with engine-managed device memory.
    /// 创建一个由 engine 管理设备内存的 TensorRT execution context。
    /// </summary>
    /// <returns>A managed execution-context wrapper. 托管 execution context 封装。</returns>
    public TensorRtExecutionContext CreateExecutionContext()
    {
        return new TensorRtExecutionContext(Line, NativeBridgeApi.CreateExecutionContext(Line, _handle));
    }

    /// <summary>
    /// Creates an execution context without internally allocated device memory.
    /// 创建一个不由 TensorRT 内部分配 device memory 的 execution context。
    /// </summary>
    /// <returns>A managed execution context that requires device memory to be set before enqueue. 需要在 enqueue 前设置 device memory 的托管 execution context。</returns>
    public TensorRtExecutionContext CreateExecutionContextWithoutDeviceMemory()
    {
        return new TensorRtExecutionContext(Line, NativeBridgeApi.CreateExecutionContextWithoutDeviceMemory(Line, _handle));
    }

    /// <summary>
    /// Creates a refitter for updating refittable weights in this engine.
    /// 为当前 engine 创建用于更新可 refit 权重的 refitter。
    /// </summary>
    /// <param name="logger">The TensorRT logger used by the refitter. Refitter 使用的 TensorRT logger。</param>
    /// <returns>A managed refitter wrapper. 托管 refitter 封装对象。</returns>
    public TensorRtRefitter CreateRefitter(TensorRtLogger logger)
    {
        if (logger == null)
        {
            throw new ArgumentNullException(nameof(logger));
        }

        if (logger.Line != Line)
        {
            throw new ArgumentException("Logger and engine must belong to the same TensorRT API line.", nameof(logger));
        }

        return new TensorRtRefitter(Line, NativeBridgeApi.CreateRefitter(Line, _handle, logger.Handle));
    }

    /// <summary>
    /// Creates a TensorRT engine inspector for this engine.
    /// 为当前 engine 创建一个 TensorRT engine inspector。
    /// </summary>
    /// <returns>A managed engine-inspector wrapper. 托管 engine inspector 封装。</returns>
    public TensorRtEngineInspector CreateInspector()
    {
        return new TensorRtEngineInspector(Line, NativeBridgeApi.CreateEngineInspector(Line, _handle));
    }

    /// <summary>
    /// Releases the TensorRT engine handle.
    /// 释放 TensorRT engine 句柄。
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
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

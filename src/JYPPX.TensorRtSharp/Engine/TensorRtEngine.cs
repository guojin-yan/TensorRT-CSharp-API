using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Shared.Interop;
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
    private readonly TensorRtGpuAllocatorCallbackOwner? _gpuAllocatorKeepAlive;
    private readonly TensorRtStreamReader? _streamReaderKeepAlive;
    private bool _disposed;

    internal TensorRtEngine(TensorRtApiLine line, SafeTensorRtObjectHandle handle)
        : this(line, handle, null, null)
    {
    }

    internal TensorRtEngine(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle handle,
        TensorRtGpuAllocatorCallbackOwner? gpuAllocatorKeepAlive)
        : this(line, handle, gpuAllocatorKeepAlive, null)
    {
    }

    internal TensorRtEngine(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle handle,
        TensorRtGpuAllocatorCallbackOwner? gpuAllocatorKeepAlive,
        TensorRtStreamReader? streamReaderKeepAlive)
    {
        Line = line;
        _handle = handle;
        _gpuAllocatorKeepAlive = gpuAllocatorKeepAlive;
        _streamReaderKeepAlive = streamReaderKeepAlive;
        _gpuAllocatorKeepAlive?.AttachEngineBorrower(line);
        try
        {
            _streamReaderKeepAlive?.RetainEngineBorrower(line);
        }
        catch
        {
            _gpuAllocatorKeepAlive?.DetachEngineBorrower();
            throw;
        }
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
    /// Gets whether this legacy TensorRT 8/10 engine reports an implicit batch dimension.
    /// 获取 legacy TensorRT 8/10 engine 是否报告 implicit batch dimension。
    /// </summary>
    /// <remarks>
    /// This compatibility query is for deprecated implicit-batch engines. Modern explicit-batch code should not use it for control flow.
    /// 该兼容查询面向已弃用的 implicit-batch engine；现代 explicit-batch 代码不应依赖它做流程控制。
    /// </remarks>
    public bool HasImplicitBatchDimensionCompatibility => NativeBridgeApi.HasEngineImplicitBatchDimensionCompatibility(Line, _handle);

    /// <summary>
    /// Releases the TensorRT engine handle.
    /// 释放 TensorRT engine 句柄。
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        try
        {
            _handle.Dispose();
        }
        finally
        {
            try
            {
                _streamReaderKeepAlive?.ReleaseEngineBorrower();
            }
            finally
            {
                _gpuAllocatorKeepAlive?.DetachEngineBorrower();
            }
        }
        GC.SuppressFinalize(this);
    }

}

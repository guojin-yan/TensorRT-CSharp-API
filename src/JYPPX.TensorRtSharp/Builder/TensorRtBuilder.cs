using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Managed wrapper around a TensorRT builder.
/// TensorRT builder 的托管封装。
/// </summary>
public sealed partial class TensorRtBuilder : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;
    private readonly TensorRtLogger _loggerKeepAlive;
    private bool _disposed;

    /// <summary>
    /// Creates a TensorRT builder from a logger.
    /// 使用 logger 创建一个 TensorRT builder。
    /// </summary>
    /// <param name="logger">The TensorRT logger used by the builder. builder 使用的 TensorRT logger。</param>
    /// <remarks>
    /// TensorRT borrows the logger pointer. This builder keeps the managed logger attached until the builder is disposed.
    /// TensorRT 只借用 logger 指针；当前 builder 会保持托管 logger 借用关系直到 builder 释放。
    /// </remarks>
    public TensorRtBuilder(TensorRtLogger logger)
    {
        if (logger == null)
        {
            throw new ArgumentNullException(nameof(logger));
        }

        Line = logger.Line;
        _loggerKeepAlive = logger;
        _loggerKeepAlive.AttachBorrower(Line);
        try
        {
            _handle = NativeBridgeApi.CreateBuilder(Line, logger.Handle);
        }
        catch
        {
            _loggerKeepAlive.DetachBorrower();
            throw;
        }
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    /// <summary>
    /// Gets the TensorRT API line used by this builder.
    /// 获取当前 builder 使用的 TensorRT API line。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets whether the platform reports fast FP16 support.
    /// 获取平台是否报告支持快速 FP16。
    /// </summary>
    public bool PlatformHasFastFp16 => NativeBridgeApi.BuilderPlatformHasFastFp16(Line, _handle);

    /// <summary>
    /// Gets whether the platform reports fast INT8 support.
    /// 获取平台是否报告支持快速 INT8。
    /// </summary>
    public bool PlatformHasFastInt8 => NativeBridgeApi.BuilderPlatformHasFastInt8(Line, _handle);

    /// <summary>
    /// Gets whether the platform reports TF32 support.
    /// 获取平台是否报告支持 TF32。
    /// </summary>
    public bool PlatformHasTf32 => NativeBridgeApi.BuilderPlatformHasTf32(Line, _handle);

    /// <summary>
    /// Gets the number of available DLA cores.
    /// 获取可用 DLA core 数量。
    /// </summary>
    public int DlaCoreCount => NativeBridgeApi.GetBuilderDlaCoreCount(Line, _handle);

    /// <summary>
    /// Creates a builder configuration object.
    /// 创建一个 builder 配置对象。
    /// </summary>
    /// <returns>A TensorRT builder configuration wrapper. TensorRT builder 配置封装。</returns>
    public TensorRtBuilderConfig CreateBuilderConfig()
    {
        return new TensorRtBuilderConfig(Line, NativeBridgeApi.CreateBuilderConfig(Line, _handle));
    }

    /// <summary>
    /// Creates an optimization profile.
    /// 创建一个优化 profile。
    /// </summary>
    /// <returns>A TensorRT optimization-profile wrapper. TensorRT optimization profile 封装。</returns>
    public TensorRtOptimizationProfile CreateOptimizationProfile()
    {
        return new TensorRtOptimizationProfile(Line, NativeBridgeApi.CreateOptimizationProfile(Line, _handle));
    }

    /// <summary>
    /// Creates a network definition owned by this builder.
    /// 创建一个由当前 builder 拥有的 network definition。
    /// </summary>
    /// <param name="flags">The network-definition creation flags. network definition 创建标志。</param>
    /// <returns>A TensorRT network-definition wrapper. TensorRT network definition 封装。</returns>
    public TensorRtNetworkDefinition CreateNetwork(TensorRtNetworkDefinitionCreationFlags flags = TensorRtNetworkDefinitionCreationFlags.None)
    {
        return new TensorRtNetworkDefinition(Line, NativeBridgeApi.CreateNetwork(Line, _handle, (uint)flags));
    }

    /// <summary>
    /// Creates an explicit-batch network and maps strongly typed creation to the selected TensorRT API line.
    /// 创建 explicit-batch network，并将 strongly typed 创建语义映射到所选 TensorRT API 版本。
    /// </summary>
    /// <param name="stronglyTyped">Whether a strongly typed network is required. 是否要求 strongly typed network。</param>
    /// <returns>A caller-owned TensorRT network definition. 由调用方拥有的 TensorRT network definition。</returns>
    /// <remarks>
    /// TensorRT 10 uses bit 1, TensorRT 11 is always strongly typed, and TensorRT 8 does not expose this policy.
    /// TensorRT 10 使用 bit 1，TensorRT 11 始终为 strongly typed，TensorRT 8 不公开该策略。
    /// </remarks>
    public TensorRtNetworkDefinition CreateNetwork(bool stronglyTyped)
    {
        if (!stronglyTyped)
        {
            return CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
        }

        return Line switch
        {
            TensorRtApiLine.TensorRt8 => throw new NotSupportedException("Strongly typed network creation is not exposed by the TensorRT 8 adapter."),
            TensorRtApiLine.TensorRt10 => CreateNetwork(
                TensorRtNetworkDefinitionCreationFlags.ExplicitBatch |
                TensorRtNetworkDefinitionCreationFlags.StronglyTypedTensorRt10),
            TensorRtApiLine.TensorRt11 => CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch),
            _ => throw new ArgumentOutOfRangeException(nameof(Line), Line, "Unsupported TensorRT API line.")
        };
    }

    /// <summary>
    /// Builds a serialized TensorRT engine from a network and configuration.
    /// 使用 network 和配置构建序列化的 TensorRT engine。
    /// </summary>
    /// <param name="network">The source network definition. 源 network definition。</param>
    /// <param name="config">The builder configuration. builder 配置。</param>
    /// <returns>The serialized engine memory. 序列化 engine 内存。</returns>
    public TensorRtHostMemory BuildSerializedNetwork(TensorRtNetworkDefinition network, TensorRtBuilderConfig config)
    {
        if (network == null)
        {
            throw new ArgumentNullException(nameof(network));
        }

        if (config == null)
        {
            throw new ArgumentNullException(nameof(config));
        }

        if (network.Line != Line || config.Line != Line)
        {
            throw new ArgumentException("Network and config must belong to the same TensorRT API line as the builder.");
        }

        return ExecuteWithGpuAllocatorLease(() =>
            new TensorRtHostMemory(Line, NativeBridgeApi.BuildSerializedNetwork(Line, _handle, network.Handle, config.Handle)));
    }

    /// <summary>
    /// Releases the TensorRT builder handle.
    /// 释放 TensorRT builder 句柄。
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        ReleaseManagedGpuAllocatorForDispose();
        _disposed = true;
        _handle.Dispose();
        GC.KeepAlive(_loggerKeepAlive);
        _loggerKeepAlive.DetachBorrower();
        GC.SuppressFinalize(this);
    }
}

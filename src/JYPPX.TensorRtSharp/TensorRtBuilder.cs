using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtBuilder : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;

    public TensorRtBuilder(TensorRtLogger logger)
    {
        if (logger == null)
        {
            throw new ArgumentNullException(nameof(logger));
        }

        Line = logger.Line;
        _handle = NativeBridgeApi.CreateBuilder(Line, logger.Handle);
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    public TensorRtApiLine Line { get; }

    public bool PlatformHasFastFp16 => NativeBridgeApi.BuilderPlatformHasFastFp16(Line, _handle);

    public bool PlatformHasFastInt8 => NativeBridgeApi.BuilderPlatformHasFastInt8(Line, _handle);

    public bool PlatformHasTf32 => NativeBridgeApi.BuilderPlatformHasTf32(Line, _handle);

    public int DlaCoreCount => NativeBridgeApi.GetBuilderDlaCoreCount(Line, _handle);

    public TensorRtBuilderConfig CreateBuilderConfig()
    {
        return new TensorRtBuilderConfig(Line, NativeBridgeApi.CreateBuilderConfig(Line, _handle));
    }

    public TensorRtOptimizationProfile CreateOptimizationProfile()
    {
        return new TensorRtOptimizationProfile(Line, NativeBridgeApi.CreateOptimizationProfile(Line, _handle));
    }

    public TensorRtNetworkDefinition CreateNetwork(TensorRtNetworkDefinitionCreationFlags flags = TensorRtNetworkDefinitionCreationFlags.None)
    {
        return new TensorRtNetworkDefinition(Line, NativeBridgeApi.CreateNetwork(Line, _handle, (uint)flags));
    }

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

        return new TensorRtHostMemory(Line, NativeBridgeApi.BuildSerializedNetwork(Line, _handle, network.Handle, config.Handle));
    }

    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}

using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtBuilder
{
    /// <summary>
    /// Builds a TensorRT 11 CUDA engine directly from a network and builder configuration.
    /// 直接从 network 和 builder config 构建 TensorRT 11 CUDA engine。
    /// </summary>
    /// <param name="network">The TensorRT network definition to build. 要构建的 TensorRT network definition。</param>
    /// <param name="config">The TensorRT builder configuration. TensorRT 构建配置。</param>
    /// <returns>A managed CUDA engine wrapper. 托管 CUDA engine 封装对象。</returns>
    /// <remarks>
    /// This exposes TensorRT's <c>IBuilder::buildEngineWithConfig</c> path. For version-compatible deployment and
    /// persistence, prefer <see cref="BuildSerializedNetwork(TensorRtNetworkDefinition, TensorRtBuilderConfig)"/>.
    /// 该方法对应 TensorRT 的 <c>IBuilder::buildEngineWithConfig</c> 路径；如果需要版本兼容部署或持久化，请优先使用
    /// <see cref="BuildSerializedNetwork(TensorRtNetworkDefinition, TensorRtBuilderConfig)"/>。
    /// </remarks>
    public TensorRtEngine BuildEngineWithConfig(TensorRtNetworkDefinition network, TensorRtBuilderConfig config)
    {
        ValidateBuildInputs(network, config);
        return new TensorRtEngine(Line, NativeBridgeApi.BuildEngineWithConfig(Line, _handle, network.Handle, config.Handle));
    }

    /// <summary>
    /// Builds a serialized TensorRT 11 network and captures optional kernel text emitted by TensorRT.
    /// 构建 TensorRT 11 序列化网络，并获取 TensorRT 可选生成的 kernel text。
    /// </summary>
    /// <param name="network">The TensorRT network definition to build. 要构建的 TensorRT network definition。</param>
    /// <param name="config">The TensorRT builder configuration. TensorRT 构建配置。</param>
    /// <returns>The serialized plan plus optional kernel text. 序列化 plan 以及可选 kernel text。</returns>
    public TensorRtSerializedNetworkWithKernelText BuildSerializedNetworkWithKernelText(TensorRtNetworkDefinition network, TensorRtBuilderConfig config)
    {
        ValidateBuildInputs(network, config);
        NativeTensorRtSerializedNetworkWithKernelText result = NativeBridgeApi.BuildSerializedNetworkWithKernelText(Line, _handle, network.Handle, config.Handle);
        TensorRtHostMemory plan = new TensorRtHostMemory(Line, result.Plan);
        TensorRtHostMemory? kernelText = result.KernelText == null ? null : new TensorRtHostMemory(Line, result.KernelText);
        return new TensorRtSerializedNetworkWithKernelText(plan, kernelText);
    }

    private void ValidateBuildInputs(TensorRtNetworkDefinition network, TensorRtBuilderConfig config)
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
    }
}

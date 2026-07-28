using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtEngine
{
    /// <summary>
    /// Serializes this TensorRT engine into host memory using TensorRT's default serialization behavior.
    /// 使用 TensorRT 默认序列化行为将当前 engine 序列化到 host memory。
    /// </summary>
    /// <returns>A host-memory buffer containing the serialized engine. / 包含序列化 engine 的 host-memory 缓冲区。</returns>
    public TensorRtHostMemory Serialize()
    {
        return new TensorRtHostMemory(Line, NativeBridgeApi.SerializeEngine(Line, _handle));
    }

    /// <summary>
    /// Creates a TensorRT serialization config for this engine when the active line supports it.
    /// 在当前 TensorRT 版本线支持时，为当前 engine 创建 serialization config。
    /// </summary>
    /// <returns>A managed serialization config wrapper. / 托管 serialization config 封装。</returns>
    /// <remarks>
    /// This bridge currently supports this path for TensorRT 10 and TensorRT 11.
    /// 当前桥接库支持 TensorRT 10 和 TensorRT 11 的该路径。
    /// </remarks>
    public TensorRtSerializationConfig CreateSerializationConfig()
    {
        return new TensorRtSerializationConfig(Line, NativeBridgeApi.CreateSerializationConfig(Line, _handle));
    }

    /// <summary>
    /// Serializes this engine using an explicit TensorRT serialization config.
    /// 使用显式 TensorRT serialization config 序列化当前 engine。
    /// </summary>
    /// <param name="config">The serialization config to apply. / 要应用的 serialization config。</param>
    /// <returns>A host-memory buffer containing the serialized engine. / 包含序列化 engine 的 host-memory 缓冲区。</returns>
    /// <remarks>
    /// This overload is available for TensorRT 10 and TensorRT 11. TensorRT 8 callers should use <see cref="Serialize()"/>.
    /// 该重载适用于 TensorRT 10 和 TensorRT 11；TensorRT 8 调用方应使用 <see cref="Serialize()"/>。
    /// </remarks>
    public TensorRtHostMemory Serialize(TensorRtSerializationConfig config)
    {
        if (config == null)
        {
            throw new ArgumentNullException(nameof(config));
        }

        if (config.Line != Line)
        {
            throw new ArgumentException("Serialization config and engine must belong to the same TensorRT API line.", nameof(config));
        }

        return new TensorRtHostMemory(Line, NativeBridgeApi.SerializeEngineWithConfig(Line, _handle, config.Handle));
    }

    /// <summary>
    /// Creates a TensorRT runtime config for execution-context creation when supported by the active line.
    /// 在当前 TensorRT 版本线支持时，创建用于 execution context 创建的 runtime config。
    /// </summary>
    /// <returns>A managed runtime config wrapper. / 托管 runtime config 封装。</returns>
    /// <remarks>
    /// This bridge currently supports this path for TensorRT 10 and TensorRT 11.
    /// 当前桥接库支持 TensorRT 10 和 TensorRT 11 的该路径。
    /// </remarks>
    public TensorRtRuntimeConfig CreateRuntimeConfig()
    {
        return new TensorRtRuntimeConfig(Line, NativeBridgeApi.CreateRuntimeConfig(Line, _handle));
    }

    /// <summary>
    /// Creates an execution context with a TensorRT allocation strategy.
    /// 使用 TensorRT allocation strategy 创建 execution context。
    /// </summary>
    /// <param name="strategy">The allocation strategy requested from TensorRT. / 请求 TensorRT 使用的分配策略。</param>
    /// <returns>A managed execution context wrapper. / 托管 execution context 封装。</returns>
    /// <remarks>
    /// The allocation-strategy overload is currently a TensorRT 11 path. Use <see cref="CreateExecutionContext(TensorRtRuntimeConfig)"/> for TensorRT 10 runtime-config based creation.
    /// allocation-strategy 重载当前为 TensorRT 11 路径；TensorRT 10 可使用 <see cref="CreateExecutionContext(TensorRtRuntimeConfig)"/> 进行 runtime-config 创建。
    /// </remarks>
    public TensorRtExecutionContext CreateExecutionContext(TensorRtExecutionContextAllocationStrategy strategy)
    {
        return new TensorRtExecutionContext(Line, NativeBridgeApi.CreateExecutionContext(Line, _handle, strategy));
    }

    /// <summary>
    /// Creates an execution context with an explicit TensorRT runtime config.
    /// 使用显式 TensorRT runtime config 创建 execution context。
    /// </summary>
    /// <param name="runtimeConfig">The runtime config to apply. / 要应用的 runtime config。</param>
    /// <returns>A managed execution context wrapper. / 托管 execution context 封装。</returns>
    /// <remarks>
    /// This bridge currently supports this path for TensorRT 10 and TensorRT 11.
    /// 当前桥接库支持 TensorRT 10 和 TensorRT 11 的该路径。
    /// </remarks>
    public TensorRtExecutionContext CreateExecutionContext(TensorRtRuntimeConfig runtimeConfig)
    {
        if (runtimeConfig == null)
        {
            throw new ArgumentNullException(nameof(runtimeConfig));
        }

        if (runtimeConfig.Line != Line)
        {
            throw new ArgumentException("Runtime config and engine must belong to the same TensorRT API line.", nameof(runtimeConfig));
        }

        return new TensorRtExecutionContext(Line, NativeBridgeApi.CreateExecutionContext(Line, _handle, runtimeConfig.Handle));
    }
}

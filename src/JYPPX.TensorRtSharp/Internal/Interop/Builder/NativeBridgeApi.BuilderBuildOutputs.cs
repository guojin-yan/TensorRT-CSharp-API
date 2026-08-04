using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal readonly struct NativeTensorRtSerializedNetworkWithKernelText
{
    public NativeTensorRtSerializedNetworkWithKernelText(SafeTensorRtObjectHandle plan, SafeTensorRtObjectHandle? kernelText)
    {
        Plan = plan;
        KernelText = kernelText;
    }

    public SafeTensorRtObjectHandle Plan { get; }

    public SafeTensorRtObjectHandle? KernelText { get; }
}

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle BuildEngineWithConfig(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle builder,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle config)
    {
        SafeTensorRtObjectHandle engine;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_build_engine_with_config(builder, network, config, out engine),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_build_engine_with_config(builder, network, config, out engine),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_build_engine_with_config(builder, network, config, out engine),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return engine;
    }

    public static NativeTensorRtSerializedNetworkWithKernelText BuildSerializedNetworkWithKernelText(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle builder,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle config)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(BuildSerializedNetworkWithKernelText));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_builder_build_serialized_network_with_kernel_text(builder, network, config, out SafeTensorRtObjectHandle plan, out SafeTensorRtObjectHandle kernelText);
        NativeStatus.ThrowIfFailed(status);
        if (kernelText.IsInvalid)
        {
            kernelText.Dispose();
            return new NativeTensorRtSerializedNetworkWithKernelText(plan, null);
        }

        return new NativeTensorRtSerializedNetworkWithKernelText(plan, kernelText);
    }

}

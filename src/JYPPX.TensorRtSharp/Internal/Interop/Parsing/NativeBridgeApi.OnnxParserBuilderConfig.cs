using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static bool SetOnnxParserBuilderConfig(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle parser,
        SafeTensorRtObjectHandle config)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetOnnxParserBuilderConfig));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_onnx_parser_set_builder_config_safe(
            parser,
            config,
            out int set);
        NativeStatus.ThrowIfFailed(status);
        return set != 0;
    }
}

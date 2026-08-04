using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle CreateOnnxParserRefitter(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle refitter,
        SafeTensorRtObjectHandle logger)
    {
        SafeTensorRtObjectHandle parserRefitter;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_parser_refitter_create(refitter, logger, out parserRefitter),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_parser_refitter_create(refitter, logger, out parserRefitter),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser refitter diagnostics are available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return parserRefitter;
    }

    public static int GetOnnxParserRefitterErrorCount(TensorRtApiLine line, SafeTensorRtObjectHandle parserRefitter)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_parser_refitter_get_error_count(parserRefitter, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_parser_refitter_get_error_count(parserRefitter, out count),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser refitter diagnostics are available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static TensorRtParserErrorInfo GetOnnxParserRefitterError(TensorRtApiLine line, SafeTensorRtObjectHandle parserRefitter, int index)
    {
        NativeTensorRtParserErrorInfo error;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_parser_refitter_get_error(parserRefitter, index, out error),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_parser_refitter_get_error(parserRefitter, index, out error),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser refitter diagnostics are available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return BridgeInfoMapper.ToManaged(error);
    }

    public static TensorRtOnnxParserDiagnostic GetOnnxParserRefitterDiagnostic(TensorRtApiLine line, SafeTensorRtObjectHandle parserRefitter, int index)
    {
        NativeTensorRtParserErrorInfo error;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_parser_refitter_get_error(parserRefitter, index, out error),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_parser_refitter_get_error(parserRefitter, index, out error),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser refitter diagnostics are available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);

        string description = ReadOnnxParserRefitterErrorString(line, parserRefitter, index, 0);
        string file = ReadOnnxParserRefitterErrorString(line, parserRefitter, index, 1);
        string functionName = ReadOnnxParserRefitterErrorString(line, parserRefitter, index, 2);
        string nodeName = ReadOnnxParserRefitterErrorString(line, parserRefitter, index, 3);
        string nodeOperator = ReadOnnxParserRefitterErrorString(line, parserRefitter, index, 4);
        IReadOnlyList<string> localFunctionStack = GetOnnxParserRefitterLocalFunctionStack(line, parserRefitter, index);

        return new TensorRtOnnxParserDiagnostic(
            index,
            error.Code,
            error.Line,
            error.Node,
            description,
            file,
            functionName,
            nodeName,
            nodeOperator,
            localFunctionStack);
    }

    public static IReadOnlyList<string> GetOnnxParserRefitterLocalFunctionStack(TensorRtApiLine line, SafeTensorRtObjectHandle parserRefitter, int index)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_parser_refitter_error_get_local_function_stack_size(parserRefitter, index, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_parser_refitter_error_get_local_function_stack_size(parserRefitter, index, out count),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser refitter diagnostics are available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        if (count <= 0)
        {
            return Array.Empty<string>();
        }

        string[] stack = new string[count];
        for (int stackIndex = 0; stackIndex < count; stackIndex++)
        {
            int currentIndex = stackIndex;
            stack[stackIndex] = ReadParserErrorString(
                (IntPtr buffer, UIntPtr size, out UIntPtr required) => line switch
                {
                    TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_parser_refitter_error_get_local_function_stack_entry(parserRefitter, index, currentIndex, buffer, size, out required),
                    TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_parser_refitter_error_get_local_function_stack_entry(parserRefitter, index, currentIndex, buffer, size, out required),
                    TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser refitter diagnostics are available for TensorRT 10 and TensorRT 11 adapters."),
                    _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
                });
        }

        return stack;
    }

    public static void ClearOnnxParserRefitterErrors(TensorRtApiLine line, SafeTensorRtObjectHandle parserRefitter)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_parser_refitter_clear_errors(parserRefitter),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_parser_refitter_clear_errors(parserRefitter),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser refitter diagnostics are available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    private static string ReadOnnxParserRefitterErrorString(TensorRtApiLine line, SafeTensorRtObjectHandle parserRefitter, int index, int field)
    {
        return ReadParserErrorString(
            (IntPtr buffer, UIntPtr size, out UIntPtr required) =>
            {
                return field switch
                {
                    0 => line switch
                    {
                        TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_parser_refitter_error_get_description(parserRefitter, index, buffer, size, out required),
                        TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_parser_refitter_error_get_description(parserRefitter, index, buffer, size, out required),
                        TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser refitter diagnostics are available for TensorRT 10 and TensorRT 11 adapters."),
                        _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
                    },
                    1 => line switch
                    {
                        TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_parser_refitter_error_get_file(parserRefitter, index, buffer, size, out required),
                        TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_parser_refitter_error_get_file(parserRefitter, index, buffer, size, out required),
                        TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser refitter diagnostics are available for TensorRT 10 and TensorRT 11 adapters."),
                        _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
                    },
                    2 => line switch
                    {
                        TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_parser_refitter_error_get_function(parserRefitter, index, buffer, size, out required),
                        TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_parser_refitter_error_get_function(parserRefitter, index, buffer, size, out required),
                        TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser refitter diagnostics are available for TensorRT 10 and TensorRT 11 adapters."),
                        _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
                    },
                    3 => line switch
                    {
                        TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_parser_refitter_error_get_node_name(parserRefitter, index, buffer, size, out required),
                        TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_parser_refitter_error_get_node_name(parserRefitter, index, buffer, size, out required),
                        TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser refitter diagnostics are available for TensorRT 10 and TensorRT 11 adapters."),
                        _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
                    },
                    4 => line switch
                    {
                        TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_parser_refitter_error_get_node_operator(parserRefitter, index, buffer, size, out required),
                        TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_parser_refitter_error_get_node_operator(parserRefitter, index, buffer, size, out required),
                        TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser refitter diagnostics are available for TensorRT 10 and TensorRT 11 adapters."),
                        _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
                    },
                    _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.TensorRt, "Unsupported ONNX parser-refitter diagnostic string field.")
                };
            });
    }
}

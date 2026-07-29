using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static int GetOnnxParserErrorCount(TensorRtApiLine line, SafeTensorRtObjectHandle parser)
    {
        int count;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_onnx_parser_get_error_count(parser, out count);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_onnx_parser_get_error_count(parser, out count);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_onnx_parser_get_error_count(parser, out count);
                break;
            default:
                throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.");
        }

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static TensorRtParserErrorInfo GetOnnxParserError(TensorRtApiLine line, SafeTensorRtObjectHandle parser, int index)
    {
        NativeTensorRtParserErrorInfo error;
        BridgeStatusCode status;
        switch (line)
        {
            case TensorRtApiLine.TensorRt8:
                status = NativeMethodsTensorRt.jyppx_trt8_onnx_parser_get_error(parser, index, out error);
                break;
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_onnx_parser_get_error(parser, index, out error);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_onnx_parser_get_error(parser, index, out error);
                break;
            default:
                throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.");
        }

        NativeStatus.ThrowIfFailed(status);
        return BridgeInfoMapper.ToManaged(error);
    }

    public static TensorRtOnnxParserDiagnostic GetOnnxParserDiagnostic(TensorRtApiLine line, SafeTensorRtObjectHandle parser, int index)
    {
        int code;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_parser_error_get_code(parser, index, out code),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_error_get_code(parser, index, out code),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_error_get_code(parser, index, out code),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);

        int lineNumber;
        status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_parser_error_get_line(parser, index, out lineNumber),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_error_get_line(parser, index, out lineNumber),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_error_get_line(parser, index, out lineNumber),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);

        int node;
        status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_parser_error_get_node(parser, index, out node),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_error_get_node(parser, index, out node),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_error_get_node(parser, index, out node),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);

        string description = ReadOnnxParserErrorString(line, parser, index, 0);
        string file = ReadOnnxParserErrorString(line, parser, index, 1);
        string functionName = ReadOnnxParserErrorString(line, parser, index, 2);
        string nodeName = ReadOnnxParserErrorString(line, parser, index, 3);
        string nodeOperator = ReadOnnxParserErrorString(line, parser, index, 4);
        IReadOnlyList<string> localFunctionStack = GetOnnxParserLocalFunctionStack(line, parser, index);

        return new TensorRtOnnxParserDiagnostic(
            index,
            code,
            lineNumber,
            node,
            description,
            file,
            functionName,
            nodeName,
            nodeOperator,
            localFunctionStack);
    }

    public static IReadOnlyList<string> GetOnnxParserLocalFunctionStack(TensorRtApiLine line, SafeTensorRtObjectHandle parser, int index)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_parser_error_get_local_function_stack_size(parser, index, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_error_get_local_function_stack_size(parser, index, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_error_get_local_function_stack_size(parser, index, out count),
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
                    TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_parser_error_get_local_function_stack_entry(parser, index, currentIndex, buffer, size, out required),
                    TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_error_get_local_function_stack_entry(parser, index, currentIndex, buffer, size, out required),
                    TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_error_get_local_function_stack_entry(parser, index, currentIndex, buffer, size, out required),
                    _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
                });
        }

        return stack;
    }

    public static void ClearOnnxParserErrors(TensorRtApiLine line, SafeTensorRtObjectHandle parser)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_parser_clear_errors(parser),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_clear_errors(parser),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_clear_errors(parser),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
    }

    private static string ReadOnnxParserErrorString(TensorRtApiLine line, SafeTensorRtObjectHandle parser, int index, int field)
    {
        return ReadParserErrorString(
            (IntPtr buffer, UIntPtr size, out UIntPtr required) =>
            {
                return field switch
                {
                    0 => line switch
                    {
                        TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_parser_error_get_description(parser, index, buffer, size, out required),
                        TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_error_get_description(parser, index, buffer, size, out required),
                        TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_error_get_description(parser, index, buffer, size, out required),
                        _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
                    },
                    1 => line switch
                    {
                        TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_parser_error_get_file(parser, index, buffer, size, out required),
                        TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_error_get_file(parser, index, buffer, size, out required),
                        TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_error_get_file(parser, index, buffer, size, out required),
                        _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
                    },
                    2 => line switch
                    {
                        TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_parser_error_get_function(parser, index, buffer, size, out required),
                        TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_error_get_function(parser, index, buffer, size, out required),
                        TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_error_get_function(parser, index, buffer, size, out required),
                        _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
                    },
                    3 => line switch
                    {
                        TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_parser_error_get_node_name(parser, index, buffer, size, out required),
                        TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_error_get_node_name(parser, index, buffer, size, out required),
                        TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_error_get_node_name(parser, index, buffer, size, out required),
                        _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
                    },
                    4 => line switch
                    {
                        TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_parser_error_get_node_operator(parser, index, buffer, size, out required),
                        TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_parser_error_get_node_operator(parser, index, buffer, size, out required),
                        TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_error_get_node_operator(parser, index, buffer, size, out required),
                        _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
                    },
                    _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.TensorRt, "Unsupported ONNX parser diagnostic string field.")
                };
            });
    }

}

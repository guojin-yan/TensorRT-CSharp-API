using System;
using System.Runtime.InteropServices;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static bool LoadOnnxParserModelProto(TensorRtApiLine line, SafeTensorRtObjectHandle parser, byte[] modelData, string? modelPath)
    {
        ValidateNonEmptyBuffer(modelData, nameof(modelData), "ONNX model proto data must not be empty.");

        using Utf8Interop.Utf8StringScope modelPathUtf8 = Utf8Interop.ToNativeString(modelPath);
        GCHandle pinned = GCHandle.Alloc(modelData, GCHandleType.Pinned);
        try
        {
            int loaded;
            BridgeStatusCode status = line switch
            {
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_load_model_proto(parser, pinned.AddrOfPinnedObject(), new UIntPtr((ulong)modelData.Length), modelPathUtf8.Pointer, out loaded),
                TensorRtApiLine.TensorRt8 or TensorRtApiLine.TensorRt10 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser model-proto loading is available for the TensorRT 11 adapter."),
                _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
            };

            NativeStatus.ThrowIfFailed(status);
            return loaded != 0;
        }
        finally
        {
            pinned.Free();
        }
    }

    public static bool LoadOnnxParserInitializer(TensorRtApiLine line, SafeTensorRtObjectHandle parser, string name, IntPtr data, UIntPtr dataSize)
    {
        ValidateNativeInitializerInput(name, data, dataSize);

        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(name);
        int loaded;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_load_initializer(parser, nameUtf8.Pointer, data, dataSize, out loaded),
            TensorRtApiLine.TensorRt8 or TensorRtApiLine.TensorRt10 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser initializer loading is available for the TensorRT 11 adapter."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return loaded != 0;
    }

    public static bool ParseOnnxLoadedModelProto(TensorRtApiLine line, SafeTensorRtObjectHandle parser)
    {
        int parsed;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_parser_parse_model_proto(parser, out parsed),
            TensorRtApiLine.TensorRt8 or TensorRtApiLine.TensorRt10 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser model-proto parsing is available for the TensorRT 11 adapter."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return parsed != 0;
    }

    public static bool RefitOnnxParserRefitterFromBytes(TensorRtApiLine line, SafeTensorRtObjectHandle parserRefitter, byte[] modelData, string? modelPath)
    {
        ValidateNonEmptyBuffer(modelData, nameof(modelData), "ONNX model data must not be empty.");

        using Utf8Interop.Utf8StringScope modelPathUtf8 = Utf8Interop.ToNativeString(modelPath);
        GCHandle pinned = GCHandle.Alloc(modelData, GCHandleType.Pinned);
        try
        {
            int refitted;
            BridgeStatusCode status = line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_parser_refitter_refit_from_bytes(parserRefitter, pinned.AddrOfPinnedObject(), new UIntPtr((ulong)modelData.Length), modelPathUtf8.Pointer, out refitted),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_parser_refitter_refit_from_bytes(parserRefitter, pinned.AddrOfPinnedObject(), new UIntPtr((ulong)modelData.Length), modelPathUtf8.Pointer, out refitted),
                TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser-refitter refitFromBytes is available for TensorRT 10 and TensorRT 11 adapters."),
                _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
            };

            NativeStatus.ThrowIfFailed(status);
            return refitted != 0;
        }
        finally
        {
            pinned.Free();
        }
    }

    public static bool RefitOnnxParserRefitterFromFile(TensorRtApiLine line, SafeTensorRtObjectHandle parserRefitter, string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("ONNX model path must not be null or empty.", nameof(filePath));
        }

        using Utf8Interop.Utf8StringScope pathUtf8 = Utf8Interop.ToNativeString(filePath);
        int refitted;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_parser_refitter_refit_from_file(parserRefitter, pathUtf8.Pointer, out refitted),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_parser_refitter_refit_from_file(parserRefitter, pathUtf8.Pointer, out refitted),
            TensorRtApiLine.TensorRt8 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser-refitter refitFromFile is available for TensorRT 10 and TensorRT 11 adapters."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return refitted != 0;
    }

    public static bool LoadOnnxParserRefitterModelProto(TensorRtApiLine line, SafeTensorRtObjectHandle parserRefitter, byte[] modelData, string? modelPath)
    {
        ValidateNonEmptyBuffer(modelData, nameof(modelData), "ONNX model proto data must not be empty.");

        using Utf8Interop.Utf8StringScope modelPathUtf8 = Utf8Interop.ToNativeString(modelPath);
        GCHandle pinned = GCHandle.Alloc(modelData, GCHandleType.Pinned);
        try
        {
            int loaded;
            BridgeStatusCode status = line switch
            {
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_parser_refitter_load_model_proto(parserRefitter, pinned.AddrOfPinnedObject(), new UIntPtr((ulong)modelData.Length), modelPathUtf8.Pointer, out loaded),
                TensorRtApiLine.TensorRt8 or TensorRtApiLine.TensorRt10 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser-refitter model-proto loading is available for the TensorRT 11 adapter."),
                _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
            };

            NativeStatus.ThrowIfFailed(status);
            return loaded != 0;
        }
        finally
        {
            pinned.Free();
        }
    }

    public static bool LoadOnnxParserRefitterInitializer(TensorRtApiLine line, SafeTensorRtObjectHandle parserRefitter, string name, IntPtr data, UIntPtr dataSize)
    {
        ValidateNativeInitializerInput(name, data, dataSize);

        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(name);
        int loaded;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_parser_refitter_load_initializer(parserRefitter, nameUtf8.Pointer, data, dataSize, out loaded),
            TensorRtApiLine.TensorRt8 or TensorRtApiLine.TensorRt10 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser-refitter initializer loading is available for the TensorRT 11 adapter."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return loaded != 0;
    }

    public static bool RefitOnnxParserRefitterLoadedModel(TensorRtApiLine line, SafeTensorRtObjectHandle parserRefitter)
    {
        int refitted;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_parser_refitter_refit_model_proto(parserRefitter, out refitted),
            TensorRtApiLine.TensorRt8 or TensorRtApiLine.TensorRt10 => throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, "ONNX parser-refitter loaded model-proto refit is available for the TensorRT 11 adapter."),
            _ => throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.")
        };

        NativeStatus.ThrowIfFailed(status);
        return refitted != 0;
    }

    private static void ValidateNonEmptyBuffer(byte[] buffer, string argumentName, string emptyMessage)
    {
        if (buffer == null)
        {
            throw new ArgumentNullException(argumentName);
        }

        if (buffer.Length == 0)
        {
            throw new ArgumentException(emptyMessage, argumentName);
        }
    }

    private static void ValidateNativeInitializerInput(string name, IntPtr data, UIntPtr dataSize)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("ONNX initializer name must not be null or empty.", nameof(name));
        }

        if (data == IntPtr.Zero || dataSize == UIntPtr.Zero)
        {
            throw new ArgumentException("ONNX initializer native data must not be null or empty.", nameof(data));
        }
    }
}

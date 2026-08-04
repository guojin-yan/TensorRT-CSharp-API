using System;
using System.Text;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    private delegate BridgeStatusCode OnnxConfigStringGetter(SafeTensorRtObjectHandle config, byte[] outputBuffer, UIntPtr outputBufferSize, out UIntPtr requiredSize);

    public static SafeTensorRtObjectHandle CreateOnnxConfig(TensorRtApiLine line)
    {
        SafeTensorRtObjectHandle config;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_config_create(out config),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_config_create(out config),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_config_create(out config),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return config;
    }

    public static TensorRtDataType GetOnnxConfigModelDataType(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int dataType;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_config_get_model_dtype(config, out dataType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_config_get_model_dtype(config, out dataType),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_config_get_model_dtype(config, out dataType),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return Enum.IsDefined(typeof(TensorRtDataType), dataType)
            ? (TensorRtDataType)dataType
            : TensorRtDataType.Unknown;
    }

    public static void SetOnnxConfigModelDataType(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtDataType dataType)
    {
        ValidateOnnxConfigModelDataType(dataType);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_config_set_model_dtype(config, (int)dataType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_config_set_model_dtype(config, (int)dataType),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_config_set_model_dtype(config, (int)dataType),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static int GetOnnxConfigVerbosityLevel(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int verbosity;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_config_get_verbosity_level(config, out verbosity),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_config_get_verbosity_level(config, out verbosity),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_config_get_verbosity_level(config, out verbosity),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return verbosity;
    }

    public static void SetOnnxConfigVerbosityLevel(TensorRtApiLine line, SafeTensorRtObjectHandle config, int verbosity)
    {
        if (verbosity < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(verbosity), "ONNX config verbosity must be greater than or equal to zero.");
        }

        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_config_set_verbosity_level(config, verbosity),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_config_set_verbosity_level(config, verbosity),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_config_set_verbosity_level(config, verbosity),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static void IncreaseOnnxConfigVerbosity(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_config_add_verbosity(config),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_config_add_verbosity(config),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_config_add_verbosity(config),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static void DecreaseOnnxConfigVerbosity(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_config_reduce_verbosity(config),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_config_reduce_verbosity(config),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_config_reduce_verbosity(config),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static string GetOnnxConfigModelFileName(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        return ReadOnnxConfigString(line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_config_get_model_file_name,
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_config_get_model_file_name,
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_config_get_model_file_name,
            _ => throw UnsupportedLine()
        }, config, "ONNX config model file name is too large for the managed buffer.");
    }

    public static void SetOnnxConfigModelFileName(TensorRtApiLine line, SafeTensorRtObjectHandle config, string? value)
    {
        using Utf8Interop.Utf8StringScope valueUtf8 = Utf8Interop.ToNativeString(value ?? string.Empty);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_config_set_model_file_name(config, valueUtf8.Pointer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_config_set_model_file_name(config, valueUtf8.Pointer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_config_set_model_file_name(config, valueUtf8.Pointer),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static string GetOnnxConfigTextFileName(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        return ReadOnnxConfigString(line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_config_get_text_file_name,
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_config_get_text_file_name,
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_config_get_text_file_name,
            _ => throw UnsupportedLine()
        }, config, "ONNX config text file name is too large for the managed buffer.");
    }

    public static void SetOnnxConfigTextFileName(TensorRtApiLine line, SafeTensorRtObjectHandle config, string? value)
    {
        using Utf8Interop.Utf8StringScope valueUtf8 = Utf8Interop.ToNativeString(value ?? string.Empty);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_config_set_text_file_name(config, valueUtf8.Pointer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_config_set_text_file_name(config, valueUtf8.Pointer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_config_set_text_file_name(config, valueUtf8.Pointer),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static string GetOnnxConfigFullTextFileName(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        return ReadOnnxConfigString(line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_config_get_full_text_file_name,
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_config_get_full_text_file_name,
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_config_get_full_text_file_name,
            _ => throw UnsupportedLine()
        }, config, "ONNX config full text file name is too large for the managed buffer.");
    }

    public static void SetOnnxConfigFullTextFileName(TensorRtApiLine line, SafeTensorRtObjectHandle config, string? value)
    {
        using Utf8Interop.Utf8StringScope valueUtf8 = Utf8Interop.ToNativeString(value ?? string.Empty);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_config_set_full_text_file_name(config, valueUtf8.Pointer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_config_set_full_text_file_name(config, valueUtf8.Pointer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_config_set_full_text_file_name(config, valueUtf8.Pointer),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
    }

    public static bool GetOnnxConfigPrintLayerInfo(TensorRtApiLine line, SafeTensorRtObjectHandle config)
    {
        int enabled;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_config_get_print_layer_info(config, out enabled),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_config_get_print_layer_info(config, out enabled),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_config_get_print_layer_info(config, out enabled),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return enabled != 0;
    }

    public static void SetOnnxConfigPrintLayerInfo(TensorRtApiLine line, SafeTensorRtObjectHandle config, bool enabled)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_onnx_config_set_print_layer_info(config, enabled ? 1 : 0),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_onnx_config_set_print_layer_info(config, enabled ? 1 : 0),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_onnx_config_set_print_layer_info(config, enabled ? 1 : 0),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
    }

    private static void ValidateOnnxConfigModelDataType(TensorRtDataType dataType)
    {
        if (dataType != TensorRtDataType.Float &&
            dataType != TensorRtDataType.Half &&
            dataType != TensorRtDataType.Int8)
        {
            throw new ArgumentOutOfRangeException(nameof(dataType), "ONNX config model data type must be Float, Half, or Int8.");
        }
    }

    private static string ReadOnnxConfigString(OnnxConfigStringGetter getter, SafeTensorRtObjectHandle config, string tooLargeMessage)
    {
        BridgeStatusCode status = getter(config, Array.Empty<byte>(), UIntPtr.Zero, out UIntPtr requiredSize);
        NativeStatus.ThrowIfFailed(status);

        ulong required = requiredSize.ToUInt64();
        if (required == 0)
        {
            return string.Empty;
        }

        if (required > int.MaxValue)
        {
            throw new BridgeProbeException(BridgeStatusCode.BufferTooSmall, BridgeErrorCategory.TensorRt, tooLargeMessage);
        }

        byte[] buffer = new byte[checked((int)required)];
        status = getter(config, buffer, requiredSize, out _);
        NativeStatus.ThrowIfFailed(status);

        int terminator = Array.IndexOf(buffer, (byte)0);
        int length = terminator >= 0 ? terminator : buffer.Length;
        return length == 0 ? string.Empty : Encoding.UTF8.GetString(buffer, 0, length);
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

internal static partial class TrtexecLikeBuildPolicy
{
    private static (string Token, TensorRtDataType DataType) ParseDataType(string value, string optionName)
    {
        return value.Trim().ToLowerInvariant() switch
        {
            "fp32" => ("fp32", TensorRtDataType.Float),
            "fp16" => ("fp16", TensorRtDataType.Half),
            "bf16" => ("bf16", TensorRtDataType.BFloat16),
            "int32" => ("int32", TensorRtDataType.Int32),
            "int64" => ("int64", TensorRtDataType.Int64),
            "int8" => ("int8", TensorRtDataType.Int8),
            "uint8" => ("uint8", TensorRtDataType.UInt8),
            "bool" => ("bool", TensorRtDataType.Bool),
            _ => throw new ArgumentException(optionName + " contains an unsupported data type. Expected fp32, fp16, bf16, int32, int64, int8, uint8, or bool.")
        };
    }

    private static (string Token, TensorRtTensorFormats Format) ParseTensorFormat(string value, string optionName)
    {
        return value.Trim().ToLowerInvariant() switch
        {
            "chw" => ("chw", TensorRtTensorFormats.Linear),
            "chw2" => ("chw2", TensorRtTensorFormats.Chw2),
            "chw4" => ("chw4", TensorRtTensorFormats.Chw4),
            "hwc8" => ("hwc8", TensorRtTensorFormats.Hwc8),
            "chw16" => ("chw16", TensorRtTensorFormats.Chw16),
            "chw32" => ("chw32", TensorRtTensorFormats.Chw32),
            "dhwc8" => ("dhwc8", TensorRtTensorFormats.Dhwc8),
            "cdhw32" => ("cdhw32", TensorRtTensorFormats.Cdhw32),
            "hwc" => ("hwc", TensorRtTensorFormats.Hwc),
            "dhwc" => ("dhwc", TensorRtTensorFormats.Dhwc),
            "dla_linear" => ("dla_linear", TensorRtTensorFormats.DlaLinear),
            "hwc16" => ("hwc16", TensorRtTensorFormats.Hwc16),
            "dla_hwc4" => ("dla_hwc4", TensorRtTensorFormats.DlaHwc4),
            _ => throw new ArgumentException(optionName + " contains an unsupported tensor format.")
        };
    }
}

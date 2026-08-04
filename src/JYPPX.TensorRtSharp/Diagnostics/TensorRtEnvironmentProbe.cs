using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Queries the current bridge state without requiring TensorRT inference to succeed.
/// 在不要求 TensorRT 推理成功的前提下查询当前 bridge 状态。
/// </summary>
public static partial class TensorRtEnvironmentProbe
{
    private static bool IsProbeException(Exception exception)
    {
        return exception is BridgeProbeException ||
               exception is NotSupportedException ||
               exception is InvalidOperationException ||
               exception is DllNotFoundException ||
               exception is BadImageFormatException ||
               exception is EntryPointNotFoundException ||
               exception is SEHException ||
               exception is AccessViolationException;
    }

    private static string FormatProbeException(string stageName, Exception exception)
    {
        if (exception is SEHException)
        {
            return $"{stageName} raised SEHException: {exception.Message}";
        }

        if (exception is AccessViolationException)
        {
            return $"{stageName} raised AccessViolationException: {exception.Message}";
        }

        return exception.Message;
    }

    private static bool TryAddProbeStage<T>(List<TensorRtRuntimeProbeStage> stages, string stageName, Func<T> action, Func<T, string> formatMessage, out T value)
    {
        try
        {
            value = action();
            stages.Add(new TensorRtRuntimeProbeStage(stageName, succeeded: true, formatMessage(value)));
            return true;
        }
        catch (Exception exception) when (IsProbeException(exception))
        {
            value = default!;
            stages.Add(new TensorRtRuntimeProbeStage(stageName, succeeded: false, FormatProbeException(stageName, exception)));
            return false;
        }
    }
}

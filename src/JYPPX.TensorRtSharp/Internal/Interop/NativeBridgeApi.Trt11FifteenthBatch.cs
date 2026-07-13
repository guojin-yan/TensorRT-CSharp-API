using System;
using System.Runtime.InteropServices;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static int[] GetEngineProfileTensorValues(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName, int profileIndex, TensorRtOptimizationProfileSelector selector, int valueCount)
    {
        EnsureTensorRt10(line, nameof(GetEngineProfileTensorValues));
        ValidateProfileTensorValuesInput(tensorName, profileIndex, valueCount);

        int[] values = new int[valueCount];
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        GCHandle pinned = GCHandle.Alloc(values, GCHandleType.Pinned);
        try
        {
            BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt10_engine_get_profile_tensor_values(engine, tensorNameUtf8.Pointer, profileIndex, (int)selector, pinned.AddrOfPinnedObject(), values.Length, out int actualCount);
            NativeStatus.ThrowIfFailed(status);
            if (actualCount <= 0)
            {
                return Array.Empty<int>();
            }

            if (actualCount == values.Length)
            {
                return values;
            }

            int[] trimmed = new int[Math.Min(actualCount, values.Length)];
            Array.Copy(values, trimmed, trimmed.Length);
            return trimmed;
        }
        finally
        {
            pinned.Free();
        }
    }

    public static long[] GetEngineProfileTensorValuesV2(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName, int profileIndex, TensorRtOptimizationProfileSelector selector, int valueCount)
    {
        EnsureTensorRt10Or11(line, nameof(GetEngineProfileTensorValuesV2));
        ValidateProfileTensorValuesInput(tensorName, profileIndex, valueCount);

        long[] values = new long[valueCount];
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        GCHandle pinned = GCHandle.Alloc(values, GCHandleType.Pinned);
        try
        {
            int actualCount;
            BridgeStatusCode status = line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_profile_tensor_values_v2(engine, tensorNameUtf8.Pointer, profileIndex, (int)selector, pinned.AddrOfPinnedObject(), values.Length, out actualCount),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_profile_tensor_values_v2(engine, tensorNameUtf8.Pointer, profileIndex, (int)selector, pinned.AddrOfPinnedObject(), values.Length, out actualCount),
                _ => throw UnsupportedLine()
            };
            NativeStatus.ThrowIfFailed(status);
            if (actualCount <= 0)
            {
                return Array.Empty<long>();
            }

            if (actualCount == values.Length)
            {
                return values;
            }

            long[] trimmed = new long[Math.Min(actualCount, values.Length)];
            Array.Copy(values, trimmed, trimmed.Length);
            return trimmed;
        }
        finally
        {
            pinned.Free();
        }
    }

    private static void ValidateProfileTensorValuesInput(string tensorName, int profileIndex, int valueCount)
    {
        ValidateTensorName(tensorName);
        if (profileIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(profileIndex), "Profile index must be greater than or equal to zero.");
        }

        if (valueCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(valueCount), "Value count must be greater than zero because TensorRT does not expose a count query for engine profile tensor values.");
        }
    }

    private static void EnsureTensorRt10(TensorRtApiLine line, string apiName)
    {
        if (line != TensorRtApiLine.TensorRt10)
        {
            throw new BridgeProbeException(BridgeStatusCode.NotSupported, BridgeErrorCategory.TensorRt, $"{apiName} is available for TensorRT 10 adapters.");
        }
    }

    public static void ClearEngineInspectorErrorRecorder(TensorRtApiLine line, SafeTensorRtObjectHandle inspector)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_inspector_clear_error_recorder(inspector),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_inspector_clear_error_recorder(inspector),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_inspector_clear_error_recorder(inspector),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static ulong GetExecutionContextInputConsumedEventAddressValue(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetExecutionContextInputConsumedEventAddressValue));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_get_input_consumed_event_value(context, out UIntPtr address);
        NativeStatus.ThrowIfFailed(status);
        return address.ToUInt64();
    }

    public static TensorRtExecutionContextAllocationStrategy GetExecutionContextRuntimeConfigAllocationStrategy(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetExecutionContextRuntimeConfigAllocationStrategy));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_get_runtime_config_allocation_strategy(context, out int strategy);
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtExecutionContextAllocationStrategy)strategy;
    }

    public static string GetExecutionContextEngineName(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetExecutionContextEngineName));
        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt11_execution_context_get_engine_name(context, buffer, size, out required),
            "Execution context engine name is too large for the managed buffer.");
    }

    public static int GetExecutionContextEngineIOTensorCount(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetExecutionContextEngineIOTensorCount));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_get_engine_io_tensor_count(context, out int count);
        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static int GetExecutionContextEngineLayerCount(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetExecutionContextEngineLayerCount));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_get_engine_layer_count(context, out int count);
        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static int GetExecutionContextEngineOptimizationProfileCount(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(GetExecutionContextEngineOptimizationProfileCount));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_get_engine_optimization_profile_count(context, out int count);
        NativeStatus.ThrowIfFailed(status);
        return count;
    }
}

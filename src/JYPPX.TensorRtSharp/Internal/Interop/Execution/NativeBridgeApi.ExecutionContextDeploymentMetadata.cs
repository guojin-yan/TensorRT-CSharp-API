using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static TensorRtDims GetExecutionContextTensorShape(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        NativeTensorRtDims dims;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_get_tensor_shape(context, tensorNameUtf8.Pointer, out dims),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_get_tensor_shape(context, tensorNameUtf8.Pointer, out dims),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_get_tensor_shape(context, tensorNameUtf8.Pointer, out dims),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(dims);
    }

    public static int[] GetExecutionContextShapeBinding(TensorRtApiLine line, SafeTensorRtObjectHandle context, int bindingIndex)
    {
        EnsureTensorRt8Only(line, nameof(GetExecutionContextShapeBinding));
        return ReadInt32Array(
            static (handle, index, _, _, buffer, bufferCount, out actualCount) => NativeMethodsTensorRt.jyppx_trt8_execution_context_get_shape_binding(handle, index, buffer, bufferCount, out actualCount),
            context,
            bindingIndex,
            0,
            0);
    }

    public static bool SetExecutionContextInputShapeBinding(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle context,
        int bindingIndex,
        IReadOnlyList<int> values)
    {
        EnsureTensorRt8Only(line, nameof(SetExecutionContextInputShapeBinding));
        if (bindingIndex < 0) { throw new ArgumentOutOfRangeException(nameof(bindingIndex)); }
        if (values == null) { throw new ArgumentNullException(nameof(values)); }
        if (values.Count <= 0 || values.Count > 1_000_000)
        {
            throw new ArgumentOutOfRangeException(nameof(values), "Shape binding values must contain 1 to 1000000 elements.");
        }

        int[] copiedValues = new int[values.Count];
        for (int index = 0; index < values.Count; ++index)
        {
            copiedValues[index] = values[index];
        }
        GCHandle pinned = GCHandle.Alloc(copiedValues, GCHandleType.Pinned);
        try
        {
            BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt8_execution_context_set_input_shape_binding(
                context,
                bindingIndex,
                pinned.AddrOfPinnedObject(),
                copiedValues.Length,
                out int set);
            NativeStatus.ThrowIfFailed(status);
            return set != 0;
        }
        finally
        {
            pinned.Free();
        }
    }

    public static TensorRtDims GetExecutionContextTensorStrides(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        NativeTensorRtDims dims;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_get_tensor_strides(context, tensorNameUtf8.Pointer, out dims),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_get_tensor_strides(context, tensorNameUtf8.Pointer, out dims),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_get_tensor_strides(context, tensorNameUtf8.Pointer, out dims),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(dims);
    }

    public static bool IsExecutionContextTensorAddressBound(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        int value;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_is_tensor_address_bound(context, tensorNameUtf8.Pointer, out value),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_is_tensor_address_bound(context, tensorNameUtf8.Pointer, out value),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_is_tensor_address_bound(context, tensorNameUtf8.Pointer, out value),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return value != 0;
    }

    public static long GetExecutionContextMaxOutputSize(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        long size;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_get_max_output_size(context, tensorNameUtf8.Pointer, out size),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_get_max_output_size(context, tensorNameUtf8.Pointer, out size),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_get_max_output_size(context, tensorNameUtf8.Pointer, out size),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return size;
    }

    public static void SetExecutionContextTensorDebugState(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName, bool debugState)
    {
        SetContextTensorBool(line, context, tensorName, debugState, NativeMethodsTensorRt.jyppx_trt8_execution_context_set_tensor_debug_state, NativeMethodsTensorRt.jyppx_trt10_execution_context_set_tensor_debug_state, NativeMethodsTensorRt.jyppx_trt11_execution_context_set_tensor_debug_state);
    }

    public static bool GetExecutionContextTensorDebugState(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName)
    {
        return GetContextTensorBool(line, context, tensorName, NativeMethodsTensorRt.jyppx_trt8_execution_context_get_tensor_debug_state, NativeMethodsTensorRt.jyppx_trt10_execution_context_get_tensor_debug_state, NativeMethodsTensorRt.jyppx_trt11_execution_context_get_tensor_debug_state);
    }

    public static void SetAllExecutionContextTensorsDebugState(TensorRtApiLine line, SafeTensorRtObjectHandle context, bool debugState)
    {
        SetContextBool(line, context, debugState, NativeMethodsTensorRt.jyppx_trt8_execution_context_set_all_tensors_debug_state, NativeMethodsTensorRt.jyppx_trt10_execution_context_set_all_tensors_debug_state, NativeMethodsTensorRt.jyppx_trt11_execution_context_set_all_tensors_debug_state);
    }

    public static void SetExecutionContextDebugSync(TensorRtApiLine line, SafeTensorRtObjectHandle context, bool debugSync)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_set_debug_sync(context, debugSync ? 1 : 0),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_set_debug_sync(context, debugSync ? 1 : 0),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_set_debug_sync(context, debugSync ? 1 : 0),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static bool GetExecutionContextDebugSync(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        return GetContextBool(line, context, NativeMethodsTensorRt.jyppx_trt8_execution_context_get_debug_sync, NativeMethodsTensorRt.jyppx_trt10_execution_context_get_debug_sync, NativeMethodsTensorRt.jyppx_trt11_execution_context_get_debug_sync);
    }

    public static string GetExecutionContextName(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        return ReadUtf8Buffer(line switch
        {
            TensorRtApiLine.TensorRt8 => (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt8_execution_context_get_name(context, buffer, size, out required),
            TensorRtApiLine.TensorRt10 => (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt10_execution_context_get_name(context, buffer, size, out required),
            TensorRtApiLine.TensorRt11 => (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt11_execution_context_get_name(context, buffer, size, out required),
            _ => throw UnsupportedLine()
        }, "Execution context name is too large for the managed buffer.");
    }

    public static string GetExecutionContextErrorBuffer(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        if (line != TensorRtApiLine.TensorRt8)
        {
            throw new BridgeProbeException(
                BridgeStatusCode.NotSupported,
                BridgeErrorCategory.TensorRt,
                "IExecutionContext::getErrorBuffer is unavailable on the standard nvinfer1::IExecutionContext vendor type supported by this bridge; the compatibility API remains diagnostic-only.");
        }

        return ReadUtf8Buffer(
            (byte[] buffer, UIntPtr size, out UIntPtr required) =>
                NativeMethodsTensorRt.jyppx_trt8_execution_context_get_error_buffer_copy(context, buffer, size, out required),
            "Execution context error buffer is too large for the managed buffer.");
    }

    public static void SetExecutionContextName(TensorRtApiLine line, SafeTensorRtObjectHandle context, string name)
    {
        using Utf8Interop.Utf8StringScope nameUtf8 = Utf8Interop.ToNativeString(name ?? string.Empty);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_set_name(context, nameUtf8.Pointer),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_set_name(context, nameUtf8.Pointer),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_set_name(context, nameUtf8.Pointer),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static int GetExecutionContextOptimizationProfile(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        int profileIndex;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_get_optimization_profile(context, out profileIndex),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_get_optimization_profile(context, out profileIndex),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_get_optimization_profile(context, out profileIndex),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return profileIndex;
    }

    public static void SetExecutionContextOptimizationProfileAsync(TensorRtApiLine line, SafeTensorRtObjectHandle context, int profileIndex, SafeCudaStreamHandle stream)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_set_optimization_profile_async(context, profileIndex, stream),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_set_optimization_profile_async(context, profileIndex, stream),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_set_optimization_profile_async(context, profileIndex, stream),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static bool AllInputDimensionsSpecified(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        return GetContextBool(line, context, NativeMethodsTensorRt.jyppx_trt8_execution_context_all_input_dimensions_specified, NativeMethodsTensorRt.jyppx_trt10_execution_context_all_input_dimensions_specified, NativeMethodsTensorRt.jyppx_trt11_execution_context_all_input_dimensions_specified);
    }

    public static bool AllInputShapesSpecified(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        return GetContextBool(line, context, NativeMethodsTensorRt.jyppx_trt8_execution_context_all_input_shapes_specified, NativeMethodsTensorRt.jyppx_trt10_execution_context_all_input_shapes_specified, NativeMethodsTensorRt.jyppx_trt11_execution_context_all_input_shapes_specified);
    }

    public static int InferExecutionContextShapes(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        int missingCount;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_infer_shapes(context, out missingCount),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_infer_shapes(context, out missingCount),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_infer_shapes(context, out missingCount),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return missingCount;
    }

    public static bool GetExecutionContextEnqueueEmitsProfile(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        return GetContextBool(line, context, NativeMethodsTensorRt.jyppx_trt8_execution_context_get_enqueue_emits_profile, NativeMethodsTensorRt.jyppx_trt10_execution_context_get_enqueue_emits_profile, NativeMethodsTensorRt.jyppx_trt11_execution_context_get_enqueue_emits_profile);
    }

    public static void SetExecutionContextEnqueueEmitsProfile(TensorRtApiLine line, SafeTensorRtObjectHandle context, bool enqueueEmitsProfile)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_set_enqueue_emits_profile(context, enqueueEmitsProfile ? 1 : 0),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_set_enqueue_emits_profile(context, enqueueEmitsProfile ? 1 : 0),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_set_enqueue_emits_profile(context, enqueueEmitsProfile ? 1 : 0),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static bool ReportExecutionContextToProfiler(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        int reported;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_report_to_profiler(context, out reported),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_report_to_profiler(context, out reported),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_report_to_profiler(context, out reported),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return reported != 0;
    }

    public static void SetExecutionContextDeviceMemory(TensorRtApiLine line, SafeTensorRtObjectHandle context, SafeCudaMemoryHandle memory)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_set_device_memory(context, memory),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_set_device_memory(context, memory),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_set_device_memory(context, memory),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetExecutionContextDeviceMemoryV2(TensorRtApiLine line, SafeTensorRtObjectHandle context, SafeCudaMemoryHandle memory)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_set_device_memory_v2(context, memory),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_set_device_memory_v2(context, memory),
            TensorRtApiLine.TensorRt8 => throw new NotSupportedException("TensorRT 8 does not expose setDeviceMemoryV2."),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static ulong GetExecutionContextDeviceMemorySize(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        UIntPtr size;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_get_device_memory_size(context, out size),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_get_device_memory_size(context, out size),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_get_device_memory_size(context, out size),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return size.ToUInt64();
    }

    public static ulong UpdateExecutionContextDeviceMemorySizeForShapes(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        UIntPtr size;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_update_device_memory_size_for_shapes(context, out size),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_update_device_memory_size_for_shapes(context, out size),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_update_device_memory_size_for_shapes(context, out size),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return size.ToUInt64();
    }

    public static ulong GetExecutionContextPersistentCacheLimit(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        UIntPtr size;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_get_persistent_cache_limit(context, out size),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_get_persistent_cache_limit(context, out size),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_get_persistent_cache_limit(context, out size),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return size.ToUInt64();
    }

    public static void SetExecutionContextPersistentCacheLimit(TensorRtApiLine line, SafeTensorRtObjectHandle context, ulong cacheSize)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_set_persistent_cache_limit(context, (UIntPtr)cacheSize),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_set_persistent_cache_limit(context, (UIntPtr)cacheSize),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_set_persistent_cache_limit(context, (UIntPtr)cacheSize),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    public static void SetExecutionContextInputConsumedEvent(TensorRtApiLine line, SafeTensorRtObjectHandle context, SafeCudaEventHandle eventHandle)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_execution_context_set_input_consumed_event(context, eventHandle),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_execution_context_set_input_consumed_event(context, eventHandle),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_execution_context_set_input_consumed_event(context, eventHandle),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

}

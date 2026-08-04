using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static int GetEngineLayerCount(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        return GetEngineInt(line, engine, NativeMethodsTensorRt.jyppx_trt8_engine_get_layer_count, NativeMethodsTensorRt.jyppx_trt10_engine_get_layer_count, NativeMethodsTensorRt.jyppx_trt11_engine_get_layer_count);
    }

    public static bool IsEngineRefittable(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        return GetEngineBool(line, engine, NativeMethodsTensorRt.jyppx_trt8_engine_is_refittable, NativeMethodsTensorRt.jyppx_trt10_engine_is_refittable, NativeMethodsTensorRt.jyppx_trt11_engine_is_refittable);
    }

    public static SafeTensorRtObjectHandle CreateRefitter(TensorRtApiLine line, SafeTensorRtObjectHandle engine, SafeTensorRtObjectHandle logger)
    {
        SafeTensorRtObjectHandle refitter;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_create_refitter(engine, logger, out refitter),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_create_refitter(engine, logger, out refitter),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_create_refitter(engine, logger, out refitter),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return refitter;
    }

    public static string GetEngineName(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        return ReadUtf8Buffer(line switch
        {
            TensorRtApiLine.TensorRt8 => (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt8_engine_get_name(engine, buffer, size, out required),
            TensorRtApiLine.TensorRt10 => (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt10_engine_get_name(engine, buffer, size, out required),
            TensorRtApiLine.TensorRt11 => (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt11_engine_get_name(engine, buffer, size, out required),
            _ => throw UnsupportedLine()
        }, "Engine name is too large for the managed buffer.");
    }

    public static TensorRtTensorLocation GetEngineTensorLocation(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        int location;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_get_tensor_location(engine, tensorNameUtf8.Pointer, out location),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_tensor_location(engine, tensorNameUtf8.Pointer, out location),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_tensor_location(engine, tensorNameUtf8.Pointer, out location),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtTensorLocation)location;
    }

    public static bool IsEngineShapeInferenceIO(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName)
    {
        return GetEngineTensorBool(line, engine, tensorName, NativeMethodsTensorRt.jyppx_trt8_engine_is_shape_inference_io, NativeMethodsTensorRt.jyppx_trt10_engine_is_shape_inference_io, NativeMethodsTensorRt.jyppx_trt11_engine_is_shape_inference_io);
    }

    public static int GetEngineTensorBytesPerComponent(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName)
    {
        return GetEngineTensorInt(line, engine, tensorName, NativeMethodsTensorRt.jyppx_trt8_engine_get_tensor_bytes_per_component, NativeMethodsTensorRt.jyppx_trt10_engine_get_tensor_bytes_per_component, NativeMethodsTensorRt.jyppx_trt11_engine_get_tensor_bytes_per_component);
    }

    public static int GetEngineTensorComponentsPerElement(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName)
    {
        return GetEngineTensorInt(line, engine, tensorName, NativeMethodsTensorRt.jyppx_trt8_engine_get_tensor_components_per_element, NativeMethodsTensorRt.jyppx_trt10_engine_get_tensor_components_per_element, NativeMethodsTensorRt.jyppx_trt11_engine_get_tensor_components_per_element);
    }

    public static TensorRtTensorFormat GetEngineTensorFormat(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        int format;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_get_tensor_format(engine, tensorNameUtf8.Pointer, out format),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_tensor_format(engine, tensorNameUtf8.Pointer, out format),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_tensor_format(engine, tensorNameUtf8.Pointer, out format),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return (TensorRtTensorFormat)format;
    }

    public static string GetEngineTensorFormatDescription(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        return ReadUtf8Buffer(line switch
        {
            TensorRtApiLine.TensorRt8 => (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt8_engine_get_tensor_format_desc(engine, tensorNameUtf8.Pointer, buffer, size, out required),
            TensorRtApiLine.TensorRt10 => (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt10_engine_get_tensor_format_desc(engine, tensorNameUtf8.Pointer, buffer, size, out required),
            TensorRtApiLine.TensorRt11 => (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt11_engine_get_tensor_format_description(engine, tensorNameUtf8.Pointer, buffer, size, out required),
            _ => throw UnsupportedLine()
        }, "Engine tensor format description is too large for the managed buffer.");
    }

    public static int GetEngineTensorVectorizedDimension(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName)
    {
        return GetEngineTensorInt(line, engine, tensorName, NativeMethodsTensorRt.jyppx_trt8_engine_get_tensor_vectorized_dim, NativeMethodsTensorRt.jyppx_trt10_engine_get_tensor_vectorized_dim, NativeMethodsTensorRt.jyppx_trt11_engine_get_tensor_vectorized_dim);
    }

    public static int GetEngineTensorBytesPerComponent(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName, int profileIndex)
    {
        return GetEngineTensorProfileInt(line, engine, tensorName, profileIndex, NativeMethodsTensorRt.jyppx_trt8_engine_get_tensor_bytes_per_component_for_profile, NativeMethodsTensorRt.jyppx_trt10_engine_get_tensor_bytes_per_component_for_profile, NativeMethodsTensorRt.jyppx_trt11_engine_get_tensor_bytes_per_component_for_profile);
    }

    public static int GetEngineTensorComponentsPerElement(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName, int profileIndex)
    {
        return GetEngineTensorProfileInt(line, engine, tensorName, profileIndex, NativeMethodsTensorRt.jyppx_trt8_engine_get_tensor_components_per_element_for_profile, NativeMethodsTensorRt.jyppx_trt10_engine_get_tensor_components_per_element_for_profile, NativeMethodsTensorRt.jyppx_trt11_engine_get_tensor_components_per_element_for_profile);
    }

    public static TensorRtTensorFormat GetEngineTensorFormat(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName, int profileIndex)
    {
        return (TensorRtTensorFormat)GetEngineTensorProfileInt(line, engine, tensorName, profileIndex, NativeMethodsTensorRt.jyppx_trt8_engine_get_tensor_format_for_profile, NativeMethodsTensorRt.jyppx_trt10_engine_get_tensor_format_for_profile, NativeMethodsTensorRt.jyppx_trt11_engine_get_tensor_format_for_profile);
    }

    public static string GetEngineTensorFormatDescription(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName, int profileIndex)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        return ReadUtf8Buffer(line switch
        {
            TensorRtApiLine.TensorRt8 => (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt8_engine_get_tensor_format_desc_for_profile(engine, tensorNameUtf8.Pointer, profileIndex, buffer, size, out required),
            TensorRtApiLine.TensorRt10 => (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt10_engine_get_tensor_format_desc_for_profile(engine, tensorNameUtf8.Pointer, profileIndex, buffer, size, out required),
            TensorRtApiLine.TensorRt11 => (byte[] buffer, UIntPtr size, out UIntPtr required) => NativeMethodsTensorRt.jyppx_trt11_engine_get_tensor_format_desc_for_profile(engine, tensorNameUtf8.Pointer, profileIndex, buffer, size, out required),
            _ => throw UnsupportedLine()
        }, "Engine tensor profile format description is too large for the managed buffer.");
    }

    public static int GetEngineTensorVectorizedDimension(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName, int profileIndex)
    {
        return GetEngineTensorProfileInt(line, engine, tensorName, profileIndex, NativeMethodsTensorRt.jyppx_trt8_engine_get_tensor_vectorized_dim_for_profile, NativeMethodsTensorRt.jyppx_trt10_engine_get_tensor_vectorized_dim_for_profile, NativeMethodsTensorRt.jyppx_trt11_engine_get_tensor_vectorized_dim_for_profile);
    }

    public static TensorRtDims GetEngineProfileShape(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName, int profileIndex, TensorRtOptimizationProfileSelector selector)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        NativeTensorRtDims shape;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_engine_get_profile_shape(engine, tensorNameUtf8.Pointer, profileIndex, (int)selector, out shape),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_engine_get_profile_shape(engine, tensorNameUtf8.Pointer, profileIndex, (int)selector, out shape),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_engine_get_profile_shape(engine, tensorNameUtf8.Pointer, profileIndex, (int)selector, out shape),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(shape);
    }

    public static int[] GetEngineProfileShapeValues(TensorRtApiLine line, SafeTensorRtObjectHandle engine, int bindingIndex, int profileIndex, TensorRtOptimizationProfileSelector selector)
    {
        EnsureTensorRt8Only(line, nameof(GetEngineProfileShapeValues));
        return ReadInt32Array(
            static (handle, index, prof, sel, buffer, bufferCount, out actualCount) => NativeMethodsTensorRt.jyppx_trt8_cuda_engine_get_profile_shape_values(handle, index, prof, sel, buffer, bufferCount, out actualCount),
            engine,
            bindingIndex,
            profileIndex,
            (int)selector);
    }

    public static TensorRtEngineCapability GetEngineCapability(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        return (TensorRtEngineCapability)GetEngineInt(line, engine, NativeMethodsTensorRt.jyppx_trt8_engine_get_engine_capability, NativeMethodsTensorRt.jyppx_trt10_engine_get_engine_capability, NativeMethodsTensorRt.jyppx_trt11_engine_get_engine_capability);
    }

    public static TensorRtTacticSources GetEngineTacticSources(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        return (TensorRtTacticSources)GetEngineUInt(line, engine, NativeMethodsTensorRt.jyppx_trt8_engine_get_tactic_sources, NativeMethodsTensorRt.jyppx_trt10_engine_get_tactic_sources, NativeMethodsTensorRt.jyppx_trt11_engine_get_tactic_sources);
    }

    public static TensorRtProfilingVerbosity GetEngineProfilingVerbosity(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        return (TensorRtProfilingVerbosity)GetEngineInt(line, engine, NativeMethodsTensorRt.jyppx_trt8_engine_get_profiling_verbosity, NativeMethodsTensorRt.jyppx_trt10_engine_get_profiling_verbosity, NativeMethodsTensorRt.jyppx_trt11_engine_get_profiling_verbosity);
    }

    public static int GetEngineMaxBatchSizeCompatibility(TensorRtApiLine line, SafeTensorRtObjectHandle engine)
    {
        return GetEngineInt(line, engine, NativeMethodsTensorRt.jyppx_trt8_engine_get_max_batch_size, NativeMethodsTensorRt.jyppx_trt10_engine_get_max_batch_size, NativeMethodsTensorRt.jyppx_trt11_engine_get_max_batch_size);
    }

}

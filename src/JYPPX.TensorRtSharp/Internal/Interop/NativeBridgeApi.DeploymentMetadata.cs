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
    private delegate BridgeStatusCode Utf8BufferGetter(byte[] outputBuffer, UIntPtr outputBufferSize, out UIntPtr requiredSize);
    private delegate BridgeStatusCode Int32ArrayGetter(SafeTensorRtObjectHandle handle, int index, int profileIndex, int selector, IntPtr outputValues, int outputCount, out int actualCount);
    private delegate BridgeStatusCode EngineIntGetter(SafeTensorRtObjectHandle engine, out int value);
    private delegate BridgeStatusCode EngineUIntGetter(SafeTensorRtObjectHandle engine, out uint value);
    private delegate BridgeStatusCode EngineBoolGetter(SafeTensorRtObjectHandle engine, out int value);
    private delegate BridgeStatusCode EngineTensorIntGetter(SafeTensorRtObjectHandle engine, IntPtr tensorName, out int value);
    private delegate BridgeStatusCode EngineTensorBoolGetter(SafeTensorRtObjectHandle engine, IntPtr tensorName, out int value);
    private delegate BridgeStatusCode EngineTensorProfileIntGetter(SafeTensorRtObjectHandle engine, IntPtr tensorName, int profileIndex, out int value);
    private delegate BridgeStatusCode ContextBoolGetter(SafeTensorRtObjectHandle context, out int value);
    private delegate BridgeStatusCode ContextBoolSetter(SafeTensorRtObjectHandle context, int value);
    private delegate BridgeStatusCode ContextDimsGetter(SafeTensorRtObjectHandle context, IntPtr tensorName, out NativeTensorRtDims dims);
    private delegate BridgeStatusCode ContextTensorLongGetter(SafeTensorRtObjectHandle context, IntPtr tensorName, out long value);
    private delegate BridgeStatusCode ContextTensorBoolGetter(SafeTensorRtObjectHandle context, IntPtr tensorName, out int value);
    private delegate BridgeStatusCode ContextTensorBoolSetter(SafeTensorRtObjectHandle context, IntPtr tensorName, int value);
    private delegate BridgeStatusCode LayerIntGetter(SafeTensorRtObjectHandle layer, out int value);
    private delegate BridgeStatusCode LayerUIntGetter(SafeTensorRtObjectHandle layer, out uint value);
    private delegate BridgeStatusCode LayerInt64Getter(SafeTensorRtObjectHandle layer, out long value);
    private delegate BridgeStatusCode LayerDoubleGetter(SafeTensorRtObjectHandle layer, out double value);
    private delegate BridgeStatusCode LayerDimsGetter(SafeTensorRtObjectHandle layer, out NativeTensorRtDims dims);
    private delegate BridgeStatusCode LayerIntSetter(SafeTensorRtObjectHandle layer, int value);
    private delegate BridgeStatusCode LayerUIntSetter(SafeTensorRtObjectHandle layer, uint value);
    private delegate BridgeStatusCode LayerInt64Setter(SafeTensorRtObjectHandle layer, long value);
    private delegate BridgeStatusCode LayerDoubleSetter(SafeTensorRtObjectHandle layer, double value);
    private delegate BridgeStatusCode LayerDimsSetter(SafeTensorRtObjectHandle layer, ref NativeTensorRtDims dims);
    private delegate BridgeStatusCode RefitterEntriesGetter(SafeTensorRtObjectHandle refitter, NativeTensorRtRefitEntryInfo[] outputEntries, int outputCount, out int count);

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

    public static int GetRefitterMissingCount(TensorRtApiLine line, SafeTensorRtObjectHandle refitter)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_refitter_get_missing_count(refitter, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_get_missing_count(refitter, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_refitter_get_missing_count(refitter, out count),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static int GetRefitterAllCount(TensorRtApiLine line, SafeTensorRtObjectHandle refitter)
    {
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_refitter_get_all_count(refitter, out count),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_get_all_count(refitter, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_refitter_get_all_count(refitter, out count),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static NativeTensorRtRefitEntryInfo[] GetRefitterMissingEntries(TensorRtApiLine line, SafeTensorRtObjectHandle refitter)
    {
        return GetRefitterEntries(
            line,
            refitter,
            GetRefitterMissingCount(line, refitter),
            NativeMethodsTensorRt.jyppx_trt8_refitter_get_missing_entries,
            NativeMethodsTensorRt.jyppx_trt10_refitter_get_missing_entries,
            NativeMethodsTensorRt.jyppx_trt11_refitter_get_missing_entries);
    }

    public static NativeTensorRtRefitEntryInfo[] GetRefitterAllEntries(TensorRtApiLine line, SafeTensorRtObjectHandle refitter)
    {
        return GetRefitterEntries(
            line,
            refitter,
            GetRefitterAllCount(line, refitter),
            NativeMethodsTensorRt.jyppx_trt8_refitter_get_all_entries,
            NativeMethodsTensorRt.jyppx_trt10_refitter_get_all_entries,
            NativeMethodsTensorRt.jyppx_trt11_refitter_get_all_entries);
    }

    public static bool SetRefitterWeights(TensorRtApiLine line, SafeTensorRtObjectHandle refitter, string layerName, TensorRtWeightsRole role, TensorRtRefitWeightsBuffer weights)
    {
        if (weights == null)
        {
            throw new ArgumentNullException(nameof(weights));
        }

        if (string.IsNullOrWhiteSpace(layerName))
        {
            throw new ArgumentException("Layer name must not be null or empty.", nameof(layerName));
        }

        using Utf8Interop.Utf8StringScope layerNameUtf8 = Utf8Interop.ToNativeString(layerName);
        int set;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_refitter_set_weights(refitter, layerNameUtf8.Pointer, (int)role, (int)weights.DataType, weights.Pointer, weights.ElementCount, out set),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_set_weights(refitter, layerNameUtf8.Pointer, (int)role, (int)weights.DataType, weights.Pointer, weights.ElementCount, out set),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_refitter_set_weights(refitter, layerNameUtf8.Pointer, (int)role, (int)weights.DataType, weights.Pointer, weights.ElementCount, out set),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return set != 0;
    }

    public static bool RefitCudaEngine(TensorRtApiLine line, SafeTensorRtObjectHandle refitter)
    {
        int refitted;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_refitter_refit_cuda_engine(refitter, out refitted),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_refitter_refit_cuda_engine(refitter, out refitted),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_refitter_refit_cuda_engine(refitter, out refitted),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return refitted != 0;
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

    public static void SetReduceOperation(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtReduceOperation operation)
    {
        SetLayerInt(line, layer, (int)operation, NativeMethodsTensorRt.jyppx_trt8_reduce_layer_set_operation, NativeMethodsTensorRt.jyppx_trt10_reduce_layer_set_operation, NativeMethodsTensorRt.jyppx_trt11_reduce_layer_set_operation);
    }

    public static void SetReduceAxes(TensorRtApiLine line, SafeTensorRtObjectHandle layer, uint axes)
    {
        SetLayerUInt(line, layer, axes, NativeMethodsTensorRt.jyppx_trt8_reduce_layer_set_axes, NativeMethodsTensorRt.jyppx_trt10_reduce_layer_set_axes, NativeMethodsTensorRt.jyppx_trt11_reduce_layer_set_axes);
    }

    public static void SetReduceKeepDimensions(TensorRtApiLine line, SafeTensorRtObjectHandle layer, bool keepDimensions)
    {
        SetLayerInt(line, layer, keepDimensions ? 1 : 0, NativeMethodsTensorRt.jyppx_trt8_reduce_layer_set_keep_dimensions, NativeMethodsTensorRt.jyppx_trt10_reduce_layer_set_keep_dimensions, NativeMethodsTensorRt.jyppx_trt11_reduce_layer_set_keep_dimensions);
    }

    public static void SetUnaryOperation(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtUnaryOperation operation)
    {
        SetLayerInt(line, layer, (int)operation, NativeMethodsTensorRt.jyppx_trt8_unary_layer_set_operation, NativeMethodsTensorRt.jyppx_trt10_unary_layer_set_operation, NativeMethodsTensorRt.jyppx_trt11_unary_layer_set_operation);
    }

    public static void SetTopKOperation(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtTopKOperation operation)
    {
        SetLayerInt(line, layer, (int)operation, NativeMethodsTensorRt.jyppx_trt8_topk_layer_set_operation, NativeMethodsTensorRt.jyppx_trt10_topk_layer_set_operation, NativeMethodsTensorRt.jyppx_trt11_topk_layer_set_operation);
    }

    public static void SetTopKValue(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int k)
    {
        SetLayerInt(line, layer, k, NativeMethodsTensorRt.jyppx_trt8_topk_layer_set_k, NativeMethodsTensorRt.jyppx_trt10_topk_layer_set_k, NativeMethodsTensorRt.jyppx_trt11_topk_layer_set_k);
    }

    public static void SetTopKAxes(TensorRtApiLine line, SafeTensorRtObjectHandle layer, uint axes)
    {
        SetLayerUInt(line, layer, axes, NativeMethodsTensorRt.jyppx_trt8_topk_layer_set_axes, NativeMethodsTensorRt.jyppx_trt10_topk_layer_set_axes, NativeMethodsTensorRt.jyppx_trt11_topk_layer_set_axes);
    }

    public static void SetGatherAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int axis)
    {
        SetLayerInt(line, layer, axis, NativeMethodsTensorRt.jyppx_trt8_gather_layer_set_axis, NativeMethodsTensorRt.jyppx_trt10_gather_layer_set_axis, NativeMethodsTensorRt.jyppx_trt11_gather_layer_set_axis);
    }

    public static TensorRtElementWiseOperation GetElementWiseOperation(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtElementWiseOperation)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_elementwise_layer_get_operation, NativeMethodsTensorRt.jyppx_trt10_elementwise_layer_get_operation, NativeMethodsTensorRt.jyppx_trt11_elementwise_layer_get_operation);
    }

    public static void SetElementWiseOperation(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtElementWiseOperation operation)
    {
        SetLayerInt(line, layer, (int)operation, NativeMethodsTensorRt.jyppx_trt8_elementwise_layer_set_operation, NativeMethodsTensorRt.jyppx_trt10_elementwise_layer_set_operation, NativeMethodsTensorRt.jyppx_trt11_elementwise_layer_set_operation);
    }

    public static void SetActivationType(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtActivationType activationType)
    {
        SetLayerInt(line, layer, (int)activationType, NativeMethodsTensorRt.jyppx_trt8_activation_layer_set_type, NativeMethodsTensorRt.jyppx_trt10_activation_layer_set_type, NativeMethodsTensorRt.jyppx_trt11_activation_layer_set_type);
    }

    public static double GetActivationAlpha(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDouble(line, layer, NativeMethodsTensorRt.jyppx_trt8_activation_layer_get_alpha, NativeMethodsTensorRt.jyppx_trt10_activation_layer_get_alpha, NativeMethodsTensorRt.jyppx_trt11_activation_layer_get_alpha);
    }

    public static void SetActivationAlpha(TensorRtApiLine line, SafeTensorRtObjectHandle layer, double alpha)
    {
        SetLayerDouble(line, layer, alpha, NativeMethodsTensorRt.jyppx_trt8_activation_layer_set_alpha, NativeMethodsTensorRt.jyppx_trt10_activation_layer_set_alpha, NativeMethodsTensorRt.jyppx_trt11_activation_layer_set_alpha);
    }

    public static double GetActivationBeta(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDouble(line, layer, NativeMethodsTensorRt.jyppx_trt8_activation_layer_get_beta, NativeMethodsTensorRt.jyppx_trt10_activation_layer_get_beta, NativeMethodsTensorRt.jyppx_trt11_activation_layer_get_beta);
    }

    public static void SetActivationBeta(TensorRtApiLine line, SafeTensorRtObjectHandle layer, double beta)
    {
        SetLayerDouble(line, layer, beta, NativeMethodsTensorRt.jyppx_trt8_activation_layer_set_beta, NativeMethodsTensorRt.jyppx_trt10_activation_layer_set_beta, NativeMethodsTensorRt.jyppx_trt11_activation_layer_set_beta);
    }

    public static void SetPoolingType(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtPoolingType poolingType)
    {
        SetLayerInt(line, layer, (int)poolingType, NativeMethodsTensorRt.jyppx_trt8_pooling_layer_set_type, NativeMethodsTensorRt.jyppx_trt10_pooling_layer_set_type, NativeMethodsTensorRt.jyppx_trt11_pooling_layer_set_type);
    }

    public static double GetPoolingBlendFactor(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDouble(line, layer, NativeMethodsTensorRt.jyppx_trt8_pooling_layer_get_blend_factor, NativeMethodsTensorRt.jyppx_trt10_pooling_layer_get_blend_factor, NativeMethodsTensorRt.jyppx_trt11_pooling_layer_get_blend_factor);
    }

    public static void SetPoolingBlendFactor(TensorRtApiLine line, SafeTensorRtObjectHandle layer, double blendFactor)
    {
        SetLayerDouble(line, layer, blendFactor, NativeMethodsTensorRt.jyppx_trt8_pooling_layer_set_blend_factor, NativeMethodsTensorRt.jyppx_trt10_pooling_layer_set_blend_factor, NativeMethodsTensorRt.jyppx_trt11_pooling_layer_set_blend_factor);
    }

    public static bool GetPoolingAverageCountExcludesPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_pooling_layer_get_average_count_excludes_padding, NativeMethodsTensorRt.jyppx_trt10_pooling_layer_get_average_count_excludes_padding, NativeMethodsTensorRt.jyppx_trt11_pooling_layer_get_average_count_excludes_padding) != 0;
    }

    public static void SetPoolingAverageCountExcludesPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer, bool excludesPadding)
    {
        SetLayerInt(line, layer, excludesPadding ? 1 : 0, NativeMethodsTensorRt.jyppx_trt8_pooling_layer_set_average_count_excludes_padding, NativeMethodsTensorRt.jyppx_trt10_pooling_layer_set_average_count_excludes_padding, NativeMethodsTensorRt.jyppx_trt11_pooling_layer_set_average_count_excludes_padding);
    }

    public static TensorRtDims GetPoolingPrePadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_pooling_layer_get_pre_padding, NativeMethodsTensorRt.jyppx_trt10_pooling_layer_get_pre_padding, NativeMethodsTensorRt.jyppx_trt11_pooling_layer_get_pre_padding);
    }

    public static void SetPoolingPrePadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims padding)
    {
        SetLayerDims(line, layer, padding, NativeMethodsTensorRt.jyppx_trt8_pooling_layer_set_pre_padding, NativeMethodsTensorRt.jyppx_trt10_pooling_layer_set_pre_padding, NativeMethodsTensorRt.jyppx_trt11_pooling_layer_set_pre_padding);
    }

    public static TensorRtDims GetPoolingPostPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_pooling_layer_get_post_padding, NativeMethodsTensorRt.jyppx_trt10_pooling_layer_get_post_padding, NativeMethodsTensorRt.jyppx_trt11_pooling_layer_get_post_padding);
    }

    public static void SetPoolingPostPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims padding)
    {
        SetLayerDims(line, layer, padding, NativeMethodsTensorRt.jyppx_trt8_pooling_layer_set_post_padding, NativeMethodsTensorRt.jyppx_trt10_pooling_layer_set_post_padding, NativeMethodsTensorRt.jyppx_trt11_pooling_layer_set_post_padding);
    }

    public static TensorRtPaddingMode GetPoolingPaddingMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtPaddingMode)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_pooling_layer_get_padding_mode, NativeMethodsTensorRt.jyppx_trt10_pooling_layer_get_padding_mode, NativeMethodsTensorRt.jyppx_trt11_pooling_layer_get_padding_mode);
    }

    public static void SetPoolingPaddingMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtPaddingMode paddingMode)
    {
        SetLayerInt(line, layer, (int)paddingMode, NativeMethodsTensorRt.jyppx_trt8_pooling_layer_set_padding_mode, NativeMethodsTensorRt.jyppx_trt10_pooling_layer_set_padding_mode, NativeMethodsTensorRt.jyppx_trt11_pooling_layer_set_padding_mode);
    }

    public static int GetConvolutionOutputMaps(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_get_nb_output_maps, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_get_nb_output_maps, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_get_nb_output_maps);
    }

    public static void SetConvolutionOutputMaps(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int outputMaps)
    {
        SetLayerInt(line, layer, outputMaps, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_set_nb_output_maps, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_set_nb_output_maps, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_set_nb_output_maps);
    }

    public static int GetConvolutionGroups(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_get_nb_groups, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_get_nb_groups, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_get_nb_groups);
    }

    public static void SetConvolutionGroups(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int groups)
    {
        SetLayerInt(line, layer, groups, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_set_nb_groups, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_set_nb_groups, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_set_nb_groups);
    }

    public static TensorRtDims GetConvolutionStride(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_get_stride_nd, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_get_stride_nd, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_get_stride_nd);
    }

    public static void SetConvolutionStride(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims stride)
    {
        SetLayerDims(line, layer, stride, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_set_stride_nd, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_set_stride_nd, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_set_stride_nd);
    }

    public static TensorRtDims GetConvolutionPrePadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_get_pre_padding, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_get_pre_padding, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_get_pre_padding);
    }

    public static void SetConvolutionPrePadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims padding)
    {
        SetLayerDims(line, layer, padding, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_set_pre_padding, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_set_pre_padding, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_set_pre_padding);
    }

    public static TensorRtDims GetConvolutionPostPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_get_post_padding, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_get_post_padding, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_get_post_padding);
    }

    public static void SetConvolutionPostPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims padding)
    {
        SetLayerDims(line, layer, padding, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_set_post_padding, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_set_post_padding, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_set_post_padding);
    }

    public static TensorRtDims GetConvolutionDilation(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_get_dilation_nd, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_get_dilation_nd, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_get_dilation_nd);
    }

    public static void SetConvolutionDilation(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims dilation)
    {
        SetLayerDims(line, layer, dilation, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_set_dilation_nd, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_set_dilation_nd, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_set_dilation_nd);
    }

    public static TensorRtPaddingMode GetConvolutionPaddingMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtPaddingMode)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_get_padding_mode, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_get_padding_mode, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_get_padding_mode);
    }

    public static void SetConvolutionPaddingMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtPaddingMode paddingMode)
    {
        SetLayerInt(line, layer, (int)paddingMode, NativeMethodsTensorRt.jyppx_trt8_convolution_layer_set_padding_mode, NativeMethodsTensorRt.jyppx_trt10_convolution_layer_set_padding_mode, NativeMethodsTensorRt.jyppx_trt11_convolution_layer_set_padding_mode);
    }

    public static int GetDeconvolutionOutputMaps(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_get_nb_output_maps, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_get_nb_output_maps, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_get_nb_output_maps);
    }

    public static void SetDeconvolutionOutputMaps(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int outputMaps)
    {
        SetLayerInt(line, layer, outputMaps, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_set_nb_output_maps, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_set_nb_output_maps, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_set_nb_output_maps);
    }

    public static int GetDeconvolutionGroups(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_get_nb_groups, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_get_nb_groups, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_get_nb_groups);
    }

    public static void SetDeconvolutionGroups(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int groups)
    {
        SetLayerInt(line, layer, groups, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_set_nb_groups, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_set_nb_groups, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_set_nb_groups);
    }

    public static TensorRtDims GetDeconvolutionKernelSize(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_get_kernel_size_nd, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_get_kernel_size_nd, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_get_kernel_size_nd);
    }

    public static void SetDeconvolutionKernelSize(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims kernelSize)
    {
        SetLayerDims(line, layer, kernelSize, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_set_kernel_size_nd, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_set_kernel_size_nd, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_set_kernel_size_nd);
    }

    public static TensorRtDims GetDeconvolutionStride(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_get_stride_nd, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_get_stride_nd, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_get_stride_nd);
    }

    public static void SetDeconvolutionStride(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims stride)
    {
        SetLayerDims(line, layer, stride, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_set_stride_nd, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_set_stride_nd, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_set_stride_nd);
    }

    public static TensorRtDims GetDeconvolutionPrePadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_get_pre_padding, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_get_pre_padding, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_get_pre_padding);
    }

    public static void SetDeconvolutionPrePadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims padding)
    {
        SetLayerDims(line, layer, padding, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_set_pre_padding, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_set_pre_padding, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_set_pre_padding);
    }

    public static TensorRtDims GetDeconvolutionPostPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_get_post_padding, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_get_post_padding, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_get_post_padding);
    }

    public static void SetDeconvolutionPostPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims padding)
    {
        SetLayerDims(line, layer, padding, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_set_post_padding, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_set_post_padding, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_set_post_padding);
    }

    public static TensorRtDims GetDeconvolutionDilation(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_get_dilation_nd, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_get_dilation_nd, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_get_dilation_nd);
    }

    public static void SetDeconvolutionDilation(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims dilation)
    {
        SetLayerDims(line, layer, dilation, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_set_dilation_nd, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_set_dilation_nd, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_set_dilation_nd);
    }

    public static TensorRtPaddingMode GetDeconvolutionPaddingMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtPaddingMode)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_get_padding_mode, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_get_padding_mode, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_get_padding_mode);
    }

    public static void SetDeconvolutionPaddingMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtPaddingMode paddingMode)
    {
        SetLayerInt(line, layer, (int)paddingMode, NativeMethodsTensorRt.jyppx_trt8_deconvolution_layer_set_padding_mode, NativeMethodsTensorRt.jyppx_trt10_deconvolution_layer_set_padding_mode, NativeMethodsTensorRt.jyppx_trt11_deconvolution_layer_set_padding_mode);
    }

    public static TensorRtScaleMode GetScaleMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtScaleMode)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_scale_layer_get_mode, NativeMethodsTensorRt.jyppx_trt10_scale_layer_get_mode, NativeMethodsTensorRt.jyppx_trt11_scale_layer_get_mode);
    }

    public static void SetScaleMode(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtScaleMode mode)
    {
        SetLayerInt(line, layer, (int)mode, NativeMethodsTensorRt.jyppx_trt8_scale_layer_set_mode, NativeMethodsTensorRt.jyppx_trt10_scale_layer_set_mode, NativeMethodsTensorRt.jyppx_trt11_scale_layer_set_mode);
    }

    public static int GetScaleChannelAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_scale_layer_get_channel_axis, NativeMethodsTensorRt.jyppx_trt10_scale_layer_get_channel_axis, NativeMethodsTensorRt.jyppx_trt11_scale_layer_get_channel_axis);
    }

    public static void SetScaleChannelAxis(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int channelAxis)
    {
        SetLayerInt(line, layer, channelAxis, NativeMethodsTensorRt.jyppx_trt8_scale_layer_set_channel_axis, NativeMethodsTensorRt.jyppx_trt10_scale_layer_set_channel_axis, NativeMethodsTensorRt.jyppx_trt11_scale_layer_set_channel_axis);
    }

    public static TensorRtDims GetPaddingPrePadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_padding_layer_get_pre_padding_nd, NativeMethodsTensorRt.jyppx_trt10_padding_layer_get_pre_padding_nd, NativeMethodsTensorRt.jyppx_trt11_padding_layer_get_pre_padding_nd);
    }

    public static void SetPaddingPrePadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims padding)
    {
        SetLayerDims(line, layer, padding, NativeMethodsTensorRt.jyppx_trt8_padding_layer_set_pre_padding_nd, NativeMethodsTensorRt.jyppx_trt10_padding_layer_set_pre_padding_nd, NativeMethodsTensorRt.jyppx_trt11_padding_layer_set_pre_padding_nd);
    }

    public static TensorRtDims GetPaddingPostPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDims(line, layer, NativeMethodsTensorRt.jyppx_trt8_padding_layer_get_post_padding_nd, NativeMethodsTensorRt.jyppx_trt10_padding_layer_get_post_padding_nd, NativeMethodsTensorRt.jyppx_trt11_padding_layer_get_post_padding_nd);
    }

    public static void SetPaddingPostPadding(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims padding)
    {
        SetLayerDims(line, layer, padding, NativeMethodsTensorRt.jyppx_trt8_padding_layer_set_post_padding_nd, NativeMethodsTensorRt.jyppx_trt10_padding_layer_set_post_padding_nd, NativeMethodsTensorRt.jyppx_trt11_padding_layer_set_post_padding_nd);
    }

    public static TensorRtResizeCoordinateTransformation GetResizeCoordinateTransformation(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtResizeCoordinateTransformation)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_resize_layer_get_coordinate_transformation, NativeMethodsTensorRt.jyppx_trt10_resize_layer_get_coordinate_transformation, NativeMethodsTensorRt.jyppx_trt11_resize_layer_get_coordinate_transformation);
    }

    public static void SetResizeCoordinateTransformation(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtResizeCoordinateTransformation value)
    {
        SetLayerInt(line, layer, (int)value, NativeMethodsTensorRt.jyppx_trt8_resize_layer_set_coordinate_transformation, NativeMethodsTensorRt.jyppx_trt10_resize_layer_set_coordinate_transformation, NativeMethodsTensorRt.jyppx_trt11_resize_layer_set_coordinate_transformation);
    }

    public static TensorRtResizeSelector GetResizeSelectorForSinglePixel(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtResizeSelector)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_resize_layer_get_selector_for_single_pixel, NativeMethodsTensorRt.jyppx_trt10_resize_layer_get_selector_for_single_pixel, NativeMethodsTensorRt.jyppx_trt11_resize_layer_get_selector_for_single_pixel);
    }

    public static void SetResizeSelectorForSinglePixel(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtResizeSelector value)
    {
        SetLayerInt(line, layer, (int)value, NativeMethodsTensorRt.jyppx_trt8_resize_layer_set_selector_for_single_pixel, NativeMethodsTensorRt.jyppx_trt10_resize_layer_set_selector_for_single_pixel, NativeMethodsTensorRt.jyppx_trt11_resize_layer_set_selector_for_single_pixel);
    }

    public static TensorRtResizeRoundMode GetResizeNearestRounding(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return (TensorRtResizeRoundMode)GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_resize_layer_get_nearest_rounding, NativeMethodsTensorRt.jyppx_trt10_resize_layer_get_nearest_rounding, NativeMethodsTensorRt.jyppx_trt11_resize_layer_get_nearest_rounding);
    }

    public static void SetResizeNearestRounding(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtResizeRoundMode value)
    {
        SetLayerInt(line, layer, (int)value, NativeMethodsTensorRt.jyppx_trt8_resize_layer_set_nearest_rounding, NativeMethodsTensorRt.jyppx_trt10_resize_layer_set_nearest_rounding, NativeMethodsTensorRt.jyppx_trt11_resize_layer_set_nearest_rounding);
    }

    public static double GetResizeCubicCoefficient(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerDouble(line, layer, NativeMethodsTensorRt.jyppx_trt8_resize_layer_get_cubic_coeff, NativeMethodsTensorRt.jyppx_trt10_resize_layer_get_cubic_coeff, NativeMethodsTensorRt.jyppx_trt11_resize_layer_get_cubic_coeff);
    }

    public static void SetResizeCubicCoefficient(TensorRtApiLine line, SafeTensorRtObjectHandle layer, double value)
    {
        SetLayerDouble(line, layer, value, NativeMethodsTensorRt.jyppx_trt8_resize_layer_set_cubic_coeff, NativeMethodsTensorRt.jyppx_trt10_resize_layer_set_cubic_coeff, NativeMethodsTensorRt.jyppx_trt11_resize_layer_set_cubic_coeff);
    }

    public static bool GetResizeExcludeOutside(TensorRtApiLine line, SafeTensorRtObjectHandle layer)
    {
        return GetLayerInt(line, layer, NativeMethodsTensorRt.jyppx_trt8_resize_layer_get_exclude_outside, NativeMethodsTensorRt.jyppx_trt10_resize_layer_get_exclude_outside, NativeMethodsTensorRt.jyppx_trt11_resize_layer_get_exclude_outside) != 0;
    }

    public static void SetResizeExcludeOutside(TensorRtApiLine line, SafeTensorRtObjectHandle layer, bool value)
    {
        SetLayerInt(line, layer, value ? 1 : 0, NativeMethodsTensorRt.jyppx_trt8_resize_layer_set_exclude_outside, NativeMethodsTensorRt.jyppx_trt10_resize_layer_set_exclude_outside, NativeMethodsTensorRt.jyppx_trt11_resize_layer_set_exclude_outside);
    }

    private static int GetEngineInt(TensorRtApiLine line, SafeTensorRtObjectHandle engine, EngineIntGetter trt8, EngineIntGetter trt10, EngineIntGetter? trt11 = null)
    {
        int value;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => trt8(engine, out value),
            TensorRtApiLine.TensorRt10 => trt10(engine, out value),
            TensorRtApiLine.TensorRt11 when trt11 != null => trt11(engine, out value),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    private static NativeTensorRtRefitEntryInfo[] GetRefitterEntries(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle refitter,
        int initialCount,
        RefitterEntriesGetter trt8,
        RefitterEntriesGetter trt10,
        RefitterEntriesGetter? trt11 = null)
    {
        if (initialCount <= 0)
        {
            return Array.Empty<NativeTensorRtRefitEntryInfo>();
        }

        NativeTensorRtRefitEntryInfo[] entries = CreateRefitterEntryBuffer(initialCount);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => trt8(refitter, entries, entries.Length, out _),
            TensorRtApiLine.TensorRt10 => trt10(refitter, entries, entries.Length, out _),
            TensorRtApiLine.TensorRt11 when trt11 != null => trt11(refitter, entries, entries.Length, out _),
            _ => throw UnsupportedLine()
        };

        if (status == BridgeStatusCode.BufferTooSmall)
        {
            int requiredCount = GetRefitterEntryRequiredCount(line, refitter, trt8, trt10, trt11);
            entries = CreateRefitterEntryBuffer(requiredCount);
            status = line switch
            {
                TensorRtApiLine.TensorRt8 => trt8(refitter, entries, entries.Length, out _),
                TensorRtApiLine.TensorRt10 => trt10(refitter, entries, entries.Length, out _),
                TensorRtApiLine.TensorRt11 when trt11 != null => trt11(refitter, entries, entries.Length, out _),
                _ => throw UnsupportedLine()
            };
        }

        NativeStatus.ThrowIfFailed(status);
        return entries;
    }

    private static int GetRefitterEntryRequiredCount(TensorRtApiLine line, SafeTensorRtObjectHandle refitter, RefitterEntriesGetter trt8, RefitterEntriesGetter trt10, RefitterEntriesGetter? trt11 = null)
    {
        NativeTensorRtRefitEntryInfo[] empty = Array.Empty<NativeTensorRtRefitEntryInfo>();
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => trt8(refitter, empty, 0, out count),
            TensorRtApiLine.TensorRt10 => trt10(refitter, empty, 0, out count),
            TensorRtApiLine.TensorRt11 when trt11 != null => trt11(refitter, empty, 0, out count),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    private static NativeTensorRtRefitEntryInfo[] CreateRefitterEntryBuffer(int count)
    {
        NativeTensorRtRefitEntryInfo[] entries = new NativeTensorRtRefitEntryInfo[count];
        for (int index = 0; index < entries.Length; index++)
        {
            entries[index].LayerName = new byte[256];
        }

        return entries;
    }

    private static uint GetEngineUInt(TensorRtApiLine line, SafeTensorRtObjectHandle engine, EngineUIntGetter trt8, EngineUIntGetter trt10, EngineUIntGetter? trt11 = null)
    {
        uint value;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => trt8(engine, out value),
            TensorRtApiLine.TensorRt10 => trt10(engine, out value),
            TensorRtApiLine.TensorRt11 when trt11 != null => trt11(engine, out value),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    private static bool GetEngineBool(TensorRtApiLine line, SafeTensorRtObjectHandle engine, EngineBoolGetter trt8, EngineBoolGetter trt10, EngineBoolGetter? trt11 = null)
    {
        int value;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => trt8(engine, out value),
            TensorRtApiLine.TensorRt10 => trt10(engine, out value),
            TensorRtApiLine.TensorRt11 when trt11 != null => trt11(engine, out value),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return value != 0;
    }

    private static int GetEngineTensorInt(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName, EngineTensorIntGetter trt8, EngineTensorIntGetter trt10, EngineTensorIntGetter? trt11 = null)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        int value;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => trt8(engine, tensorNameUtf8.Pointer, out value),
            TensorRtApiLine.TensorRt10 => trt10(engine, tensorNameUtf8.Pointer, out value),
            TensorRtApiLine.TensorRt11 when trt11 != null => trt11(engine, tensorNameUtf8.Pointer, out value),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    private static bool GetEngineTensorBool(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName, EngineTensorBoolGetter trt8, EngineTensorBoolGetter trt10, EngineTensorBoolGetter? trt11 = null)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        int value;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => trt8(engine, tensorNameUtf8.Pointer, out value),
            TensorRtApiLine.TensorRt10 => trt10(engine, tensorNameUtf8.Pointer, out value),
            TensorRtApiLine.TensorRt11 when trt11 != null => trt11(engine, tensorNameUtf8.Pointer, out value),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return value != 0;
    }

    private static int GetEngineTensorProfileInt(TensorRtApiLine line, SafeTensorRtObjectHandle engine, string tensorName, int profileIndex, EngineTensorProfileIntGetter trt8, EngineTensorProfileIntGetter trt10, EngineTensorProfileIntGetter? trt11 = null)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        int value;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => trt8(engine, tensorNameUtf8.Pointer, profileIndex, out value),
            TensorRtApiLine.TensorRt10 => trt10(engine, tensorNameUtf8.Pointer, profileIndex, out value),
            TensorRtApiLine.TensorRt11 when trt11 != null => trt11(engine, tensorNameUtf8.Pointer, profileIndex, out value),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    private static bool GetContextBool(TensorRtApiLine line, SafeTensorRtObjectHandle context, ContextBoolGetter trt8, ContextBoolGetter trt10, ContextBoolGetter? trt11 = null)
    {
        int value;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => trt8(context, out value),
            TensorRtApiLine.TensorRt10 => trt10(context, out value),
            TensorRtApiLine.TensorRt11 when trt11 != null => trt11(context, out value),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return value != 0;
    }

    private static void SetContextBool(TensorRtApiLine line, SafeTensorRtObjectHandle context, bool value, ContextBoolSetter trt8, ContextBoolSetter trt10, ContextBoolSetter? trt11 = null)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => trt8(context, value ? 1 : 0),
            TensorRtApiLine.TensorRt10 => trt10(context, value ? 1 : 0),
            TensorRtApiLine.TensorRt11 when trt11 != null => trt11(context, value ? 1 : 0),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    private static TensorRtDims GetContextTensorDims(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName, ContextDimsGetter trt8, ContextDimsGetter trt10)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        NativeTensorRtDims dims;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => trt8(context, tensorNameUtf8.Pointer, out dims),
            TensorRtApiLine.TensorRt10 => trt10(context, tensorNameUtf8.Pointer, out dims),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(dims);
    }

    private static long GetContextTensorLong(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName, ContextTensorLongGetter trt8, ContextTensorLongGetter trt10)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        long value;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => trt8(context, tensorNameUtf8.Pointer, out value),
            TensorRtApiLine.TensorRt10 => trt10(context, tensorNameUtf8.Pointer, out value),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    private static bool GetContextTensorBool(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName, ContextTensorBoolGetter trt8, ContextTensorBoolGetter trt10, ContextTensorBoolGetter? trt11 = null)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        int value;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => trt8(context, tensorNameUtf8.Pointer, out value),
            TensorRtApiLine.TensorRt10 => trt10(context, tensorNameUtf8.Pointer, out value),
            TensorRtApiLine.TensorRt11 when trt11 != null => trt11(context, tensorNameUtf8.Pointer, out value),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return value != 0;
    }

    private static void SetContextTensorBool(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName, bool value, ContextTensorBoolSetter trt8, ContextTensorBoolSetter trt10, ContextTensorBoolSetter? trt11 = null)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => trt8(context, tensorNameUtf8.Pointer, value ? 1 : 0),
            TensorRtApiLine.TensorRt10 => trt10(context, tensorNameUtf8.Pointer, value ? 1 : 0),
            TensorRtApiLine.TensorRt11 when trt11 != null => trt11(context, tensorNameUtf8.Pointer, value ? 1 : 0),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    private static int GetLayerInt(TensorRtApiLine line, SafeTensorRtObjectHandle layer, LayerIntGetter trt8, LayerIntGetter trt10, LayerIntGetter? trt11 = null)
    {
        int value;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => trt8(layer, out value),
            TensorRtApiLine.TensorRt10 => trt10(layer, out value),
            TensorRtApiLine.TensorRt11 when trt11 != null => trt11(layer, out value),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    private static double GetLayerDouble(TensorRtApiLine line, SafeTensorRtObjectHandle layer, LayerDoubleGetter trt8, LayerDoubleGetter trt10, LayerDoubleGetter? trt11 = null)
    {
        double value;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => trt8(layer, out value),
            TensorRtApiLine.TensorRt10 => trt10(layer, out value),
            TensorRtApiLine.TensorRt11 when trt11 != null => trt11(layer, out value),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    private static uint GetLayerUInt(TensorRtApiLine line, SafeTensorRtObjectHandle layer, LayerUIntGetter trt8, LayerUIntGetter trt10, LayerUIntGetter? trt11 = null)
    {
        uint value;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => trt8(layer, out value),
            TensorRtApiLine.TensorRt10 => trt10(layer, out value),
            TensorRtApiLine.TensorRt11 when trt11 != null => trt11(layer, out value),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    private static long GetLayerInt64(TensorRtApiLine line, SafeTensorRtObjectHandle layer, LayerInt64Getter trt8, LayerInt64Getter trt10, LayerInt64Getter? trt11 = null)
    {
        long value;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => trt8(layer, out value),
            TensorRtApiLine.TensorRt10 => trt10(layer, out value),
            TensorRtApiLine.TensorRt11 when trt11 != null => trt11(layer, out value),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return value;
    }

    private static TensorRtDims GetLayerDims(TensorRtApiLine line, SafeTensorRtObjectHandle layer, LayerDimsGetter trt8, LayerDimsGetter trt10, LayerDimsGetter? trt11 = null)
    {
        NativeTensorRtDims dims;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => trt8(layer, out dims),
            TensorRtApiLine.TensorRt10 => trt10(layer, out dims),
            TensorRtApiLine.TensorRt11 when trt11 != null => trt11(layer, out dims),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return TensorRtDims.FromNative(dims);
    }

    private static void SetLayerInt(TensorRtApiLine line, SafeTensorRtObjectHandle layer, int value, LayerIntSetter trt8, LayerIntSetter trt10, LayerIntSetter? trt11 = null)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => trt8(layer, value),
            TensorRtApiLine.TensorRt10 => trt10(layer, value),
            TensorRtApiLine.TensorRt11 when trt11 != null => trt11(layer, value),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    private static void SetLayerUInt(TensorRtApiLine line, SafeTensorRtObjectHandle layer, uint value, LayerUIntSetter trt8, LayerUIntSetter trt10, LayerUIntSetter? trt11 = null)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => trt8(layer, value),
            TensorRtApiLine.TensorRt10 => trt10(layer, value),
            TensorRtApiLine.TensorRt11 when trt11 != null => trt11(layer, value),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    private static void SetLayerInt64(TensorRtApiLine line, SafeTensorRtObjectHandle layer, long value, LayerInt64Setter trt8, LayerInt64Setter trt10, LayerInt64Setter? trt11 = null)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => trt8(layer, value),
            TensorRtApiLine.TensorRt10 => trt10(layer, value),
            TensorRtApiLine.TensorRt11 when trt11 != null => trt11(layer, value),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    private static void SetLayerDouble(TensorRtApiLine line, SafeTensorRtObjectHandle layer, double value, LayerDoubleSetter trt8, LayerDoubleSetter trt10, LayerDoubleSetter? trt11 = null)
    {
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => trt8(layer, value),
            TensorRtApiLine.TensorRt10 => trt10(layer, value),
            TensorRtApiLine.TensorRt11 when trt11 != null => trt11(layer, value),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    private static void SetLayerDims(TensorRtApiLine line, SafeTensorRtObjectHandle layer, TensorRtDims dims, LayerDimsSetter trt8, LayerDimsSetter trt10, LayerDimsSetter? trt11 = null)
    {
        if (dims == null)
        {
            throw new ArgumentNullException(nameof(dims));
        }

        NativeTensorRtDims nativeDims = dims.ToNative();
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => trt8(layer, ref nativeDims),
            TensorRtApiLine.TensorRt10 => trt10(layer, ref nativeDims),
            TensorRtApiLine.TensorRt11 when trt11 != null => trt11(layer, ref nativeDims),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
    }

    private static string ReadUtf8Buffer(Utf8BufferGetter getter, string tooLargeMessage)
    {
        BridgeStatusCode status = getter(Array.Empty<byte>(), UIntPtr.Zero, out UIntPtr requiredSize);
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
        status = getter(buffer, requiredSize, out _);
        NativeStatus.ThrowIfFailed(status);
        int terminator = Array.IndexOf(buffer, (byte)0);
        int length = terminator >= 0 ? terminator : buffer.Length;
        return length == 0 ? string.Empty : Encoding.UTF8.GetString(buffer, 0, length);
    }

    private static void ValidateTensorName(string tensorName)
    {
        if (string.IsNullOrWhiteSpace(tensorName))
        {
            throw new ArgumentException("Tensor name must not be null or empty.", nameof(tensorName));
        }
    }

    private static int[] ReadInt32Array(Int32ArrayGetter getter, SafeTensorRtObjectHandle handle, int index, int profileIndex, int selector)
    {
        BridgeStatusCode status = getter(handle, index, profileIndex, selector, IntPtr.Zero, 0, out int requiredCount);
        NativeStatus.ThrowIfFailed(status);
        if (requiredCount <= 0)
        {
            return Array.Empty<int>();
        }

        int[] values = new int[requiredCount];
        GCHandle pinned = GCHandle.Alloc(values, GCHandleType.Pinned);
        try
        {
            status = getter(handle, index, profileIndex, selector, pinned.AddrOfPinnedObject(), values.Length, out int actualCount);
            NativeStatus.ThrowIfFailed(status);
            if (actualCount == values.Length)
            {
                return values;
            }

            int[] trimmed = new int[Math.Max(actualCount, 0)];
            Array.Copy(values, trimmed, trimmed.Length);
            return trimmed;
        }
        finally
        {
            pinned.Free();
        }
    }

    private static BridgeProbeException UnsupportedLine()
    {
        return new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.");
    }
}

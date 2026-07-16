using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal readonly struct NativeTensorRtSerializedNetworkWithKernelText
{
    public NativeTensorRtSerializedNetworkWithKernelText(SafeTensorRtObjectHandle plan, SafeTensorRtObjectHandle? kernelText)
    {
        Plan = plan;
        KernelText = kernelText;
    }

    public SafeTensorRtObjectHandle Plan { get; }

    public SafeTensorRtObjectHandle? KernelText { get; }
}

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle BuildEngineWithConfig(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle builder,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle config)
    {
        SafeTensorRtObjectHandle engine;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_build_engine_with_config(builder, network, config, out engine),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_build_engine_with_config(builder, network, config, out engine),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_build_engine_with_config(builder, network, config, out engine),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return engine;
    }

    public static NativeTensorRtSerializedNetworkWithKernelText BuildSerializedNetworkWithKernelText(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle builder,
        SafeTensorRtObjectHandle network,
        SafeTensorRtObjectHandle config)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(BuildSerializedNetworkWithKernelText));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_builder_build_serialized_network_with_kernel_text(builder, network, config, out SafeTensorRtObjectHandle plan, out SafeTensorRtObjectHandle kernelText);
        NativeStatus.ThrowIfFailed(status);
        if (kernelText.IsInvalid)
        {
            kernelText.Dispose();
            return new NativeTensorRtSerializedNetworkWithKernelText(plan, null);
        }

        return new NativeTensorRtSerializedNetworkWithKernelText(plan, kernelText);
    }

    public static TensorRtDataType GetHostMemoryDataType(TensorRtApiLine line, SafeTensorRtObjectHandle hostMemory)
    {
        int dataType;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_host_memory_get_type(hostMemory, out dataType),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_host_memory_get_type(hostMemory, out dataType),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_host_memory_get_type(hostMemory, out dataType),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return (TensorRtDataType)dataType;
    }

    public static bool SetOptimizationProfileShapeValuesV2(TensorRtApiLine line, SafeTensorRtObjectHandle profile, string inputName, TensorRtOptimizationProfileSelector selector, IReadOnlyList<long> values)
    {
        EnsureTensorRt10Or11(line, nameof(SetOptimizationProfileShapeValuesV2));
        if (string.IsNullOrWhiteSpace(inputName))
        {
            throw new ArgumentException("Input name must not be null or empty.", nameof(inputName));
        }

        if (values == null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        if (values.Count == 0)
        {
            throw new ArgumentException("Shape values must not be empty.", nameof(values));
        }

        long[] valueArray = new long[values.Count];
        for (int i = 0; i < values.Count; i++)
        {
            valueArray[i] = values[i];
        }

        using Utf8Interop.Utf8StringScope inputNameUtf8 = Utf8Interop.ToNativeString(inputName);
        GCHandle pinned = GCHandle.Alloc(valueArray, GCHandleType.Pinned);
        try
        {
            int set;
            BridgeStatusCode status = line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_optimization_profile_set_shape_values_v2(profile, inputNameUtf8.Pointer, (int)selector, pinned.AddrOfPinnedObject(), valueArray.Length, out set),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_optimization_profile_set_shape_values_v2(profile, inputNameUtf8.Pointer, (int)selector, pinned.AddrOfPinnedObject(), valueArray.Length, out set),
                _ => throw UnsupportedLine()
            };
            NativeStatus.ThrowIfFailed(status);
            return set != 0;
        }
        finally
        {
            pinned.Free();
        }
    }

    public static int GetOptimizationProfileShapeValueCountV2(TensorRtApiLine line, SafeTensorRtObjectHandle profile, string inputName)
    {
        EnsureTensorRt10Or11(line, nameof(GetOptimizationProfileShapeValueCountV2));
        if (string.IsNullOrWhiteSpace(inputName))
        {
            throw new ArgumentException("Input name must not be null or empty.", nameof(inputName));
        }

        using Utf8Interop.Utf8StringScope inputNameUtf8 = Utf8Interop.ToNativeString(inputName);
        int count;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_optimization_profile_get_shape_value_count_v2(profile, inputNameUtf8.Pointer, out count),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_optimization_profile_get_shape_value_count_v2(profile, inputNameUtf8.Pointer, out count),
            _ => throw UnsupportedLine()
        };
        NativeStatus.ThrowIfFailed(status);
        return count;
    }

    public static long[] GetOptimizationProfileShapeValuesV2(TensorRtApiLine line, SafeTensorRtObjectHandle profile, string inputName, TensorRtOptimizationProfileSelector selector)
    {
        int count = GetOptimizationProfileShapeValueCountV2(line, profile, inputName);
        if (count <= 0)
        {
            return Array.Empty<long>();
        }

        long[] values = new long[count];
        using Utf8Interop.Utf8StringScope inputNameUtf8 = Utf8Interop.ToNativeString(inputName);
        GCHandle pinned = GCHandle.Alloc(values, GCHandleType.Pinned);
        try
        {
            int actualCount;
            BridgeStatusCode status = line switch
            {
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_optimization_profile_get_shape_values_v2(profile, inputNameUtf8.Pointer, (int)selector, pinned.AddrOfPinnedObject(), values.Length, out actualCount),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_optimization_profile_get_shape_values_v2(profile, inputNameUtf8.Pointer, (int)selector, pinned.AddrOfPinnedObject(), values.Length, out actualCount),
                _ => throw UnsupportedLine()
            };
            NativeStatus.ThrowIfFailed(status);
            if (actualCount == values.Length)
            {
                return values;
            }

            long[] trimmed = new long[Math.Max(actualCount, 0)];
            Array.Copy(values, trimmed, trimmed.Length);
            return trimmed;
        }
        finally
        {
            pinned.Free();
        }
    }

    public static void ClearBuilderConfigFlag(TensorRtApiLine line, SafeTensorRtObjectHandle config, TensorRtBuilderFlag flag)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(ClearBuilderConfigFlag));
        NativeStatus.ThrowIfFailed(NativeMethodsTensorRt.jyppx_trt11_builder_config_clear_flag(config, (int)flag));
    }

    public static bool SetBuilderConfigPluginsToSerialize(TensorRtApiLine line, SafeTensorRtObjectHandle config, IReadOnlyList<string> pluginLibraryPaths)
    {
        if (pluginLibraryPaths == null)
        {
            throw new ArgumentNullException(nameof(pluginLibraryPaths));
        }

        if (pluginLibraryPaths.Count == 0)
        {
            int cleared;
            BridgeStatusCode clearStatus = line switch
            {
                TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_plugins_to_serialize(config, IntPtr.Zero, 0, out cleared),
                TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_plugins_to_serialize(config, IntPtr.Zero, 0, out cleared),
                TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_plugins_to_serialize(config, IntPtr.Zero, 0, out cleared),
                _ => throw UnsupportedLine()
            };

            NativeStatus.ThrowIfFailed(clearStatus);
            return cleared != 0;
        }

        Utf8Interop.Utf8StringScope[] scopes = new Utf8Interop.Utf8StringScope[pluginLibraryPaths.Count];
        IntPtr[] pointers = new IntPtr[pluginLibraryPaths.Count];
        try
        {
            for (int i = 0; i < pluginLibraryPaths.Count; i++)
            {
                if (string.IsNullOrWhiteSpace(pluginLibraryPaths[i]))
                {
                    throw new ArgumentException("Plugin library path entries must not be null or empty.", nameof(pluginLibraryPaths));
                }

                scopes[i] = Utf8Interop.ToNativeString(pluginLibraryPaths[i]);
                pointers[i] = scopes[i].Pointer;
            }

            GCHandle pinned = GCHandle.Alloc(pointers, GCHandleType.Pinned);
            try
            {
                int set;
                BridgeStatusCode status = line switch
                {
                    TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_builder_config_set_plugins_to_serialize(config, pinned.AddrOfPinnedObject(), pointers.Length, out set),
                    TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_builder_config_set_plugins_to_serialize(config, pinned.AddrOfPinnedObject(), pointers.Length, out set),
                    TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_builder_config_set_plugins_to_serialize(config, pinned.AddrOfPinnedObject(), pointers.Length, out set),
                    _ => throw UnsupportedLine()
                };

                NativeStatus.ThrowIfFailed(status);
                return set != 0;
            }
            finally
            {
                pinned.Free();
            }
        }
        finally
        {
            for (int i = 0; i < scopes.Length; i++)
            {
                scopes[i]?.Dispose();
            }
        }
    }

    public static bool ClearExecutionContextTensorAddress(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(ClearExecutionContextTensorAddress));
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_clear_tensor_address(context, tensorNameUtf8.Pointer, out int cleared);
        NativeStatus.ThrowIfFailed(status);
        return cleared != 0;
    }

    public static bool ClearExecutionContextInputTensorAddress(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(ClearExecutionContextInputTensorAddress));
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_clear_input_tensor_address(context, tensorNameUtf8.Pointer, out int cleared);
        NativeStatus.ThrowIfFailed(status);
        return cleared != 0;
    }

    public static bool ClearExecutionContextOutputTensorAddress(TensorRtApiLine line, SafeTensorRtObjectHandle context, string tensorName)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(ClearExecutionContextOutputTensorAddress));
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_clear_output_tensor_address(context, tensorNameUtf8.Pointer, out int cleared);
        NativeStatus.ThrowIfFailed(status);
        return cleared != 0;
    }

    public static void ClearExecutionContextDeviceMemory(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(ClearExecutionContextDeviceMemory));
        NativeStatus.ThrowIfFailed(NativeMethodsTensorRt.jyppx_trt11_execution_context_clear_device_memory(context));
    }

    public static bool ClearExecutionContextInputConsumedEvent(TensorRtApiLine line, SafeTensorRtObjectHandle context)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(ClearExecutionContextInputConsumedEvent));
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt11_execution_context_clear_input_consumed_event(context, out int cleared);
        NativeStatus.ThrowIfFailed(status);
        return cleared != 0;
    }

    public static void SetExecutionContextAuxStreams(TensorRtApiLine line, SafeTensorRtObjectHandle context, IReadOnlyList<SafeCudaStreamHandle> streams)
    {
        EnsureTensorRt11DeploymentApi(line, nameof(SetExecutionContextAuxStreams));
        if (streams == null)
        {
            throw new ArgumentNullException(nameof(streams));
        }

        if (streams.Count == 0)
        {
            NativeStatus.ThrowIfFailed(NativeMethodsTensorRt.jyppx_trt11_execution_context_set_aux_streams(context, IntPtr.Zero, 0));
            return;
        }

        IntPtr[] streamPointers = new IntPtr[streams.Count];
        for (int i = 0; i < streams.Count; i++)
        {
            if (streams[i] == null || streams[i].IsInvalid)
            {
                throw new ArgumentException("Auxiliary streams must contain valid CUDA stream handles.", nameof(streams));
            }

            streamPointers[i] = streams[i].DangerousGetHandle();
        }

        GCHandle pinned = GCHandle.Alloc(streamPointers, GCHandleType.Pinned);
        try
        {
            NativeStatus.ThrowIfFailed(NativeMethodsTensorRt.jyppx_trt11_execution_context_set_aux_streams(context, pinned.AddrOfPinnedObject(), streamPointers.Length));
        }
        finally
        {
            pinned.Free();
        }
    }
}

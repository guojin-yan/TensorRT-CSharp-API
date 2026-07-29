using System;
using System.Runtime.InteropServices;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle DeserializeHostMemory(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle runtime,
        SafeTensorRtObjectHandle hostMemory)
    {
        TensorRtLineBindings bindings = GetBindings(line);
        BridgeStatusCode status = bindings.DeserializeHostMemory(runtime, hostMemory, out SafeTensorRtObjectHandle engine);
        NativeStatus.ThrowIfFailed(status);
        return engine;
    }

    public static SafeTensorRtObjectHandle DeserializeEngineData(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle runtime,
        byte[] engineData)
    {
        if (engineData == null)
        {
            throw new ArgumentNullException(nameof(engineData));
        }

        if (engineData.Length == 0)
        {
            throw new ArgumentException("Serialized engine data must not be empty.", nameof(engineData));
        }

        GCHandle pinned = GCHandle.Alloc(engineData, GCHandleType.Pinned);
        try
        {
            SafeTensorRtObjectHandle engine;
            BridgeStatusCode status;
            switch (line)
            {
                case TensorRtApiLine.TensorRt8:
                    status = NativeMethodsTensorRt.jyppx_trt8_runtime_deserialize_engine(runtime, pinned.AddrOfPinnedObject(), (UIntPtr)engineData.Length, out engine);
                    break;
                case TensorRtApiLine.TensorRt10:
                    status = NativeMethodsTensorRt.jyppx_trt10_runtime_deserialize_engine(runtime, pinned.AddrOfPinnedObject(), (UIntPtr)engineData.Length, out engine);
                    break;
                case TensorRtApiLine.TensorRt11:
                    status = NativeMethodsTensorRt.jyppx_trt11_runtime_deserialize_engine(runtime, pinned.AddrOfPinnedObject(), (UIntPtr)engineData.Length, out engine);
                    break;
                default:
                    throw new BridgeProbeException(BridgeStatusCode.InvalidArgument, BridgeErrorCategory.Common, "Unsupported TensorRT API line.");
            }

            NativeStatus.ThrowIfFailed(status);
            return engine;
        }
        finally
        {
            pinned.Free();
        }
    }

}

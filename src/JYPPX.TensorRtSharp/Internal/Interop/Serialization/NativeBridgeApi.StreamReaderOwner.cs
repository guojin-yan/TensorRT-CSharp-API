using System;
using System.Runtime.InteropServices;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle CreateStreamReaderV2Owner(TensorRtApiLine line, byte[] data)
    {
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }
        if (data.Length == 0)
        {
            throw new ArgumentException("Stream reader data must not be empty.", nameof(data));
        }

        GCHandle pinned = GCHandle.Alloc(data, GCHandleType.Pinned);
        try
        {
            BridgeStatusCode status;
            SafeTensorRtObjectHandle owner;
            switch (line)
            {
                case TensorRtApiLine.TensorRt10:
                    status = NativeMethodsTensorRt.jyppx_trt10_stream_reader_v2_owner_create(
                        pinned.AddrOfPinnedObject(), (UIntPtr)data.Length, out owner);
                    break;
                case TensorRtApiLine.TensorRt11:
                    status = NativeMethodsTensorRt.jyppx_trt11_stream_reader_v2_owner_create(
                        pinned.AddrOfPinnedObject(), (UIntPtr)data.Length, out owner);
                    break;
                default:
                    throw new NotSupportedException("IStreamReaderV2 requires TensorRT 10 or TensorRT 11.");
            }

            NativeStatus.ThrowIfFailed(status);
            return owner;
        }
        finally
        {
            pinned.Free();
        }
    }

    public static NativeTensorRtStreamReaderOwnerInfo GetStreamReaderV2OwnerInfo(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle owner)
    {
        BridgeStatusCode status;
        NativeTensorRtStreamReaderOwnerInfo info;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_stream_reader_v2_owner_get_info(owner, out info);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_stream_reader_v2_owner_get_info(owner, out info);
                break;
            default:
                throw new NotSupportedException("IStreamReaderV2 requires TensorRT 10 or TensorRT 11.");
        }

        NativeStatus.ThrowIfFailed(status);
        return info;
    }

    public static SafeTensorRtObjectHandle DeserializeEngineFromStreamReaderV2(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle runtime,
        SafeTensorRtObjectHandle owner)
    {
        BridgeStatusCode status;
        SafeTensorRtObjectHandle engine;
        switch (line)
        {
            case TensorRtApiLine.TensorRt10:
                status = NativeMethodsTensorRt.jyppx_trt10_runtime_deserialize_stream_reader_v2(runtime, owner, out engine);
                break;
            case TensorRtApiLine.TensorRt11:
                status = NativeMethodsTensorRt.jyppx_trt11_runtime_deserialize_stream_reader_v2(runtime, owner, out engine);
                break;
            default:
                throw new NotSupportedException("IStreamReaderV2 requires TensorRT 10 or TensorRT 11.");
        }

        NativeStatus.ThrowIfFailed(status);
        return engine;
    }
}

using System;
using System.IO;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static TensorRtLegacyUffRequiredVersionSnapshot GetLegacyUffRequiredVersion(TensorRtApiLine line)
    {
        EnsureLegacyParserLine(line);
        BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt8_legacy_uff_get_required_version(
            out int major,
            out int minor,
            out int patch);
        NativeStatus.ThrowIfFailed(status);
        return new TensorRtLegacyUffRequiredVersionSnapshot(line, major, minor, patch);
    }

    public static TensorRtCaffeBinaryProtoSnapshot ReadLegacyCaffeBinaryProto(
        TensorRtApiLine line,
        string filePath)
    {
        EnsureLegacyParserLine(line);
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("Caffe binaryproto path must not be null or empty.", nameof(filePath));
        }

        string fullPath = Path.GetFullPath(filePath);
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException("Caffe binaryproto file was not found.", fullPath);
        }

        using Utf8Interop.Utf8StringScope pathUtf8 = Utf8Interop.ToNativeString(fullPath);
        for (int attempt = 0; attempt < 2; attempt++)
        {
            BridgeStatusCode status = NativeMethodsTensorRt.jyppx_trt8_legacy_caffe_binary_proto_copy(
                pathUtf8.Pointer,
                Array.Empty<byte>(),
                UIntPtr.Zero,
                out UIntPtr requiredSize,
                out NativeTensorRtDims queriedDimensions,
                out int queriedDataType);
            NativeStatus.ThrowIfFailed(status);

            ulong required = requiredSize.ToUInt64();
            if (required > int.MaxValue)
            {
                throw new BridgeProbeException(
                    BridgeStatusCode.BufferTooSmall,
                    BridgeErrorCategory.TensorRt,
                    "Caffe binaryproto data is too large for a managed byte array.");
            }

            byte[] data = new byte[checked((int)required)];
            status = NativeMethodsTensorRt.jyppx_trt8_legacy_caffe_binary_proto_copy(
                pathUtf8.Pointer,
                data,
                new UIntPtr((uint)data.Length),
                out UIntPtr copiedSize,
                out NativeTensorRtDims copiedDimensions,
                out int copiedDataType);
            if (status == BridgeStatusCode.BufferTooSmall && attempt == 0)
            {
                continue;
            }

            NativeStatus.ThrowIfFailed(status);
            ulong copied = copiedSize.ToUInt64();
            if (copied > (ulong)data.Length)
            {
                throw new BridgeProbeException(
                    BridgeStatusCode.BufferTooSmall,
                    BridgeErrorCategory.TensorRt,
                    "Caffe binaryproto changed while it was being copied.");
            }
            if (copied < (ulong)data.Length)
            {
                Array.Resize(ref data, checked((int)copied));
            }

            TensorRtDims queriedShape = TensorRtDims.FromNative(queriedDimensions);
            TensorRtDims copiedShape = TensorRtDims.FromNative(copiedDimensions);
            if (queriedDataType != copiedDataType || !HaveSameDimensions(queriedShape, copiedShape))
            {
                throw new IOException("Caffe binaryproto metadata changed while it was being copied.");
            }

            return new TensorRtCaffeBinaryProtoSnapshot(
                line,
                Path.GetFileName(fullPath),
                copiedShape,
                (TensorRtDataType)copiedDataType,
                data);
        }

        throw new BridgeProbeException(
            BridgeStatusCode.BufferTooSmall,
            BridgeErrorCategory.TensorRt,
            "Caffe binaryproto changed repeatedly while it was being copied.");
    }

    private static bool HaveSameDimensions(TensorRtDims left, TensorRtDims right)
    {
        if (left.Rank != right.Rank)
        {
            return false;
        }

        for (int index = 0; index < left.Rank; index++)
        {
            if (left.Values[index] != right.Values[index])
            {
                return false;
            }
        }

        return true;
    }

    private static void EnsureLegacyParserLine(TensorRtApiLine line)
    {
        if (line != TensorRtApiLine.TensorRt8)
        {
            throw new BridgeProbeException(
                BridgeStatusCode.NotSupported,
                BridgeErrorCategory.TensorRt,
                "Legacy UFF and Caffe parser diagnostics are available only for the TensorRT 8 adapter.");
        }
    }
}

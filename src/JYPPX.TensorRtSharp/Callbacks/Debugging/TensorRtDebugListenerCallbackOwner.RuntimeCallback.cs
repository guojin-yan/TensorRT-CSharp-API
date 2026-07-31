using System;
using System.Runtime.InteropServices;
using System.Text;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtDebugListenerCallbackOwner
{
    [ThreadStatic]
    private static int s_runtimeCallbackDepth;

    internal static bool IsExecutingRuntimeCallbackOnCurrentThread => s_runtimeCallbackDepth > 0;

    private static BridgeStatusCode InvokeManagedDebugListener(
        uint line,
        IntPtr tensorName,
        UIntPtr tensorNameLength,
        int dataType,
        int location,
        int shapeRank,
        long dim0,
        long dim1,
        long dim2,
        long dim3,
        long dim4,
        long dim5,
        long dim6,
        long dim7,
        IntPtr userState)
    {
        CallbackState? state = null;
        s_runtimeCallbackDepth++;
        try
        {
            if (userState == IntPtr.Zero)
            {
                return BridgeStatusCode.InvalidArgument;
            }

            state = GCHandle.FromIntPtr(userState).Target as CallbackState;
            if (state == null || state.Handler == null)
            {
                return BridgeStatusCode.InvalidState;
            }

            state.EnterCallback();
            state.RecordInvocation();

            TensorRtApiLine apiLine = line == (uint)TensorRtApiLine.TensorRt10
                ? TensorRtApiLine.TensorRt10
                : TensorRtApiLine.TensorRt11;
            TensorRtDataType copiedDataType = Enum.IsDefined(typeof(TensorRtDataType), dataType)
                ? (TensorRtDataType)dataType
                : TensorRtDataType.Unknown;
            TensorRtTensorLocation copiedLocation = Enum.IsDefined(typeof(TensorRtTensorLocation), location)
                ? (TensorRtTensorLocation)location
                : TensorRtTensorLocation.Device;
            long[] dimensions = CopyShape(shapeRank, dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7);
            string copiedName = DecodeUtf8(tensorName, tensorNameLength);
            string shapeSummary = FormatShape(
                dimensions.Length,
                GetDimension(dimensions, 0),
                GetDimension(dimensions, 1),
                GetDimension(dimensions, 2),
                GetDimension(dimensions, 3),
                GetDimension(dimensions, 4),
                GetDimension(dimensions, 5),
                GetDimension(dimensions, 6),
                GetDimension(dimensions, 7));

            state.RecordRequest(
                apiLine,
                copiedName,
                copiedDataType,
                copiedLocation,
                dimensions.Length,
                shapeSummary,
                isInput: false,
                isOutput: false,
                isShapeTensor: false,
                isExecutionTensor: true);

            TensorRtDebugTensorMetadataSnapshot metadata = new TensorRtDebugTensorMetadataSnapshot(
                copiedName,
                copiedName.Length,
                copiedDataType,
                copiedLocation,
                dimensions.Length,
                shapeSummary,
                isInput: false,
                isOutput: false,
                isShapeTensor: false,
                isExecutionTensor: true,
                metadataCopied: true);
            bool succeeded = state.Handler(metadata);
            if (!succeeded)
            {
                state.RecordReturnedFailure(
                    "Managed debug listener handler returned false.",
                    BridgeStatusCode.InvalidState);
                return BridgeStatusCode.InvalidState;
            }

            state.RecordStatus(
                BridgeStatusCode.Ok,
                "TensorRT IDebugListener::processDebugTensor copied metadata into the managed handler.");
            return BridgeStatusCode.Ok;
        }
        catch (Exception exception)
        {
            state?.RecordFailure(
                exception,
                "Managed debug listener handler threw " + exception.GetType().Name + ": " + exception.Message,
                BridgeStatusCode.InvalidState);
            return BridgeStatusCode.InvalidState;
        }
        finally
        {
            state?.ExitCallback();
            s_runtimeCallbackDepth--;
        }
    }

    private static long[] CopyShape(
        int shapeRank,
        long dim0,
        long dim1,
        long dim2,
        long dim3,
        long dim4,
        long dim5,
        long dim6,
        long dim7)
    {
        if (shapeRank < 0 || shapeRank > MaxShapeRank)
        {
            throw new InvalidOperationException("TensorRT debug listener reported an invalid shape rank.");
        }

        long[] allDimensions = { dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7 };
        long[] dimensions = new long[shapeRank];
        Array.Copy(allDimensions, dimensions, shapeRank);
        return dimensions;
    }

    private static string DecodeUtf8(IntPtr value, UIntPtr length)
    {
        if (value == IntPtr.Zero || length == UIntPtr.Zero)
        {
            return string.Empty;
        }

        ulong byteLength = length.ToUInt64();
        if (byteLength > 255UL)
        {
            throw new InvalidOperationException("TensorRT debug tensor name exceeded the copied native metadata limit.");
        }

        byte[] buffer = new byte[checked((int)byteLength)];
        Marshal.Copy(value, buffer, 0, buffer.Length);
        return Encoding.UTF8.GetString(buffer);
    }
}

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

using System;
using System.Runtime.InteropServices;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp.Internal.Interop;

[StructLayout(LayoutKind.Sequential)]
internal struct NativeBuildInfo
{
    public uint AbiVersion;
    public uint BridgeVersionMajor;
    public uint BridgeVersionMinor;
    public uint BridgeVersionPatch;
    public IntPtr BridgeName;
    public IntPtr BridgeBanner;
    public IntPtr CompilerId;
    public IntPtr CompilerVersion;
    public IntPtr SystemName;
    public IntPtr SystemProcessor;
    public IntPtr BuildConfiguration;
    public IntPtr CudaToolkitVersion;
    public IntPtr TensorRtVersion;
    public int HasCudaToolkit;
    public int HasTensorRt;
    public int CudaBindingsEnabled;
    public int TensorRtBindingsEnabled;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeRuntimeInfo
{
    public uint AbiVersion;
    public IntPtr BridgeName;
    public IntPtr BridgeBanner;
    public IntPtr LastErrorMessage;
    public BridgeErrorCategory LastErrorCategory;
    public int CudaToolkitAvailable;
    public int TensorRtAvailable;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCapabilityInfo
{
    public int SupportsTrt8Adapter;
    public int SupportsTrt10Adapter;
    public int SupportsTrt11Adapter;
    public int SupportsTrt8RuntimeCreation;
    public int SupportsTrt10RuntimeCreation;
    public int SupportsTrt11RuntimeCreation;
    public int SupportsTrt8BuilderCreation;
    public int SupportsTrt10BuilderCreation;
    public int SupportsTrt11BuilderCreation;
    public int SupportsLastErrorQuery;
    public int SupportsBuildInfoQuery;
    public int SupportsRuntimeInfoQuery;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeTensorRtAdapterInfo
{
    public uint Line;
    public int VendorDependencyAvailable;
    public int RuntimeCreationSupported;
    public int BuilderCreationSupported;
    public int NetworkCreationSupported;
    public int EngineDeserializationSupported;
    public IntPtr DetectedVersion;
    public IntPtr StatusMessage;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeTensorRtDims
{
    public const int MaxDimensionCount = 8;

    public int NbDims;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = MaxDimensionCount)]
    public int[] D;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeTensorRtDims64
{
    public const int MaxDimensionCount = 8;

    public int NbDims;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = MaxDimensionCount)]
    public long[] D;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeTensorRtTensorInfo
{
    public int Index;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
    public byte[] Name;
    public int DataType;
    public int IoMode;
    public NativeTensorRtDims Shape;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeTensorRtParserErrorInfo
{
    public int Index;
    public int Code;
    public int Line;
    public int Node;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 1024)]
    public byte[] Description;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
    public byte[] File;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
    public byte[] FunctionName;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
    public byte[] NodeName;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 128)]
    public byte[] NodeOperator;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeTensorRtRefitEntryInfo
{
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
    public byte[] LayerName;
    public int Role;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeTensorRtWeightsInfo
{
    public int DataType;
    public long Count;
    public int HasValues;
}

using System;
using System.Runtime.InteropServices;
using JYPPX.TensorRtSharp.Shared.Interop;

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

[StructLayout(LayoutKind.Sequential)]
internal struct NativeTensorRtErrorRecorderSnapshotInfo
{
    public int HasRecorder;
    public int ErrorCount;
    public int HasOverflowed;
    public int InterfaceInfoAvailable;
    public int InterfaceInfoMajor;
    public int InterfaceInfoMinor;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 128)]
    public byte[] InterfaceInfoKind;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeTensorRtErrorRecordInfo
{
    public int Index;
    public int Code;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 1024)]
    public byte[] Description;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeTensorRtAllocatorOwnerDiagnosticInfo
{
    public uint Line;
    public ulong InvocationCount;
    public ulong FailureCount;
    public int LastStatus;
    public int IsAttached;
    public ulong LastSize;
    public ulong LastAlignment;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 1024)]
    public byte[] LastDiagnostic;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeTensorRtAllocatorOwnerStateInfo
{
    public uint Line;
    public ulong OwnerId;
    public ulong StateTransitionCount;
    public ulong LedgerAllocationCount;
    public ulong LedgerReleaseCount;
    public ulong LedgerFailureCount;
    public ulong LastAllocationId;
    public ulong LastReleaseAllocationId;
    public ulong LastSize;
    public ulong LastAlignment;
    public ulong LastStreamValue;
    public int AttachState;
    public int LastStatus;
    public int IsAttached;
    public int HasLiveAllocation;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 64)]
    public byte[] LastOperation;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 1024)]
    public byte[] LastDiagnostic;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeTensorRtDebugListenerOwnerInfo
{
    public uint Line;
    public ulong OwnerId;
    public ulong InvocationCount;
    public ulong FailureCount;
    public ulong InFlightCallbackCount;
    public ulong MaxInFlightCallbackCount;
    public ulong AttachCount;
    public ulong DetachCount;
    public int LastStatus;
    public int IsAttached;
    public int LastCallbackSucceeded;
    public int LastDataType;
    public int LastLocation;
    public int LastShapeRank;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
    public long[] LastShape;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
    public byte[] LastTensorName;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 1024)]
    public byte[] LastDiagnostic;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeTensorRtOutputAllocatorOwnerInfo
{
    public uint Line;
    public ulong OwnerId;
    public ulong InvocationCount;
    public ulong NotifyShapeCount;
    public ulong ReallocateOutputCount;
    public ulong FailureCount;
    public ulong InFlightCallbackCount;
    public ulong MaxInFlightCallbackCount;
    public ulong AttachCount;
    public ulong DetachCount;
    public ulong AllocationCount;
    public ulong ReuseCount;
    public ulong ReleaseCount;
    public ulong LiveAllocationCount;
    public ulong LiveAllocationBytes;
    public ulong PeakLiveAllocationBytes;
    public ulong LastRequestedSize;
    public ulong LastAlignment;
    public int LastStatus;
    public int IsAttached;
    public int LastCallbackSucceeded;
    public int LastAllocationSucceeded;
    public int LastHadCurrentMemory;
    public int LastHadStream;
    public int LastCallbackKind;
    public int LastShapeRank;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
    public long[] LastShape;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
    public byte[] LastTensorName;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 1024)]
    public byte[] LastDiagnostic;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeTensorRtRuntimeCreateDiagnosticInfo
{
    public uint Line;
    public int Attempted;
    public int LoggerHandlePresent;
    public int LoggerPayloadPresent;
    public int CreateInferRuntimeReturnedNonNull;
    public int CreateInferRuntimeReturnedNull;
    public int LastStatus;
    public int TensorRtAvailable;
    public int ExpectedMajor;
    public int BridgeBuiltMajor;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 64)]
    public byte[] DetectedVersion;
    public int LoggerCallbackAvailable;
    public uint LoggerMessageCount;
    public int LastLoggerSeverity;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 1024)]
    public byte[] LastLoggerMessage;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 128)]
    public byte[] CreateRuntimePhase;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 1024)]
    public byte[] NativeDetail;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 1024)]
    public byte[] Diagnostic;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeTensorRtExecutionContextCallbackStateInfo
{
    public uint Line;
    public int HasOutputAllocator;
    public int HasTemporaryStorageAllocator;
    public int HasDebugListener;
    public int OutputAllocatorInterfaceInfoAvailable;
    public int TemporaryStorageAllocatorInterfaceInfoAvailable;
    public int DebugListenerInterfaceInfoAvailable;
    public int OutputAllocatorClearSupported;
    public int TemporaryStorageAllocatorClearSupported;
    public int DebugListenerClearSupported;
    public int OutputAllocatorCleared;
    public int TemporaryStorageAllocatorCleared;
    public int DebugListenerCleared;
    public int OutputAllocatorInterfaceMajor;
    public int OutputAllocatorInterfaceMinor;
    public int TemporaryStorageAllocatorInterfaceMajor;
    public int TemporaryStorageAllocatorInterfaceMinor;
    public int DebugListenerInterfaceMajor;
    public int DebugListenerInterfaceMinor;
    public int LastStatus;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 128)]
    public byte[] OutputAllocatorInterfaceKind;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 128)]
    public byte[] TemporaryStorageAllocatorInterfaceKind;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 128)]
    public byte[] DebugListenerInterfaceKind;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 64)]
    public byte[] LastOperation;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 1024)]
    public byte[] LastDiagnostic;
}

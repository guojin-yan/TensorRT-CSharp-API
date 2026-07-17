using System;
using System.Runtime.InteropServices;
using JYPPX.Shared.Interop;

namespace JYPPX.CudaSharp.Internal.Interop;

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaKernelLibraryInventory
{
    public uint ReportedKernelCount;
    public uint EnumeratedKernelCount;
    public uint NullKernelCount;
    public int IsComplete;
}

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
    public int SupportsTrt8RuntimeCreation;
    public int SupportsTrt10RuntimeCreation;
    public int SupportsTrt8BuilderCreation;
    public int SupportsTrt10BuilderCreation;
    public int SupportsLastErrorQuery;
    public int SupportsBuildInfoQuery;
    public int SupportsRuntimeInfoQuery;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaRuntimeInfo
{
    public int VendorDependencyAvailable;
    public int SupportsStreams;
    public int SupportsEvents;
    public int SupportsMemory;
    public int RuntimeVersion;
    public int DriverVersion;
    public int DeviceCount;
    public IntPtr StatusMessage;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaDeviceInfo
{
    public int Ordinal;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
    public byte[] Name;
    public int Major;
    public int Minor;
    public int MultiProcessorCount;
    public int WarpSize;
    public int MaxThreadsPerBlock;
    public int CanMapHostMemory;
    public int Integrated;
    public ulong TotalGlobalMemory;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaDeviceSelectionRequirements
{
    public int Major;
    public int Minor;
    public int MultiProcessorCount;
    public int WarpSize;
    public int MaxThreadsPerBlock;
    public int CanMapHostMemory;
    public int Integrated;
    public ulong TotalGlobalMemory;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaMemoryInfo
{
    public ulong FreeBytes;
    public ulong TotalBytes;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaPointerAttributes
{
    public int MemoryType;
    public int Device;
    public ulong DevicePointer;
    public ulong HostPointer;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaMemRangeAttributeValue
{
    public int Attribute;
    public int Value;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaManagedMemoryBatchRange
{
    public IntPtr Memory;
    public UIntPtr Offset;
    public UIntPtr Size;
    public int DestinationDevice;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaPitchedMemoryInfo
{
    public ulong PitchBytes;
    public ulong WidthBytes;
    public ulong Height;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaChannelFormatDesc
{
    public int X;
    public int Y;
    public int Z;
    public int W;
    public int FormatKind;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaArrayExtent
{
    public ulong Width;
    public ulong Height;
    public ulong Depth;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaArrayInfo
{
    public NativeCudaChannelFormatDesc Channel;
    public NativeCudaArrayExtent Extent;
    public uint Flags;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaArrayMemoryRequirements
{
    public ulong Size;
    public ulong Alignment;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaArraySparseProperties
{
    public uint TileWidth;
    public uint TileHeight;
    public uint TileDepth;
    public uint MipTailFirstLevel;
    public ulong MipTailSize;
    public uint Flags;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaMemLocation
{
    public int Type;
    public int Id;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaMemcpyAttributes
{
    public int SrcAccessOrder;
    public NativeCudaMemLocation SrcLocationHint;
    public NativeCudaMemLocation DstLocationHint;
    public uint Flags;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaPitchedPtr
{
    public IntPtr Pointer;
    public UIntPtr Pitch;
    public UIntPtr XSize;
    public UIntPtr YSize;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaPos
{
    public UIntPtr X;
    public UIntPtr Y;
    public UIntPtr Z;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaMemcpy3DPeerParams
{
    public IntPtr SrcArray;
    public NativeCudaPos SrcPos;
    public NativeCudaPitchedPtr SrcPtr;
    public int SrcDevice;
    public IntPtr DstArray;
    public NativeCudaPos DstPos;
    public NativeCudaPitchedPtr DstPtr;
    public int DstDevice;
    public NativeCudaArrayExtent Extent;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaOffset3D
{
    public UIntPtr X;
    public UIntPtr Y;
    public UIntPtr Z;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaMemcpy3DOperandPointer
{
    public IntPtr Pointer;
    public UIntPtr RowLength;
    public UIntPtr LayerHeight;
    public NativeCudaMemLocation LocationHint;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaMemcpy3DOperandArray
{
    public IntPtr Array;
    public NativeCudaOffset3D Offset;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaMemcpy3DOperand
{
    public int Type;
    public NativeCudaMemcpy3DOperandPointer Pointer;
    public NativeCudaMemcpy3DOperandArray Array;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaMemcpy3DBatchOp
{
    public NativeCudaMemcpy3DOperand Source;
    public NativeCudaMemcpy3DOperand Destination;
    public NativeCudaArrayExtent Extent;
    public int SourceAccessOrder;
    public uint Flags;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaMemPoolPtrExportData
{
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 64)]
    public byte[] Reserved;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaGraphEdgeData
{
    public byte FromPort;
    public byte ToPort;
    public byte Type;
    public byte Reserved0;
    public byte Reserved1;
    public byte Reserved2;
    public byte Reserved3;
    public byte Reserved4;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaGraphMemsetNodeParams
{
    public ulong DestinationAddress;
    public ulong Pitch;
    public uint Value;
    public uint ElementSize;
    public ulong Width;
    public ulong Height;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaGraphMemcpyNodeParams
{
    public ulong SourceAddress;
    public ulong DestinationAddress;
    public ulong SourcePitch;
    public ulong DestinationPitch;
    public ulong SourceXSize;
    public ulong SourceYSize;
    public ulong DestinationXSize;
    public ulong DestinationYSize;
    public ulong SourcePositionX;
    public ulong SourcePositionY;
    public ulong SourcePositionZ;
    public ulong DestinationPositionX;
    public ulong DestinationPositionY;
    public ulong DestinationPositionZ;
    public ulong Width;
    public ulong Height;
    public ulong Depth;
    public int Kind;
    public uint SourceIsArray;
    public uint DestinationIsArray;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaGraphKernelNodeAttributeValue
{
    public int Attribute;
    public int IntValue;
    public uint X;
    public uint Y;
    public uint Z;
    public uint Reserved0;
    public uint Reserved1;
    public uint Reserved2;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaGraphKernelNodeParamsSnapshot
{
    public uint GridX;
    public uint GridY;
    public uint GridZ;
    public uint BlockX;
    public uint BlockY;
    public uint BlockZ;
    public uint SharedMemoryBytes;
    public int HasFunction;
    public int HasKernelParams;
    public int HasExtra;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaGraphHostNodeParamsSnapshot
{
    public int HasCallback;
    public int HasUserData;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaGraphMemAllocNodeParamsSnapshot
{
    public ulong ByteCount;
    public ulong AccessDescriptorCount;
    public int AllocationType;
    public ulong HandleTypes;
    public int LocationType;
    public int LocationId;
    public int HasAccessDescriptors;
    public int HasDevicePointer;
    public int HasSecurityAttributes;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaGraphMemFreeNodeParamsSnapshot
{
    public int HasDevicePointer;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaGraphExternalSemaphoreNodeParamsSnapshot
{
    public uint SemaphoreCount;
    public int HasSemaphoreArray;
    public int HasParameterArray;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaResourceDescriptorSnapshot
{
    public int ResourceType;
    public int HasArray;
    public int HasMipmappedArray;
    public int HasDevicePointer;
    public ulong SizeInBytes;
    public ulong Width;
    public ulong Height;
    public ulong PitchInBytes;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaTextureDescriptor
{
    public int AddressModeX;
    public int AddressModeY;
    public int AddressModeZ;
    public int FilterMode;
    public int ReadMode;
    public int Srgb;
    public float BorderColorR;
    public float BorderColorG;
    public float BorderColorB;
    public float BorderColorA;
    public int NormalizedCoordinates;
    public uint MaxAnisotropy;
    public int MipmapFilterMode;
    public float MipmapLevelBias;
    public float MinMipmapLevelClamp;
    public float MaxMipmapLevelClamp;
    public int DisableTrilinearOptimization;
    public int SeamlessCubemap;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaTextureResourceViewSnapshot
{
    public int IsSpecified;
    public int Format;
    public ulong Width;
    public ulong Height;
    public ulong Depth;
    public uint FirstMipmapLevel;
    public uint LastMipmapLevel;
    public uint FirstLayer;
    public uint LastLayer;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaDim3
{
    public uint X;
    public uint Y;
    public uint Z;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaFuncAttributes
{
    public UIntPtr SharedSizeBytes;
    public UIntPtr ConstSizeBytes;
    public UIntPtr LocalSizeBytes;
    public int MaxThreadsPerBlock;
    public int NumRegisters;
    public int PtxVersion;
    public int BinaryVersion;
    public int CacheModeCa;
    public int MaxDynamicSharedSizeBytes;
    public int PreferredSharedMemoryCarveout;
    public int ClusterDimMustBeSet;
    public int RequiredClusterWidth;
    public int RequiredClusterHeight;
    public int RequiredClusterDepth;
    public int ClusterSchedulingPolicyPreference;
    public int NonPortableClusterSizeAllowed;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeCudaLaunchConfig
{
    public NativeCudaDim3 GridDim;
    public NativeCudaDim3 BlockDim;
    public UIntPtr DynamicSharedMemoryBytes;
    public IntPtr Stream;
    public IntPtr Attributes;
    public uint AttributeCount;
}

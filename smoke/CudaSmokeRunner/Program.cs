using System;
using System.Linq;
using JYPPX.CudaSharp;

CudaEnvironmentSnapshot snapshot = CudaEnvironmentProbe.GetCurrent();
Console.WriteLine($"Bridge={snapshot.BuildInfo.BridgeName} CUDA Toolkit={snapshot.BuildInfo.CudaToolkitVersion} DeviceCount={snapshot.CudaRuntimeInfo.DeviceCount}");

if (!snapshot.CudaRuntimeInfo.VendorDependencyAvailable)
{
    Console.WriteLine(snapshot.CudaRuntimeInfo.StatusMessage);
    return;
}

foreach (CudaDeviceInfo device in snapshot.Devices)
{
    Console.WriteLine($"Device[{device.Ordinal}] {device.Name} CC={device.Major}.{device.Minor} VRAM={device.TotalGlobalMemory}");
}

Console.WriteLine($"CurrentDevice={CudaDevice.Current}");
Console.WriteLine($"CudaVersions Runtime={CudaDevice.RuntimeVersion} Driver={CudaDevice.DriverVersion}");
try
{
    string pciBusId = CudaDevice.GetPciBusId(CudaDevice.Current);
    int deviceFromPci = CudaDevice.GetByPciBusId(pciBusId);
    CudaFunctionCachePreference cacheConfig = CudaDevice.CacheConfig;
    CudaDevice.CacheConfig = cacheConfig;
    CudaSharedMemoryConfig sharedMemoryConfig = CudaDevice.SharedMemoryConfig;
    CudaDevice.SharedMemoryConfig = sharedMemoryConfig;
    CudaDeviceRuntimeFlags runtimeFlags = CudaDevice.RuntimeFlags;
    Console.WriteLine($"DeviceDeploymentConfig PciBusId={pciBusId} PciLookup={deviceFromPci} Cache={cacheConfig} SharedMemory={sharedMemoryConfig} RuntimeFlags={runtimeFlags}");
}
catch (CudaException exception)
{
    Console.WriteLine($"DeviceDeploymentConfig=Skipped Reason={exception.Message}");
}
CudaDeviceProperties properties = CudaDevice.CurrentProperties;
Console.WriteLine(
    $"DeviceProperties CC={properties.ComputeCapabilityLabel} " +
    $"Block=[{string.Join(",", properties.MaxBlockDimensions)}] Grid=[{string.Join(",", properties.MaxGridDimensions)}] " +
    $"AsyncEngines={properties.AsyncEngineCount?.ToString() ?? "n/a"} Managed={properties.ManagedMemory?.ToString() ?? "n/a"} " +
    $"UnifiedAddressing={properties.UnifiedAddressing?.ToString() ?? "n/a"} L2={properties.L2CacheSizeBytes?.ToString() ?? "n/a"}");
CudaStreamPriorityRange priorityRange = CudaStream.GetPriorityRange();
Console.WriteLine($"StreamPriorityRange {priorityRange}");
CudaMemoryInfo memoryInfo = CudaDevice.GetMemoryInfo();
Console.WriteLine($"MemoryInfo Free={memoryInfo.FreeBytes} Total={memoryInfo.TotalBytes}");
bool tryMemoryInfoOk = CudaDevice.TryGetMemoryInfo(out CudaMemoryInfo? tryMemoryInfo, out string tryMemoryInfoDiagnostic);
Console.WriteLine($"TryMemoryInfo Ok={tryMemoryInfoOk} Free={tryMemoryInfo?.FreeBytes.ToString() ?? "n/a"} Diagnostic={tryMemoryInfoDiagnostic}");
CudaMemoryPressureSnapshot memoryPressure = CudaDevice.GetMemoryPressureSnapshot();
Console.WriteLine($"MemoryPressure {memoryPressure} UsedRatio={memoryPressure.UsedRatio:0.000} LowFree10={memoryPressure.IsBelowFreeRatio(0.10)}");
try
{
    CudaMemoryPool defaultPool = CudaDevice.GetDefaultMemoryPool(CudaDevice.Current);
    CudaMemoryPool currentPool = CudaDevice.GetCurrentMemoryPool(CudaDevice.Current);
    long releaseThreshold = defaultPool.GetAttribute(CudaMemoryPoolAttribute.ReleaseThreshold);
    long usedMemoryCurrent = defaultPool.GetAttribute(CudaMemoryPoolAttribute.UsedMemoryCurrent);
    currentPool.TrimTo(0);
    CudaDevice.SetCurrentMemoryPool(defaultPool);
    defaultPool.ResetReservedMemoryHigh();
    defaultPool.ResetUsedMemoryHigh();
    Console.WriteLine($"DefaultMemoryPool Device={defaultPool.DeviceOrdinal} ReleaseThreshold={releaseThreshold} UsedCurrent={usedMemoryCurrent} ReservedHigh={defaultPool.ReservedMemoryHighBytes} UsedHigh={defaultPool.UsedMemoryHighBytes} CurrentPoolDevice={currentPool.DeviceOrdinal} Trim=True ResetHigh=True");
    CudaMemoryPoolAccessFlags selfAccess = defaultPool.GetAccess(CudaDevice.Current);
    string peerAccessState = "Skipped";
    if (snapshot.CudaRuntimeInfo.DeviceCount > 1)
    {
        int peerDevice = CudaDevice.Current == 0 ? 1 : 0;
        if (CudaDevice.CanAccessPeer(CudaDevice.Current, peerDevice))
        {
            defaultPool.SetAccess(peerDevice, CudaMemoryPoolAccessFlags.ReadWrite);
            peerAccessState = $"{peerDevice}:{defaultPool.GetAccess(peerDevice)}";
        }
        else
        {
            peerAccessState = $"{peerDevice}:PeerAccessFalse";
        }
    }

    Console.WriteLine($"MemoryPoolAccess Self={selfAccess} Peer={peerAccessState}");

    using CudaOwnedMemoryPool ownedPool = CudaMemoryPool.Create(CudaDevice.Current);
    CudaMemoryPool ownedPoolView = ownedPool.Pool;
    ownedPoolView.ReleaseThresholdBytes = releaseThreshold;
    using CudaStream poolStream = new CudaStream(CudaStreamCreationFlags.NonBlocking);
    using CudaMemory pooledMemory = ownedPool.AllocateAsync(64, poolStream);
    using CudaPinnedMemory pooledDestination = new CudaPinnedMemory(64);
    pooledMemory.FillAsync(0x3D, 64, poolStream);
    pooledMemory.CopyToAsync(pooledDestination, 64, poolStream);
    poolStream.Synchronize();
    bool pooledAllocationOk = pooledDestination.ToArray(64).All(static value => value == 0x3D);
    ownedPoolView.ResetReservedMemoryHigh();
    ownedPoolView.ResetUsedMemoryHigh();
    ownedPool.TrimTo(0);
    Console.WriteLine($"OwnedMemoryPool Device={ownedPool.DeviceOrdinal} PoolAsyncAllocation={pooledAllocationOk} ReservedCurrent={ownedPoolView.ReservedMemoryCurrentBytes} UsedCurrent={ownedPoolView.UsedMemoryCurrentBytes} ResetHigh=True Trim=True");
}
catch (CudaException exception)
{
    Console.WriteLine($"MemoryPool=Skipped Reason={exception.Message}");
}

CudaDevice.Synchronize();
Console.WriteLine("DeviceSynchronize=True");
Console.WriteLine(
    $"DeviceAttributes ManagedMemory={CudaDevice.GetBooleanAttribute(CudaDevice.Current, CudaDeviceAttribute.ManagedMemory)} " +
    $"ConcurrentManagedAccess={CudaDevice.GetBooleanAttribute(CudaDevice.Current, CudaDeviceAttribute.ConcurrentManagedAccess)} " +
    $"UnifiedAddressing={CudaDevice.GetBooleanAttribute(CudaDevice.Current, CudaDeviceAttribute.UnifiedAddressing)} " +
    $"HostRegisterSupported={CudaDevice.GetBooleanAttribute(CudaDevice.Current, CudaDeviceAttribute.HostRegisterSupported)} " +
    $"AsyncEngines={CudaDevice.GetAttribute(CudaDevice.Current, CudaDeviceAttribute.AsyncEngineCount)}");
try
{
    ulong stackLimit = CudaDevice.GetLimit(CudaDeviceLimit.StackSize);
    CudaDevice.SetLimit(CudaDeviceLimit.StackSize, stackLimit);
    ulong stackLimitAfterSet = CudaDevice.GetLimit(CudaDeviceLimit.StackSize);
    Console.WriteLine($"DeviceLimit StackSize={stackLimit} AfterSet={stackLimitAfterSet}");
}
catch (CudaException exception)
{
    Console.WriteLine($"DeviceLimit=Skipped Reason={exception.Message}");
}
int initialCudaError = CudaDevice.PeekAtLastErrorCode();
Console.WriteLine($"CudaPeekLastError={initialCudaError}:{CudaDevice.GetErrorName(initialCudaError)}:{CudaDevice.GetErrorString(initialCudaError)}");

using CudaDeviceScope deviceScope = CudaDevice.Use(0);
Console.WriteLine($"ScopedDevice={CudaDevice.Current} PreviousDevice={deviceScope.PreviousDevice}");

using CudaStream stream = new CudaStream(CudaStreamCreationFlags.NonBlocking);
using CudaStream priorityStream = new CudaStream(CudaStreamCreationFlags.NonBlocking, priorityRange.GreatestPriority);
using CudaEvent cudaEvent = new CudaEvent();
using CudaMemory memory = new CudaMemory(32);
Console.WriteLine($"StreamFlags={stream.Flags} EventFlags={cudaEvent.Flags}");
Console.WriteLine($"PriorityStream Flags={priorityStream.Flags} Priority={priorityStream.Priority}");
try
{
    ulong streamId = stream.Id;
    Console.WriteLine($"StreamId={streamId}");
}
catch (CudaException exception)
{
    Console.WriteLine($"StreamId=Skipped Reason={exception.Message}");
}
try
{
    int streamDevice = stream.DeviceOrdinal;
    Console.WriteLine($"StreamDevice={streamDevice}");
}
catch (CudaException exception)
{
    Console.WriteLine($"StreamDevice=Skipped Reason={exception.Message}");
}
try
{
    priorityStream.CopyAttributesFrom(stream);
    CudaStreamCaptureMode previousCaptureMode = CudaStream.ExchangeThreadCaptureMode(CudaStreamCaptureMode.Relaxed);
    _ = CudaStream.ExchangeThreadCaptureMode(previousCaptureMode);
    Console.WriteLine($"StreamDeploymentMetadata AttributeCopy=True PreviousThreadCaptureMode={previousCaptureMode}");
}
catch (CudaException exception)
{
    Console.WriteLine($"StreamDeploymentMetadata=Skipped Reason={exception.Message}");
}
CudaStreamCaptureInfo initialCaptureInfo = stream.GetCaptureInfo();
Console.WriteLine($"StreamCapture InitialStatus={stream.CaptureStatus} Info={initialCaptureInfo.Status}:{initialCaptureInfo.CaptureId}");
bool tryCaptureInfoOk = stream.TryGetCaptureInfo(out CudaStreamCaptureInfo tryCaptureInfo, out string tryCaptureInfoDiagnostic);
Console.WriteLine($"TryStreamCaptureInfo Ok={tryCaptureInfoOk} Info={tryCaptureInfo} Diagnostic={tryCaptureInfoDiagnostic}");
CudaPointerAttributes pointerAttributes = memory.GetPointerAttributes();
Console.WriteLine($"PointerAttributes Type={pointerAttributes.MemoryType} Device={pointerAttributes.DeviceOrdinal} DevicePtr=0x{pointerAttributes.DevicePointerAddress:X}");

byte[] source = Enumerable.Range(1, 32).Select(static i => (byte)i).ToArray();
memory.CopyFrom(source);
byte[] roundTrip = new byte[source.Length];
memory.CopyTo(roundTrip);

bool roundTripOk = source.SequenceEqual(roundTrip);
Console.WriteLine($"RoundTrip={roundTripOk}");

float[] floatSource = { 1.25f, -2.5f, 3.75f, 8.0f };
using CudaMemory floatMemory = new CudaMemory(floatSource.Length * sizeof(float));
floatMemory.CopyFrom(floatSource);
float[] floatRoundTrip = floatMemory.ToSingleArray(floatSource.Length);
bool floatRoundTripOk = floatSource.SequenceEqual(floatRoundTrip);
Console.WriteLine($"FloatRoundTrip={floatRoundTripOk}");

if (CudaDevice.GetBooleanAttribute(CudaDevice.Current, CudaDeviceAttribute.ManagedMemory))
{
    using CudaManagedMemory managedMemory = new CudaManagedMemory(floatSource.Length * sizeof(float));
    managedMemory.CopyFrom(floatSource);
    try
    {
        managedMemory.Advise(managedMemory.SizeInBytes, CudaMemoryAdvice.SetPreferredLocation, CudaDevice.Current);
        managedMemory.PrefetchAsync(managedMemory.SizeInBytes, CudaDevice.Current, stream);
        stream.Synchronize();
        Console.WriteLine("ManagedMemoryAdvice=True");
    }
    catch (CudaException exception)
    {
        Console.WriteLine($"ManagedMemoryAdvice=Skipped Reason={exception.Message}");
        _ = CudaDevice.GetLastErrorCode();
    }

    float[] managedRoundTrip = managedMemory.ToSingleArray(floatSource.Length);
    bool managedRoundTripOk = floatSource.SequenceEqual(managedRoundTrip);
    Console.WriteLine($"ManagedMemoryRoundTrip={managedRoundTripOk} Attachment={managedMemory.AttachmentFlags}");
}
else
{
    Console.WriteLine("ManagedMemoryRoundTrip=Skipped Reason=DeviceAttributeManagedMemoryFalse");
}

memory.Fill(0x7F, 32);
byte[] filled = memory.ToArray(32);
bool fillOk = filled.All(static value => value == 0x7F);
Console.WriteLine($"Fill={fillOk}");

memory.Fill(0x11);
bool fullFillOk = memory.ToArray(memory.SizeInBytes).All(static value => value == 0x11);

cudaEvent.Record(stream);
cudaEvent.Record(stream, CudaEventRecordFlags.Default);
stream.Synchronize();
cudaEvent.Synchronize();
Console.WriteLine($"StreamAndEvent=True EventReady={cudaEvent.IsReady()} PriorityStreamReady={priorityStream.IsReady()}");

using CudaMemory destinationMemory = new CudaMemory(32);
memory.CopyTo(destinationMemory, 32);
bool deviceToDeviceOk = destinationMemory.ToArray(32).All(static value => value == 0x7F);
Console.WriteLine($"DeviceToDevice={deviceToDeviceOk}");

using CudaMemory autoCopyMemory = new CudaMemory(32);
memory.CopyToAuto(autoCopyMemory, 32);
bool autoCopyOk = autoCopyMemory.ToArray(32).All(static value => value == 0x7F);
memory.FillAsync(0x5A, 32, stream);
memory.CopyToAutoAsync(autoCopyMemory, 32, stream);
stream.Synchronize();
bool autoCopyAsyncOk = autoCopyMemory.ToArray(32).All(static value => value == 0x5A);
Console.WriteLine($"MemcpyDefault Sync={autoCopyOk} Async={autoCopyAsyncOk}");

float helperElapsedMs = stream.MeasureElapsedTime(cudaStream =>
{
    memory.FillAsync(0x22, cudaStream);
    memory.CopyToAsync(autoCopyMemory, cudaStream);
});
bool helperAsyncCopyOk = autoCopyMemory.ToArray(autoCopyMemory.SizeInBytes).All(static value => value == 0x22);
Console.WriteLine($"CudaDeploymentHelpers FullFill={fullFillOk} FullAsyncCopy={helperAsyncCopyOk} EventElapsedMs={helperElapsedMs:0.###}");

try
{
    using CudaMemory asyncAllocatedMemory = CudaMemory.AllocateAsync(64, stream);
    using CudaPinnedMemory asyncAllocatedDestination = new CudaPinnedMemory(64);
    asyncAllocatedMemory.FillAsync(0x5C, 64, stream);
    asyncAllocatedMemory.CopyToAsync(asyncAllocatedDestination, 64, stream);
    asyncAllocatedMemory.FreeAsync(stream);
    stream.Synchronize();
    bool asyncAllocationOk = asyncAllocatedDestination.ToArray(64).All(static value => value == 0x5C);
    Console.WriteLine($"AsyncMemoryPoolAllocation={asyncAllocationOk}");
}
catch (CudaException exception)
{
    Console.WriteLine($"AsyncMemoryPoolAllocation=Skipped Reason={exception.Message}");
}

using CudaPinnedMemory pinnedSource = new CudaPinnedMemory(32);
using CudaPinnedMemory pinnedDestination = new CudaPinnedMemory(32);
using CudaMemory asyncMemory = new CudaMemory(32);
using CudaEvent startEvent = new CudaEvent();
using CudaEvent endEvent = new CudaEvent();
using CudaEvent disableTimingEvent = new CudaEvent(CudaEventCreationFlags.DisableTiming);
using CudaPinnedMemory portablePinned = new CudaPinnedMemory(16, CudaPinnedMemoryAllocationFlags.Portable);
Console.WriteLine($"PinnedFlags Source={pinnedSource.Flags} Portable={portablePinned.Flags} DisableTimingEventFlags={disableTimingEvent.Flags}");

CudaDeviceInfo currentDevice = CudaDevice.GetInfo(CudaDevice.Current);
if (currentDevice.CanMapHostMemory)
{
    try
    {
        using CudaPinnedMemory mappedPinned = new CudaPinnedMemory(16, CudaPinnedMemoryAllocationFlags.Mapped);
        Console.WriteLine($"MappedPinned IsMapped={mappedPinned.IsMapped} DevicePointer=0x{mappedPinned.MappedDevicePointerAddress:X}");
    }
    catch (CudaException exception)
    {
        Console.WriteLine($"MappedPinned Skipped=True Reason={exception.Message}");
    }
}
else
{
    Console.WriteLine("MappedPinned Skipped=True Reason=DeviceCannotMapHostMemory");
}

pinnedSource.CopyFrom(source);
disableTimingEvent.Record(stream);
startEvent.Record(stream);
asyncMemory.CopyFromAsync(pinnedSource, stream);
asyncMemory.CopyToAsync(pinnedDestination, stream);
endEvent.Record(stream);
endEvent.Synchronize();
disableTimingEvent.Synchronize();
float elapsedMs = endEvent.ElapsedTimeSince(startEvent);
byte[] asyncRoundTrip = pinnedDestination.ToArray(32);
bool asyncRoundTripOk = source.SequenceEqual(asyncRoundTrip);
Console.WriteLine($"PinnedAsyncRoundTrip={asyncRoundTripOk} ElapsedMs={elapsedMs}");
Console.WriteLine($"StreamReadyAfterSync={stream.IsReady()}");

if (CudaDevice.GetBooleanAttribute(CudaDevice.Current, CudaDeviceAttribute.HostRegisterSupported))
{
    try
    {
        byte[] registeredSourceBuffer = Enumerable.Range(0, 48).Select(static value => (byte)(value + 11)).ToArray();
        byte[] registeredDestinationBuffer = new byte[registeredSourceBuffer.Length];
        using CudaRegisteredHostMemory registeredSource = new CudaRegisteredHostMemory(registeredSourceBuffer);
        using CudaRegisteredHostMemory registeredDestination = new CudaRegisteredHostMemory(registeredDestinationBuffer);
        using CudaMemory registeredMemory = new CudaMemory(registeredSourceBuffer.Length);

        registeredMemory.CopyFromAsync(registeredSource, registeredSourceBuffer.Length, stream);
        registeredMemory.CopyToAsync(registeredDestination, registeredDestinationBuffer.Length, stream);
        stream.Synchronize();

        bool registeredRoundTripOk = registeredSourceBuffer.SequenceEqual(registeredDestinationBuffer);
        Console.WriteLine($"RegisteredHostMemoryAsyncRoundTrip={registeredRoundTripOk} Flags={registeredSource.Flags}");
    }
    catch (CudaException exception)
    {
        Console.WriteLine($"RegisteredHostMemoryAsyncRoundTrip=Skipped Reason={exception.Message}");
    }
}
else
{
    Console.WriteLine("RegisteredHostMemoryAsyncRoundTrip=Skipped Reason=HostRegisterSupportedFalse");
}

using CudaStream producerStream = new CudaStream();
using CudaStream consumerStream = new CudaStream();
using CudaEvent producerReady = new CudaEvent();
using CudaPinnedMemory waitDestination = new CudaPinnedMemory(32);
using CudaMemory waitMemory = new CudaMemory(32);

waitMemory.FillAsync(0x2A, 32, producerStream);
producerReady.Record(producerStream);
consumerStream.WaitFor(producerReady);
waitMemory.CopyToAsync(waitDestination, 32, consumerStream);
consumerStream.Synchronize();
byte[] waitRoundTrip = waitDestination.ToArray(32);
bool streamWaitEventOk = waitRoundTrip.All(static value => value == 0x2A);
Console.WriteLine($"StreamWaitEvent={streamWaitEventOk}");

const int pitchedWidth = 8;
const int pitchedHeight = 4;
byte[] pitchedSource = Enumerable.Range(0, pitchedWidth * pitchedHeight).Select(static value => (byte)(value + 3)).ToArray();
using CudaPitchedMemory pitchedMemory = new CudaPitchedMemory(pitchedWidth, pitchedHeight);
using CudaPitchedMemory pitchedCopy = new CudaPitchedMemory(pitchedWidth, pitchedHeight);
pitchedMemory.CopyFrom2D(pitchedSource, pitchedWidth);
byte[] pitchedRoundTrip = pitchedMemory.ToArray2D(pitchedWidth);
bool pitchedRoundTripOk = pitchedSource.SequenceEqual(pitchedRoundTrip);
pitchedMemory.CopyTo(pitchedCopy);
bool pitchedDeviceToDeviceOk = pitchedSource.SequenceEqual(pitchedCopy.ToArray2D(pitchedWidth));
pitchedCopy.Fill2D(0x31);
bool pitchedFillOk = pitchedCopy.ToArray2D(pitchedWidth).All(static value => value == 0x31);
Console.WriteLine($"PitchedMemory SyncRoundTrip={pitchedRoundTripOk} DeviceToDevice={pitchedDeviceToDeviceOk} Fill2D={pitchedFillOk} Width={pitchedMemory.WidthInBytes} Height={pitchedMemory.Height} Pitch={pitchedMemory.PitchInBytes}");

using CudaPinnedMemory pitchedPinnedSource = new CudaPinnedMemory(pitchedSource.Length);
using CudaPinnedMemory pitchedPinnedDestination = new CudaPinnedMemory(pitchedSource.Length);
using CudaPitchedMemory pitchedAsyncMemory = new CudaPitchedMemory(pitchedWidth, pitchedHeight);
pitchedPinnedSource.CopyFrom(pitchedSource);
pitchedAsyncMemory.CopyFrom2DAsync(pitchedPinnedSource, pitchedWidth, stream);
pitchedAsyncMemory.CopyTo2DAsync(pitchedPinnedDestination, pitchedWidth, stream);
stream.Synchronize();
byte[] pitchedAsyncRoundTrip = pitchedPinnedDestination.ToArray(pitchedSource.Length);
bool pitchedAsyncRoundTripOk = pitchedSource.SequenceEqual(pitchedAsyncRoundTrip);
pitchedAsyncMemory.Fill2DAsync(0x44, stream);
pitchedAsyncMemory.CopyTo2DAsync(pitchedPinnedDestination, pitchedWidth, stream);
stream.Synchronize();
bool pitchedAsyncFillOk = pitchedPinnedDestination.ToArray(pitchedSource.Length).All(static value => value == 0x44);
Console.WriteLine($"PitchedMemoryAsync RoundTrip={pitchedAsyncRoundTripOk} Fill2DAsync={pitchedAsyncFillOk}");

const int volumeWidth = 8;
const int volumeHeight = 2;
const int volumeDepth = 3;
byte[] volumeSource = Enumerable.Range(0, volumeWidth * volumeHeight * volumeDepth).Select(static value => (byte)(value + 17)).ToArray();
using CudaPitchedMemory volumeMemory = CudaPitchedMemory.Allocate3D(volumeWidth, volumeHeight, volumeDepth);
using CudaPitchedMemory volumeCopy = CudaPitchedMemory.Allocate3D(volumeWidth, volumeHeight, volumeDepth);
volumeMemory.CopyFrom3D(volumeSource, volumeWidth, volumeHeight, volumeDepth);
byte[] volumeRoundTrip = volumeMemory.ToArray3D(volumeWidth, volumeHeight, volumeDepth);
bool volumeRoundTripOk = volumeSource.SequenceEqual(volumeRoundTrip);
volumeMemory.CopyTo3D(volumeCopy, volumeWidth, volumeHeight, volumeDepth);
bool volumeDeviceToDeviceOk = volumeSource.SequenceEqual(volumeCopy.ToArray3D(volumeWidth, volumeHeight, volumeDepth));
volumeCopy.Fill3D(0x23, volumeHeight, volumeDepth);
bool volumeFill3DOk = volumeCopy.ToArray3D(volumeWidth, volumeHeight, volumeDepth).All(static value => value == 0x23);
using CudaPinnedMemory volumePinnedSource = new CudaPinnedMemory(volumeSource.Length);
using CudaPinnedMemory volumePinnedDestination = new CudaPinnedMemory(volumeSource.Length);
using CudaPitchedMemory volumeAsyncMemory = CudaPitchedMemory.Allocate3D(volumeWidth, volumeHeight, volumeDepth);
volumePinnedSource.CopyFrom(volumeSource);
volumeAsyncMemory.CopyFrom3DAsync(volumePinnedSource, volumeWidth, volumeHeight, volumeDepth, stream);
volumeAsyncMemory.CopyTo3DAsync(volumePinnedDestination, volumeWidth, volumeHeight, volumeDepth, stream);
stream.Synchronize();
bool volumeAsyncRoundTripOk = volumeSource.SequenceEqual(volumePinnedDestination.ToArray(volumeSource.Length));
volumeAsyncMemory.Fill3DAsync(0x31, volumeHeight, volumeDepth, stream);
volumeAsyncMemory.CopyTo3DAsync(volumePinnedDestination, volumeWidth, volumeHeight, volumeDepth, stream);
stream.Synchronize();
bool volumeAsyncFill3DOk = volumePinnedDestination.ToArray(volumeSource.Length).All(static value => value == 0x31);
Console.WriteLine($"PitchedMemory3D SyncRoundTrip={volumeRoundTripOk} DeviceToDevice={volumeDeviceToDeviceOk} Fill3D={volumeFill3DOk} AsyncRoundTrip={volumeAsyncRoundTripOk} Fill3DAsync={volumeAsyncFill3DOk} Width={volumeWidth} SliceHeight={volumeHeight} Depth={volumeDepth} AllocatedRows={volumeMemory.Height} Pitch={volumeMemory.PitchInBytes}");

const int arrayWidth = 8;
const int arrayHeight = 4;
byte[] arraySource = Enumerable.Range(0, arrayWidth * arrayHeight).Select(static value => (byte)(value + 41)).ToArray();
using CudaArray cudaArray = new CudaArray(CudaChannelFormatDescriptor.UInt8, arrayWidth, arrayHeight);
using CudaArray cudaArrayCopy = new CudaArray(CudaChannelFormatDescriptor.UInt8, arrayWidth, arrayHeight);
cudaArray.CopyFrom2D(arraySource, arrayWidth, arrayWidth, arrayHeight);
byte[] arrayRoundTrip = cudaArray.ToArray2D(arrayWidth, arrayWidth, arrayHeight);
bool arrayRoundTripOk = arraySource.SequenceEqual(arrayRoundTrip);
cudaArray.CopyTo2D(cudaArrayCopy, arrayWidth, arrayHeight);
bool arrayToArrayOk = arraySource.SequenceEqual(cudaArrayCopy.ToArray2D(arrayWidth, arrayWidth, arrayHeight));
using CudaPinnedMemory arrayPinnedSource = new CudaPinnedMemory(arraySource.Length);
using CudaPinnedMemory arrayPinnedDestination = new CudaPinnedMemory(arraySource.Length);
arrayPinnedSource.CopyFrom(arraySource);
cudaArrayCopy.CopyFrom2DAsync(arrayPinnedSource, arrayWidth, arrayWidth, arrayHeight, stream);
cudaArrayCopy.CopyTo2DAsync(arrayPinnedDestination, arrayWidth, arrayWidth, arrayHeight, stream);
stream.Synchronize();
bool arrayAsyncRoundTripOk = arraySource.SequenceEqual(arrayPinnedDestination.ToArray(arraySource.Length));
CudaArrayInfo arrayInfo = cudaArray.Info;
CudaChannelFormatDescriptor channelDescriptor = cudaArray.ChannelDescriptor;
bool arrayMemoryRequirementsOk = cudaArray.TryGetMemoryRequirements(CudaDevice.Current, out CudaArrayMemoryRequirements arrayRequirements, out string arrayMemoryRequirementsDiagnostic);
bool arraySparseOk = cudaArray.TryGetSparseProperties(out CudaArraySparseProperties arraySparseProperties, out string arraySparseDiagnostic);
string textureLinearMaxWidth = "n/a";
bool textureLinearMaxWidthOk = true;
try
{
    textureLinearMaxWidth = CudaDevice.GetTexture1DLinearMaxWidth(CudaChannelFormatDescriptor.UInt8, CudaDevice.Current).ToString();
}
catch (CudaException exception)
{
    textureLinearMaxWidthOk = false;
    textureLinearMaxWidth = $"Skipped:{exception.Message}";
}

if (!arrayMemoryRequirementsOk || !arraySparseOk || !textureLinearMaxWidthOk)
{
    _ = CudaDevice.GetLastErrorCode();
}

Console.WriteLine($"CudaArray RoundTrip={arrayRoundTripOk} ArrayToArray={arrayToArrayOk} AsyncRoundTrip={arrayAsyncRoundTripOk} Info={arrayInfo.Extent} Channel={channelDescriptor} MemReq={arrayMemoryRequirementsOk}:{arrayRequirements.SizeBytes}/{arrayRequirements.AlignmentBytes}:{arrayMemoryRequirementsDiagnostic} Sparse={arraySparseOk}:{arraySparseProperties.Flags}:{arraySparseDiagnostic} Texture1DMax={textureLinearMaxWidth}");

using CudaArray cudaArray3D = CudaArray.Create3D(CudaChannelFormatDescriptor.UInt8, new CudaArrayExtent(4, 2, 2));
byte[] array3DSource = Enumerable.Range(0, 4 * 2 * 2).Select(static value => (byte)(value + 71)).ToArray();
cudaArray3D.CopyFrom3D(array3DSource, 4, 4, 2, 4, 2, 2);
byte[] array3DRoundTrip = cudaArray3D.ToArray3D(4, 4, 2, 4, 2, 2);
bool array3DRoundTripOk = array3DSource.SequenceEqual(array3DRoundTrip);
using CudaArray cudaArray3DCopy = CudaArray.Create3D(CudaChannelFormatDescriptor.UInt8, new CudaArrayExtent(4, 2, 2));
cudaArray3D.CopyTo3D(cudaArray3DCopy, 4, 2, 2);
bool array3DToArrayOk = array3DSource.SequenceEqual(cudaArray3DCopy.ToArray3D(4, 4, 2, 4, 2, 2));
Console.WriteLine($"CudaArray3D RoundTrip={array3DRoundTripOk} ArrayToArray={array3DToArrayOk} Info={cudaArray3D.Info.Extent}");

using CudaMipmappedArray mipmappedArray = new CudaMipmappedArray(CudaChannelFormatDescriptor.UInt8, new CudaArrayExtent(4, 2, 2), 2);
CudaArrayInfo mipLevelInfo = mipmappedArray.GetLevelInfo(0);
bool mipMemoryRequirementsOk = mipmappedArray.TryGetMemoryRequirements(CudaDevice.Current, out CudaArrayMemoryRequirements mipRequirements, out string mipMemoryRequirementsDiagnostic);
bool mipSparseOk = mipmappedArray.TryGetSparseProperties(out CudaArraySparseProperties mipSparseProperties, out string mipSparseDiagnostic);
if (!mipMemoryRequirementsOk || !mipSparseOk)
{
    _ = CudaDevice.GetLastErrorCode();
}

Console.WriteLine($"CudaMipmappedArray Level0={mipLevelInfo.Extent} Levels={mipmappedArray.LevelCount} MemReq={mipMemoryRequirementsOk}:{mipRequirements.SizeBytes}/{mipRequirements.AlignmentBytes}:{mipMemoryRequirementsDiagnostic} Sparse={mipSparseOk}:{mipSparseProperties.Flags}:{mipSparseDiagnostic}");

try
{
    CudaDevice.ResetPersistingL2Cache();
    Console.WriteLine("PersistingL2Reset=True");
}
catch (CudaException exception)
{
    Console.WriteLine($"PersistingL2Reset=Skipped Reason={exception.Message}");
}

if (CudaDevice.Count > 1)
{
    bool canAccessPeer = CudaDevice.CanAccessPeer(0, 1);
    int p2pAccessSupported = CudaDevice.GetP2PAttribute(CudaDeviceP2PAttribute.AccessSupported, 0, 1);
    int p2pPerformanceRank = CudaDevice.GetP2PAttribute(CudaDeviceP2PAttribute.PerformanceRank, 0, 1);
    Console.WriteLine($"PeerAccess Device0To1={canAccessPeer} AttributeAccessSupported={p2pAccessSupported} PerformanceRank={p2pPerformanceRank}");
    if (canAccessPeer)
    {
        bool peerAccessEnabled = false;
        try
        {
            CudaDevice.EnablePeerAccess(1);
            peerAccessEnabled = true;
            Console.WriteLine("PeerAccessEnable=True");
            using CudaMemory peerSource = new CudaMemory(32);
            peerSource.Fill(0x6A, 32);
            CudaDevice.SetCurrent(1);
            using CudaMemory peerDestination = new CudaMemory(32);
            CudaDevice.SetCurrent(0);
            peerSource.CopyToPeer(peerDestination, 0, 1, 32);
            CudaDevice.SetCurrent(1);
            bool peerCopyOk = peerDestination.ToArray(32).All(static value => value == 0x6A);
            CudaDevice.SetCurrent(0);
            using CudaStream peerStream = new CudaStream();
            peerSource.Fill(0x6B, 32);
            peerSource.CopyToPeerAsync(peerDestination, 0, 1, 32, peerStream);
            peerStream.Synchronize();
            CudaDevice.SetCurrent(1);
            bool peerCopyAsyncOk = peerDestination.ToArray(32).All(static value => value == 0x6B);
            CudaDevice.SetCurrent(0);
            Console.WriteLine($"PeerCopy Sync={peerCopyOk} Async={peerCopyAsyncOk}");
        }
        catch (CudaException exception)
        {
            Console.WriteLine($"PeerAccessEnable=False Reason={exception.Message}");
        }
        finally
        {
            CudaDevice.SetCurrent(0);
            if (peerAccessEnabled)
            {
                try
                {
                    CudaDevice.DisablePeerAccess(1);
                    Console.WriteLine("PeerAccessDisable=True");
                }
                catch (CudaException exception)
                {
                    Console.WriteLine($"PeerAccessDisable=False Reason={exception.Message}");
                }
            }
        }
    }
}
else
{
    Console.WriteLine("PeerAccess Skipped=True Reason=SingleDevice");
}

try
{
    using CudaGraph emptyGraph = CudaGraph.Create();
    using CudaGraph emptyGraphClone = emptyGraph.Clone();
    Console.WriteLine($"CudaGraphStatic EmptyNodes={emptyGraph.NodeCount} EmptyRoots={emptyGraph.RootNodeCount} EmptyEdges={emptyGraph.EdgeCount} CloneNodes={emptyGraphClone.NodeCount}");
}
catch (CudaException exception)
{
    Console.WriteLine($"CudaGraphStatic=Skipped Reason={exception.Message}");
}

float[] pinnedFloatSourceValues = { -1.0f, 0.5f, 2.25f, 16.0f, 32.5f, -8.0f };
int pinnedFloatByteCount = pinnedFloatSourceValues.Length * sizeof(float);
using CudaPinnedMemory pinnedFloatSource = new CudaPinnedMemory(pinnedFloatByteCount);
using CudaPinnedMemory pinnedFloatDestination = new CudaPinnedMemory(pinnedFloatByteCount);
using CudaMemory pinnedFloatDevice = new CudaMemory(pinnedFloatByteCount);

pinnedFloatSource.CopyFrom(pinnedFloatSourceValues);
pinnedFloatDevice.CopyFromAsync(pinnedFloatSource, pinnedFloatByteCount, stream);
pinnedFloatDevice.CopyToAsync(pinnedFloatDestination, pinnedFloatByteCount, stream);
stream.Synchronize();
float[] pinnedFloatRoundTrip = pinnedFloatDestination.ToSingleArray(pinnedFloatSourceValues.Length);
bool pinnedFloatAsyncRoundTripOk = pinnedFloatSourceValues.SequenceEqual(pinnedFloatRoundTrip);
Console.WriteLine($"PinnedFloatAsyncRoundTrip={pinnedFloatAsyncRoundTripOk}");
int finalCudaError = CudaDevice.GetLastErrorCode();
Console.WriteLine($"CudaGetLastError={finalCudaError}:{CudaDevice.GetErrorName(finalCudaError)}:{CudaDevice.GetErrorString(finalCudaError)}");

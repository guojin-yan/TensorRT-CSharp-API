using JYPPX.TensorRtSharp.Cuda;
using JYPPX.TensorRtSharp.Cuda.Enum;
using JYPPX.TensorRtSharp.Cuda.Struct;
using JYPPX.TensorRtSharp.Exceptions;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;


namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {

   

        [Pure, DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDeviceReset",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaDeviceReset();

        [Pure, DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDeviceSynchronize",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaDeviceSynchronize();

        [Pure, DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDeviceSetLimit",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaDeviceSetLimit(
            CudaLimit limit,
            ulong value);

        [Pure, DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDeviceGetLimit",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaDeviceGetLimit(
            out ulong pValue,
            CudaLimit limit);

        [Pure, DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDeviceGetTexture1DLinearMaxWidth",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaDeviceGetTexture1DLinearMaxWidth(
            out ulong maxWidthInElements,
            ref CudaChannelFormatDesc fmtDesc,
            int device);



        [Pure, DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDeviceGetCacheConfig",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaDeviceGetCacheConfig(
            out CudaFuncCache pCacheConfig);

        [Pure, DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDeviceGetStreamPriorityRange",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaDeviceGetStreamPriorityRange(
            out int leastPriority,
            out int greatestPriority);

        [Pure, DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDeviceSetCacheConfig",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaDeviceSetCacheConfig(
            CudaFuncCache cacheConfig);

        [Pure, DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDeviceGetSharedMemConfig",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaDeviceGetSharedMemConfig(
            out CudaSharedMemConfig pConfig);

        [Pure, DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDeviceSetSharedMemConfig",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaDeviceSetSharedMemConfig(
            CudaSharedMemConfig config);

        [Pure, DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDeviceGetByPCIBusId",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaDeviceGetByPCIBusId(
            out int device,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string pciBusId);

        [Pure, DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDeviceGetPCIBusId",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaDeviceGetPCIBusId(
            [Out] StringBuilder pciBusId,
            int len,
            int device);

        [Pure, DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaIpcGetEventHandle",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaIpcGetEventHandle(
            out CudaIpcEventHandle_t handle,
            IntPtr cudaEvent);

        [Pure, DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaIpcOpenEventHandle",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaIpcOpenEventHandle(
            out IntPtr cudaEvent,
            CudaIpcEventHandle_t handle);

        [Pure, DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaIpcGetMemHandle",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaIpcGetMemHandle(
            out CudaIpcMemHandle_t handle,
            IntPtr devPtr);

        [Pure, DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaIpcOpenMemHandle",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaIpcOpenMemHandle(
            out IntPtr devPtr,
            CudaIpcMemHandle_t handle,
            uint flags);

        [Pure, DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaIpcCloseMemHandle",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaIpcCloseMemHandle(
            IntPtr devPtr);



        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGetLastError",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaGetLastError();


        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaPeekAtLastError",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaPeekAtLastError();


        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGetErrorName",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static IntPtr cudaRuntime_cudaGetErrorName(CudaExceptionStatus error);

  
        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGetErrorString",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static IntPtr cudaRuntime_cudaGetErrorString(CudaExceptionStatus error);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGetDeviceCount",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaGetDeviceCount(out int count);


        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGetDeviceProperties",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaGetDeviceProperties(ref CudaDeviceProp prop, int device);


        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDeviceGetAttribute",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaDeviceGetAttribute(out int value, CudaDeviceAttr attr, int device);


        /// <returns>CUDA 成功状态码。</returns>
        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDeviceGetDefaultMemPool",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaDeviceGetDefaultMemPool(out CudaMemPool_t memPool, int device);


        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDeviceSetMemPool",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaDeviceSetMemPool(int device, CudaMemPool_t memPool);


        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDeviceGetMemPool",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaDeviceGetMemPool(out CudaMemPool_t memPool, int device);


        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDeviceGetP2PAttribute",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaDeviceGetP2PAttribute(
            out int value,
            CudaDeviceP2PAttr attr,
            int srcDevice,
            int dstDevice);


        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaChooseDevice",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaChooseDevice(out int device, ref CudaDeviceProp prop);


        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaSetDevice",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaSetDevice(int device);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGetDevice",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaGetDevice(out int device);


        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaSetValidDevices",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaSetValidDevices(int[] device_arr, int len);


        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaSetDeviceFlags",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaSetDeviceFlags(uint flags);


        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGetDeviceFlags",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaGetDeviceFlags(out uint flags);



        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDeviceFlushGPUDirectRDMAWrites",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaDeviceFlushGPUDirectRDMAWrites(
            CudaFlushGPUDirectRDMAWritesTarget target,
            CudaFlushGPUDirectRDMAWritesScope scope);


        // 已添加
        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaStreamCreate",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaStreamCreate(out IntPtr pStream);

        // 已添加
        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaStreamCreateWithFlags",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaStreamCreateWithFlags(out IntPtr pStream, uint flags);

        // 已添加
        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaStreamCreateWithPriority",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaStreamCreateWithPriority(out IntPtr pStream, uint flags, int priority);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaStreamGetPriority",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaStreamGetPriority(IntPtr hStream, out int priority);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaStreamGetFlags",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaStreamGetFlags(IntPtr hStream, out uint flags);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaCtxResetPersistingL2Cache",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaCtxResetPersistingL2Cache();

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaStreamCopyAttributes",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaStreamCopyAttributes(IntPtr dst, IntPtr src);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaStreamGetAttribute",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaStreamGetAttribute(
            IntPtr hStream,
            CudaStreamAttrID attr,
            ref CudaStreamAttrValue value_out);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaStreamSetAttribute",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaStreamSetAttribute(
            IntPtr hStream,
            CudaStreamAttrID attr,
            ref CudaStreamAttrValue value);

        // 已添加
        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaStreamDestroy",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaStreamDestroy(IntPtr stream);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaStreamWaitEvent",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaStreamWaitEvent(
            IntPtr stream,
            IntPtr cudaEvent, 
            uint flags);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaStreamAddCallback",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaStreamAddCallback(
            IntPtr stream,
            CudaStreamCallback callback,
            IntPtr userData,
            uint flags);


        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaStreamSynchronize",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaStreamSynchronize(IntPtr stream);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaStreamQuery",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaStreamQuery(IntPtr stream);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaStreamAttachMemAsync",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaStreamAttachMemAsync(
            IntPtr stream,
            IntPtr devPtr,
            ulong length,
            uint flags);


        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaStreamBeginCapture",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaStreamBeginCapture(
            IntPtr stream,
            CudaStreamCaptureMode mode);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaThreadExchangeStreamCaptureMode",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaThreadExchangeStreamCaptureMode(out CudaStreamCaptureMode mode);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaStreamEndCapture",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaStreamEndCapture(
            IntPtr stream,
            out CudaGraph_t pGraph);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaStreamIsCapturing",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaStreamIsCapturing(
            IntPtr stream,
            out CudaStreamCaptureStatus pCaptureStatus);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaStreamGetCaptureInfo",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaStreamGetCaptureInfo(
            IntPtr stream,
            out CudaStreamCaptureStatus pCaptureStatus,
            out ulong pId);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaStreamGetCaptureInfo_v2",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaStreamGetCaptureInfo_v2(
            IntPtr stream,
            out CudaStreamCaptureStatus captureStatus_out,
            out ulong id_out,
            out CudaGraph_t graph_out,
            [In, Out] IntPtr[] dependencies_out, // CudaGraphNode_t** 的处理较为复杂，这里用IntPtr[]近似
            out ulong numDependencies_out);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaStreamUpdateCaptureDependencies",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaStreamUpdateCaptureDependencies(
            IntPtr stream,
            IntPtr[] dependencies, // CudaGraphNode_t*
            ulong numDependencies,
            uint flags);


        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaEventCreate",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaEventCreate(out IntPtr cudaEvent);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaEventCreateWithFlags",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaEventCreateWithFlags(out IntPtr cudaEvent, uint flags);


        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaEventRecord",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaEventRecord(IntPtr cudaEvent, IntPtr stream);


        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaEventRecordWithFlags",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaEventRecordWithFlags(IntPtr cudaEvent, IntPtr stream, uint flags);


        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaEventQuery",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaEventQuery(IntPtr cudaEvent);


        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaEventSynchronize",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaEventSynchronize(IntPtr cudaEvent);


        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaEventDestroy",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaEventDestroy(IntPtr cudaEvent);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaEventElapsedTime",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaEventElapsedTime(out float ms, IntPtr start, IntPtr end);


        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaImportExternalMemory",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaImportExternalMemory(
        //    out CudaExternalMemory_t extMem_out,
        //    ref CudaExternalMemoryHandleDesc memHandleDesc);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaExternalMemoryGetMappedBuffer",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaExternalMemoryGetMappedBuffer(
        //    out IntPtr devPtr,
        //    CudaExternalMemory_t extMem,
        //    ref CudaExternalMemoryBufferDesc bufferDesc);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaExternalMemoryGetMappedMipmappedArray",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaExternalMemoryGetMappedMipmappedArray(
        //    out CudaMipmappedArray_t mipmap,
        //    CudaExternalMemory_t extMem,
        //    ref CudaExternalMemoryMipmappedArrayDesc mipmapDesc);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDestroyExternalMemory",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaDestroyExternalMemory(CudaExternalMemory_t extMem);


        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaImportExternalSemaphore",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaImportExternalSemaphore(
        //    out CudaExternalSemaphore_t extSem_out,
        //    ref CudaExternalSemaphoreHandleDesc semHandleDesc);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaSignalExternalSemaphoresAsync",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaSignalExternalSemaphoresAsync(
        //    CudaExternalSemaphore_t[] extSemArray, // 在 C# 中使用数组封送 C 数组
        //    CudaExternalSemaphoreSignalParams[] paramsArray,
        //    uint numExtSems,
        //    IntPtr stream);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaWaitExternalSemaphoresAsync",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaWaitExternalSemaphoresAsync(
        //    CudaExternalSemaphore_t[] extSemArray,
        //    CudaExternalSemaphoreWaitParams[] paramsArray,
        //    uint numExtSems,
        //    IntPtr stream);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDestroyExternalSemaphore",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaDestroyExternalSemaphore(CudaExternalSemaphore_t extSem);


        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaLaunchKernel",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaLaunchKernel(
        //    IntPtr func, 
        //    dim3 gridDim,
        //    dim3 blockDim,
        //    IntPtr[] args, 
        //    uint sharedMem, 
        //    IntPtr stream);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaLaunchCooperativeKernel",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaLaunchCooperativeKernel(
        //    IntPtr func,
        //    dim3 gridDim,
        //    dim3 blockDim,
        //    IntPtr[] args,
        //    uint sharedMem,
        //    IntPtr stream);


        //// ===================================================================
        //// Function Configuration APIs
        //// ===================================================================

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaFuncSetCacheConfig",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaFuncSetCacheConfig(
        //    IntPtr func,
        //    CudaFuncCache cacheConfig);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaFuncSetSharedMemConfig",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaFuncSetSharedMemConfig(
        //    IntPtr func,
        //    CudaSharedMemConfig config);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaFuncGetAttributes",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaFuncGetAttributes(
        //    ref CudaFuncAttributes attr,
        //    IntPtr func);


        //// ===================================================================
        //// Function Attribute Management
        //// ===================================================================

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaFuncSetAttribute",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaFuncSetAttribute(
        //    IntPtr func,
        //    CudaFuncAttribute attr,
        //    int value);


        //// ===================================================================
        //// Host Function APIs
        //// ===================================================================

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaLaunchHostFunc",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaLaunchHostFunc(
        //    IntPtr stream,
        //    CudaHostFn fn,
        //    IntPtr userData);


        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaOccupancyMaxActiveBlocksPerMultiprocessor",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        //    out int numBlocks,
        //    IntPtr func,
        //    int blockSize,
        //    ulong dynamicSMemSize // ulong -> UIntPtr
        //);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaOccupancyAvailableDynamicSMemPerBlock",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaOccupancyAvailableDynamicSMemPerBlock(
        //    out ulong dynamicSmemSize, // ulong -> UIntPtr
        //    IntPtr func,
        //    int numBlocks,
        //    int blockSize);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        //    out int numBlocks,
        //    IntPtr func,
        //    int blockSize,
        //    ulong dynamicSMemSize, // ulong -> UIntPtr
        //    uint flags);



        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMallocManaged",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMallocManaged(
            out IntPtr devPtr,
            ulong size, // ulong -> UIntPtr
            CudaMemAttach flags);


        // ===================================================================
        // Device Memory Allocation
        // ===================================================================

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMalloc",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMalloc(
            out IntPtr devPtr,
            ulong size); // ulong -> UIntPtr


        // ===================================================================
        // Host Pinned Memory Allocation
        // ===================================================================

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMallocHost",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMallocHost(
            out IntPtr ptr,
            ulong size); // ulong -> UIntPtr


        // ===================================================================
        // Pitched Memory Allocation
        // ===================================================================

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMallocPitch",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMallocPitch(
            out IntPtr devPtr,
            out ulong pitch, // ulong -> UIntPtr
            ulong width,
            ulong height); // ulong -> UIntPtr


        // ===================================================================
        // Array Memory Allocation
        // ===================================================================

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMallocArray",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMallocArray(
            out IntPtr array,
            ref CudaChannelFormatDesc desc,
            ulong width,
            ulong height,
            uint flags); // ulong -> UIntPtr


        // ===================================================================
        // Memory Deallocation
        // ===================================================================

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaFree",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaFree(IntPtr devPtr);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaFreeHost",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaFreeHost(IntPtr ptr);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaFreeArray",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaFreeArray(IntPtr array);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaFreeMipmappedArray",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaFreeMipmappedArray(CudaMipmappedArray_t mipmappedArray);


        // ===================================================================
        // Host Memory Management
        // ===================================================================

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaHostAlloc",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaHostAlloc(
            out IntPtr pHost,
            ulong size, // ulong -> UIntPtr
            uint flags);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaHostRegister",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaHostRegister(
            IntPtr ptr,
            ulong size, // ulong -> UIntPtr
            uint flags);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaHostUnregister",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaHostUnregister(IntPtr ptr);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaHostGetDevicePointer",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaHostGetDevicePointer(
            out IntPtr pDevice,
            IntPtr pHost,
            uint flags);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaHostGetFlags",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaHostGetFlags(
            out uint pFlags,
            IntPtr pHost);


        // ===================================================================
        // 3D Memory Allocation
        // ===================================================================

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMalloc3D",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMalloc3D(
            ref CudaPitchedPtr pitchedDevPtr,
            CudaExtent extent);



        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMalloc3DArray",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMalloc3DArray(
            out IntPtr array,
            ref CudaChannelFormatDesc desc,
            CudaExtent extent,
            uint flags);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMallocMipmappedArray",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMallocMipmappedArray(
            out CudaMipmappedArray_t mipmappedArray,
            ref CudaChannelFormatDesc desc,
            CudaExtent extent,
            uint numLevels,
            uint flags);


        // ===================================================================
        // Mipmap Array Access
        // ===================================================================

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGetMipmappedArrayLevel",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaGetMipmappedArrayLevel(
            out IntPtr levelArray,
            CudaMipmappedArray_const_t mipmappedArray,
            uint level);


        // ===================================================================
        // 3D & Peer-to-Peer Memory Copy
        // ===================================================================

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemcpy3D",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemcpy3D(ref CudaMemcpy3DParms p);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemcpy3DPeer",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemcpy3DPeer(ref CudaMemcpy3DPeerParms p);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemcpy3DAsync",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemcpy3DAsync(
            ref CudaMemcpy3DParms p,
            IntPtr stream);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemcpy3DPeerAsync",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemcpy3DPeerAsync(
            ref CudaMemcpy3DPeerParms p,
            IntPtr stream);


        // ===================================================================
        // Memory & Array Information
        // ===================================================================

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemGetInfo",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemGetInfo(
            out ulong free, // ulong -> UIntPtr
            out ulong total); // ulong -> UIntPtr

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaArrayGetInfo",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaArrayGetInfo(
            out CudaChannelFormatDesc desc,
            out CudaExtent extent,
            out uint flags,
            IntPtr array);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaArrayGetPlane",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaArrayGetPlane(
            out IntPtr pPlaneArray,
            IntPtr hArray,
            uint planeIdx);



        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaArrayGetSparseProperties",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaArrayGetSparseProperties(
            ref CudaArraySparseProperties sparseProperties,
            IntPtr array);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMipmappedArrayGetSparseProperties",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMipmappedArrayGetSparseProperties(
            ref CudaArraySparseProperties sparseProperties,
            CudaMipmappedArray_t mipmap);


        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemcpy",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemcpy(
            IntPtr dst,
            IntPtr src,
            ulong count, // ulong -> UIntPtr
            CudaMemcpyKind kind);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemcpyPeer",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemcpyPeer(
            IntPtr dst,
            int dstDevice,
            IntPtr src,
            int srcDevice,
            ulong count); // ulong -> UIntPtr

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemcpy2D",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemcpy2D(
            IntPtr dst,
            ulong dpitch, // ulong -> UIntPtr
            IntPtr src,
            ulong spitch, // ulong -> UIntPtr
            ulong width,  // ulong -> UIntPtr
            ulong height, // ulong -> UIntPtr
            CudaMemcpyKind kind);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemcpy2DToArray",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemcpy2DToArray(
            IntPtr dst,
            ulong wOffset, // ulong -> UIntPtr
            ulong hOffset, // ulong -> UIntPtr
            IntPtr src,
            ulong spitch, // ulong -> UIntPtr
            ulong width,  // ulong -> UIntPtr
            ulong height, // ulong -> UIntPtr
            CudaMemcpyKind kind);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemcpy2DFromArray",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemcpy2DFromArray(
            IntPtr dst,
            ulong dpitch, // ulong -> UIntPtr
            CudaArray_const_t src,
            ulong wOffset, // ulong -> UIntPtr
            ulong hOffset, // ulong -> UIntPtr
            ulong width,  // ulong -> UIntPtr
            ulong height, // ulong -> UIntPtr
            CudaMemcpyKind kind);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemcpy2DArrayToArray",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemcpy2DArrayToArray(
            IntPtr dst,
            ulong wOffsetDst, // ulong -> UIntPtr
            ulong hOffsetDst, // ulong -> UIntPtr
            CudaArray_const_t src,
            ulong wOffsetSrc, // ulong -> UIntPtr
            ulong hOffsetSrc, // ulong -> UIntPtr
            ulong width,      // ulong -> UIntPtr
            ulong height,     // ulong -> UIntPtr
            CudaMemcpyKind kind);

        // ===================================================================
        // Global Memory (Symbol) APIs
        // ===================================================================

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemcpyToSymbol",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemcpyToSymbol(
            IntPtr symbol,
            IntPtr src,
            ulong count, // ulong -> UIntPtr
            ulong offset, // ulong -> UIntPtr
            CudaMemcpyKind kind);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemcpyFromSymbol",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemcpyFromSymbol(
            IntPtr dst,
            IntPtr symbol,
            ulong count, // ulong -> UIntPtr
            ulong offset, // ulong -> UIntPtr
            CudaMemcpyKind kind);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGetSymbolAddress",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaGetSymbolAddress(
            out IntPtr devPtr,
            IntPtr symbol);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGetSymbolSize",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaGetSymbolSize(
            out ulong size, // ulong -> UIntPtr
            IntPtr symbol);

        // ===================================================================
        // Asynchronous Memory Copy APIs
        // ===================================================================

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemcpyAsync",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemcpyAsync(
            IntPtr dst,
            IntPtr src,
            ulong count, // ulong -> UIntPtr
            CudaMemcpyKind kind,
            IntPtr stream);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemcpyPeerAsync",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemcpyPeerAsync(
            IntPtr dst,
            int dstDevice,
            IntPtr src,
            int srcDevice,
            ulong count, // ulong -> UIntPtr
            IntPtr stream);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemcpy2DAsync",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemcpy2DAsync(
            IntPtr dst,
            ulong dpitch, // ulong -> UIntPtr
            IntPtr src,
            ulong spitch, // ulong -> UIntPtr
            ulong width,  // ulong -> UIntPtr
            ulong height, // ulong -> UIntPtr
            CudaMemcpyKind kind,
            IntPtr stream);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemcpyToSymbolAsync",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemcpyToSymbolAsync(
            IntPtr symbol,
            IntPtr src,
            ulong count, // ulong -> UIntPtr
            ulong offset, // ulong -> UIntPtr
            CudaMemcpyKind kind,
            IntPtr stream);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemcpyFromSymbolAsync",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemcpyFromSymbolAsync(
            IntPtr dst,
            IntPtr symbol,
            ulong count, // ulong -> UIntPtr
            ulong offset, // ulong -> UIntPtr
            CudaMemcpyKind kind,
            IntPtr stream);

        // ===================================================================
        // Memory Set APIs
        // ===================================================================

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemset",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemset(
            IntPtr devPtr,
            int value,
            ulong count); // ulong -> UIntPtr

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemset2D",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemset2D(
            IntPtr devPtr,
            ulong pitch, // ulong -> UIntPtr
            int value,
            ulong width,  // ulong -> UIntPtr
            ulong height); // ulong -> UIntPtr

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemset3D",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemset3D(
            ref CudaPitchedPtr pitchedDevPtr,
            int value,
            CudaExtent extent);


        // ===================================================================
        // Asynchronous Memory Set & Prefetch APIs
        // ===================================================================

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemsetAsync",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemsetAsync(
            IntPtr devPtr,
            int value,
            ulong count,
            IntPtr stream);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemset2DAsync",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemset2DAsync(
            IntPtr devPtr,
            ulong pitch,
            int value,
            ulong width,
            ulong height,
            IntPtr stream);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemset3DAsync",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemset3DAsync(
            ref CudaPitchedPtr pitchedDevPtr,
            int value,
            CudaExtent extent,
            IntPtr stream);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemPrefetchAsync",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemPrefetchAsync(
            IntPtr devPtr,
            ulong count,
            int dstDevice,
            IntPtr stream);


        // ===================================================================
        // Memory Management APIs(Advise, Get Attributes)
        // ===================================================================

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemAdvise",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemAdvise(
            IntPtr devPtr,
            ulong count,
            CudaMemoryAdvise advice,
            int device);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemRangeGetAttribute",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemRangeGetAttribute(
            IntPtr data,             // C: void*
            ulong dataSize,          // C: ulong
            CudaMemRangeAttribute attribute,
            IntPtr devPtr,           // C: const void*
            ulong count);            // C: ulong

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemRangeGetAttributes",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemRangeGetAttributes(
            IntPtr data,             // C: void** -> 需要使用IntPtr数组的指针, 或 [In, Out] IntPtr[]
            IntPtr dataSizes,        // C: ulong* -> 需要使用ulong数组的指针, 或 [In, Out] ulong[]
            IntPtr attributes,       // C: enum cudaMemRangeAttribute* -> 需要使用枚举数组的指针, 或 [In, Out] CudaMemRangeAttribute[]
            ulong numAttributes,
            IntPtr devPtr,           // C: const void*
            ulong count);
        // 注意: 这三个输出指针参数在C#中处理较为复杂，通常需要固定大小的缓冲区或将它们封装到一个特殊的结构体中。
        // 上述简化声明可能不完全适用。更复杂的版本可能需要使用 Marshal.AllocHGlobal 并手动管理内存。


        // ===================================================================
        // Memory Pools APIs
        // ===================================================================

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemPoolCreate",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemPoolCreate(
            out CudaMemPool_t memPool,
            ref CudaMemPoolProps poolProps);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemPoolDestroy",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemPoolDestroy(CudaMemPool_t memPool);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemPoolGetAttribute",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemPoolGetAttribute(
            CudaMemPool_t memPool,
            CudaMemPoolAttr attr,
            IntPtr value); // C: void* -> 将取出的值封送到此IntPtr指向的内存

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemPoolSetAttribute",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemPoolSetAttribute(
            CudaMemPool_t memPool,
            CudaMemPoolAttr attr,
            IntPtr value); // C: void* -> 从此IntPtr指向的内存读取值

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemPoolTrimTo",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemPoolTrimTo(
            CudaMemPool_t memPool,
            ulong minBytesToKeep);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemPoolSetAccess",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemPoolSetAccess(
            CudaMemPool_t memPool,
            IntPtr descList, // C: const struct cudaMemAccessDesc* -> 使用结构体数组/缓冲区
            ulong count);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemPoolGetAccess",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemPoolGetAccess(
            out CudaMemAccessFlags flags,
            CudaMemPool_t memPool,
            ref CudaMemLocation location);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMallocAsync",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMallocAsync(
            out IntPtr devPtr,
            ulong size,
            IntPtr hStream);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaFreeAsync",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaFreeAsync(
            IntPtr devPtr,
            IntPtr hStream);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMallocFromPoolAsync",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMallocFromPoolAsync(
            out IntPtr ptr,
            ulong size,
            CudaMemPool_t memPool,
            IntPtr stream);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemPoolExportToShareableHandle",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemPoolExportToShareableHandle(
            IntPtr shareableHandle, // C: void* -> 调用者负责分配并传入指向缓冲区的指针
            CudaMemPool_t memPool,
            CudaMemAllocationHandleType handleType,
            uint flags);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemPoolImportFromShareableHandle",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemPoolImportFromShareableHandle(
            out CudaMemPool_t memPool,
            IntPtr shareableHandle,
            CudaMemAllocationHandleType handleType,
            uint flags);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemPoolExportPointer",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemPoolExportPointer(
            IntPtr exportData, // C: struct cudaMemPoolPtrExportData* -> 传入指向结构体的指针
            IntPtr ptr);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaMemPoolImportPointer",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaMemPoolImportPointer(
            out IntPtr ptr,
            CudaMemPool_t memPool,
            IntPtr exportData); // C: struct cudaMemPoolPtrExportData*


        // ===================================================================
        // Pointer Attributes & Peer Access APIs
        // ===================================================================

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaPointerGetAttributes",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaPointerGetAttributes(
            ref CudaPointerAttributes attributes,
            IntPtr ptr);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDeviceCanAccessPeer",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaDeviceCanAccessPeer(
            out int canAccessPeer,
            int device,
            int peerDevice);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDeviceEnablePeerAccess",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaDeviceEnablePeerAccess(
            int peerDevice,
            uint flags);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDeviceDisablePeerAccess",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaDeviceDisablePeerAccess(
            int peerDevice);


        // ===================================================================
        // Cuda Graphics API for Resource Management and Mapping
        // ===================================================================

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphicsUnregisterResource",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaGraphicsUnregisterResource(
            cudaGraphicsResource_t resource);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphicsResourceSetMapFlags",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaGraphicsResourceSetMapFlags(
            cudaGraphicsResource_t resource,
            uint flags); // unsigned int -> uint

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphicsMapResources",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaGraphicsMapResources(
            int count,
            IntPtr resources, // cudaGraphicsResource_t* -> 传入指向资源句柄数组的指针
            IntPtr stream);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphicsUnmapResources",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaGraphicsUnmapResources(
            int count,
            IntPtr resources, // cudaGraphicsResource_t* -> 传入指向资源句柄数组的指针
            IntPtr stream);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphicsResourceGetMappedPointer",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaGraphicsResourceGetMappedPointer(
            out IntPtr devPtr, // void** -> 获取到的设备指针
            out ulong size,   // size_t* -> 获取到的大小，按您的要求转为 ulong
            cudaGraphicsResource_t resource);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphicsSubResourceGetMappedArray",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaGraphicsSubResourceGetMappedArray(
            out IntPtr array, // IntPtr* -> 获取到的CUDA数组
            cudaGraphicsResource_t resource,
            uint arrayIndex,      // unsigned int -> uint
            uint mipLevel);       // unsigned int -> uint

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphicsResourceGetMappedMipmappedArray",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaGraphicsResourceGetMappedMipmappedArray(
            out CudaMipmappedArray_t mipmappedArray, // cudaMipmappedArray_t* -> 获取到的mipmap数组
            cudaGraphicsResource_t resource);


        // ===================================================================
        // Cuda Texture & Surface Objects API
        // ===================================================================

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaCreateChannelDesc",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaCreateChannelDesc(
            int x, int y, int z, int w,
            cudaChannelFormatKind f,
            out cudaChannelFormatDesc formatDesc);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaCreateTextureObject",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaCreateTextureObject(
            out ulong pTexObject,
            ref cudaResourceDesc pResDesc, // 假设这些结构体已正确定义
            ref cudaTextureDesc pTexDesc,
            ref cudaResourceViewDesc pResViewDesc);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDestroyTextureObject",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaDestroyTextureObject(ulong texObject);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGetTextureObjectResourceDesc",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaGetTextureObjectResourceDesc(
            ref cudaResourceDesc pResDesc, // 使用 ref 来获取输出结构体
            ulong texObject);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGetTextureObjectTextureDesc",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaGetTextureObjectTextureDesc(
            ref cudaTextureDesc pTexDesc,
            ulong texObject);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGetTextureObjectResourceViewDesc",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaGetTextureObjectResourceViewDesc(
            ref cudaResourceViewDesc pResViewDesc,
            ulong texObject);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaCreateSurfaceObject",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaCreateSurfaceObject(
            out ulong pSurfObject,
            ref cudaResourceDesc pResDesc);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDestroySurfaceObject",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaDestroySurfaceObject(ulong surfObject);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGetSurfaceObjectResourceDesc",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaGetSurfaceObjectResourceDesc(
            ref cudaResourceDesc pResDesc,
            ulong surfObject);


        // ===================================================================
        // Cuda Versioning API
        // ===================================================================

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDriverGetVersion",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaDriverGetVersion(out int driverVersion);

        [DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaRuntimeGetVersion",
            CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static CudaExceptionStatus cudaRuntime_cudaRuntimeGetVersion(out int runtimeVersion);


        //// ===================================================================
        //// Cuda Graph API (Initial Set)
        //// ===================================================================

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphCreate",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphCreate(
        //    out CudaGraph_t pGraph,
        //    uint flags);

        //// 注意: cudaGraphAddKernelNode 和 Get/Set/Params 方法中的指针数组在 C# 中
        //// 需要特殊处理。最稳健的方式是传入一个 IntPtr，指向一个预分配的非托管数组。
        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphAddKernelNode",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphAddKernelNode(
        //    out CudaGraphNode_t pGraphNode,
        //    CudaGraph_t graph,
        //    IntPtr pDependencies,  // C: const cudaGraphNode_t*
        //    ulong numDependencies, // C: size_t
        //    ref cudaKernelNodeParams pNodeParams);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphKernelNodeGetParams",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphKernelNodeGetParams(
        //    CudaGraphNode_t node,
        //    out cudaKernelNodeParams pNodeParams);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphKernelNodeSetParams",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphKernelNodeSetParams(
        //    CudaGraphNode_t node,
        //    ref cudaKernelNodeParams pNodeParams);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphKernelNodeCopyAttributes",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphKernelNodeCopyAttributes(
        //    CudaGraphNode_t hSrc,
        //    CudaGraphNode_t hDst);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphKernelNodeGetAttribute",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphKernelNodeGetAttribute(
        //    CudaGraphNode_t hNode,
        //    cudaKernelNodeAttrID attr,
        //    ref cudaKernelNodeAttrValue value_out); // union 需要作为 ref struct

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphKernelNodeSetAttribute",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphKernelNodeSetAttribute(
        //    CudaGraphNode_t hNode,
        //    cudaKernelNodeAttrID attr,
        //    ref cudaKernelNodeAttrValue value);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphAddMemcpyNode",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphAddMemcpyNode(
        //    out CudaGraphNode_t pGraphNode,
        //    CudaGraph_t graph,
        //    IntPtr pDependencies,
        //    ulong numDependencies,
        //    ref cudaMemcpy3DParms pCopyParams);



        //// ===================================================================
        //// Conditional Compilation Block for CUDA API Version >= 11010
        //// ===================================================================


        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphAddMemcpyNodeToSymbol",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphAddMemcpyNodeToSymbol(
        //    out CudaGraphNode_t pGraphNode,
        //    CudaGraph_t graph,
        //    IntPtr pDependencies,
        //    ulong numDependencies,
        //    IntPtr symbol,
        //    IntPtr src,
        //    ulong count,
        //    ulong offset,
        //    CudaMemcpyKind kind);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphAddMemcpyNodeFromSymbol",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphAddMemcpyNodeFromSymbol(
        //    out CudaGraphNode_t pGraphNode,
        //    CudaGraph_t graph,
        //    IntPtr pDependencies,
        //    ulong numDependencies,
        //    IntPtr dst,
        //    IntPtr symbol,
        //    ulong count,
        //    ulong offset,
        //    CudaMemcpyKind kind);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphAddMemcpyNode1D",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphAddMemcpyNode1D(
        //    out CudaGraphNode_t pGraphNode,
        //    CudaGraph_t graph,
        //    IntPtr pDependencies,
        //    ulong numDependencies,
        //    IntPtr dst,
        //    IntPtr src,
        //    ulong count,
        //    CudaMemcpyKind kind);

        //// 注意: 这里的 Get/Set Params 方法因为函数签名变化，其行为也变了。
        //// 它们不再操作 cudaMemcpy3DParms 结构体。
        //// 为了防止与无版本标识的同名方法冲突，可以重命名或放在不同的部分。
        //// 这里我们保持原名，因为 C++ 的重载在 C 中是通过不同的 EntryPoint 名称解决的。

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphMemcpyNodeSetParamsToSymbol",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphMemcpyNodeSetParamsToSymbol(
        //    CudaGraphNode_t node,
        //    IntPtr symbol,
        //    IntPtr src,
        //    ulong count,
        //    ulong offset,
        //    CudaMemcpyKind kind);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphMemcpyNodeSetParamsFromSymbol",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphMemcpyNodeSetParamsFromSymbol(
        //    CudaGraphNode_t node,
        //    IntPtr dst,
        //    IntPtr symbol,
        //    ulong count,
        //    ulong offset,
        //    CudaMemcpyKind kind);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphMemcpyNodeSetParams1D",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphMemcpyNodeSetParams1D(
        //    CudaGraphNode_t node,
        //    IntPtr dst,
        //    IntPtr src,
        //    ulong count,
        //    CudaMemcpyKind kind);


        //// ===================================================================
        //// Core CUDA Graph API Nodes (No Version Condition)
        //// ===================================================================

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphMemcpyNodeGetParams",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphMemcpyNodeGetParams(
        //    CudaGraphNode_t node,
        //    out cudaMemcpy3DParms pNodeParams);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphMemcpyNodeSetParams",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphMemcpyNodeSetParams(
        //    CudaGraphNode_t node,
        //    ref cudaMemcpy3DParms pNodeParams);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphAddMemsetNode",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphAddMemsetNode(
        //    out CudaGraphNode_t pGraphNode,
        //    CudaGraph_t graph,
        //    IntPtr pDependencies,
        //    ulong numDependencies,
        //    ref cudaMemsetParams pMemsetParams);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphMemsetNodeGetParams",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphMemsetNodeGetParams(
        //    CudaGraphNode_t node,
        //    out cudaMemsetParams pNodeParams);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphMemsetNodeSetParams",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphMemsetNodeSetParams(
        //    CudaGraphNode_t node,
        //    ref cudaMemsetParams pNodeParams);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphAddHostNode",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphAddHostNode(
        //    out CudaGraphNode_t pGraphNode,
        //    CudaGraph_t graph,
        //    IntPtr pDependencies,
        //    ulong numDependencies,
        //    ref cudaHostNodeParams pNodeParams);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphHostNodeGetParams",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphHostNodeGetParams(
        //    CudaGraphNode_t node,
        //    out cudaHostNodeParams pNodeParams);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphHostNodeSetParams",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphHostNodeSetParams(
        //    CudaGraphNode_t node,
        //    ref cudaHostNodeParams pNodeParams);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphAddChildGraphNode",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphAddChildGraphNode(
        //    out CudaGraphNode_t pGraphNode,
        //    CudaGraph_t graph,
        //    IntPtr pDependencies,
        //    ulong numDependencies,
        //    CudaGraph_t childGraph);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphChildGraphNodeGetGraph",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphChildGraphNodeGetGraph(
        //    CudaGraphNode_t node,
        //    out CudaGraph_t pGraph);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphAddEmptyNode",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphAddEmptyNode(
        //    out CudaGraphNode_t pGraphNode,
        //    CudaGraph_t graph,
        //    IntPtr pDependencies,
        //    ulong numDependencies);

        //// ===================================================================
        //// Conditional Compilation Block for CUDA API Version >= 11020
        //// ===================================================================


        //// 使用 IntPtr 代表复杂的信号量参数结构体
        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphAddExternalSemaphoresSignalNode",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphAddExternalSemaphoresSignalNode(
        //    out CudaGraphNode_t pGraphNode,
        //    CudaGraph_t graph,
        //    IntPtr pDependencies,
        //    ulong numDependencies,
        //    IntPtr nodeParams); // const cudaExternalSemaphoreSignalNodeParams*

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphExternalSemaphoresSignalNodeGetParams",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphExternalSemaphoresSignalNodeGetParams(
        //    CudaGraphNode_t hNode,
        //    IntPtr params_out); // cudaExternalSemaphoreSignalNodeParams*

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphExternalSemaphoresSignalNodeSetParams",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphExternalSemaphoresSignalNodeSetParams(
        //    CudaGraphNode_t hNode,
        //    IntPtr nodeParams); // const cudaExternalSemaphoreSignalNodeParams*

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphAddExternalSemaphoresWaitNode",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphAddExternalSemaphoresWaitNode(
        //    out CudaGraphNode_t pGraphNode,
        //    CudaGraph_t graph,
        //    IntPtr pDependencies,
        //    ulong numDependencies,
        //    IntPtr nodeParams); // const cudaExternalSemaphoreWaitNodeParams*

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphExternalSemaphoresWaitNodeGetParams",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphExternalSemaphoresWaitNodeGetParams(
        //    CudaGraphNode_t hNode,
        //    IntPtr params_out); // cudaExternalSemaphoreWaitNodeParams*

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphExternalSemaphoresWaitNodeSetParams",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphExternalSemaphoresWaitNodeSetParams(
        //    CudaGraphNode_t hNode,
        //    IntPtr nodeParams); // const cudaExternalSemaphoreWaitNodeParams*


        //// ===================================================================
        //// Conditional Compilation Block for CUDA API Version >= 11040
        //// ===================================================================


        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphAddMemAllocNode",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphAddMemAllocNode(
        //    out CudaGraphNode_t pGraphNode,
        //    CudaGraph_t graph,
        //    IntPtr pDependencies,
        //    ulong numDependencies,
        //    ref cudaMemAllocNodeParams nodeParams);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphMemAllocNodeGetParams",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphMemAllocNodeGetParams(
        //    CudaGraphNode_t node,
        //    out cudaMemAllocNodeParams params_out);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphAddMemFreeNode",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphAddMemFreeNode(
        //    out CudaGraphNode_t pGraphNode,
        //    CudaGraph_t graph,
        //    IntPtr pDependencies,
        //    ulong numDependencies,
        //    IntPtr dptr); // void*

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphMemFreeNodeGetParams",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphMemFreeNodeGetParams(
        //    CudaGraphNode_t node,
        //    out IntPtr dptr_out);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDeviceGraphMemTrim",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaDeviceGraphMemTrim(int device);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDeviceGetGraphMemAttribute",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaDeviceGetGraphMemAttribute(
        //    int device,
        //    cudaGraphMemAttributeType attr,
        //    out IntPtr value); // void*, 属性值可能是一个整数或指针

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaDeviceSetGraphMemAttribute",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaDeviceSetGraphMemAttribute(
        //    int device,
        //    cudaGraphMemAttributeType attr,
        //    IntPtr value); // void*, 同上



        //// ===================================================================
        //// Cuda Graph API (Cloning, Querying, Instantiation, Execution)
        //// ===================================================================

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphClone",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphClone(
        //    out CudaGraph_t pGraphClone,
        //    CudaGraph_t originalGraph);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphNodeFindInClone",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphNodeFindInClone(
        //    out CudaGraphNode_t pNode,
        //    CudaGraphNode_t originalNode,
        //    CudaGraph_t clonedGraph);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphNodeGetType",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphNodeGetType(
        //    CudaGraphNode_t node,
        //    out cudaGraphNodeType pType);

        //// -- Array pointer handling: nodes, numNodes --
        //// 为了处理返回的数组，通常需要先调用一次函数（传入 nodes=null, numNodes=非零）
        //// 来获取节点数量，然后分配一个足够大的数组，再次调用函数来填充数据。
        //// P/Invoke 声明可以这样写：
        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphGetNodes",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphGetNodes(
        //    CudaGraph_t graph,
        //    IntPtr nodes,       // cudaGraphNode_t*
        //    out ulong numNodes); // size_t*

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphGetRootNodes",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphGetRootNodes(
        //    CudaGraph_t graph,
        //    IntPtr pRootNodes,  // cudaGraphNode_t*
        //    out ulong pNumRootNodes);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphGetEdges",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphGetEdges(
        //    CudaGraph_t graph,
        //    IntPtr from,        // cudaGraphNode_t*
        //    IntPtr to,          // cudaGraphNode_t*
        //    out ulong numEdges); // size_t*

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphNodeGetDependencies",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphNodeGetDependencies(
        //    CudaGraphNode_t node,
        //    IntPtr pDependencies, // cudaGraphNode_t*
        //    out ulong pNumDependencies);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphNodeGetDependentNodes",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphNodeGetDependentNodes(
        //    CudaGraphNode_t node,
        //    IntPtr pDependentNodes, // cudaGraphNode_t*
        //    out ulong pNumDependentNodes);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphAddDependencies",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphAddDependencies(
        //    CudaGraph_t graph,
        //    IntPtr from, // const cudaGraphNode_t*
        //    IntPtr to,   // const cudaGraphNode_t*
        //    ulong numDependencies);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphRemoveDependencies",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphRemoveDependencies(
        //    CudaGraph_t graph,
        //    IntPtr from, // const cudaGraphNode_t*
        //    IntPtr to,   // const cudaGraphNode_t*
        //    ulong numDependencies);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphDestroyNode",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphDestroyNode(CudaGraphNode_t node);

        //// -- Conditional Compilation for cudaGraphInstantiateWithFlags --


        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphInstantiateWithFlags",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphInstantiateWithFlags(
        //    out IntPtr pGraphExec,
        //    CudaGraph_t graph,
        //    ulong flags); // unsigned long long


        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphInstantiate",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphInstantiate(
        //    out IntPtr pGraphExec,
        //    CudaGraph_t graph,
        //    out CudaGraphNode_t pErrorNode,
        //    IntPtr pLogBuffer, // char*
        //    ulong bufferSize);

        //// -- Graph Execution: Setting Params (Runtime) --

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphExecKernelNodeSetParams",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphExecKernelNodeSetParams(
        //    IntPtr hGraphExec,
        //    CudaGraphNode_t node,
        //    ref cudaKernelNodeParams pNodeParams);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphExecMemcpyNodeSetParams",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphExecMemcpyNodeSetParams(
        //    IntPtr hGraphExec,
        //    CudaGraphNode_t node,
        //    ref cudaMemcpy3DParms pNodeParams);

        //// Conditional Block for CUDA >= 11010

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphExecMemcpyNodeSetParamsToSymbol",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphExecMemcpyNodeSetParamsToSymbol(
        //    IntPtr hGraphExec,
        //    CudaGraphNode_t node,
        //    IntPtr symbol,
        //    IntPtr src,
        //    ulong count,
        //    ulong offset,
        //    CudaMemcpyKind kind);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphExecMemcpyNodeSetParamsFromSymbol",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphExecMemcpyNodeSetParamsFromSymbol(
        //    IntPtr hGraphExec,
        //    CudaGraphNode_t node,
        //    IntPtr dst,
        //    IntPtr symbol,
        //    ulong count,
        //    ulong offset,
        //    CudaMemcpyKind kind);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphExecMemcpyNodeSetParams1D",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphExecMemcpyNodeSetParams1D(
        //    IntPtr hGraphExec,
        //    CudaGraphNode_t node,
        //    IntPtr dst,
        //    IntPtr src,
        //    ulong count,
        //    CudaMemcpyKind kind);


        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphExecMemsetNodeSetParams",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphExecMemsetNodeSetParams(
        //    IntPtr hGraphExec,
        //    CudaGraphNode_t node,
        //    ref cudaMemsetParams pNodeParams);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphExecHostNodeSetParams",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphExecHostNodeSetParams(
        //    IntPtr hGraphExec,
        //    CudaGraphNode_t node,
        //    ref cudaHostNodeParams pNodeParams);

        //// Conditional Block for CUDA >= 11010

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphExecChildGraphNodeSetParams",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphExecChildGraphNodeSetParams(
        //    IntPtr hGraphExec,
        //    CudaGraphNode_t node,
        //    CudaGraph_t childGraph);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphExecEventRecordNodeSetEvent",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphExecEventRecordNodeSetEvent(
        //    IntPtr hGraphExec,
        //    CudaGraphNode_t hNode,
        //    IntPtr cudaEvent);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphExecEventWaitNodeSetEvent",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphExecEventWaitNodeSetEvent(
        //    IntPtr hGraphExec,
        //    CudaGraphNode_t hNode,
        //    IntPtr cudaEvent);


        //// Conditional Block for CUDA >= 11020

        //// 使用 IntPtr 代表复杂的信号量参数结构体
        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphExecExternalSemaphoresSignalNodeSetParams",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphExecExternalSemaphoresSignalNodeSetParams(
        //    IntPtr hGraphExec,
        //    CudaGraphNode_t hNode,
        //    IntPtr nodeParams); // const cudaExternalSemaphoreSignalNodeParams*

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphExecExternalSemaphoresWaitNodeSetParams",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphExecExternalSemaphoresWaitNodeSetParams(
        //    IntPtr hGraphExec,
        //    CudaGraphNode_t hNode,
        //    IntPtr nodeParams); // const cudaExternalSemaphoreWaitNodeParams*


        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphExecUpdate",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphExecUpdate(
        //    IntPtr hGraphExec,
        //    CudaGraph_t hGraph,
        //    out CudaGraphNode_t hErrorNode_out,
        //    out cudaGraphExecUpdateResult updateResult_out);

        //// Conditional Block for CUDA >= 11010

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphUpload",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphUpload(
        //    IntPtr graphExec,
        //    IntPtr stream);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphLaunch",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphLaunch(
        //    IntPtr graphExec,
        //    IntPtr stream);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphExecDestroy",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphExecDestroy(IntPtr graphExec);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphDestroy",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphDestroy(CudaGraph_t graph);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphDebugDotPrint",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphDebugDotPrint(
        //    CudaGraph_t graph,
        //    string path,
        //    uint flags);

        //// ===================================================================
        //// User Object API
        //// ===================================================================

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaUserObjectCreate",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaUserObjectCreate(
        //    out IntPtr object_out,
        //    IntPtr ptr,
        //    cudaHostFn_t destroy,
        //    uint initialRefcount,
        //    uint flags);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaUserObjectRetain",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaUserObjectRetain(
        //    IntPtr userObject,
        //    uint count);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaUserObjectRelease",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaUserObjectRelease(
        //    IntPtr userObject,
        //    uint count);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphRetainUserObject",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphRetainUserObject(
        //    CudaGraph_t graph,
        //    IntPtr userObject,
        //    uint count,
        //    uint flags);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGraphReleaseUserObject",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGraphReleaseUserObject(
        //    CudaGraph_t graph,
        //    IntPtr userObject,
        //    uint count);

        //// ===================================================================
        //// Driver API (Interoperability)
        //// ===================================================================

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGetDriverEntryPoint",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGetDriverEntryPoint(
        //    string symbol,
        //    out IntPtr funcPtr, // void**
        //    ulong flags);      // unsigned long long

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGetExportTable",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGetExportTable(
        //    out IntPtr ppExportTable, // const void**
        //    ref cudaUUID_t pExportTableId);

        //[DllImport(dllExtern, EntryPoint = "cudaRuntime_cudaGetFuncBySymbol",
        //    CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public extern static CudaExceptionStatus cudaRuntime_cudaGetFuncBySymbol(
        //    out IntPtr functionPtr,
        //    IntPtr symbolPtr); // const void*


    }
}

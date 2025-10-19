using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using JYPPX.TensorRtSharp.Internal.Fundamentals;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    public class Runtime : DisposableTrtObject
    {


        /// <summary>
        /// Creates Build
        /// </summary>
        public Runtime()
        {
            InitHandleException.handler(
                NativeMethods.trtRuntime_createInferRuntime(out ptr));
        }
        /// <summary>
        /// Releases the resources
        /// </summary>
        public void Release()
        {
            Dispose();
        }

        /// <inheritdoc />
        /// <summary>
        /// Releases unmanaged resources
        /// </summary>
        protected override void DisposeUnmanaged()
        {
            if (ptr != IntPtr.Zero && IsEnabledDispose)
                NativeMethods.trtRuntime_free(ptr);
            base.DisposeUnmanaged();
        }


        public void setDLACore(int dlaCore)
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_setDLACore(ptr, dlaCore));
        }

        public int getDLACore()
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_getDLACore(ptr, out int coreNum));
            return coreNum;
        }

        
        public int getNbDLACores()
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_getNbDLACores(ptr, out int coresNum));
            return coresNum;
        }

       
        public void setGpuAllocator(GpuAllocator allocator)
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_setGpuAllocator(ptr, allocator.TrtPtr));
        }



        public CudaEngine deserializeCudaEngineByBlob(byte[] blob, ulong size)
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_deserializeCudaEngineByBlob(
                ptr, ref blob[0], size, out IntPtr cudaEngine));
            return new CudaEngine(cudaEngine);
        }

        
        public CudaEngine deserializeCudaEngineByFileStreamReader(FileStreamReader streamReader)
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_deserializeCudaEngineByFileStreamReader(
                ptr, streamReader.TrtPtr, out IntPtr cudaEngine));
            return new CudaEngine(cudaEngine);
        }

        
        public CudaEngine deserializeCudaEngineByAsyncStreamReader(AsyncStreamReader streamReader)
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_deserializeCudaEngineByAsyncStreamReader(
                ptr, streamReader.TrtPtr, out IntPtr cudaEngine));
            return new CudaEngine(cudaEngine);
        }

   
        public void setMaxThreads(int maxThreads)
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_setMaxThreads(ptr, maxThreads));
        }
        
        public int getMaxThreads()
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_getMaxThreads(ptr, out int maxThreads));
            return maxThreads;
        }

        public void setTemporaryDirectory(string path) { 
            if (!Directory.Exists(path))
                throw new TrtException($"The specified path does not exist: {path}");
            TrtHandleException.handler(NativeMethods.trtRuntime_setTemporaryDirectory(ptr, path));
        }


        public string getTemporaryDirectory()
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_getTemporaryDirectory(ptr, out IntPtr pathPtr));
            string path = Marshal.PtrToStringUni(pathPtr);
            Marshal.FreeHGlobal(pathPtr);
            return path;
        }

    }
}

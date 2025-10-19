using JYPPX.TensorRtSharp.Cuda;
using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using JYPPX.TensorRtSharp.Internal.Fundamentals;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;


namespace JYPPX.TensorRtSharp.Nvinfer
{
    public class ExecutionContext : DisposableTrtObject
    {


        /// <summary>
        /// Creates empty ExecutionContext
        /// </summary>
        public ExecutionContext()
        {
            //InitHandleException.handler(
            //    NativeMethods.trtBuild_createInferBuilder(out ptr));
        }

        /// <summary>
        /// Creates from native  pointer
        /// </summary>
        /// <param name="ptr"></param>
        internal ExecutionContext(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
                throw new TrtException("Native object address is NULL");
            this.ptr = ptr;
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
                NativeMethods.trtExecutionContext_free(ptr);
            base.DisposeUnmanaged();
        }



        public CudaEngine getEngine()
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getEngine(
                ptr, out IntPtr enginePtr));
            return new CudaEngine(enginePtr);
        }

        
        public void setName(string name)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setName(
                ptr, name));
        }




        public string getName()
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getName(
                ptr, out IntPtr namePtr));
            return Marshal.PtrToStringAnsi(namePtr) ?? string.Empty;
        }

        
        public void setDeviceMemory(IntPtr memory)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setDeviceMemory(
                ptr, memory));
        }
       

        public void setDeviceMemoryV2(IntPtr memory, long size)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setDeviceMemoryV2(
                ptr, memory, size));
        }

        
        public Dims getTensorStrides(string name)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getTensorStrides(
                ptr, name, out Dims dims));
            return dims;
        }

        

        public int getOptimizationProfile()
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getOptimizationProfile(
                ptr, out int profile));
            return profile;
        }

   

        public void setinputShape(string name, Dims dims)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setInputShape(
                ptr, name, dims));
        }

        
        public Dims getTensorShape(string name)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getTensorShape(
                ptr, name, out Dims dims));
            return dims;
        }
        
        public void executeV2(IntPtr[] bindings)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_executeV2(
                ptr, ref bindings[0]));
        }

        public void setTensorAddress(string name, IntPtr tensorAddress)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setTensorAddress(
                ptr, name, tensorAddress));
        }

        
        public IntPtr getTensorAddress(string name)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getTensorAddress(
                ptr, name, out IntPtr tensorAddress));
            return tensorAddress;
        }



        public void setOutputTensorAddress(string tensorName, IntPtr tensorAddress)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setOutputTensorAddress(
                ptr, tensorName, tensorAddress));
        }


        public void setInputTensorAddress(string tensorName, IntPtr tensorAddress)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setInputTensorAddress(
                ptr, tensorName, tensorAddress));
        }



        public IntPtr getOutputTensorAddress(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getOutputTensorAddress(
                ptr, tensorName, out IntPtr tensorAddress));
            return tensorAddress;
        }


        
        public long getMaxOutputSize(string name)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getMaxOutputSize(
                ptr, name, out long maxOutputSize));
            return maxOutputSize;
        }


        public void executeV3(CudaStream stream)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_executeV3(
                ptr, stream.TrtPtr));
        }
    }
}
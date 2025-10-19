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
    public class Build : DisposableTrtObject
    {


        /// <summary>
        /// Creates Build
        /// </summary>
        public Build()
        {
            InitHandleException.handler(
                NativeMethods.trtBuild_createInferBuilder(out ptr));
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
                NativeMethods.trtBuild_free(ptr);
            base.DisposeUnmanaged();
        }

        public bool platformHasFastFp16() 
        {
            TrtHandleException.handler(NativeMethods.trtBuild_platformHasFastFp16(ptr, out int flag));
            return flag != 0;
        }

        public bool platformHasFastInt8()
        {
            int flag = 0;
            TrtHandleException.handler(NativeMethods.trtBuild_platformHasFastInt8(ptr, out flag));
            return flag != 0;
        }

        public int maxDLABatchSize()
        {
            int size = 0;
            TrtHandleException.handler(NativeMethods.trtBuild_getMaxDLABatchSize(ptr, out size));
            return size;
        }

        public int nbDLACores()
        {
            int count = 0;
            TrtHandleException.handler(NativeMethods.trtBuild_getNbDLACores(ptr, out count));
            return count;
        }

        public void setGpuAllocator(GpuAllocator gpuAllocator) 
        {
            TrtHandleException.handler(NativeMethods.trtBuild_setGpuAllocator(ptr, gpuAllocator.TrtPtr));
        }

        public BuilderConfig createBuilderConfig()
        {
            IntPtr configPtr = IntPtr.Zero;
            TrtHandleException.handler(NativeMethods.trtBuild_createBuilderConfig(ptr, out configPtr));
            return new BuilderConfig(configPtr);
        }

        public NetworkDefinition createNetworkV2(TrtNetworkDefinitionCreationFlag flags)
        {
            IntPtr networkPtr = IntPtr.Zero;
            TrtHandleException.handler(NativeMethods.trtBuild_createNetworkV2(ptr, flags, out networkPtr));
            return new NetworkDefinition(networkPtr);
        }

        public OptimizationProfile createOptimizationProfile()
        {
            IntPtr profilePtr = IntPtr.Zero;
            TrtHandleException.handler(NativeMethods.trtBuild_createOptimizationProfile(ptr, out profilePtr));
            return new OptimizationProfile(profilePtr);
        }

        public void reset()
        {
            TrtHandleException.handler(NativeMethods.trtBuild_reset(ptr));
        }

        public HostMemory buildSerializedNetwork(NetworkDefinition network, BuilderConfig config)
        {
            IntPtr hostMemory = IntPtr.Zero;
            TrtHandleException.handler(NativeMethods.trtBuild_buildSerializedNetwork(ptr, network.TrtPtr, config.TrtPtr, out hostMemory));
            return new HostMemory(hostMemory);
        }

        public bool buildSerializedNetworkToStream(NetworkDefinition network, BuilderConfig config, FileStreamReader writer)
        {
            int flag = 0;
            TrtHandleException.handler(NativeMethods.trtBuild_buildSerializedNetworkToStream(ptr, network.TrtPtr, config.TrtPtr, writer.TrtPtr, out flag));
            return flag != 0;
        }


        public CudaEngine buildEngineWithConfig(NetworkDefinition network, BuilderConfig config)
        {
            IntPtr enginePtr = IntPtr.Zero;
            TrtHandleException.handler(NativeMethods.trtBuild_buildEngineWithConfig(ptr, network.TrtPtr, config.TrtPtr, out enginePtr));
            return new CudaEngine(enginePtr);
        }


        public bool isNetworkSupported(NetworkDefinition network, BuilderConfig config)
        {
            int flag = 0;
            TrtHandleException.handler(NativeMethods.trtBuild_isNetworkSupported(ptr, network.TrtPtr, config.TrtPtr, out flag));
            return flag != 0;
        }

    }
}

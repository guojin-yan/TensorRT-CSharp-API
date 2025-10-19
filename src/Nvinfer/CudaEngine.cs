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
    public class CudaEngine : DisposableTrtObject
    {


        /// <summary>
        /// Creates empty CudaEngine
        /// </summary>
        public CudaEngine()
        {
            //InitHandleException.handler(
            //    NativeMethods.trtBuild_createInferBuilder(out ptr));
        }

        /// <summary>
        /// Creates from native  pointer
        /// </summary>
        /// <param name="ptr"></param>
        internal CudaEngine(IntPtr ptr)
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
                NativeMethods.trtCudaEngine_free(ptr);
            base.DisposeUnmanaged();
        }




        
        public Dims getTensorShape(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorShape(
                ptr, tensorName, out Dims dims));
            return dims;
        }

        
        
        public TrtDataType getTensorDataType(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorDataType(
                ptr, tensorName, out TrtDataType dataType));
            return dataType;
        }

        
        public TrtTensorLocation getTensorLocation(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorLocation(
                ptr, tensorName, out TrtTensorLocation location));
            return location;
        }

        public bool isShapeInferenceIO(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_isShapeInferenceIO(
                ptr, tensorName, out int isShapeIO));
            return isShapeIO != 0;
        }

        public TrtTensorIOMode getTensorIOMode(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorIOMode(
                ptr, tensorName, out TrtTensorIOMode ioMode));
            return ioMode;
        }

        public int getTensorBytesPerComponent(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorBytesPerComponent(
                ptr, tensorName, out int bytesPerComponent));
            return bytesPerComponent;
        }

        public int getTensorBytesPerComponent(string tensorName, int profileIndex)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorBytesPerComponent_ForProfile(
                ptr, tensorName, profileIndex, out int bytesPerComponent));
            return bytesPerComponent;
        }

        public int getTensorComponentsPerElement(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorComponentsPerElement(
                ptr, tensorName, out int componentsPerElement));
            return componentsPerElement;
        }

        public int getTensorComponentsPerElement(string tensorName, int profileIndex)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorComponentsPerElement_ForProfile(
                ptr, tensorName, profileIndex, out int componentsPerElement));
            return componentsPerElement;
        }

        public TrtTensorFormat getTensorFormat(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorFormat(
                ptr, tensorName, out TrtTensorFormat format));
            return format;
        }

        public TrtTensorFormat getTensorFormat(string tensorName, int profileIndex)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorFormat_ForProfile(
                ptr, tensorName, profileIndex, out TrtTensorFormat format));
            return format;
        }

        public string getTensorFormatDesc(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorFormatDesc(
                ptr, tensorName, out IntPtr formatDescPtr));
            return Marshal.PtrToStringAnsi(formatDescPtr);
        }

        public string getTensorFormatDesc(string tensorName, int profileIndex)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorFormatDescByProfileIndex(
                ptr, tensorName, profileIndex, out IntPtr formatDescPtr));
            return Marshal.PtrToStringAnsi(formatDescPtr);
        }

        public int getTensorVectorizedDim(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorVectorizedDim(
                ptr, tensorName, out int vectorizedDim));
            return vectorizedDim;
        }

        public int getTensorVectorizedDim(string tensorName, int profileIndex)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorVectorizedDim_ForProfile(
                ptr, tensorName, profileIndex, out int vectorizedDim));
            return vectorizedDim;
        }

        public bool isDebugTensor(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_isDebugTensor(
                ptr, tensorName, out int isDebug));
            return isDebug != 0;
        }

        public int getNbLayers()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getNbLayers(
                ptr, out int nbLayers));
            return nbLayers;
        }

        public string getName()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getName(
                ptr, out IntPtr namePtr));
            return Marshal.PtrToStringAnsi(namePtr);
        }

        public int getNbOptimizationProfiles()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getNbOptimizationProfiles(
                ptr, out int nbProfiles));
            return nbProfiles;
        }

        public TrtEngineCapability getEngineCapability()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getEngineCapability(
                ptr, out TrtEngineCapability capability));
            return capability;
        }

        public bool isRefittable()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_isRefittable(
                ptr, out int refittable));
            return refittable != 0;
        }

        public int getNbIOTensors()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getNbIOTensors(
                ptr, out int nbIOTensors));
            return nbIOTensors;
        }

        public string getIOTensorName(int index)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getIOTensorName(
                ptr, index, out IntPtr tensorNamePtr));
            return Marshal.PtrToStringAnsi(tensorNamePtr);
        }

        public TrtHardwareCompatibilityLevel getHardwareCompatibilityLevel() { 
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getHardwareCompatibilityLevel(
                ptr, out TrtHardwareCompatibilityLevel level));
            return level;
        }

        public int getNbAuxStreams()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getNbAuxStreams(
                ptr, out int nbAuxStreams));
            return nbAuxStreams;
        }

        public TrtTacticSource getTacticSources()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTacticSources(
                ptr, out TrtTacticSource sources));
            return sources;
        }

        public TrtProfilingVerbosity getProfilingVerbosity()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getProfilingVerbosity(
                ptr, out TrtProfilingVerbosity verbosity));
            return verbosity;
        }

        public HostMemory serialize()
        {
            IntPtr serializedEngine;
            TrtHandleException.handler(NativeMethods.trtCudaEngine_serialize(
                ptr, out serializedEngine));
            return new HostMemory(serializedEngine);
        }

        public SerializationConfig createSerializationConfig() 
        {
            IntPtr config;
            TrtHandleException.handler(NativeMethods.trtCudaEngine_createSerializationConfig(
                ptr, out config));
            return new SerializationConfig(config);
        }

        public HostMemory serializeWithConfig(SerializationConfig config)
        {
            IntPtr serializedEngine;
            TrtHandleException.handler(NativeMethods.trtCudaEngine_serializeWithConfig(
                ptr, config.TrtPtr, out serializedEngine));
            return new HostMemory(serializedEngine);
        }

        public ExecutionContext createExecutionContext()
        {
            IntPtr context;
            TrtHandleException.handler(NativeMethods.trtCudaEngine_createExecutionContext(
                ptr, out context));
            return new ExecutionContext(context);
        }

        public ExecutionContext createExecutionContext(TrtExecutionContextAllocationStrategy strategy)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_createExecutionContextByStrategy(
                ptr, strategy,
                out IntPtr context));
            return new ExecutionContext(context);
        }


        
        public ExecutionContext createExecutionContextWithoutDeviceMemory()
        {
            IntPtr context;
            TrtHandleException.handler(NativeMethods.trtCudaEngine_createExecutionContextWithoutDeviceMemory(
                ptr, out context));
            return new ExecutionContext(context);
        }


        public ExecutionContext createExecutionContext(RuntimeConfig runtimeConfig)
        {
            IntPtr context;
            TrtHandleException.handler(NativeMethods.trtCudaEngine_createExecutionContextByRuntimeConfig(
                ptr, runtimeConfig.TrtPtr,
                out context));
            return new ExecutionContext(context);
        }

        public RuntimeConfig createRuntimeConfig()
        {
            IntPtr config;
            TrtHandleException.handler(NativeMethods.trtCudaEngine_createRuntimeConfig(
                ptr, out config));
            return new RuntimeConfig(config);
        }

        public long getDeviceMemorySize()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getDeviceMemorySize(
                ptr, out long size));
            return size;
        }
      

        public ulong getDeviceMemorySizeForProfile(int profileIndex)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getDeviceMemorySizeForProfile(
                ptr, profileIndex, out ulong size));
            return size;
        }
     


        public long getDeviceMemorySizeV2()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getDeviceMemorySizeV2(
                ptr, out long size));
            return size;
        }


        public long getDeviceMemorySizeForProfileV2(int profileIndex)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getDeviceMemorySizeForProfileV2(
                ptr, profileIndex, out long size));
            return size;
        }

        public bool setWeightStreamingBudget(long gpuMemoryBudget)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_setWeightStreamingBudget(
                ptr, gpuMemoryBudget, out int success));
            return success != 0;
        }

        public long getWeightStreamingBudget()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getWeightStreamingBudget(
                ptr, out long budget));
            return budget;
        }

        public long getMinimumWeightStreamingBudget()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getMinimumWeightStreamingBudget(
                ptr, out long budget));
            return budget;
        }
        


        public long getStreamableWeightsSize()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getStreamableWeightsSize(
                ptr, out long size));
            return size;
        }

        public bool setWeightStreamingBudgetV2(long gpuMemoryBudget)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_setWeightStreamingBudgetV2(
                ptr, gpuMemoryBudget, out int success));
            return success != 0;
        }


        public long getWeightStreamingBudgetV2()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getWeightStreamingBudgetV2(
                ptr, out long budget));
            return budget;
        }

        public long getWeightStreamingAutomaticBudget()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getWeightStreamingAutomaticBudget(
                ptr, out long budget));
            return budget;
        }

        public long getWeightStreamingScratchMemorySize() 
        { 
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getWeightStreamingScratchMemorySize(
                ptr, out long size));
            return size;
        }


        public Dims getProfileShape(string tensorName, int profileIndex, TrtOptProfileSelector select)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getProfileShape(
                ptr, tensorName, profileIndex, select, out Dims shape));
            return shape;
        }


        public IntPtr getProfileTensorValues(string tensorName, int profileIndex, TrtOptProfileSelector select)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getProfileTensorValues(
                ptr, tensorName, profileIndex, select, out IntPtr profileTensorValues));
            return profileTensorValues;
        }
        
        public IntPtr getProfileTensorValuesV2(string tensorName, int profileIndex, TrtOptProfileSelector select)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getProfileTensorValuesV2(
                ptr, tensorName, profileIndex, select, out IntPtr profileTensorValuesV2));
            return profileTensorValuesV2;
        }

        public void setErrorRecorder(ErrorRecorder recorder)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_setErrorRecorder(
                ptr, recorder.getHandle()));
        }
        

        public int hasImplicitBatchDimension()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_hasImplicitBatchDimension(
                ptr, out int hasImplicitBatch));
            return hasImplicitBatch;
        }

        public EngineInspector createEngineInspector()
        {
            IntPtr inspectorPtr;
            TrtHandleException.handler(NativeMethods.trtCudaEngine_createEngineInspector(
                ptr, out inspectorPtr));
            return new EngineInspector(inspectorPtr);
        }

    }
}

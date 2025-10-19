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
    public class BuilderConfig : DisposableTrtObject
    {


        /// <summary>
        /// Creates empty BuilderConfig
        /// </summary>
        public BuilderConfig()
        {
            //InitHandleException.handler(
            //    NativeMethods.trtBuild_createInferBuilder(out ptr));
        }

        /// <summary>
        /// Creates from native  pointer
        /// </summary>
        /// <param name="ptr"></param>
        internal BuilderConfig(IntPtr ptr)
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
            //if (ptr != IntPtr.Zero && IsEnabledDispose)
            //    NativeMethods.trtBuild_free(ptr);
            //base.DisposeUnmanaged();
        }

        public void setAvgTimingIterationsConfig(int avgTiming)
        {
            TrtHandleException.handler(NativeMethods.trtBuildertrtBuilderConfig_setAvgTimingIterationsConfig(ptr, avgTiming));
        }

        public int getAvgTimingIterationsConfig()
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_getAvgTimingIteration(ptr, out int avgTiming));
            return avgTiming;
        }
        
        public void setEngineCapability(TrtEngineCapability trtEngineCapability)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_setEngineCapability(ptr, trtEngineCapability));
        }
        
        public TrtEngineCapability getEngineCapability()
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_getEngineCapability(ptr, out TrtEngineCapability capability));
            return capability;
        }

        public void setInt8Calibrator(Int8Calibrator int8Calibrator) { 
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_setInt8Calibrator(ptr, int8Calibrator.TrtPtr));
        }

        public Int8Calibrator getInt8Calibrator()
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_getInt8Calibrator(ptr, out IntPtr calibratorPtr));
            return new Int8Calibrator(calibratorPtr);
        }


        public void setFlags(uint builderFlags)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_setFlags(ptr, builderFlags));
        }

        public uint getFlags()
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_getFlags(ptr, out uint builderFlags));
            return builderFlags;
        }

        
        public void clearFlag(TrtBuilderFlag builderFlag)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_clearFlag(ptr, builderFlag));
        }

        public void setFlag(TrtBuilderFlag builderFlag)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_setFlag(ptr, builderFlag));
        }


        
        public TrtBuilderFlag getFlag()
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_getFlag(ptr, out uint builderFlag));
            return (TrtBuilderFlag)builderFlag;
        }

        public void etLayerDeviceType(Layer layer, TrtDeviceType deviceType)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_setLayerDeviceType(ptr, layer.TrtPtr, deviceType));
        }

        public TrtDeviceType getLayerDeviceType(Layer layer)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_getLayerDeviceType(ptr, layer.TrtPtr, out TrtDeviceType deviceType));
            return deviceType;
        }


        public bool isDeviceTypeSet(Layer layer)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_isDeviceTypeSet(ptr, layer.TrtPtr, out int outState));
            return outState != 0;
        }

        
        public void resetLayerDeviceType(Layer layer)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_resetLayerDeviceType(ptr, layer.TrtPtr));
        }

        
        public bool canRunOnDLA(Layer layer)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_canRunOnDLA(ptr, layer.TrtPtr, out int outState));
            return outState != 0;
        }

        
        public void setDLACore(int dlaCore)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_setDLACore(ptr, dlaCore));
        }
        

        public int getDLACore()
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_getDLACore(ptr, out int dlaCore));
            return dlaCore;
        }

        

        public void setDefaultDeviceType(TrtDeviceType deviceType)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_setDefaultDeviceType(ptr, deviceType));
        }


        public TrtDeviceType getDefaultDeviceType()
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_getDefaultDeviceType(ptr, out TrtDeviceType deviceType));
            return deviceType;
        }

          

        public void reset()
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_reset(ptr));
        }
        

        public void setProfileStream(CudaStream stream)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_setProfileStream(ptr, stream.TrtPtr));
        }


        public CudaStream getProfileStream()
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_getProfileStream(ptr, out IntPtr streamPtr));
            return new CudaStream(streamPtr);
        }

        public int addOptimizationProfile(OptimizationProfile profile)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_addOptimizationProfile(ptr, profile.TrtPtr, out int outIndex));
            return outIndex;
        }

        

        public int getNbOptimizationProfiles()
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_getNbOptimizationProfiles(ptr, out int outCount));
            return outCount;
        }


        
        public void setProfilingVerbosity(TrtProfilingVerbosity verbosity)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_setProfilingVerbosity(ptr, verbosity));
        }

        
        public TrtProfilingVerbosity getProfilingVerbosity()
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_getProfilingVerbosity(ptr, out TrtProfilingVerbosity verbosity));
            return verbosity;
        }
    }
}

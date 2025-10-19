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
    public class Refitter : DisposableTrtObject
    {


        /// <summary>
        /// Creates empty Layer
        /// </summary>
        public Refitter(CudaEngine cudaEngine)
        {
            InitHandleException.handler(
                NativeMethods.trtRefitter_createInferRefitter(cudaEngine.TrtPtr, out ptr));
        }

        /// <summary>
        /// Creates from native  pointer
        /// </summary>
        /// <param name="ptr"></param>
        internal Refitter(IntPtr ptr)
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
                NativeMethods.trtRefitter_free(ptr);
            base.DisposeUnmanaged();
        }


        public void setWeights(string layerName, TrtWeightsRole role, Weights weights)
        {
            int success;
            TrtHandleException.handler(NativeMethods.trtRefitter_setWeights(
                ptr, layerName, role, weights.TrtPtr, out success));
            if (success == 0)
                throw new TrtException($"Failed to set weights for layer '{layerName}' with role '{role}'");
        }


        public bool refitCudaEngine()
        {
            int succ;
            TrtHandleException.handler(NativeMethods.trtRefitter_refitCudaEngine(
                ptr, out succ));
            return succ != 0;
        }


        public void getMissing(int size, string[] layerNames, TrtWeightsRole[] roles, out int count)
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_getMissing(
                ptr, size, layerNames, roles, out count));
        }
        
        public void getAll(int size, string[] layerNames, TrtWeightsRole[] roles, out int count)
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_getAll(
                ptr, size, layerNames, roles, out count));
        }
      
        public bool setDynamicRange(string tensorName, float min, float max)
        {
            int success;
            TrtHandleException.handler(NativeMethods.trtRefitter_setDynamicRange(
                ptr, tensorName, min, max, out success));
            return success != 0;
        }

        public float getDynamicRangeMin(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_getDynamicRangeMin(
                ptr, tensorName, out float min));
            return min;
        }

        public float getDynamicRangeMax(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_getDynamicRangeMax(
                ptr, tensorName, out float max));
            return max;
        }

        public int getTensorsWithDynamicRange(int size, string[] tensorNames)
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_getTensorsWithDynamicRange(
                ptr, size, tensorNames, out int count));
            return count;
        }
        

        //public extern static TrtExceptionStatus trtRefitter_setErrorRecorder(
        //    IntPtr refitter,
        //    IntPtr recorder);

        //public extern static TrtExceptionStatus trtRefitter_getErrorRecorder(
        //    IntPtr refitter,
        //    out IntPtr recorder);
        // Named Weights Operations
        

        public bool setNamedWeights(string name, Weights weights)
        {
            int success;
            TrtHandleException.handler(NativeMethods.trtRefitter_setNamedWeights(
                ptr, name, weights.TrtPtr, out success));
            return success != 0;
        }

        

        public bool setNamedWeightsWithLocation(string name, Weights weights, TrtTensorLocation location)
        {
            int success;
            TrtHandleException.handler(NativeMethods.trtRefitter_setNamedWeightsWithLocation(
                ptr, name, weights.TrtPtr, location, out success));
            return success != 0;
        }
        

        public int getMissingWeights(int size, string[] weightsNames)
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_getMissingWeights(
                ptr, size, weightsNames, out int count));
            return count;
        }

        

        public int getAllWeights(int size, string[] weightsNames)
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_getAllWeights(
                ptr, size, weightsNames, out int count));
            return count;
        }

      
        //public extern static TrtExceptionStatus trtRefitter_getLogger(
        //    IntPtr refitter,
        //    out IntPtr logger);
       

        public bool setMaxThreads(int maxThreads)
        {
            int success;
            TrtHandleException.handler(NativeMethods.trtRefitter_setMaxThreads(
                ptr, maxThreads, out success));
            return success != 0;
        }
    

        public int getMaxThreads()
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_getMaxThreads(
                ptr, out int maxThreads));
            return maxThreads;
        }
        

        public Weights getNamedWeights(string name)
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_getNamedWeights(
                ptr, name, out IntPtr weightsPtr));
            return new Weights(weightsPtr);
        }
        

        public TrtTensorLocation getWeightsLocation(string name)
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_getWeightsLocation(
                ptr, name, out TrtTensorLocation location));
            return location;
        }


        public bool unsetNamedWeights(string name)
        {
            int success;
            TrtHandleException.handler(NativeMethods.trtRefitter_unsetNamedWeights(
                ptr, name, out success));
            return success != 0;
        }

        public void setWeightsValidation(int weightsValidation)
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_setWeightsValidation(
                ptr, weightsValidation));
        }

        public int getWeightsValidation()
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_getWeightsValidation(
                ptr, out int weightsValidation));
            return weightsValidation;
        }

        public bool refitCudaEngineAsync(CudaStream stream)
        {
            int success;
            TrtHandleException.handler(NativeMethods.trtRefitter_refitCudaEngineAsync(
                ptr, stream.TrtPtr, out success));
            return success != 0;
        }

        public Weights getWeightsPrototype(string weightsName)
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_getWeightsPrototype(
                ptr, weightsName, out IntPtr weightsPtr));
            return new Weights(weightsPtr);
        }
    }
}

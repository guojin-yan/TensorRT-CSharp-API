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
    internal class Tensor : DisposableTrtObject
    {


        /// <summary>
        /// Creates empty Layer
        /// </summary>
        public Tensor()
        {
            //InitHandleException.handler(
            //    NativeMethods.trtBuild_createInferBuilder(out ptr));
        }

        /// <summary>
        /// Creates from native  pointer
        /// </summary>
        /// <param name="ptr"></param>
        internal Tensor(IntPtr ptr)
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
                NativeMethods.trtTensor_free(ptr);
            base.DisposeUnmanaged();
        }


        public void setName(string name)
        {
            TrtHandleException.handler(NativeMethods.trtTensor_setName(ptr, name));
        }

        public string getName()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_getName(ptr, out IntPtr namePtr));
            return Marshal.PtrToStringAnsi(namePtr) ?? string.Empty;
        }

        public void setDimensions(Dims dimensions)
        {
            TrtHandleException.handler(NativeMethods.trtTensor_setDimensions(ptr, dimensions));
        }
        
        public Dims getDimensions()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_getDimensions(ptr, out Dims dims));
            return dims;
        }
        
        public void setType(TrtDataType type)
        {
            TrtHandleException.handler(NativeMethods.trtTensor_setType(ptr, type));
        }
        
        public TrtDataType getType()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_getType(ptr, out TrtDataType type));
            return type;
        }
        
        public void setDynamicRange(float min, float max, out int success)
        {
            TrtHandleException.handler(NativeMethods.trtTensor_setDynamicRange(ptr, min, max, out success));
        }
        

        public bool isNetworkInput()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_isNetworkInput(ptr, out int isInput));
            return isInput != 0;
        }
       
        public bool isNetworkOutput()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_isNetworkOutput(ptr, out int isOutput));
            return isOutput != 0;
        }
       
        public void setBroadcastAcrossBatch(int broadcast)
        {
            TrtHandleException.handler(NativeMethods.trtTensor_setBroadcastAcrossBatch(ptr, broadcast));
        }
        

        public int getBroadcastAcrossBatch()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_getBroadcastAcrossBatch(ptr, out int broadcast));
            return broadcast;
        }
        

        public TrtTensorLocation getLocation()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_getLocation(ptr, 
                out TrtTensorLocation location));
            return location;
        }
        
        public void setLocation(TrtTensorLocation location)
        {
            TrtHandleException.handler(NativeMethods.trtTensor_setLocation(ptr, location));
        }


        public bool dynamicRangeIsSet()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_dynamicRangeIsSet(ptr, out int isSet));
            return isSet != 0;
        }

        public void resetDynamicRange()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_resetDynamicRange(ptr));
        }

        public float getDynamicRangeMin()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_getDynamicRangeMin(ptr, out float min));
            return min;
        }

        public float getDynamicRangeMax()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_getDynamicRangeMax(ptr, out float max));
            return max;
        }

        public TrtTensorFormat getAllowedFormats()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_getAllowedFormats(ptr, out TrtTensorFormat format));
            return format;
        }

       
       public bool isShapeTensor()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_isShapeTensor(ptr, out int isShape));
            return isShape != 0;
        }

        public bool isExecutionTensor()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_isExecutionTensor(ptr, out int isExec));
            return isExec != 0;
        }
       
        public void setDimensionName(int index, string name)
        {
            TrtHandleException.handler(NativeMethods.trtTensor_setDimensionName(ptr, index, name));
        }

        public string getDimensionName(int index)
        {
            TrtHandleException.handler(NativeMethods.trtTensor_getDimensionName(ptr, index, out IntPtr namePtr));
            return Marshal.PtrToStringAnsi(namePtr) ?? string.Empty;
        }
    }
}
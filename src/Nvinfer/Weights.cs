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
    public class Weights : DisposableTrtObject
    {


        /// <summary>
        /// Creates empty Layer
        /// </summary>
        public Weights()
        {
            //InitHandleException.handler(
            //    NativeMethods.trtBuild_createInferBuilder(out ptr));
        }

        /// <summary>
        /// Creates from native  pointer
        /// </summary>
        /// <param name="ptr"></param>
        internal Weights(IntPtr ptr)
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
                NativeMethods.trtWeight_free(ptr);
            base.DisposeUnmanaged();
        }

        public TrtDataType DataType
        {
            get => NativeMethods.trtWeight_getDataType(ptr);
        }
        
        public long Count
        {
            get => NativeMethods.trtWeight_getCount(ptr);
        }
        
        public IntPtr Values
        {
            get => NativeMethods.trtWeight_getValues(ptr);
        }

    }
}

using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.Internal.Fundamentals;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{

    public class NetworkDefinition : DisposableTrtObject
    {


        /// <summary>
        /// Creates empty NetworkDefinition
        /// </summary>
        public NetworkDefinition()
        {
            //InitHandleException.handler(
            //    NativeMethods.trtBuild_createInferBuilder(out ptr));
        }

        /// <summary>
        /// Creates from native  pointer
        /// </summary>
        /// <param name="ptr"></param>
        internal NetworkDefinition(IntPtr ptr)
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
    }
}

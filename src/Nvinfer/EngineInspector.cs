using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using JYPPX.TensorRtSharp.Internal.Fundamentals;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    public class EngineInspector : DisposableTrtObject
    {


        /// <summary>
        /// Creates empty Layer
        /// </summary>
        public EngineInspector()
        {

        }

        /// <summary>
        /// Creates from native  pointer
        /// </summary>
        /// <param name="ptr"></param>
        internal EngineInspector(IntPtr ptr)
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
                NativeMethods.trtEngineInspector_free(ptr);
            base.DisposeUnmanaged();
        }

        public bool setExecutionContext(ExecutionContext context)
        {
            TrtHandleException.handler(NativeMethods.trtEngineInspector_setExecutionContext(
                ptr, context.TrtPtr, out int success));
            return success != 0;
        }
        

        public ExecutionContext getExecutionContext()
        {
            TrtHandleException.handler(NativeMethods.trtEngineInspector_getExecutionContext(
                ptr, out IntPtr contextPtr));
            return new ExecutionContext(contextPtr);
        }
        

        public string getLayerInformation(int layerIndex, TrtLayerInformationFormat format)
        {
            TrtHandleException.handler(NativeMethods.trtEngineInspector_getLayerInformationByLayerIndex(
                ptr, layerIndex, format, out IntPtr infoPtr));
            return Marshal.PtrToStringAnsi(infoPtr);
        }
        
        public string getEngineInformation(TrtLayerInformationFormat format)
        {
            TrtHandleException.handler(NativeMethods.trtEngineInspector_getEngineInformation(
                ptr, format, out IntPtr infoPtr));
            return Marshal.PtrToStringAnsi(infoPtr);
        }

    }
}

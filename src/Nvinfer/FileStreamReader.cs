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
    public class FileStreamReader : DisposableTrtObject
    {


        /// <summary>
        /// Creates empty FileStreamReader
        /// </summary>
        public FileStreamReader()
        {
            NativeMethods.trtFileStreamReader_createFileStreamReader(out ptr);
        }

        /// <summary>
        /// Creates from native  pointer
        /// </summary>
        /// <param name="ptr"></param>
        internal FileStreamReader(IntPtr ptr)
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
                NativeMethods.trtFileStreamReader_free(ptr);
            base.DisposeUnmanaged();
        }

        public void open(string filepath)
        {
            TrtHandleException.handler(NativeMethods.trtFileStreamReader_open(
                ptr, filepath));
        }
       

        public void close()
        {
            TrtHandleException.handler(NativeMethods.trtFileStreamReader_close(
                ptr));
        }
       

        public long read(IntPtr dest, long bytes)
        {
            TrtHandleException.handler(NativeMethods.trtFileStreamReader_read(
                ptr, dest, bytes, out long bytesRead));
            return bytesRead;
        }
       

        public void reset()
        {
            TrtHandleException.handler(NativeMethods.trtFileStreamReader_reset(
                ptr));
        }
        

        public bool isOpen()
        {
            TrtHandleException.handler(NativeMethods.trtFileStreamReader_isOpen(
                ptr, out int isOpen));
            return isOpen != 0;
        }
        public extern static TrtExceptionStatus trtFileStreamReader_isOpen(
            IntPtr reader,
            out int isOpen);
    }
}

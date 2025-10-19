using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using JYPPX.TensorRtSharp.Internal.Fundamentals;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{

    public class AsyncStreamReader : DisposableTrtObject
    {


        /// <summary>
        /// Creates empty FileStreamReader
        /// </summary>
        public AsyncStreamReader()
        {
            NativeMethods.trtAsyncStreamReader_createAsyncStreamReader(out ptr);
        }

        /// <summary>
        /// Creates from native  pointer
        /// </summary>
        /// <param name="ptr"></param>
        internal AsyncStreamReader(IntPtr ptr)
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
                NativeMethods.trtAsyncStreamReader_free(ptr);
            base.DisposeUnmanaged();
        }


        public void open(string filepath)
        {
            TrtHandleException.handler(NativeMethods.trtAsyncStreamReader_open(
                ptr, filepath));
        }

        public void close()
        {
            TrtHandleException.handler(NativeMethods.trtAsyncStreamReader_close(
                ptr));
        }
       
        public long read(IntPtr dest, long bytes, IntPtr stream)
        {
            TrtHandleException.handler(NativeMethods.trtAsyncStreamReader_read(
                ptr, dest, bytes, stream, out long bytesRead));
            return bytesRead;
        }
       
        public void seek(long offset, TrtSeekPosition where)
        {
            TrtHandleException.handler(NativeMethods.trtAsyncStreamReader_seek(
                ptr, offset, where, out int success));
        }
       
        public bool isOpen()
        {
            TrtHandleException.handler(NativeMethods.trtAsyncStreamReader_isOpen(
                ptr, out int isOpen));
            return isOpen != 0;
        }
    }
}

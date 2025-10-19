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
    public class HostMemory : DisposableTrtObject
    {


        /// <summary>
        /// Creates empty HostMemory
        /// </summary>
        public HostMemory()
        {
            //InitHandleException.handler(
            //    NativeMethods.trtBuild_createInferBuilder(out ptr));
        }

        /// <summary>
        /// Creates from native  pointer
        /// </summary>
        /// <param name="ptr"></param>
        internal HostMemory(IntPtr ptr)
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
                NativeMethods.trtHostMemory_free(ptr);
            base.DisposeUnmanaged();
        }


        public IntPtr Data
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtHostMemory_data(ptr, out IntPtr data));
                return data;
            }
        }

        public long Size
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtHostMemory_size(ptr, out long size));
                return size;
            }
        }

        public TrtDataType DataType
        {
            get
            {
                TrtHandleException.handler(NativeMethods.trtHostMemory_type(ptr, out TrtDataType dataType));
                return dataType;
            }
        }

        public byte[] getByteData()
        {
            if (Data == IntPtr.Zero || Size <= 0)
                return Array.Empty<byte>();
            byte[] data = new byte[Size];
            Marshal.Copy(Data, data, 0, (int)Size);
            return data;
        }
    }
}

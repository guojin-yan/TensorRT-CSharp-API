using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorRtSharp
{
    /// <summary>
    /// DisposableObject + ICvPtrHolder
    /// </summary>
    public abstract class DisposableTrtObject : DisposableObject, ITrtPtrHolder
    {
        /// <summary>
        /// Data pointer
        /// </summary>
        protected IntPtr ptr;

        /// <summary>
        /// Default constructor
        /// </summary>
        protected DisposableTrtObject()
            : this(true)
        {
        }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="ptr"></param>
        protected DisposableTrtObject(IntPtr ptr)
            : this(ptr, true)
        {
        }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="isEnabledDispose"></param>
        protected DisposableTrtObject(bool isEnabledDispose)
            : this(IntPtr.Zero, isEnabledDispose)
        {
        }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="isEnabledDispose"></param>
        protected DisposableTrtObject(IntPtr ptr, bool isEnabledDispose)
            : base(isEnabledDispose)
        {
            this.ptr = ptr;
        }

        /// <summary>
        /// releases unmanaged resources
        /// </summary>
        protected override void DisposeUnmanaged()
        {
            ptr = IntPtr.Zero;
            base.DisposeUnmanaged();
        }

        /// <summary>
        /// Native pointer of OpenCV structure
        /// </summary>
        public IntPtr TrtPtr
        {
            get
            {
                ThrowIfDisposed();
                return ptr;
            }
        }
    }
}

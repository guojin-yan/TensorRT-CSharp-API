using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorRtSharp
{
    /// <summary>
    /// Represents a TensorRT-based class which has a native pointer. 
    /// </summary>
    public interface ITrtPtrHolder
    {
        /// <summary>
        /// Unmanaged TensorRT data pointer
        /// </summary>
        IntPtr TrtPtr { get; }
    }
}

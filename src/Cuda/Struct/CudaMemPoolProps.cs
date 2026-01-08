using JYPPX.TensorRtSharp.Cuda;
using JYPPX.TensorRtSharp.Cuda.Enum;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace JYPPX.TensorRtSharp.Cuda
{
    /// <summary>
    /// 指定从池中进行的分配的属性。
    /// Specifies the properties of allocations made from the pool.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaMemPoolProps
    {
        /// <summary>
        /// 分配类型。目前必须指定为 cudaMemAllocationTypePinned。
        /// Allocation type. Currently must be specified as cudaMemAllocationTypePinned.
        /// </summary>
        public CudaMemAllocationType allocType;
        /// <summary>
        /// 池中分配的分配所支持的句柄类型。
        /// Handle types that will be supported by allocations from the pool.
        /// </summary>
        public CudaMemAllocationHandleType handleTypes;
        /// <summary>
        /// 分配应驻留的位置。
        /// Location allocations should reside.
        /// </summary>
        public CudaMemLocation location;
        /// <summary>
        /// 指定 ::cudaMemHandleTypeWin32 时所需的 Windows 特定 LPSECURITYATTRIBUTES。
        /// 此安全属性定义了导出的分配可能被转移到的进程的范围。
        /// 在所有其他情况下，此字段必须为零。
        /// Windows-specific LPSECURITYATTRIBUTES required when
        /// ::cudaMemHandleTypeWin32 is specified. This security attribute defines
        /// the scope of which exported allocations may be tranferred to other
        /// processes. In all other cases, this field is required to be zero.
        /// </summary>
        public IntPtr win32SecurityAttributes;
        /// <summary>
        /// 最大池大小。当设置为 0 时，默认为系统依赖值。
        /// Maximum pool size. When set to 0, defaults to a system dependent value.
        /// (注意：在 64 位系统中，size_t 映射为 ulong)
        /// </summary>
        public ulong maxSize;
        /// <summary>
        /// 保留供将来使用，必须为 0。
        /// Reserved for future use, must be 0.
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 56)]
        public byte[] reserved;
        /// <summary>
        /// 初始化结构体，将保留字段置零。
        /// Initializes the struct and zeroes out the reserved field.
        /// </summary>
        public CudaMemPoolProps(bool initReserved = true)
        {
            allocType = CudaMemAllocationType.Pinned;
            handleTypes = CudaMemAllocationHandleType.None;
            location = new CudaMemLocation { type = CudaMemLocationType.Invalid, id = -1 };
            win32SecurityAttributes = IntPtr.Zero;
            maxSize = 0;

            if (initReserved)
            {
                reserved = new byte[56];
            }
            else
            {
                reserved = null; // 如果不初始化，在 Marshalling 时可能会出错，建议初始化
            }
        }
    }
}

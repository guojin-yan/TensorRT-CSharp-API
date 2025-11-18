using JYPPX.TensorRtSharp.Cuda;
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
    /// <summary>
    /// 表示一个用于分配和释放 GPU 内存的分配器。<br/>
    /// Represents an allocator used for allocating and freeing GPU memory.
    /// </summary>
    public class GpuAllocator : DisposableTrtObject
    {
        /// <summary>
        /// 初始化一个新的、空的 <see cref="GpuAllocator"/> 实例。<br/>
        /// Initializes a new, empty instance of the <see cref="GpuAllocator"/> class.
        /// </summary>
        public GpuAllocator()
        {
        }

        /// <summary>
        /// 使用一个原生（非托管）指针初始化 <see cref="GpuAllocator"/> 类的新实例。<br/>
        /// Initializes a new instance of the <see cref="GpuAllocator"/> class from a native (unmanaged) pointer.
        /// </summary>
        /// <param name="ptr">
        /// 指向原生 <c>TrtGpuAllocator</c> 对象的指针。<br/>
        /// A pointer to the native <c>TrtGpuAllocator</c> object.
        /// </param>
        /// <exception cref="TrtException">
        /// 如果 <paramref name="ptr"/> 为 <see cref="IntPtr.Zero"/>，则抛出此异常。<br/>
        /// Thrown if <paramref name="ptr"/> is <see cref="IntPtr.Zero"/>.
        /// </exception>
        internal GpuAllocator(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
                throw new TrtException("Native object address is NULL");
            this.ptr = ptr;
        }

        /// <summary>
        /// 释放由 <see cref="GpuAllocator"/> 使用的所有资源。<br/>
        /// Releases all resources used by the <see cref="GpuAllocator"/>.
        /// </summary>
        public void Release()
        {
            Dispose();
        }

        /// <inheritdoc />
        /// <summary>
        /// 释放非托管资源。<br/>
        /// Releases unmanaged resources.
        /// </summary>
        protected override void DisposeUnmanaged()
        {
            if (ptr != IntPtr.Zero && IsEnabledDispose)
                NativeMethods.trtGpuAllocator_free(ptr);
            base.DisposeUnmanaged();
        }

        /// <summary>
        /// 异步分配一块 GPU 内存。<br/>
        /// Asynchronously allocates a block of GPU memory.
        /// </summary>
        /// <param name="size">
        /// 要分配的内存大小（以字节为单位）。<br/>
        /// The size of the memory to allocate in bytes.
        /// </param>
        /// <param name="alignment">
        /// 分配的内存对齐要求（以字节为单位）。<br/>
        /// The allocation's alignment requirement in bytes.
        /// </param>
        /// <param name="flags">
        /// 分配标志。<br/>
        /// Allocation flags.
        /// </param>
        /// <param name="stream">
        ///用于执行分配的 <see cref="CudaStream"/>。<br/>
        /// The <see cref="CudaStream"/> on which to perform the allocation.
        /// </param>
        /// <param name="memoryPtr">
        /// 当方法返回时，包含指向已分配内存的指针。此参数未经初始化即被传递。<br/>
        /// When the method returns, contains a pointer to the allocated memory. This parameter is passed uninitialized.
        /// </param>
        public void allocateAsync(ulong size, ulong alignment, uint flags, CudaStream stream, out IntPtr memoryPtr)
        {
            TrtHandleException.handler(
                NativeMethods.trtGpuAllocator_allocateAsync(
                    ptr,
                    size,
                    alignment,
                    flags,
                    stream.TrtPtr,
                    out memoryPtr));
        }

        /// <summary>
        /// 异步释放一块 GPU 内存。<br/>
        /// Asynchronously deallocates a block of GPU memory.
        /// </summary>
        /// <param name="memory">
        /// 指向要释放的内存块的指针。<br/>
        /// The pointer to the memory block to deallocate.
        /// </param>
        /// <param name="stream">
        /// 用于执行释放的 <see cref="CudaStream"/>。<br/>
        /// The <see cref="CudaStream"/> on which to perform the deallocation.
        /// </param>
        /// <returns>
        /// 如果操作成功，则为 <c>true</c>；否则为 <c>false</c>。<br/>
        /// <c>true</c> if the operation was successful; otherwise, <c>false</c>.
        /// </returns>
        public bool deallocateAsync(IntPtr memory, CudaStream stream)
        {
            TrtHandleException.handler(
                NativeMethods.trtGpuAllocator_deallocateAsync(
                    ptr,
                    memory,
                    stream.TrtPtr,
                    out int successStatus));
            return successStatus != 0;
        }

        /// <summary>
        /// 重新分配内存块的大小。<br/>
        /// Reallocates a memory block to a new size.
        /// </summary>
        /// <param name="baseAddr">
        /// 指向原始内存块的指针。<br/>
        /// The pointer to the original memory block.
        /// </param>
        /// <param name="alignment">
        /// 新分配的内存对齐要求。<br/>
        /// The new allocation's alignment requirement.
        /// </param>
        /// <param name="newSize">
        /// 内存块的新大小（以字节为单位）。<br/>
        /// The new size for the memory block in bytes.
        /// </param>
        /// <param name="memoryPtr">
        /// 当方法返回时，包含指向重新分配后的内存的指针。此参数未经初始化即被传递。<br/>
        /// When the method returns, contains a pointer to the reallocated memory. This parameter is passed uninitialized.
        /// </param>
        public void reallocate(IntPtr baseAddr, ulong alignment, ulong newSize, out IntPtr memoryPtr)
        {
            TrtHandleException.handler(
                NativeMethods.trtGpuAllocator_reallocate(
                    ptr,
                    baseAddr,
                    alignment,
                    newSize,
                    out memoryPtr));
        }

        /// <summary>
        /// 获取与此分配器关联的接口信息。<br/>
        /// Gets the interface information associated with this allocator.
        /// </summary>
        /// <returns>
        /// 一个包含接口版本和类型的 <see cref="InterfaceInfo"/> 结构体。<br/>
        /// An <see cref="InterfaceInfo"/> structure containing the interface version and type.
        /// </returns>
        public InterfaceInfo getInterfaceInfo()
        {
            TrtHandleException.handler(
                NativeMethods.trtGpuAllocator_getInterfaceInfo(
                    ptr,
                    out InterfaceInfo info));
            return info;
        }

    }

}

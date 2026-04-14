using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using JYPPX.TensorRtSharp.Internal.Fundamentals;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{
#if NETCOREAPP || NET5_0_OR_GREATER
    /// <summary>
    /// 内部辅助接口，用于在支持 Span 的 .NET 环境中统一获取内存视图。
    /// Internal helper interface to unify memory view retrieval in .NET environments that support Span.
    /// </summary>
    /// <typeparam name="T">元素类型 / The element type.</typeparam>
    internal interface ISpanProvider<T> where T : struct
    {
        /// <summary>
        /// 获取数据的 Span 视图。
        /// Gets the Span view of the data.
        /// </summary>
        Span<T> GetSpan();
    }
#endif

    /// <summary>
    /// 表示 CUDA 固定内存的托管包装类。
    /// Managed wrapper class for CUDA Pinned Memory (Page-locked Memory).
    /// 
    /// 固定内存不会被操作系统交换到磁盘，这允许 CUDA 直接进行 DMA（直接内存访问）传输，
    /// 从而显著提高主机与设备之间的数据拷贝速度。
    /// Pinned memory is not swapped to disk by the OS, allowing CUDA to perform DMA (Direct Memory Access) transfers,
    /// significantly speeding up data copying between host and device.
    /// 
    /// 该类实现了 IDisposable 模式以释放非托管内存。
    /// This class implements the IDisposable pattern to release unmanaged memory.
    /// </summary>
    /// <typeparam name="T">内存中存储的数据类型（必须是非托管结构体）。/ The type of data stored in memory (must be an unmanaged struct).</typeparam>
    public unsafe class CudaPinnedMemory<T> : DisposableTrtObject
#if NETCOREAPP || NET5_0_OR_GREATER
    , ISpanProvider<T> // 仅在 Core 环境下实现该接口，以利用高性能 Span 操作 / Implements this interface only in Core environments to leverage high-performance Span operations.
#endif
        where T : struct
    {

        // 当前缓冲区的元素数量 / Number of elements in the current buffer.
        private ulong length;

        // 标识对象是否已被释放 / Flag indicating whether the object has been disposed.
        private bool _disposed;

        // 单个元素的大小（字节） / Size of a single element in bytes.
        private readonly int elementSize;

        // 总内存大小（字节），使用 long 以支持大于 2GB 的分配 / Total memory size in bytes, using long to support allocations larger than 2GB.
        private readonly ulong sizeInBytes;

        // 类型安全的指针转换属性 / Type-safe pointer conversion property.
        private T* Ptr => (T*)ptr.ToPointer();

        /// <summary>
        /// 获取非托管内存的原始指针。
        /// Gets the raw pointer to the unmanaged memory.
        /// </summary>
        public IntPtr Pointer => ptr;

        /// <summary>
        /// 获取当前缓冲区中的元素数量。
        /// Gets the number of elements in the current buffer.
        /// </summary>
        public ulong Length => (ulong)length;

        /// <summary>
        /// 获取当前缓冲区的总字节大小。
        /// Gets the total byte size of the current buffer.
        /// 支持超过 2GB 的内存查询 (long 类型)。
        /// Supports querying memory larger than 2GB (long type).
        /// </summary>
        public ulong SizeInBytes => sizeInBytes;

        /// <summary>
        /// 构造函数，分配指定数量的固定内存。
        /// Constructor, allocates a specific amount of pinned memory.
        /// </summary>
        /// <param name="count">要分配的元素数量。/ The number of elements to allocate.</param>
        /// <exception cref="ArgumentOutOfRangeException">当 count 小于等于 0 时抛出。/ Thrown when count is less than or equal to 0.</exception>
        public CudaPinnedMemory(ulong count)
        {
            if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count));
            length = count;
            elementSize = sizeof(T);
            sizeInBytes = (count * (ulong)elementSize);

            // 调用 CUDA 驱动 API 分配页锁定内存 / Call CUDA Driver API to allocate page-locked memory.
            // cudaMallocHost: 在主机上分配内存，该内存被页锁定，对设备访问可达。
            // cudaMallocHost: Allocates memory on the host that is page-locked and accessible to the device.
            CudaHandleException.handler(NativeMethods.cudaRuntime_cudaMallocHost(
               out ptr, sizeInBytes));
        }

        // ==========================================
        // 核心：条件编译的 Span 支持
        // Core: Conditionally compiled Span support
        // ==========================================
#if NETCOREAPP || NET5_0_OR_GREATER
        /// <summary>
        /// [.NET Core Only] 获取当前缓冲区的 Span 视图。
        /// [.NET Core Only] Gets the Span view of the current buffer.
        /// 
        /// 这是一个零拷贝操作，提供对内存的高性能安全访问。
        /// This is a zero-copy operation, providing high-performance safe access to memory.
        /// </summary>
        public Span<T> Span
        {
            get
            {
                if (_disposed) throw new ObjectDisposedException(nameof(CudaPinnedMemory<T>));
                // C# 7.3+ 允许从指针直接构造 Span，完全零开销抽象。
                // C# 7.3+ allows constructing Span directly from a pointer, a zero-overhead abstraction.
                return new Span<T>(Ptr, (int)length);
            }
        }

        /// <summary>
        /// 显式接口实现：获取 Span 视图。
        /// Explicit interface implementation: Get Span view.
        /// </summary>
        public Span<T> GetSpan()
        {
            return Span;
        }

        /// <summary>
        /// 获取整个缓冲区的只读 Span 视图。
        /// Gets the read-only Span view of the entire buffer.
        /// </summary>
        public ReadOnlySpan<T> ReadOnlySpan
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                if (_disposed) throw new ObjectDisposedException(nameof(CudaPinnedMemory<T>));
                return new ReadOnlySpan<T>((void*)ptr, (int)length);
            }
        }

        /// <summary>
        /// 对内存进行切片，返回一个新的 Span。
        /// Slices the memory and returns a new Span.
        /// 
        /// 此操作不分配新内存，只是引用原始内存的一个子集。
        /// This operation does not allocate new memory; it just references a subset of the original memory.
        /// </summary>
        /// <param name="start">起始索引。/ The starting index.</param>
        /// <param name="length">切片长度。/ The length of the slice.</param>
        /// <returns>指向子集的 Span。/ A Span pointing to the subset.</returns>
        public Span<T> Slice(int start, int length)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(CudaPinnedMemory<T>));
            if (start < 0 || length < 0 || start + length > length)
                throw new ArgumentOutOfRangeException();
            return new Span<T>((void*)ptr, (int)length).Slice(start, length);
        }


        /// <summary>
        /// 从源 Span 拷贝数据到当前的 Pinned Memory。
        /// Copies data from a source Span to the current Pinned Memory.
        /// 
        /// 支持任意 Span 来源（数组、切片、栈上内存等）。
        /// Supports arbitrary Span sources (arrays, slices, stack memory, etc.).
        /// </summary>
        /// <param name="source">源数据 Span。/ The source data Span.</param>
        public void CopyFrom(ReadOnlySpan<T> source)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(CudaPinnedMemory<T>));
            //if ((ulong)source.Length < length) throw new ArgumentException("源 Span 长度不足 / Source Span length is insufficient");

            // 获取源数据的固定指针，防止 GC 在拷贝期间移动内存。
            // Pins the source data to prevent GC from moving memory during the copy.
            fixed (void* srcPtr = source)
            {
                ulong size = (length < (ulong)source.Length ? length : (ulong)source.Length) * (ulong)Marshal.SizeOf(typeof(T));
                // 底层内存拷贝，确保高效传输 / Low-level memory copy ensuring efficient transfer.
                Buffer.MemoryCopy(srcPtr, (void*)ptr, size, size);
            }
        }

        /// <summary>
        /// 尝试将当前 Pinned Memory 的数据拷贝到目标 Span。
        /// Attempts to copy data from the current Pinned Memory to the target Span.
        /// </summary>
        /// <param name="destination">目标 Span。/ The target Span.</param>
        /// <returns>如果拷贝成功返回 true，否则返回 false（通常是因为目标空间不足或对象已释放）。/ Returns true if copy succeeds; otherwise, false (usually due to insufficient space or disposed object).</returns>
        public bool TryCopyTo(Span<T> destination)
        {
            if (_disposed || (ulong)destination.Length < length) return false;

            // 固定目标内存指针 / Pin the target memory pointer.
            fixed (void* destPtr = destination)
            {
                ulong size = (length < (ulong)destination.Length ? length : (ulong)destination.Length) * (ulong)Marshal.SizeOf(typeof(T));
                Buffer.MemoryCopy((void*)ptr, destPtr, size, size);
            }
            return true;
        }
#endif

        // ==========================================
        // 基础功能：所有框架通用
        // Basic Features: Universal for all frameworks
        // ==========================================

        /// <summary>
        /// 将当前内存的数据拷贝到目标托管数组。
        /// Copies data from the current memory to a target managed array.
        /// </summary>
        /// <param name="destination">目标数组。/ The target array.</param>
        public void CopyTo(T[] destination)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(CudaPinnedMemory<T>));
            if (destination == null) throw new ArgumentNullException(nameof(destination));
            //if ((ulong)destination.Length < length) throw new ArgumentException("目标数组长度不足 / Destination array length is insufficient");

            // 固定数组并拷贝 / Pin array and copy.
            fixed (T* destPtr = destination)
            {
                ulong size = (length < (ulong)destination.Length ? length : (ulong)destination.Length) * (ulong)Marshal.SizeOf(typeof(T));
                Buffer.MemoryCopy(ptr.ToPointer(), destPtr, size, size);


            }
        }




        /// <summary>
        /// 从源托管数组拷贝数据到当前内存。
        /// Copies data from a source managed array to the current memory.
        /// </summary>
        /// <param name="source">源数组。/ The source array.</param>
        public void CopyFrom(T[] source)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(CudaPinnedMemory<T>));
            if (source == null) throw new ArgumentNullException(nameof(source));
            //if ((ulong)source.Length < length) throw new ArgumentException("源数组长度不足 / Source array length is insufficient");

            // 固定数组并拷贝 / Pin array and copy.
            fixed (T* srcPtr = source)
            {

                ulong size = (length < (ulong)source.Length ? length : (ulong)source.Length) * (ulong)Marshal.SizeOf(typeof(T));

                Buffer.MemoryCopy(srcPtr, ptr.ToPointer(), size, size);
            }
        }

        /// <summary>
        /// 索引器，允许像数组一样直接访问内存中的元素。
        /// Indexer, allows direct access to elements in memory like an array.
        /// </summary>
        /// <param name="index">元素索引。/ Element index.</param>
        /// <returns>指定索引处的元素。/ The element at the specified index.</returns>
        public T this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)] // 强制内联，消除属性调用的开销，类似原生数组访问速度。
            get
            {
                // 使用 uint 进行范围检查是常见的优化技巧，只需一条指令即可同时检查 <0 和 >=length。
                // Using uint for range check is a common optimization trick to check both <0 and >=length in a single instruction.
                if ((uint)index >= (uint)length) throw new IndexOutOfRangeException();
                return ((T*)ptr)[index];
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set
            {
                if ((uint)index >= (uint)length) throw new IndexOutOfRangeException();
                ((T*)ptr)[index] = value;
            }
        }

        /// <summary>
        /// 释放当前对象持有的所有资源。此方法为 Dispose 的显式别名。
        /// Releases all resources held by the current object. This method is an explicit alias for Dispose.
        /// </summary>
        public void Release()
        {
            Dispose();
        }

        /// <inheritdoc />
        /// <summary>
        /// 释放所有非托管资源，即释放 CUDA 固定内存。
        /// Releases all unmanaged resources, i.e., frees the CUDA pinned memory.
        /// 
        /// 此方法由 Dispose 模式调用，不应直接调用。
        /// This method is called by the Dispose pattern and should not be called directly.
        /// </summary>
        protected override void DisposeUnmanaged()
        {
            // 确保指针有效且允许释放
            // Ensure the pointer is valid and disposal is enabled.
            if (ptr != IntPtr.Zero && IsEnabledDispose)
            {
                // 调用 CUDA API 释放页锁定内存 / Call CUDA API to free page-locked memory.
                CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaFreeHost(ptr));
            }
            base.DisposeUnmanaged();
        }


    }
}

using System;
using System.Collections.Generic;
using System.Text;

namespace JYPPX.TensorRtSharp.Cuda
{
    /// <summary>
    /// CUDA memory pool attributes
    /// CUDA 内存池属性
    /// </summary>
    public enum CudaMemPoolAttr
    {
        /// <summary>
        /// (value type = int)
        /// Allow cuMemAllocAsync to use memory asynchronously freed
        /// in another streams as long as a stream ordering dependency
        /// of the allocating stream on the free action exists.
        /// Cuda events and null stream interactions can create the required
        /// stream ordered dependencies. (default enabled)
        /// (值类型 = int)
        /// 只要存在分配流对释放操作的流排序依赖，
        /// 就允许 cuMemAllocAsync 使用在另一个流中异步释放的内存。
        /// CUDA 事件和空流交互可以创建所需的流排序依赖。（默认启用）
        /// </summary>
        ReuseFollowEventDependencies = 0x1,

        /// <summary>
        /// (value type = int)
        /// Allow reuse of already completed frees when there is no dependency
        /// between the free and allocation. (default enabled)
        /// (值类型 = int)
        /// 允许在释放和分配之间没有依赖关系时，
        /// 重用已完成的释放。（默认启用）
        /// </summary>
        ReuseAllowOpportunistic = 0x2,

        /// <summary>
        /// (value type = int)
        /// Allow cuMemAllocAsync to insert new stream dependencies
        /// in order to establish the stream ordering required to reuse
        /// a piece of memory released by cuFreeAsync (default enabled).
        /// (值类型 = int)
        /// 允许 cuMemAllocAsync 插入新的流依赖项，
        /// 以建立重用 cuFreeAsync 释放的内存块所需的流排序（默认启用）。
        /// </summary>
        ReuseAllowInternalDependencies = 0x3,

        /// <summary>
        /// (value type = cuuint64_t)
        /// Amount of reserved memory in bytes to hold onto before trying
        /// to release memory back to the OS. When more than the release
        /// threshold bytes of memory are held by the memory pool, the
        /// allocator will try to release memory back to the OS on the
        /// next call to stream, event or context synchronize. (default 0)
        /// (值类型 = cuuint64_t)
        /// 在尝试将内存释放回操作系统之前要保留的预留内存量（字节）。
        /// 当内存池持有的内存超过释放阈值字节时，分配器将在下次调用流、事件或上下文同步时
        /// 尝试将内存释放回操作系统。（默认为 0）
        /// </summary>
        ReleaseThreshold = 0x4,

        /// <summary>
        /// (value type = cuuint64_t)
        /// Amount of backing memory currently allocated for the mempool.
        /// (值类型 = cuuint64_t)
        /// 当前为内存池分配的支持存储内存量。
        /// </summary>
        ReservedMemCurrent = 0x5,

        /// <summary>
        /// (value type = cuuint64_t)
        /// High watermark of backing memory allocated for the mempool since the
        /// last time it was reset. High watermark can only be reset to zero.
        /// (值类型 = cuuint64_t)
        /// 自上次重置以来为内存池分配的支持存储内存的高水位线。
        /// 高水位线只能重置为零。
        /// </summary>
        ReservedMemHigh = 0x6,

        /// <summary>
        /// (value type = cuuint64_t)
        /// Amount of memory from the pool that is currently in use by the application.
        /// (值类型 = cuuint64_t)
        /// 应用程序当前正在使用的内存池中的内存量。
        /// </summary>
        UsedMemCurrent = 0x7,

        /// <summary>
        /// (value type = cuuint64_t)
        /// High watermark of the amount of memory from the pool that was in use by the application since
        /// the last time it was reset. High watermark can only be reset to zero.
        /// (值类型 = cuuint64_t)
        /// 自上次重置以来应用程序使用的内存池内存量的高水位线。
        /// 高水位线只能重置为零。
        /// </summary>
        UsedMemHigh = 0x8
    }

}

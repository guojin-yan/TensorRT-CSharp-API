using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 临时文件控制标志枚举
    /// Flags used to control TensorRT's behavior when creating executable temporary files
    /// </summary>
    /// <remarks>
    /// 在某些平台上，TensorRT运行时可能需要在临时目录中创建文件，或使用特定于平台的API在内存中创建文件，
    /// 以加载实现运行时代码的临时DLL。这些标志允许应用程序显式控制TensorRT对这些文件的使用。
    /// 这将禁止使用某些TensorRT API来反序列化和加载精简运行时。
    /// On some platforms the TensorRT runtime may need to create files in a temporary directory or use platform-specific
    //! APIs to create files in-memory to load temporary DLLs that implement runtime code. These flags allow the
    //! application to explicitly control TensorRT's use of these files. This will preclude the use of certain TensorRT
    //! APIs for deserializing and loading lean runtimes.
    /// </remarks>
    public enum TrtTempfileControlFlag : int
    {
        /// <summary>
        /// 允许在内存（或未命名文件）中创建和加载文件
        /// Allow creating and loading files in-memory (or unnamed files)
        /// </summary>
        //! Allow creating and loading files in-memory (or unnamed files).
        kALLOW_IN_MEMORY_FILES = 0,

        /// <summary>
        /// 允许在文件系统的临时目录中创建和加载命名文件
        /// Allow creating and loading named files in a temporary directory on the filesystem
        /// </summary>
        /// <see cref="IRuntime::setTemporaryDirectory()"/>
        //! Allow creating and loading named files in a temporary directory on the filesystem.
        //!
        //! \see IRuntime::setTemporaryDirectory()
        kALLOW_TEMPORARY_FILES = 1,
    };


}

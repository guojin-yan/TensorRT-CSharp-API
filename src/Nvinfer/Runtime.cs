using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using JYPPX.TensorRtSharp.Internal.Fundamentals;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{

    /// <summary>
    /// TensorRT运行时类，负责反序列化引擎以执行推理。继承自DisposableTrtObject。
    /// The TensorRT runtime class responsible for deserializing engines to perform inference. Inherits from DisposableTrtObject.
    /// </summary>
    public class Runtime : DisposableTrtObject
    {

        /// <summary>
        /// 创建一个新的 Runtime 实例。
        /// Creates a new Runtime instance.
        /// </summary>
        /// <exception cref="TrtException">如果创建运行时失败。/ If the creation of the runtime fails.</exception>
        public Runtime()
        {
            InitHandleException.handler(
                NativeMethods.trtRuntime_createInferRuntime(out ptr));
        }

        /// <summary>
        /// 使用一个原生指针来初始化 Runtime 实例。主要用于内部封装。
        /// Initializes a Runtime instance from a native pointer. Primarily used for internal wrapping.
        /// </summary>
        /// <param name="ptr">指向原生对象的非托管指针。/ The unmanaged pointer to the native object.</param>
        internal Runtime(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
                throw new TrtException("Native object address is NULL");
            this.ptr = ptr;
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
        /// 释放所有非托管资源。此方法由 Dispose 模式调用，不应直接调用。
        /// Releases all unmanaged resources. This method is called by the Dispose pattern and should not be called directly.
        /// </summary>
        protected override void DisposeUnmanaged()
        {
            //if (ptr != IntPtr.Zero && IsEnabledDispose)
            //    NativeMethods.trtRuntime_free(ptr);
            base.DisposeUnmanaged();
        }

        /// <summary>
        /// 设置要用于推理的DLA（深度学习加速器）核心。
        /// Sets the DLA (Deep Learning Accelerator) core to use for inference.
        /// </summary>
        /// <param name="dlaCore">要使用的DLA核心索引。使用-1表示不使用DLA。/ The index of the DLA core to use. Use -1 to not use DLA.</param>
        public void setDLACore(int dlaCore)
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_setDLACore(ptr, dlaCore));
        }

        /// <summary>
        /// 获取当前设置的用于推理的DLA核心索引。
        /// Gets the index of the DLA core currently set for inference.
        /// </summary>
        /// <returns>DLA核心索引。如果未设置DLA，则为-1。/ The DLA core index. -1 if DLA is not set.</returns>
        public int getDLACore()
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_getDLACore(ptr, out int coreNum));
            return coreNum;
        }

        /// <summary>
        /// 获取系统中可用的DLA核心总数。
        /// Gets the total number of DLA cores available on the system.
        /// </summary>
        /// <returns>可用的DLA核心数量。/ The number of available DLA cores.</returns>
        public int getNbDLACores()
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_getNbDLACores(ptr, out int coresNum));
            return coresNum;
        }

        /// <summary>
        /// 设置一个自定义的GPU内存分配器。
        /// Sets a custom GPU memory allocator.
        /// </summary>
        /// <param name="allocator">一个实现了 GpuAllocator 接口的对象。/ An object that implements the GpuAllocator interface.</param>
        public void setGpuAllocator(GpuAllocator allocator)
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_setGpuAllocator(ptr, allocator.TrtPtr));
        }

        /// <summary>
        /// 从一个包含序列化引擎数据的字节数组（内存块）反序列化创建一个 CudaEngine。
        /// Deserializes and creates a CudaEngine from a byte array (memory block) containing the serialized engine data.
        /// </summary>
        /// <param name="blob">包含序列化引擎的字节数组。/ A byte array containing the serialized engine.</param>
        /// <param name="size">内存块的大小。/ The size of the memory block.</param>
        /// <returns>一个新创建的 CudaEngine 实例。/ A newly created CudaEngine instance.</returns>
        public CudaEngine deserializeCudaEngineByBlob(byte[] blob, ulong size)
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_deserializeCudaEngineByBlob(
                ptr, ref blob[0], size, out IntPtr cudaEngine));
            return new CudaEngine(cudaEngine);
        }

        /// <summary>
        /// 从一个 FileStreamReader 反序列化创建一个 CudaEngine。
        /// Deserializes and creates a CudaEngine from a FileStreamReader.
        /// </summary>
        /// <param name="streamReader">指向序列化引擎文件的文件流读取器。/ A file stream reader pointing to the serialized engine file.</param>
        /// <returns>一个新创建的 CudaEngine 实例。/ A newly created CudaEngine instance.</returns>
        public CudaEngine deserializeCudaEngineByFileStreamReader(FileStreamReader streamReader)
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_deserializeCudaEngineByFileStreamReader(
                ptr, streamReader.TrtPtr, out IntPtr cudaEngine));
            return new CudaEngine(cudaEngine);
        }

        /// <summary>
        /// 从一个 AsyncStreamReader 反序列化创建一个 CudaEngine。
        /// Deserializes and creates a CudaEngine from an AsyncStreamReader.
        /// </summary>
        /// <param name="streamReader">指向序列化引擎文件的异步流读取器。/ An async stream reader pointing to the serialized engine file.</param>
        /// <returns>一个新创建的 CudaEngine 实例。/ A newly created CudaEngine instance.</returns>
        public CudaEngine deserializeCudaEngineByAsyncStreamReader(AsyncStreamReader streamReader)
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_deserializeCudaEngineByAsyncStreamReader(
                ptr, streamReader.TrtPtr, out IntPtr cudaEngine));
            return new CudaEngine(cudaEngine);
        }

        /// <summary>
        /// 设置运行时可以使用的最大线程数。
        /// Sets the maximum number of threads that the runtime can use.
        /// </summary>
        /// <param name="maxThreads">最大线程数。/ The maximum number of threads.</param>
        public void setMaxThreads(int maxThreads)
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_setMaxThreads(ptr, maxThreads));
        }

        /// <summary>
        /// 获取运行时可以使用的最大线程数。
        /// Gets the maximum number of threads that the runtime can use.
        /// </summary>
        /// <returns>最大线程数。/ The maximum number of threads.</returns>
        public int getMaxThreads()
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_getMaxThreads(ptr, out int maxThreads));
            return maxThreads;
        }

        /// <summary>
        /// 设置用于存储临时文件的目录。
        /// Sets the directory to use for storing temporary files.
        /// </summary>
        /// <param name="path">临时目录的路径。/ The path to the temporary directory.</param>
        /// <exception cref="TrtException">如果指定的路径不存在。/ If the specified path does not exist.</exception>
        public void setTemporaryDirectory(string path)
        {
            if (!Directory.Exists(path))
                throw new TrtException($"The specified path does not exist: {path}");
            TrtHandleException.handler(NativeMethods.trtRuntime_setTemporaryDirectory(ptr, path));
        }

        /// <summary>
        /// 获取当前设置的临时文件目录路径。
        /// Gets the currently set path for the temporary files directory.
        /// </summary>
        /// <returns>临时目录的路径。/ The path of the temporary directory.</returns>
        public string getTemporaryDirectory()
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_getTemporaryDirectory(ptr, out IntPtr pathPtr));
            string path = Marshal.PtrToStringUni(pathPtr);
            Marshal.FreeHGlobal(pathPtr);
            return path;
        }

        /// <summary>
        /// 设置控制临时文件使用的标志。
        /// Sets flags controlling the use of temporary files.
        /// </summary>
        /// <param name="tempfileControlFlag">临时文件控制标志。/ The temporary file control flag.</param>
        public void setTempfileControlFlags(TrtTempfileControlFlag tempfileControlFlag)
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_setTempfileControlFlags(ptr, tempfileControlFlag));
        }

        /// <summary>
        /// 获取当前控制临时文件使用的标志。
        /// Gets the current flags controlling the use of temporary files.
        /// </summary>
        /// <returns>当前的临时文件控制标志。/ The current temporary file control flag.</returns>
        public TrtTempfileControlFlag getTempfileControlFlags()
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_getTempfileControlFlags(ptr, out TrtTempfileControlFlag tempfileControlFlag));
            return tempfileControlFlag;
        }

        /// <summary>
        /// 获取插件注册表，用于查找和创建插件。
        /// Gets the plugin registry, used for looking up and creating plugins.
        /// </summary>
        /// <returns>一个 PluginRegistry 实例。/ An instance of PluginRegistry.</returns>
        public PluginRegistry getPluginRegistry()
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_getPluginRegistry(ptr, out IntPtr pluginRegistryPtr));
            return new PluginRegistry(pluginRegistryPtr);
        }

        /// <summary>
        /// 从指定的库文件加载一个运行时插件。
        /// Loads a runtime plugin from the specified library file.
        /// </summary>
        /// <param name="path">包含插件的共享库文件路径。/ The file path to the shared library containing the plugin.</param>
        /// <returns>一个代表新加载运行时的 Runtime 实例。/ A Runtime instance representing the newly loaded runtime.</returns>
        public Runtime loadRuntime(string path)
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_loadRuntime(
                ptr, path, out IntPtr runtimePtr));
            return new Runtime(runtimePtr);
        }

        /// <summary>
        /// 设置是否允许引擎包含主机端代码（如Tactic源代码）。
        /// Sets whether engines are allowed to contain host-side code (e.g., Tactic source code).
        /// </summary>
        /// <param name="allowed">如果为 true，则允许包含主机端代码。/ If true, allows inclusion of host-side code.</param>
        public void setEngineHostCodeAllowed(bool allowed)
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_setEngineHostCodeAllowed(
                ptr, allowed ? 1 : 0));
        }

        /// <summary>
        /// 获取是否允许引擎包含主机端代码。
        /// Gets whether engines are allowed to contain host-side code.
        /// </summary>
        /// <returns>如果允许，则为 true；否则为 false。/ True if allowed, otherwise false.</returns>
        public bool getEngineHostCodeAllowed()
        {
            TrtHandleException.handler(NativeMethods.trtRuntime_getEngineHostCodeAllowed(
                ptr, out int allowed));
            return allowed != 0;
        }

        /// <summary>
        /// 获取与此运行时关联的 Logger
        /// Gets the logger associated with this runtime
        /// </summary>
        /// <returns>Logger 实例 / Logger instance</returns>
        public Logger getLogger()
        {
            // Runtime 使用全局 Logger 实例，返回单例
            return Logger.Instance;
        }
    }

}

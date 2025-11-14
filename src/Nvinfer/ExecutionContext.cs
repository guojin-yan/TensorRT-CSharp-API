using JYPPX.TensorRtSharp.Cuda;
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
using System.Xml.Linq;


namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 执行上下文类，用于执行TensorRT引擎推理
    /// Execution context class for executing TensorRT engine inference
    /// </summary>
    /// <remarks>
    /// 继承自DisposableTrtObject，实现了IDisposable接口，可以安全地释放资源
    /// Inherits from DisposableTrtObject and implements IDisposable interface for safe resource disposal
    /// </remarks>
    public class ExecutionContext : DisposableTrtObject
    {
        /// <summary>
        /// 创建空的执行上下文对象
        /// Creates empty ExecutionContext
        /// </summary>
        public ExecutionContext()
        {
            // InitHandleException.handler已注释，表示此构造函数未实现
            // InitHandleException.handler(
            //    NativeMethods.trtBuild_createInferBuilder(out ptr));
        }

        /// <summary>
        /// 从原生指针创建执行上下文对象
        /// Creates from native pointer
        /// </summary>
        /// <param name="ptr">原生对象指针，Native object pointer</param>
        internal ExecutionContext(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
                throw new TrtException("Native object address is NULL");
            this.ptr = ptr;
        }

        /// <summary>
        /// 释放资源
        /// Releases the resources
        /// </summary>
        public void Release()
        {
            Dispose();
        }

        /// <summary>
        /// 释放非托管资源
        /// Releases unmanaged resources
        /// </summary>
        /// <inheritdoc />
        protected override void DisposeUnmanaged()
        {
            if (ptr != IntPtr.Zero && IsEnabledDispose)
                NativeMethods.trtExecutionContext_free(ptr);
            base.DisposeUnmanaged();
        }

        /// <summary>
        /// 获取关联的引擎
        /// Gets the associated engine
        /// </summary>
        /// <returns>CUDA引擎实例，CUDA engine instance</returns>
        public CudaEngine getEngine()
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getEngine(
                ptr, out IntPtr enginePtr));
            return new CudaEngine(enginePtr);
        }

        /// <summary>
        /// 设置执行上下文的名称
        /// Sets the name of the execution context
        /// </summary>
        /// <param name="name">上下文名称，Context name</param>
        public void setName(string name)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setName(
                ptr, name));
        }

        /// <summary>
        /// 获取执行上下文的名称
        /// Gets the name of the execution context
        /// </summary>
        /// <returns>上下文名称字符串，Context name string</returns>
        public string getName()
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getName(
                ptr, out IntPtr namePtr));
            return Marshal.PtrToStringAnsi(namePtr) ?? string.Empty;
        }

        /// <summary>
        /// 设置设备内存
        /// Sets the device memory
        /// </summary>
        /// <param name="memory">设备内存指针，Device memory pointer</param>
        public void setDeviceMemory(IntPtr memory)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setDeviceMemory(
                ptr, memory));
        }

        /// <summary>
        /// 设置设备内存（版本2）
        /// Sets the device memory (version 2)
        /// </summary>
        /// <param name="memory">设备内存指针，Device memory pointer</param>
        /// <param name="size">内存大小，Memory size</param>
        public void setDeviceMemoryV2(IntPtr memory, long size)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setDeviceMemoryV2(
                ptr, memory, size));
        }

        /// <summary>
        /// 获取张量的步长（stride）
        /// Gets the tensor strides
        /// </summary>
        /// <param name="name">张量名称，Tensor name</param>
        /// <returns>张量步长对象，Tensor stride dimensions object</returns>
        public Dims getTensorStrides(string name)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getTensorStrides(
                ptr, name, out Dims dims));
            return dims;
        }

        /// <summary>
        /// 获取当前优化配置文件的索引
        /// Gets the index of the current optimization profile
        /// </summary>
        /// <returns>优化配置文件索引，Optimization profile index</returns>
        public int getOptimizationProfile()
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getOptimizationProfile(
                ptr, out int profile));
            return profile;
        }

        /// <summary>
        /// 设置输入张量的形状
        /// Sets the shape of an input tensor
        /// </summary>
        /// <param name="name">张量名称，Tensor name</param>
        /// <param name="dims">张量尺寸，Tensor dimensions</param>
        public void setinputShape(string name, Dims dims)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setInputShape(
                ptr, name, dims));
        }

        /// <summary>
        /// 获取张量的形状
        /// Gets the shape of a tensor
        /// </summary>
        /// <param name="name">张量名称，Tensor name</param>
        /// <returns>张量尺寸对象，Tensor dimensions object</returns>
        public Dims getTensorShape(string name)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getTensorShape(
                ptr, name, out Dims dims));
            return dims;
        }

        /// <summary>
        /// 执行推理（版本2）
        /// Executes inference (version 2) using bindings array
        /// </summary>
        /// <param name="bindings">绑定的设备内存指针数组，Array of bound device memory pointers</param>
        public void executeV2(IntPtr[] bindings)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_executeV2(
                ptr, ref bindings[0]));
        }

        /// <summary>
        /// 设置张量的地址
        /// Sets the address for a tensor
        /// </summary>
        /// <param name="name">张量名称，Tensor name</param>
        /// <param name="tensorAddress">张量设备内存地址，Tensor device memory address</param>
        public void setTensorAddress(string name, IntPtr tensorAddress)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setTensorAddress(
                ptr, name, tensorAddress));
        }

        /// <summary>
        /// 获取张量的地址
        /// Gets the address for a tensor
        /// </summary>
        /// <param name="name">张量名称，Tensor name</param>
        /// <returns>张量设备内存地址，Tensor device memory address</returns>
        public IntPtr getTensorAddress(string name)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getTensorAddress(
                ptr, name, out IntPtr tensorAddress));
            return tensorAddress;
        }

        /// <summary>
        /// 设置输出张量的地址
        /// Sets the address for an output tensor
        /// </summary>
        /// <param name="tensorName">张量名称，Tensor name</param>
        /// <param name="tensorAddress">张量设备内存地址，Tensor device memory address</param>
        public void setOutputTensorAddress(string tensorName, IntPtr tensorAddress)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setOutputTensorAddress(
                ptr, tensorName, tensorAddress));
        }

        /// <summary>
        /// 设置输入张量的地址
        /// Sets the address for an input tensor
        /// </summary>
        /// <param name="tensorName">张量名称，Tensor name</param>
        /// <param name="tensorAddress">张量设备内存地址，Tensor device memory address</param>
        public void setInputTensorAddress(string tensorName, IntPtr tensorAddress)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setInputTensorAddress(
                ptr, tensorName, tensorAddress));
        }

        /// <summary>
        /// 获取输出张量的地址
        /// Gets the address for an output tensor
        /// </summary>
        /// <param name="tensorName">张量名称，Tensor name</param>
        /// <returns>张量设备内存地址，Tensor device memory address</returns>
        public IntPtr getOutputTensorAddress(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getOutputTensorAddress(
                ptr, tensorName, out IntPtr tensorAddress));
            return tensorAddress;
        }

        /// <summary>
        /// 获取指定输出张量的最大可能大小
        /// Gets the maximum possible size for the specified output tensor
        /// </summary>
        /// <param name="name">张量名称，Tensor name</param>
        /// <returns>最大输出大小（字节），Maximum output size (in bytes)</returns>
        public long getMaxOutputSize(string name)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getMaxOutputSize(
                ptr, name, out long maxOutputSize));
            return maxOutputSize;
        }

        /// <summary>
        /// 在指定流上执行推理（版本3）
        /// Executes inference on the specified stream (version 3)
        /// </summary>
        /// <param name="stream">CUDA流实例，CUDA stream instance</param>
        public void executeV3(CudaStream stream)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_executeV3(
                ptr, stream.TrtPtr));
        }

        /// <summary>
        /// 设置同步调试模式
        /// Sets synchronous debug mode
        /// </summary>
        /// <param name="sync">是否开启调试同步，Whether to enable debug sync</param>
        public void setDebugSync(bool sync)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setDebugSync(
                ptr, sync ? 1 : 0));
        }

        /// <summary>
        /// 获取同步调试模式状态
        /// Gets the synchronous debug mode state
        /// </summary>
        /// <returns>如果开启调试同步返回true，否则返回false，Returns true if debug sync is enabled, false otherwise</returns>
        public bool getDebugSync()
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getDebugSync(
                ptr, out int sync));
            return sync != 0;
        }

        /// <summary>
        /// 设置性能分析器
        /// Sets the profiler
        /// </summary>
        /// <param name="profiler">性能分析器实例，Profiler instance</param>
        public void setProfiler(Profiler profiler)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setProfiler(
                ptr, profiler.TrtPtr));
        }

        /// <summary>
        /// 获取性能分析器
        /// Gets the profiler
        /// </summary>
        /// <returns>性能分析器实例，Profiler instance</returns>
        public Profiler getProfiler()
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getProfiler(
                ptr, out IntPtr profilerPtr));
            return new Profiler(profilerPtr);
        }

        /// <summary>
        /// 检查是否已指定所有输入维度
        /// Checks if all input dimensions have been specified
        /// </summary>
        /// <returns>如果已全部指定返回true，否则返回false，Returns true if all specified, false otherwise</returns>
        public bool allInputDimensionsSpecified()
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_allInputDimensionsSpecified(
                ptr, out int specified));
            return specified != 0;
        }

        /// <summary>
        /// 检查是否已指定所有输入形状
        /// Checks if all input shapes have been specified
        /// </summary>
        /// <returns>如果已全部指定返回true，否则返回false，Returns true if all specified, false otherwise</returns>
        public bool allInputShapesSpecified()
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_allInputShapesSpecified(
                ptr, out int specified));
            return specified != 0;
        }

        /// <summary>
        /// 异步设置优化配置文件
        /// Asynchronously sets the optimization profile
        /// </summary>
        /// <param name="profileIndex">配置文件索引，Profile index</param>
        /// <param name="stream">CUDA流实例，CUDA stream instance</param>
        /// <returns>设置成功返回true，否则返回false，Returns true if set successfully, false otherwise</returns>
        public bool setOptimizationProfileAsync(int profileIndex, CudaStream stream)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setOptimizationProfileAsync(
                ptr, profileIndex, stream.TrtPtr, out int success));
            return success != 0;
        }

        /// <summary>
        /// 设置入队是否输出性能分析信息
        /// Sets whether enqueue emits profile information
        /// </summary>
        /// <param name="emitProfile">是否输出性能信息，Whether to emit profile info</param>
        public void setEnqueueEmitsProfile(bool emitProfile)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setEnqueueEmitsProfile(
                ptr, emitProfile ? 1 : 0));
        }

        /// <summary>
        /// 获取入队是否输出性能分析信息的状态
        /// Gets the state of whether enqueue emits profile information
        /// </summary>
        /// <returns>如果输出返回true，否则返回false，Returns true if emits, false otherwise</returns>
        public bool getEnqueueEmitsProfile()
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getEnqueueEmitsProfile(
                ptr, out int emitProfile));
            return emitProfile != 0;
        }

        /// <summary>
        /// 向性能分析器报告
        /// Reports to the profiler
        /// </summary>
        /// <returns>报告成功返回true，否则返回false，Returns true if report was successful, false otherwise</returns>
        public bool reportToProfiler()
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_reportToProfiler(
                ptr, out int success));
            return success != 0;
        }

        /// <summary>
        /// 推断形状
        /// Infers shapes
        /// </summary>
        /// <param name="tensorNames">需要推断形状的张量名称数组，Array of tensor names to infer shapes for</param>
        /// <returns>推断结果代码，Inference result code</returns>
        public int inferShapes(string[] tensorNames)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_inferShapes(
                ptr, tensorNames.Length, tensorNames, out int result));
            return result;
        }

        /// <summary>
        /// 更新当前形状所需的设备内存大小
        /// Updates the device memory size required for the current shapes
        /// </summary>
        /// <returns>设备内存大小（字节），Device memory size (in bytes)</returns>
        public ulong updateDeviceMemorySizeForShapes()
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_updateDeviceMemorySizeForShapes(
                ptr, out ulong size));
            return size;
        }

        /// <summary>
        /// 设置输入消耗事件
        /// Sets the input consumed event
        /// </summary>
        /// <param name="cudaEvent">CUDA事件实例，CUDA event instance</param>
        /// <returns>设置成功返回true，否则返回false，Returns true if set successfully, false otherwise</returns>
        public bool setInputConsumedEvent(CudaEvent cudaEvent)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setInputConsumedEvent(
                ptr, cudaEvent.TrtPtr, out int success));
            return success != 0;
        }

        /// <summary>
        /// 获取输入消耗事件
        /// Gets the input consumed event
        /// </summary>
        /// <param name="cudaEvent">（参数未使用）CUDA事件实例，(Parameter unused) CUDA event instance</param>
        /// <returns>一个新的CUDA事件实例，A new CUDA event instance</returns>
        public CudaEvent getInputConsumedEvent(CudaEvent cudaEvent)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getInputConsumedEvent(
                ptr, out IntPtr outptr));
            return new CudaEvent(outptr);
        }

        /// <summary>
        /// 设置输出分配器
        /// Sets the output allocator for a specific tensor
        /// </summary>
        /// <param name="tensorName">张量名称，Tensor name</param>
        /// <param name="outputAllocator">输出分配器实例，Output allocator instance</param>
        /// <returns>设置成功返回true，否则返回false，Returns true if set successfully, false otherwise</returns>
        public bool setOutputAllocator(string tensorName, OutputAllocator outputAllocator)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setOutputAllocator(
                ptr, tensorName, outputAllocator.TrtPtr, out int success));
            return success != 0;
        }

        /// <summary>
        /// 获取输出分配器
        /// Gets the output allocator for a specific tensor
        /// </summary>
        /// <param name="tensorName">张量名称，Tensor name</param>
        /// <returns>输出分配器实例，Output allocator instance</returns>
        public OutputAllocator getOutputAllocator(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getOutputAllocator(
                ptr, tensorName, out IntPtr outptr));
            return new OutputAllocator(outptr);
        }

        /// <summary>
        /// 设置临时存储分配器
        /// Sets the temporary storage allocator
        /// </summary>
        /// <param name="allocator">GPU分配器实例，GPU allocator instance</param>
        /// <returns>设置成功返回true，否则返回false，Returns true if set successfully, false otherwise</returns>
        public bool setTemporaryStorageAllocator(GpuAllocator allocator)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setTemporaryStorageAllocator(
                ptr, allocator.TrtPtr, out int success));
            return success != 0;
        }

        /// <summary>
        /// 获取临时存储分配器
        /// Gets the temporary storage allocator
        /// </summary>
        /// <returns>GPU分配器实例，GPU allocator instance</returns>
        public GpuAllocator getTemporaryStorageAllocator()
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getTemporaryStorageAllocator(
                ptr, out IntPtr outptr));
            return new GpuAllocator(outptr);
        }

        /// <summary>
        /// 设置持久缓存限制
        /// Sets the persistent cache limit
        /// </summary>
        /// <param name="size">缓存大小限制（字节），Cache size limit (in bytes)</param>
        /// <returns>总是返回true，Always returns true</returns>
        public bool setPersistentCacheLimit(ulong size)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setPersistentCacheLimit(
                ptr, size));
            return true;
        }

        /// <summary>
        /// 获取持久缓存限制
        /// Gets the persistent cache limit
        /// </summary>
        /// <returns>当前缓存大小限制（字节），Current cache size limit (in bytes)</returns>
        public ulong getPersistentCacheLimit()
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getPersistentCacheLimit(
                ptr, out ulong size));
            return size;
        }

        /// <summary>
        /// 设置NVTX详细程度
        /// Sets the NVTX verbosity
        /// </summary>
        /// <param name="verbosity">性能分析详细程度枚举，Profiling verbosity enum</param>
        /// <returns>设置成功返回true，否则返回false，Returns true if set successfully, false otherwise</returns>
        public bool setNvtxVerbosity(TrtProfilingVerbosity verbosity)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setNvtxVerbosity(
                ptr, verbosity, out int success));
            return success != 0;
        }

        /// <summary>
        /// 获取NVTX详细程度
        /// Gets the NVTX verbosity
        /// </summary>
        /// <returns>性能分析详细程度枚举，Profiling verbosity enum</returns>
        public TrtProfilingVerbosity getNvtxVerbosity()
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getNvtxVerbosity(
                ptr, out TrtProfilingVerbosity verbosity));
            return verbosity;
        }

        /// <summary>
        /// 设置辅助流
        /// Sets the auxiliary streams
        /// </summary>
        /// <param name="auxStreams">CUDA流数组，CUDA stream array</param>
        /// <param name="nbStreams">流数量，Number of streams</param>
        public void setAuxStreams(CudaStream auxStreams, int nbStreams)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setAuxStreams(
                ptr, auxStreams.TrtPtr, nbStreams));
        }

        /// <summary>
        /// 设置调试监听器
        /// Sets the debug listener
        /// </summary>
        /// <param name="debugListener">调试监听器实例，Debug listener instance</param>
        /// <returns>设置成功返回true，否则返回false，Returns true if set successfully, false otherwise</returns>
        public bool setDebugListener(DebugListener debugListener)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setDebugListener(
                ptr, debugListener.TrtPtr, out int success));
            return success != 0;
        }

        /// <summary>
        /// 获取调试监听器
        /// Gets the debug listener
        /// </summary>
        /// <returns>调试监听器实例，Debug listener instance</returns>
        public DebugListener getDebugListener()
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getDebugListener(
                ptr, out IntPtr listener));
            return new DebugListener(listener);
        }

        /// <summary>
        /// 设置指定张量的调试状态
        /// Sets the debug state for a specific tensor
        /// </summary>
        /// <param name="name">张量名称，Tensor name</param>
        /// <param name="flag">调试标志，Debug flag</param>
        /// <returns>设置成功返回true，否则返回false，Returns true if set successfully, false otherwise</returns>
        public bool setTensorDebugState(string name, bool flag)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setTensorDebugState(
                ptr, name, flag ? 1 : 0, out int success));
            return success != 0;
        }

        /// <summary>
        /// 获取指定张量的调试状态
        /// Gets the debug state for a specific tensor
        /// </summary>
        /// <param name="name">张量名称，Tensor name</param>
        /// <returns>调试状态，Debug state</returns>
        public bool getDebugState(string name)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getDebugState(
                ptr, name, out int success));
            return success != 0;
        }

        /// <summary>
        /// 获取运行时配置
        /// Gets the runtime configuration
        /// </summary>
        /// <returns>运行时配置实例，Runtime configuration instance</returns>
        public RuntimeConfig getRuntimeConfig()
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_getRuntimeConfig(
                ptr, out IntPtr outptr));
            return new RuntimeConfig(outptr);
        }

        /// <summary>
        /// 设置所有张量的调试状态
        /// Sets the debug state for all tensors
        /// </summary>
        /// <param name="flag">调试标志，Debug flag</param>
        /// <returns>设置成功返回true，否则返回false，Returns true if set successfully, false otherwise</returns>
        public bool setAllTensorsDebugState(bool flag)
        {
            TrtHandleException.handler(NativeMethods.trtExecutionContext_setAllTensorsDebugState(
                ptr, flag ? 1 : 0, out int success));
            return success != 0;
        }
    }

}
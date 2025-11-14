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
    /// <summary>
    /// CUDA引擎类，用于执行TensorRT模型推理
    /// CUDA engine class for executing TensorRT model inference
    /// </summary>
    /// <remarks>
    /// 继承自DisposableTrtObject，实现了IDisposable接口，可以安全地释放资源
    /// Inherits from DisposableTrtObject and implements IDisposable interface for safe resource disposal
    /// </remarks>
    public class CudaEngine : DisposableTrtObject
    {
        /// <summary>
        /// 创建空的CUDA引擎对象
        /// Creates empty CudaEngine
        /// </summary>
        public CudaEngine()
        {
            // InitHandleException.handler已注释，表示此构造函数未实现
            // InitHandleException.handler(
            //    NativeMethods.trtBuild_createInferBuilder(out ptr));
        }

        /// <summary>
        /// 从原生指针创建CUDA引擎对象
        /// Creates from native pointer
        /// </summary>
        /// <param name="ptr">原生对象指针，Native object pointer</param>
        internal CudaEngine(IntPtr ptr)
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
                NativeMethods.trtCudaEngine_free(ptr);
            base.DisposeUnmanaged();
        }

        /// <summary>
        /// 获取张量的形状
        /// Gets the tensor shape
        /// </summary>
        /// <param name="tensorName">张量名称，Tensor name</param>
        /// <returns>张量尺寸对象，Tensor dimensions object</returns>
        public Dims getTensorShape(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorShape(
                ptr, tensorName, out Dims dims));
            return dims;
        }

        /// <summary>
        /// 获取张量的数据类型
        /// Gets the tensor data type
        /// </summary>
        /// <param name="tensorName">张量名称，Tensor name</param>
        /// <returns>张量数据类型枚举，Tensor data type enum</returns>
        public TrtDataType getTensorDataType(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorDataType(
                ptr, tensorName, out TrtDataType dataType));
            return dataType;
        }

        /// <summary>
        /// 获取张量的位置（主机或设备）
        /// Gets the tensor location (host or device)
        /// </summary>
        /// <param name="tensorName">张量名称，Tensor name</param>
        /// <returns>张量位置枚举，Tensor location enum</returns>
        public TrtTensorLocation getTensorLocation(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorLocation(
                ptr, tensorName, out TrtTensorLocation location));
            return location;
        }

        /// <summary>
        /// 检查张量是否用于形状推断
        /// Checks if tensor is used for shape inference
        /// </summary>
        /// <param name="tensorName">张量名称，Tensor name</param>
        /// <returns>如果用于形状推断返回true，否则返回false，Returns true if used for shape inference, false otherwise</returns>
        public bool isShapeInferenceIO(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_isShapeInferenceIO(
                ptr, tensorName, out int isShapeIO));
            return isShapeIO != 0;
        }

        /// <summary>
        /// 获取张量的IO模式
        /// Gets the tensor IO mode
        /// </summary>
        /// <param name="tensorName">张量名称，Tensor name</param>
        /// <returns>张量IO模式枚举，Tensor IO mode enum</returns>
        public TrtTensorIOMode getTensorIOMode(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorIOMode(
                ptr, tensorName, out TrtTensorIOMode ioMode));
            return ioMode;
        }

        /// <summary>
        /// 获取张量每个组件的字节数
        /// Gets the bytes per component of the tensor
        /// </summary>
        /// <param name="tensorName">张量名称，Tensor name</param>
        /// <returns>每个组件的字节数，Bytes per component</returns>
        public int getTensorBytesPerComponent(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorBytesPerComponent(
                ptr, tensorName, out int bytesPerComponent));
            return bytesPerComponent;
        }

        /// <summary>
        /// 获取指定配置文件中张量每个组件的字节数
        /// Gets the bytes per component of the tensor for a specific profile
        /// </summary>
        /// <param name="tensorName">张量名称，Tensor name</param>
        /// <param name="profileIndex">配置文件索引，Profile index</param>
        /// <returns>每个组件的字节数，Bytes per component</returns>
        public int getTensorBytesPerComponent(string tensorName, int profileIndex)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorBytesPerComponent_ForProfile(
                ptr, tensorName, profileIndex, out int bytesPerComponent));
            return bytesPerComponent;
        }

        /// <summary>
        /// 获取张量每个元素的组件数
        /// Gets the components per element of the tensor
        /// </summary>
        /// <param name="tensorName">张量名称，Tensor name</param>
        /// <returns>每个元素的组件数，Components per element</returns>
        public int getTensorComponentsPerElement(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorComponentsPerElement(
                ptr, tensorName, out int componentsPerElement));
            return componentsPerElement;
        }

        /// <summary>
        /// 获取指定配置文件中张量每个元素的组件数
        /// Gets the components per element of the tensor for a specific profile
        /// </summary>
        /// <param name="tensorName">张量名称，Tensor name</param>
        /// <param name="profileIndex">配置文件索引，Profile index</param>
        /// <returns>每个元素的组件数，Components per element</returns>
        public int getTensorComponentsPerElement(string tensorName, int profileIndex)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorComponentsPerElement_ForProfile(
                ptr, tensorName, profileIndex, out int componentsPerElement));
            return componentsPerElement;
        }

        /// <summary>
        /// 获取张量的格式
        /// Gets the tensor format
        /// </summary>
        /// <param name="tensorName">张量名称，Tensor name</param>
        /// <returns>张量格式枚举，Tensor format enum</returns>
        public TrtTensorFormat getTensorFormat(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorFormat(
                ptr, tensorName, out TrtTensorFormat format));
            return format;
        }

        /// <summary>
        /// 获取指定配置文件中张量的格式
        /// Gets the tensor format for a specific profile
        /// </summary>
        /// <param name="tensorName">张量名称，Tensor name</param>
        /// <param name="profileIndex">配置文件索引，Profile index</param>
        /// <returns>张量格式枚举，Tensor format enum</returns>
        public TrtTensorFormat getTensorFormat(string tensorName, int profileIndex)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorFormat_ForProfile(
                ptr, tensorName, profileIndex, out TrtTensorFormat format));
            return format;
        }

        /// <summary>
        /// 获取张量格式描述
        /// Gets the tensor format description
        /// </summary>
        /// <param name="tensorName">张量名称，Tensor name</param>
        /// <returns>张量格式描述字符串，Tensor format description string</returns>
        public string getTensorFormatDesc(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorFormatDesc(
                ptr, tensorName, out IntPtr formatDescPtr));
            return Marshal.PtrToStringAnsi(formatDescPtr);
        }

        /// <summary>
        /// 获取指定配置文件中张量格式描述
        /// Gets the tensor format description for a specific profile
        /// </summary>
        /// <param name="tensorName">张量名称，Tensor name</param>
        /// <param name="profileIndex">配置文件索引，Profile index</param>
        /// <returns>张量格式描述字符串，Tensor format description string</returns>
        public string getTensorFormatDesc(string tensorName, int profileIndex)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorFormatDescByProfileIndex(
                ptr, tensorName, profileIndex, out IntPtr formatDescPtr));
            return Marshal.PtrToStringAnsi(formatDescPtr);
        }

        /// <summary>
        /// 获取张量向量化的维度
        /// Gets the vectorized dimension of the tensor
        /// </summary>
        /// <param name="tensorName">张量名称，Tensor name</param>
        /// <returns>向量化的维度，Vectorized dimension</returns>
        public int getTensorVectorizedDim(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorVectorizedDim(
                ptr, tensorName, out int vectorizedDim));
            return vectorizedDim;
        }

        /// <summary>
        /// 获取指定配置文件中张量向量化的维度
        /// Gets the vectorized dimension of the tensor for a specific profile
        /// </summary>
        /// <param name="tensorName">张量名称，Tensor name</param>
        /// <param name="profileIndex">配置文件索引，Profile index</param>
        /// <returns>向量化的维度，Vectorized dimension</returns>
        public int getTensorVectorizedDim(string tensorName, int profileIndex)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTensorVectorizedDim_ForProfile(
                ptr, tensorName, profileIndex, out int vectorizedDim));
            return vectorizedDim;
        }

        /// <summary>
        /// 检查张量是否为调试张量
        /// Checks if tensor is a debug tensor
        /// </summary>
        /// <param name="tensorName">张量名称，Tensor name</param>
        /// <returns>如果为调试张量返回true，否则返回false，Returns true if debug tensor, false otherwise</returns>
        public bool isDebugTensor(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_isDebugTensor(
                ptr, tensorName, out int isDebug));
            return isDebug != 0;
        }

        /// <summary>
        /// 获取层的数量
        /// Gets the number of layers
        /// </summary>
        /// <returns>层的数量，Number of layers</returns>
        public int getNbLayers()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getNbLayers(
                ptr, out int nbLayers));
            return nbLayers;
        }

        /// <summary>
        /// 获取引擎名称
        /// Gets the engine name
        /// </summary>
        /// <returns>引擎名称字符串，Engine name string</returns>
        public string getName()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getName(
                ptr, out IntPtr namePtr));
            return Marshal.PtrToStringAnsi(namePtr);
        }

        /// <summary>
        /// 获取优化配置文件的数量
        /// Gets the number of optimization profiles
        /// </summary>
        /// <returns>优化配置文件的数量，Number of optimization profiles</returns>
        public int getNbOptimizationProfiles()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getNbOptimizationProfiles(
                ptr, out int nbProfiles));
            return nbProfiles;
        }

        /// <summary>
        /// 获取引擎能力
        /// Gets the engine capability
        /// </summary>
        /// <returns>引擎能力枚举，Engine capability enum</returns>
        public TrtEngineCapability getEngineCapability()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getEngineCapability(
                ptr, out TrtEngineCapability capability));
            return capability;
        }

        /// <summary>
        /// 检查引擎是否可以重新拟合
        /// Checks if the engine is refittable
        /// </summary>
        /// <returns>如果可以重新拟合返回true，否则返回false，Returns true if refittable, false otherwise</returns>
        public bool isRefittable()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_isRefittable(
                ptr, out int refittable));
            return refittable != 0;
        }

        /// <summary>
        /// 获取IO张量的数量
        /// Gets the number of IO tensors
        /// </summary>
        /// <returns>IO张量的数量，Number of IO tensors</returns>
        public int getNbIOTensors()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getNbIOTensors(
                ptr, out int nbIOTensors));
            return nbIOTensors;
        }

        /// <summary>
        /// 获取指定索引的IO张量名称
        /// Gets the IO tensor name at the specified index
        /// </summary>
        /// <param name="index">张量索引，Tensor index</param>
        /// <returns>IO张量名称字符串，IO tensor name string</returns>
        public string getIOTensorName(int index)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getIOTensorName(
                ptr, index, out IntPtr tensorNamePtr));
            return Marshal.PtrToStringAnsi(tensorNamePtr);
        }

        /// <summary>
        /// 获取硬件兼容性级别
        /// Gets the hardware compatibility level
        /// </summary>
        /// <returns>硬件兼容性级别枚举，Hardware compatibility level enum</returns>
        public TrtHardwareCompatibilityLevel getHardwareCompatibilityLevel()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getHardwareCompatibilityLevel(
                ptr, out TrtHardwareCompatibilityLevel level));
            return level;
        }

        /// <summary>
        /// 获取辅助流的数量
        /// Gets the number of auxiliary streams
        /// </summary>
        /// <returns>辅助流的数量，Number of auxiliary streams</returns>
        public int getNbAuxStreams()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getNbAuxStreams(
                ptr, out int nbAuxStreams));
            return nbAuxStreams;
        }

        /// <summary>
        /// 获取策略源
        /// Gets the tactic sources
        /// </summary>
        /// <returns>策略源枚举，Tactic sources enum</returns>
        public TrtTacticSource getTacticSources()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getTacticSources(
                ptr, out TrtTacticSource sources));
            return sources;
        }

        /// <summary>
        /// 获取性能分析详细程度
        /// Gets the profiling verbosity
        /// </summary>
        /// <returns>性能分析详细程度枚举，Profiling verbosity enum</returns>
        public TrtProfilingVerbosity getProfilingVerbosity()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getProfilingVerbosity(
                ptr, out TrtProfilingVerbosity verbosity));
            return verbosity;
        }

        /// <summary>
        /// 序列化引擎
        /// Serializes the engine
        /// </summary>
        /// <returns>包含序列化引擎数据的主机内存实例，Host memory instance containing serialized engine data</returns>
        public HostMemory serialize()
        {
            IntPtr serializedEngine;
            TrtHandleException.handler(NativeMethods.trtCudaEngine_serialize(
                ptr, out serializedEngine));
            return new HostMemory(serializedEngine);
        }

        /// <summary>
        /// 创建序列化配置
        /// Creates a serialization configuration
        /// </summary>
        /// <returns>序列化配置实例，Serialization configuration instance</returns>
        public SerializationConfig createSerializationConfig()
        {
            IntPtr config;
            TrtHandleException.handler(NativeMethods.trtCudaEngine_createSerializationConfig(
                ptr, out config));
            return new SerializationConfig(config);
        }

        /// <summary>
        /// 使用配置序列化引擎
        /// Serializes the engine with configuration
        /// </summary>
        /// <param name="config">序列化配置实例，Serialization configuration instance</param>
        /// <returns>包含序列化引擎数据的主机内存实例，Host memory instance containing serialized engine data</returns>
        public HostMemory serializeWithConfig(SerializationConfig config)
        {
            IntPtr serializedEngine;
            TrtHandleException.handler(NativeMethods.trtCudaEngine_serializeWithConfig(
                ptr, config.TrtPtr, out serializedEngine));
            return new HostMemory(serializedEngine);
        }

        /// <summary>
        /// 创建执行上下文
        /// Creates an execution context
        /// </summary>
        /// <returns>执行上下文实例，Execution context instance</returns>
        public ExecutionContext createExecutionContext()
        {
            IntPtr context;
            TrtHandleException.handler(NativeMethods.trtCudaEngine_createExecutionContext(
                ptr, out context));
            return new ExecutionContext(context);
        }

        /// <summary>
        /// 使用分配策略创建执行上下文
        /// Creates an execution context with allocation strategy
        /// </summary>
        /// <param name="strategy">执行上下文分配策略，Execution context allocation strategy</param>
        /// <returns>执行上下文实例，Execution context instance</returns>
        public ExecutionContext createExecutionContext(TrtExecutionContextAllocationStrategy strategy)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_createExecutionContextByStrategy(
                ptr, strategy,
                out IntPtr context));
            return new ExecutionContext(context);
        }

        /// <summary>
        /// 创建没有设备内存的执行上下文
        /// Creates an execution context without device memory
        /// </summary>
        /// <returns>执行上下文实例，Execution context instance</returns>
        public ExecutionContext createExecutionContextWithoutDeviceMemory()
        {
            IntPtr context;
            TrtHandleException.handler(NativeMethods.trtCudaEngine_createExecutionContextWithoutDeviceMemory(
                ptr, out context));
            return new ExecutionContext(context);
        }

        /// <summary>
        /// 使用运行时配置创建执行上下文
        /// Creates an execution context with runtime configuration
        /// </summary>
        /// <param name="runtimeConfig">运行时配置实例，Runtime configuration instance</param>
        /// <returns>执行上下文实例，Execution context instance</returns>
        public ExecutionContext createExecutionContext(RuntimeConfig runtimeConfig)
        {
            IntPtr context;
            TrtHandleException.handler(NativeMethods.trtCudaEngine_createExecutionContextByRuntimeConfig(
                ptr, runtimeConfig.TrtPtr,
                out context));
            return new ExecutionContext(context);
        }

        /// <summary>
        /// 创建运行时配置
        /// Creates a runtime configuration
        /// </summary>
        /// <returns>运行时配置实例，Runtime configuration instance</returns>
        public RuntimeConfig createRuntimeConfig()
        {
            IntPtr config;
            TrtHandleException.handler(NativeMethods.trtCudaEngine_createRuntimeConfig(
                ptr, out config));
            return new RuntimeConfig(config);
        }

        /// <summary>
        /// 获取设备内存大小
        /// Gets the device memory size
        /// </summary>
        /// <returns>设备内存大小（字节），Device memory size (in bytes)</returns>
        public long getDeviceMemorySize()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getDeviceMemorySize(
                ptr, out long size));
            return size;
        }

        /// <summary>
        /// 获取指定配置文件中设备内存大小
        /// Gets the device memory size for a specific profile
        /// </summary>
        /// <param name="profileIndex">配置文件索引，Profile index</param>
        /// <returns>设备内存大小（字节），Device memory size (in bytes)</returns>
        public ulong getDeviceMemorySizeForProfile(int profileIndex)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getDeviceMemorySizeForProfile(
                ptr, profileIndex, out ulong size));
            return size;
        }

        /// <summary>
        /// 获取设备内存大小（版本2）
        /// Gets the device memory size (version 2)
        /// </summary>
        /// <returns>设备内存大小（字节），Device memory size (in bytes)</returns>
        public long getDeviceMemorySizeV2()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getDeviceMemorySizeV2(
                ptr, out long size));
            return size;
        }

        /// <summary>
        /// 获取指定配置文件中设备内存大小（版本2）
        /// Gets the device memory size for a specific profile (version 2)
        /// </summary>
        /// <param name="profileIndex">配置文件索引，Profile index</param>
        /// <returns>设备内存大小（字节），Device memory size (in bytes)</returns>
        public long getDeviceMemorySizeForProfileV2(int profileIndex)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getDeviceMemorySizeForProfileV2(
                ptr, profileIndex, out long size));
            return size;
        }

        /// <summary>
        /// 设置权重流式传输预算
        /// Sets the weight streaming budget
        /// </summary>
        /// <param name="gpuMemoryBudget">GPU内存预算，GPU memory budget</param>
        /// <returns>设置成功返回true，否则返回false，Returns true if set successfully, false otherwise</returns>
        public bool setWeightStreamingBudget(long gpuMemoryBudget)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_setWeightStreamingBudget(
                ptr, gpuMemoryBudget, out int success));
            return success != 0;
        }

        /// <summary>
        /// 获取权重流式传输预算
        /// Gets the weight streaming budget
        /// </summary>
        /// <returns>权重流式传输预算，Weight streaming budget</returns>
        public long getWeightStreamingBudget()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getWeightStreamingBudget(
                ptr, out long budget));
            return budget;
        }

        /// <summary>
        /// 获取最小权重流式传输预算
        /// Gets the minimum weight streaming budget
        /// </summary>
        /// <returns>最小权重流式传输预算，Minimum weight streaming budget</returns>
        public long getMinimumWeightStreamingBudget()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getMinimumWeightStreamingBudget(
                ptr, out long budget));
            return budget;
        }

        /// <summary>
        /// 获取可流式传输权重的大小
        /// Gets the size of streamable weights
        /// </summary>
        /// <returns>可流式传输权重的大小（字节），Size of streamable weights (in bytes)</returns>
        public long getStreamableWeightsSize()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getStreamableWeightsSize(
                ptr, out long size));
            return size;
        }

        /// <summary>
        /// 设置权重流式传输预算（版本2）
        /// Sets the weight streaming budget (version 2)
        /// </summary>
        /// <param name="gpuMemoryBudget">GPU内存预算，GPU memory budget</param>
        /// <returns>设置成功返回true，否则返回false，Returns true if set successfully, false otherwise</returns>
        public bool setWeightStreamingBudgetV2(long gpuMemoryBudget)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_setWeightStreamingBudgetV2(
                ptr, gpuMemoryBudget, out int success));
            return success != 0;
        }

        /// <summary>
        /// 获取权重流式传输预算（版本2）
        /// Gets the weight streaming budget (version 2)
        /// </summary>
        /// <returns>权重流式传输预算，Weight streaming budget</returns>
        public long getWeightStreamingBudgetV2()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getWeightStreamingBudgetV2(
                ptr, out long budget));
            return budget;
        }

        /// <summary>
        /// 获取权重流式传输自动预算
        /// Gets the weight streaming automatic budget
        /// </summary>
        /// <returns>自动预算值，Automatic budget value</returns>
        public long getWeightStreamingAutomaticBudget()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getWeightStreamingAutomaticBudget(
                ptr, out long budget));
            return budget;
        }

        /// <summary>
        /// 获取权重流式传输暂存内存大小
        /// Gets the weight streaming scratch memory size
        /// </summary>
        /// <returns>暂存内存大小（字节），Scratch memory size (in bytes)</returns>
        public long getWeightStreamingScratchMemorySize()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getWeightStreamingScratchMemorySize(
                ptr, out long size));
            return size;
        }

        /// <summary>
        /// 获取指定配置文件中的张量形状
        /// Gets the tensor shape for a specific profile
        /// </summary>
        /// <param name="tensorName">张量名称，Tensor name</param>
        /// <param name="profileIndex">配置文件索引，Profile index</param>
        /// <param name="select">优化配置选择器，Optimization profile selector</param>
        /// <returns>张量尺寸对象，Tensor dimensions object</returns>
        public Dims getProfileShape(string tensorName, int profileIndex, TrtOptProfileSelector select)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getProfileShape(
                ptr, tensorName, profileIndex, select, out Dims shape));
            return shape;
        }

        /// <summary>
        /// 获取指定配置文件中的张量值指针
        /// Gets the tensor values pointer for a specific profile
        /// </summary>
        /// <param name="tensorName">张量名称，Tensor name</param>
        /// <param name="profileIndex">配置文件索引，Profile index</param>
        /// <param name="select">优化配置选择器，Optimization profile selector</param>
        /// <returns>张量值指针，Tensor values pointer</returns>
        public IntPtr getProfileTensorValues(string tensorName, int profileIndex, TrtOptProfileSelector select)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getProfileTensorValues(
                ptr, tensorName, profileIndex, select, out IntPtr profileTensorValues));
            return profileTensorValues;
        }

        /// <summary>
        /// 获取指定配置文件中的张量值指针（版本2）
        /// Gets the tensor values pointer for a specific profile (version 2)
        /// </summary>
        /// <param name="tensorName">张量名称，Tensor name</param>
        /// <param name="profileIndex">配置文件索引，Profile index</param>
        /// <param name="select">优化配置选择器，Optimization profile selector</param>
        /// <returns>张量值指针，Tensor values pointer</returns>
        public IntPtr getProfileTensorValuesV2(string tensorName, int profileIndex, TrtOptProfileSelector select)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_getProfileTensorValuesV2(
                ptr, tensorName, profileIndex, select, out IntPtr profileTensorValuesV2));
            return profileTensorValuesV2;
        }

        /// <summary>
        /// 设置错误记录器
        /// Sets the error recorder
        /// </summary>
        /// <param name="recorder">错误记录器实例，Error recorder instance</param>
        public void setErrorRecorder(ErrorRecorder recorder)
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_setErrorRecorder(
                ptr, recorder.getHandle()));
        }

        /// <summary>
        /// 检查是否有隐含批次维度
        /// Checks if there is an implicit batch dimension
        /// </summary>
        /// <returns>如果有隐含批次维度返回true，否则返回false，Returns true if has implicit batch dimension, false otherwise</returns>
        public int hasImplicitBatchDimension()
        {
            TrtHandleException.handler(NativeMethods.trtCudaEngine_hasImplicitBatchDimension(
                ptr, out int hasImplicitBatch));
            return hasImplicitBatch;
        }

        /// <summary>
        /// 创建引擎检查器
        /// Creates an engine inspector
        /// </summary>
        /// <returns>引擎检查器实例，Engine inspector instance</returns>
        public EngineInspector createEngineInspector()
        {
            IntPtr inspectorPtr;
            TrtHandleException.handler(NativeMethods.trtCudaEngine_createEngineInspector(
                ptr, out inspectorPtr));
            return new EngineInspector(inspectorPtr);
        }
    }

}

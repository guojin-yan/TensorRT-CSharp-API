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
    /// 构建类，用于创建TensorRT网络和引擎
    /// Build class for creating TensorRT networks and engines
    /// </summary>
    /// <remarks>
    /// 继承自DisposableTrtObject，实现了IDisposable接口，可以安全地释放资源
    /// Inherits from DisposableTrtObject and implements IDisposable interface for safe resource disposal
    /// </remarks>
    public class Builder : DisposableTrtObject
    {
        /// <summary>
        /// 创建构建器实例
        /// Creates Build
        /// </summary>
        public Builder()
        {
            InitHandleException.handler(
                NativeMethods.trtBuild_createInferBuilder(out ptr));
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
            //if (ptr != IntPtr.Zero && IsEnabledDispose)
            //    NativeMethods.trtBuild_free(ptr);
            base.DisposeUnmanaged();
        }

        /// <summary>
        /// 检查平台是否支持快速FP16计算
        /// Checks if the platform supports fast FP16 computation
        /// </summary>
        /// <returns>如果支持快速FP16计算则返回true，否则返回false，Returns true if fast FP16 is supported, false otherwise</returns>
        public bool platformHasFastFp16()
        {
            TrtHandleException.handler(NativeMethods.trtBuild_platformHasFastFp16(ptr, out int flag));
            return flag != 0;
        }

        /// <summary>
        /// 检查平台是否支持快速INT8计算
        /// Checks if the platform supports fast INT8 computation
        /// </summary>
        /// <returns>如果支持快速INT8计算则返回true，否则返回false，Returns true if fast INT8 is supported, false otherwise</returns>
        public bool platformHasFastInt8()
        {
            int flag = 0;
            TrtHandleException.handler(NativeMethods.trtBuild_platformHasFastInt8(ptr, out flag));
            return flag != 0;
        }

        /// <summary>
        /// 获取DLA的最大批处理大小
        /// Gets the maximum batch size for DLA
        /// </summary>
        /// <returns>DLA最大批处理大小，Maximum DLA batch size</returns>
        public int maxDLABatchSize()
        {
            int size = 0;
            TrtHandleException.handler(NativeMethods.trtBuild_getMaxDLABatchSize(ptr, out size));
            return size;
        }

        /// <summary>
        /// 获取DLA核心数量
        /// Gets the number of DLA cores
        /// </summary>
        /// <returns>DLA核心数量，Number of DLA cores</returns>
        public int nbDLACores()
        {
            int count = 0;
            TrtHandleException.handler(NativeMethods.trtBuild_getNbDLACores(ptr, out count));
            return count;
        }

        /// <summary>
        /// 设置GPU内存分配器
        /// Sets the GPU memory allocator
        /// </summary>
        /// <param name="gpuAllocator">GPU内存分配器实例，GPU memory allocator instance</param>
        public void setGpuAllocator(GpuAllocator gpuAllocator)
        {
            TrtHandleException.handler(NativeMethods.trtBuild_setGpuAllocator(ptr, gpuAllocator.TrtPtr));
        }

        /// <summary>
        /// 创建构建器配置对象
        /// Creates a builder configuration object
        /// </summary>
        /// <returns>构建器配置实例，Builder configuration instance</returns>
        public BuilderConfig createBuilderConfig()
        {
            IntPtr configPtr = IntPtr.Zero;
            TrtHandleException.handler(NativeMethods.trtBuild_createBuilderConfig(ptr, out configPtr));
            return new BuilderConfig(configPtr);
        }

        /// <summary>
        /// 创建网络定义
        /// Creates a network definition
        /// </summary>
        /// <param name="flags">网络定义创建标志，Network definition creation flags</param>
        /// <returns>网络定义实例，Network definition instance</returns>
        public NetworkDefinition createNetworkV2(TrtNetworkDefinitionCreationFlag flags)
        {
            IntPtr networkPtr = IntPtr.Zero;
            TrtHandleException.handler(NativeMethods.trtBuild_createNetworkV2(ptr, flags, out networkPtr));
            return new NetworkDefinition(networkPtr);
        }

        /// <summary>
        /// 创建优化配置文件
        /// Creates an optimization profile
        /// </summary>
        /// <returns>优化配置文件实例，Optimization profile instance</returns>
        public OptimizationProfile createOptimizationProfile()
        {
            IntPtr profilePtr = IntPtr.Zero;
            TrtHandleException.handler(NativeMethods.trtBuild_createOptimizationProfile(ptr, out profilePtr));
            return new OptimizationProfile(profilePtr);
        }

        /// <summary>
        /// 重置构建器状态
        /// Resets the builder state
        /// </summary>
        public void reset()
        {
            TrtHandleException.handler(NativeMethods.trtBuild_reset(ptr));
        }

        /// <summary>
        /// 构建序列化网络
        /// Builds a serialized network
        /// </summary>
        /// <param name="network">网络定义实例，Network definition instance</param>
        /// <param name="config">构建器配置实例，Builder configuration instance</param>
        /// <returns>主机内存实例，包含序列化的网络数据，Host memory instance containing serialized network data</returns>
        public HostMemory buildSerializedNetwork(NetworkDefinition network, BuilderConfig config)
        {
            IntPtr hostMemory = IntPtr.Zero;
            TrtHandleException.handler(NativeMethods.trtBuild_buildSerializedNetwork(ptr, network.TrtPtr, config.TrtPtr, out hostMemory));
            return new HostMemory(hostMemory);
        }

        /// <summary>
        /// 将序列化网络构建写入流
        /// Builds a serialized network to stream
        /// </summary>
        /// <param name="network">网络定义实例，Network definition instance</param>
        /// <param name="config">构建器配置实例，Builder configuration instance</param>
        /// <param name="writer">文件流读取器实例，File stream reader instance</param>
        /// <returns>构建成功返回true，否则返回false，Returns true if build succeeded, false otherwise</returns>
        public bool buildSerializedNetworkToStream(NetworkDefinition network, BuilderConfig config, FileStreamReader writer)
        {
            int flag = 0;
            TrtHandleException.handler(NativeMethods.trtBuild_buildSerializedNetworkToStream(ptr, network.TrtPtr, config.TrtPtr, writer.TrtPtr, out flag));
            return flag != 0;
        }

        /// <summary>
        /// 使用配置构建引擎
        /// Builds an engine with configuration
        /// </summary>
        /// <param name="network">网络定义实例，Network definition instance</param>
        /// <param name="config">构建器配置实例，Builder configuration instance</param>
        /// <returns>CUDA引擎实例，CUDA engine instance</returns>
        public CudaEngine buildEngineWithConfig(NetworkDefinition network, BuilderConfig config)
        {
            IntPtr enginePtr = IntPtr.Zero;
            TrtHandleException.handler(NativeMethods.trtBuild_buildEngineWithConfig(ptr, network.TrtPtr, config.TrtPtr, out enginePtr));
            return new CudaEngine(enginePtr);
        }

        /// <summary>
        /// 检查网络是否支持指定的配置
        /// Checks if the network is supported with the specified configuration
        /// </summary>
        /// <param name="network">网络定义实例，Network definition instance</param>
        /// <param name="config">构建器配置实例，Builder configuration instance</param>
        /// <returns>如果支持返回true，否则返回false，Returns true if supported, false otherwise</returns>
        public bool isNetworkSupported(NetworkDefinition network, BuilderConfig config)
        {
            int flag = 0;
            TrtHandleException.handler(NativeMethods.trtBuild_isNetworkSupported(ptr, network.TrtPtr, config.TrtPtr, out flag));
            return flag != 0;
        }

        /// <summary>
        /// 设置最大线程数
        /// Sets the maximum number of threads
        /// </summary>
        /// <param name="maxThreads">最大线程数，Maximum number of threads</param>
        /// <returns>设置成功返回true，否则返回false，Returns true if set successfully, false otherwise</returns>
        public bool setMaxThreads(int maxThreads)
        {
            int flag = 0;
            TrtHandleException.handler(NativeMethods.trtBuild_setMaxThreads(ptr, maxThreads, out flag));
            return flag != 0;
        }

        /// <summary>
        /// 获取最大线程数
        /// Gets the maximum number of threads
        /// </summary>
        /// <returns>当前设置的最大线程数，Current maximum number of threads</returns>
        public int getMaxThreads()
        {
            int maxThreads = 0;
            TrtHandleException.handler(NativeMethods.trtBuild_getMaxThreads(ptr, out maxThreads));
            return maxThreads;
        }

        /// <summary>
        /// 获取插件注册表
        /// Gets the plugin registry
        /// </summary>
        /// <returns>插件注册表实例，Plugin registry instance</returns>
        public PluginRegistry getPluginRegistry()
        {
            IntPtr pluginRegistryPtr = IntPtr.Zero;
            TrtHandleException.handler(NativeMethods.trtBuild_getPluginRegistry(ptr, out pluginRegistryPtr));
            return new PluginRegistry(pluginRegistryPtr);
        }

        /// <summary>
        /// 获取与此构建器关联的 Logger
        /// Gets the logger associated with this builder
        /// </summary>
        /// <returns>Logger 实例 / Logger instance</returns>
        public Logger getLogger()
        {
            // Builder 使用全局 Logger 实例，返回单例
            return Logger.Instance;
        }
    }

}

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

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 构建器配置类，用于配置TensorRT网络构建参数
    /// Builder configuration class for configuring TensorRT network building parameters
    /// </summary>
    /// <remarks>
    /// 继承自DisposableTrtObject，实现了IDisposable接口，可以安全地释放资源
    /// Inherits from DisposableTrtObject and implements IDisposable interface for safe resource disposal
    /// </remarks>
    public class BuilderConfig : DisposableTrtObject
    {
        /// <summary>
        /// 创建空的构建器配置对象
        /// Creates empty BuilderConfig
        /// </summary>
        public BuilderConfig()
        {
            // InitHandleException.handler已注释，表示此构造函数未实现
            // InitHandleException.handler(
            //    NativeMethods.trtBuild_createInferBuilder(out ptr));
        }

        /// <summary>
        /// 从原生指针创建构建器配置对象
        /// Creates from native pointer
        /// </summary>
        /// <param name="ptr">原生对象指针，Native object pointer</param>
        internal BuilderConfig(IntPtr ptr)
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
            // DisposeUnmanaged方法已注释，表示此方法未实现
            // if (ptr != IntPtr.Zero && IsEnabledDispose)
            //    NativeMethods.trtBuild_free(ptr);
            // base.DisposeUnmanaged();
        }

        /// <summary>
        /// 设置平均时间迭代配置
        /// Sets the average timing iterations configuration
        /// </summary>
        /// <param name="avgTiming">平均时间迭代次数，Average timing iterations</param>
        public void setAvgTimingIterationsConfig(int avgTiming)
        {
            TrtHandleException.handler(NativeMethods.trtBuildertrtBuilderConfig_setAvgTimingIterationsConfig(ptr, avgTiming));
        }

        /// <summary>
        /// 获取平均时间迭代配置
        /// Gets the average timing iterations configuration
        /// </summary>
        /// <returns>平均时间迭代次数，Average timing iterations</returns>
        public int getAvgTimingIterationsConfig()
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_getAvgTimingIteration(ptr, out int avgTiming));
            return avgTiming;
        }

        /// <summary>
        /// 设置引擎能力
        /// Sets the engine capability
        /// </summary>
        /// <param name="trtEngineCapability">引擎能力枚举，Engine capability enum</param>
        public void setEngineCapability(TrtEngineCapability trtEngineCapability)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_setEngineCapability(ptr, trtEngineCapability));
        }

        /// <summary>
        /// 获取引擎能力
        /// Gets the engine capability
        /// </summary>
        /// <returns>引擎能力枚举，Engine capability enum</returns>
        public TrtEngineCapability getEngineCapability()
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_getEngineCapability(ptr, out TrtEngineCapability capability));
            return capability;
        }


        /// <summary>
        /// 设置构建标志
        /// Sets the builder flags
        /// </summary>
        /// <param name="builderFlags">构建标志位，Builder flag bits</param>
        public void setFlags(uint builderFlags)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_setFlags(ptr, builderFlags));
        }

        /// <summary>
        /// 获取构建标志
        /// Gets the builder flags
        /// </summary>
        /// <returns>构建标志位，Builder flag bits</returns>
        public uint getFlags()
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_getFlags(ptr, out uint builderFlags));
            return builderFlags;
        }

        /// <summary>
        /// 清除构建标志
        /// Clears a builder flag
        /// </summary>
        /// <param name="builderFlag">要清除的构建标志，Builder flag to clear</param>
        public void clearFlag(TrtBuilderFlag builderFlag)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_clearFlag(ptr, builderFlag));
        }

        /// <summary>
        /// 设置构建标志
        /// Sets a builder flag
        /// </summary>
        /// <param name="builderFlag">要设置的构建标志，Builder flag to set</param>
        public void setFlag(TrtBuilderFlag builderFlag)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_setFlag(ptr, builderFlag));
        }

        /// <summary>
        /// 获取构建标志
        /// Gets the builder flag
        /// </summary>
        /// <returns>构建标志枚举，Builder flag enum</returns>
        public TrtBuilderFlag getFlag()
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_getFlag(ptr, out uint builderFlag));
            return (TrtBuilderFlag)builderFlag;
        }

        /// <summary>
        /// 设置层设备类型
        /// Sets the device type for a layer
        /// </summary>
        /// <param name="layer">层实例，Layer instance</param>
        /// <param name="deviceType">设备类型枚举，Device type enum</param>
        public void etLayerDeviceType(Layer layer, TrtDeviceType deviceType)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_setLayerDeviceType(ptr, layer.TrtPtr, deviceType));
        }

        /// <summary>
        /// 获取层设备类型
        /// Gets the device type for a layer
        /// </summary>
        /// <param name="layer">层实例，Layer instance</param>
        /// <returns>设备类型枚举，Device type enum</returns>
        public TrtDeviceType getLayerDeviceType(Layer layer)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_getLayerDeviceType(ptr, layer.TrtPtr, out TrtDeviceType deviceType));
            return deviceType;
        }

        /// <summary>
        /// 检查设备类型是否已设置
        /// Checks if the device type is set for a layer
        /// </summary>
        /// <param name="layer">层实例，Layer instance</param>
        /// <returns>如果已设置返回true，否则返回false，Returns true if set, false otherwise</returns>
        public bool isDeviceTypeSet(Layer layer)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_isDeviceTypeSet(ptr, layer.TrtPtr, out int outState));
            return outState != 0;
        }

        /// <summary>
        /// 重置层设备类型
        /// Resets the device type for a layer
        /// </summary>
        /// <param name="layer">层实例，Layer instance</param>
        public void resetLayerDeviceType(Layer layer)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_resetLayerDeviceType(ptr, layer.TrtPtr));
        }

        /// <summary>
        /// 检查层是否可以在DLA上运行
        /// Checks if a layer can run on DLA
        /// </summary>
        /// <param name="layer">层实例，Layer instance</param>
        /// <returns>如果可以在DLA上运行返回true，否则返回false，Returns true if can run on DLA, false otherwise</returns>
        public bool canRunOnDLA(Layer layer)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_canRunOnDLA(ptr, layer.TrtPtr, out int outState));
            return outState != 0;
        }

        /// <summary>
        /// 设置DLA核心
        /// Sets the DLA core
        /// </summary>
        /// <param name="dlaCore">DLA核心索引，DLA core index</param>
        public void setDLACore(int dlaCore)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_setDLACore(ptr, dlaCore));
        }

        /// <summary>
        /// 获取DLA核心
        /// Gets the DLA core
        /// </summary>
        /// <returns>DLA核心索引，DLA core index</returns>
        public int getDLACore()
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_getDLACore(ptr, out int dlaCore));
            return dlaCore;
        }

        /// <summary>
        /// 设置默认设备类型
        /// Sets the default device type
        /// </summary>
        /// <param name="deviceType">设备类型枚举，Device type enum</param>
        public void setDefaultDeviceType(TrtDeviceType deviceType)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_setDefaultDeviceType(ptr, deviceType));
        }

        /// <summary>
        /// 获取默认设备类型
        /// Gets the default device type
        /// </summary>
        /// <returns>设备类型枚举，Device type enum</returns>
        public TrtDeviceType getDefaultDeviceType()
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_getDefaultDeviceType(ptr, out TrtDeviceType deviceType));
            return deviceType;
        }

        /// <summary>
        /// 重置构建器配置
        /// Resets the builder configuration
        /// </summary>
        public void reset()
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_reset(ptr));
        }

        /// <summary>
        /// 设置配置文件流
        /// Sets the profile stream
        /// </summary>
        /// <param name="stream">CUDA流实例，CUDA stream instance</param>
        public void setProfileStream(CudaStream stream)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_setProfileStream(ptr, stream.TrtPtr));
        }

        /// <summary>
        /// 获取配置文件流
        /// Gets the profile stream
        /// </summary>
        /// <returns>CUDA流实例，CUDA stream instance</returns>
        public CudaStream getProfileStream()
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_getProfileStream(ptr, out IntPtr streamPtr));
            return new CudaStream(streamPtr);
        }

        /// <summary>
        /// 添加优化配置文件
        /// Adds an optimization profile
        /// </summary>
        /// <param name="profile">优化配置文件实例，Optimization profile instance</param>
        /// <returns>优化配置文件的索引，Index of the optimization profile</returns>
        public int addOptimizationProfile(OptimizationProfile profile)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_addOptimizationProfile(ptr, profile.TrtPtr, out int outIndex));
            return outIndex;
        }

        /// <summary>
        /// 获取优化配置文件数量
        /// Gets the number of optimization profiles
        /// </summary>
        /// <returns>优化配置文件的数量，Number of optimization profiles</returns>
        public int getNbOptimizationProfiles()
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_getNbOptimizationProfiles(ptr, out int outCount));
            return outCount;
        }

        /// <summary>
        /// 设置性能分析详细程度
        /// Sets the profiling verbosity
        /// </summary>
        /// <param name="verbosity">性能分析详细程度枚举，Profiling verbosity enum</param>
        public void setProfilingVerbosity(TrtProfilingVerbosity verbosity)
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_setProfilingVerbosity(ptr, verbosity));
        }

        /// <summary>
        /// 获取性能分析详细程度
        /// Gets the profiling verbosity
        /// </summary>
        /// <returns>性能分析详细程度枚举，Profiling verbosity enum</returns>
        public TrtProfilingVerbosity getProfilingVerbosity()
        {
            TrtHandleException.handler(NativeMethods.trtBuilderConfig_getProfilingVerbosity(ptr, out TrtProfilingVerbosity verbosity));
            return verbosity;
        }
    }

}

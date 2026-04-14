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
    /// 插件注册表类，用于管理和查找TensorRT插件
    /// Plugin registry class for managing and looking up TensorRT plugins
    /// </summary>
    public class PluginRegistry : DisposableTrtObject
    {
        /// <summary>
        /// 创建空的 PluginRegistry
        /// Creates an empty PluginRegistry
        /// </summary>
        public PluginRegistry()
        {
            // PluginRegistry 通常通过 Builder 或 Runtime 的 getPluginRegistry() 获取
            // PluginRegistry is usually obtained via getPluginRegistry() from Builder or Runtime
        }

        /// <summary>
        /// 从原生指针创建
        /// Creates from native pointer
        /// </summary>
        /// <param name="ptr">原生指针对象 / Native pointer</param>
        internal PluginRegistry(IntPtr ptr)
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

        /// <inheritdoc />
        /// <summary>
        /// 释放非托管资源
        /// Releases unmanaged resources
        /// </summary>
        protected override void DisposeUnmanaged()
        {
            // PluginRegistry 由 TensorRT 内部管理，不需要手动释放
            // PluginRegistry is managed internally by TensorRT, no manual release needed
            //if (ptr != IntPtr.Zero && IsEnabledDispose)
            //    NativeMethods.trtBuild_free(ptr);
            //base.DisposeUnmanaged();
        }

        #region Error Recorder

        /// <summary>
        /// 设置错误记录器
        /// Sets the error recorder
        /// </summary>
        /// <param name="errorRecorder">错误记录器实例 / Error recorder instance</param>
        public void setErrorRecorder(ErrorRecorder errorRecorder)
        {
            IntPtr recorderPtr = errorRecorder?.getHandle() ?? IntPtr.Zero;
            TrtHandleException.handler(NativeMethods.trtPluginRegistry_setErrorRecorder(ptr, recorderPtr));
        }

        /// <summary>
        /// 获取错误记录器
        /// Gets the error recorder
        /// </summary>
        /// <returns>错误记录器实例，如果没有设置则返回null / Error recorder instance, or null if not set</returns>
        public ErrorRecorder getErrorRecorder()
        {
            TrtHandleException.handler(NativeMethods.trtPluginRegistry_getErrorRecorder(ptr, out IntPtr recorderPtr));
            if (recorderPtr == IntPtr.Zero)
                return null;
            // 返回单例实例
            return ErrorRecorder.Instance;
        }

        #endregion

        #region Parent Search Control

        /// <summary>
        /// 检查是否启用了父注册表搜索
        /// Checks if parent registry search is enabled
        /// </summary>
        /// <returns>如果启用返回true，否则返回false / Returns true if enabled, false otherwise</returns>
        public bool isParentSearchEnabled()
        {
            TrtHandleException.handler(NativeMethods.trtPluginRegistry_isParentSearchEnabled(ptr, out int enabled));
            return enabled != 0;
        }

        /// <summary>
        /// 设置是否启用父注册表搜索
        /// Sets whether parent registry search is enabled
        /// </summary>
        /// <param name="enabled">是否启用 / Whether to enable</param>
        public void setParentSearchEnabled(bool enabled)
        {
            TrtHandleException.handler(NativeMethods.trtPluginRegistry_setParentSearchEnabled(ptr, enabled ? 1 : 0));
        }

        #endregion

        #region Library Management

        /// <summary>
        /// 加载并注册插件库
        /// Loads and registers a plugin library
        /// </summary>
        /// <param name="pluginPath">插件库路径 / Plugin library path</param>
        /// <returns>插件库句柄 / Plugin library handle</returns>
        public IntPtr loadLibrary(string pluginPath)
        {
            TrtHandleException.handler(NativeMethods.trtPluginRegistry_loadLibrary(ptr, pluginPath, out IntPtr handle));
            return handle;
        }

        /// <summary>
        /// 注销插件库
        /// Deregisters a plugin library
        /// </summary>
        /// <param name="handle">插件库句柄 / Plugin library handle</param>
        public void deregisterLibrary(IntPtr handle)
        {
            TrtHandleException.handler(NativeMethods.trtPluginRegistry_deregisterLibrary(ptr, handle));
        }

        #endregion

        #region Plugin Creator Interface API (TensorRT 10.0+)

        /// <summary>
        /// 注册插件创建器接口
        /// Registers a plugin creator interface
        /// </summary>
        /// <param name="creator">插件创建器 / Plugin creator</param>
        /// <param name="pluginNamespace">插件命名空间 / Plugin namespace</param>
        /// <returns>注册成功返回true，否则返回false / Returns true if successful, false otherwise</returns>
        public bool registerCreatorInterface(IntPtr creator, string pluginNamespace)
        {
            var status = NativeMethods.trtPluginRegistry_registerCreatorInterface(ptr, creator, pluginNamespace);
            return status == TrtExceptionStatus.TENSORRT_SUCCESS;
        }

        /// <summary>
        /// 获取插件创建器接口
        /// Gets a plugin creator interface
        /// </summary>
        /// <param name="pluginName">插件名称 / Plugin name</param>
        /// <param name="pluginVersion">插件版本 / Plugin version</param>
        /// <param name="pluginNamespace">插件命名空间 / Plugin namespace</param>
        /// <returns>插件创建器指针 / Plugin creator pointer</returns>
        public IntPtr getCreatorInterface(string pluginName, string pluginVersion, string pluginNamespace = "")
        {
            TrtHandleException.handler(NativeMethods.trtPluginRegistry_getCreatorInterface(
                ptr, pluginName, pluginVersion, pluginNamespace, out IntPtr creator));
            return creator;
        }

        /// <summary>
        /// 注销插件创建器接口
        /// Deregisters a plugin creator interface
        /// </summary>
        /// <param name="creator">插件创建器指针 / Plugin creator pointer</param>
        /// <returns>注销成功返回true，否则返回false / Returns true if successful, false otherwise</returns>
        public bool deregisterCreatorInterface(IntPtr creator)
        {
            var status = NativeMethods.trtPluginRegistry_deregisterCreatorInterface(ptr, creator);
            return status == TrtExceptionStatus.TENSORRT_SUCCESS;
        }

        /// <summary>
        /// 获取所有插件创建器接口
        /// Gets all plugin creator interfaces
        /// </summary>
        /// <returns>插件创建器指针数组 / Array of plugin creator pointers</returns>
        public IntPtr[] getAllCreatorsInterface()
        {
            TrtHandleException.handler(NativeMethods.trtPluginRegistry_getAllCreatorsInterface(
                ptr, out int numCreators, out IntPtr creatorsPtr));

            if (numCreators == 0 || creatorsPtr == IntPtr.Zero)
                return new IntPtr[0];

            // 从非托管数组读取指针
            IntPtr[] creators = new IntPtr[numCreators];
            for (int i = 0; i < numCreators; i++)
            {
                creators[i] = Marshal.ReadIntPtr(creatorsPtr, i * IntPtr.Size);
            }

            return creators;
        }

        /// <summary>
        /// 递归获取所有插件创建器接口
        /// Gets all plugin creator interfaces recursively
        /// </summary>
        /// <returns>插件创建器指针数组 / Array of plugin creator pointers</returns>
        public IntPtr[] getAllCreatorsRecursive()
        {
            TrtHandleException.handler(NativeMethods.trtPluginRegistry_getAllCreatorsRecursive(
                ptr, out int numCreators, out IntPtr creatorsPtr));

            if (numCreators == 0 || creatorsPtr == IntPtr.Zero)
                return new IntPtr[0];

            IntPtr[] creators = new IntPtr[numCreators];
            for (int i = 0; i < numCreators; i++)
            {
                creators[i] = Marshal.ReadIntPtr(creatorsPtr, i * IntPtr.Size);
            }

            return creators;
        }

        #endregion

        #region Deprecated IPluginCreator API (for backward compatibility)

        /// <summary>
        /// 注册插件创建器（已弃用，请使用 registerCreatorInterface）
        /// Registers a plugin creator (deprecated, use registerCreatorInterface instead)
        /// </summary>
        /// <param name="creator">插件创建器 / Plugin creator</param>
        /// <param name="pluginNamespace">插件命名空间 / Plugin namespace</param>
        /// <returns>注册成功返回true，否则返回false / Returns true if successful, false otherwise</returns>
        [Obsolete("Use registerCreatorInterface instead")]
        public bool registerCreator(IntPtr creator, string pluginNamespace)
        {
            var status = NativeMethods.trtPluginRegistry_registerCreator(ptr, creator, pluginNamespace);
            return status == TrtExceptionStatus.TENSORRT_SUCCESS;
        }

        /// <summary>
        /// 获取插件创建器（已弃用，请使用 getCreatorInterface）
        /// Gets a plugin creator (deprecated, use getCreatorInterface instead)
        /// </summary>
        /// <param name="pluginName">插件名称 / Plugin name</param>
        /// <param name="pluginVersion">插件版本 / Plugin version</param>
        /// <param name="pluginNamespace">插件命名空间 / Plugin namespace</param>
        /// <returns>插件创建器指针 / Plugin creator pointer</returns>
        [Obsolete("Use getCreatorInterface instead")]
        public IntPtr getPluginCreator(string pluginName, string pluginVersion, string pluginNamespace = "")
        {
            TrtHandleException.handler(NativeMethods.trtPluginRegistry_getPluginCreator(
                ptr, pluginName, pluginVersion, pluginNamespace, out IntPtr creator));
            return creator;
        }

        /// <summary>
        /// 注销插件创建器（已弃用，请使用 deregisterCreatorInterface）
        /// Deregisters a plugin creator (deprecated, use deregisterCreatorInterface instead)
        /// </summary>
        /// <param name="creator">插件创建器指针 / Plugin creator pointer</param>
        /// <returns>注销成功返回true，否则返回false / Returns true if successful, false otherwise</returns>
        [Obsolete("Use deregisterCreatorInterface instead")]
        public bool deregisterCreator(IntPtr creator)
        {
            var status = NativeMethods.trtPluginRegistry_deregisterCreator(ptr, creator);
            return status == TrtExceptionStatus.TENSORRT_SUCCESS;
        }

        #endregion

        #region Plugin Resource Management

        /// <summary>
        /// 获取插件资源
        /// Acquires a plugin resource
        /// </summary>
        /// <param name="key">资源键 / Resource key</param>
        /// <param name="resource">资源对象（可为null）/ Resource object (can be null)</param>
        /// <returns>获取的资源对象指针 / Acquired resource object pointer</returns>
        public IntPtr acquirePluginResource(string key, IntPtr resource)
        {
            TrtHandleException.handler(NativeMethods.trtPluginRegistry_acquirePluginResource(
                ptr, key, resource, out IntPtr acquiredResource));
            return acquiredResource;
        }

        /// <summary>
        /// 释放插件资源
        /// Releases a plugin resource
        /// </summary>
        /// <param name="key">资源键 / Resource key</param>
        /// <returns>操作结果，0表示成功 / Operation result, 0 means success</returns>
        public int releasePluginResource(string key)
        {
            TrtHandleException.handler(NativeMethods.trtPluginRegistry_releasePluginResource(
                ptr, key, out int result));
            return result;
        }

        #endregion
    }
}

using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda
{
    /// <summary>
    /// 运行时核心与工具模块。
    /// 提供错误处理、版本查询、驱动程序交互以及用户对象管理功能。
    /// Runtime Core and Utilities Module.
    /// Provides error handling, version querying, driver interaction, and user object management features.
    /// </summary>
    public static class CudaRuntime
    {
        #region 错误处理 / Error Handling
        /// <summary>
        /// 获取并清除最后一个错误。
        /// Retrieves and clears the last error.
        /// </summary>
        /// <returns>最后一个错误的状态。/ The status of the last error.</returns>
        public static CudaExceptionStatus GetLastError()
        {
            return NativeMethods.cudaRuntime_cudaGetLastError();
        }
        /// <summary>
        /// 获取最后一个错误，但不将其清除。
        /// Retrieves the last error without clearing it.
        /// </summary>
        /// <returns>最后一个错误的状态。/ The status of the last error.</returns>
        public static CudaExceptionStatus PeekAtLastError()
        {
            return NativeMethods.cudaRuntime_cudaPeekAtLastError();
        }
        /// <summary>
        /// 获取错误状态的名称字符串。
        /// Gets the name string for an error status.
        /// </summary>
        /// <param name="error">错误状态。/ The error status.</param>
        /// <returns>错误名称。/ The error name.</returns>
        public static string GetErrorName(CudaExceptionStatus error)
        {
            IntPtr pName = NativeMethods.cudaRuntime_cudaGetErrorName(error);
            return Marshal.PtrToStringAnsi(pName);
        }
        /// <summary>
        /// 获取错误状态的描述字符串。
        /// Gets the description string for an error status.
        /// </summary>
        /// <param name="error">错误状态。/ The error status.</param>
        /// <returns>错误描述。/ The error description.</returns>
        public static string GetErrorString(CudaExceptionStatus error)
        {
            IntPtr pStr = NativeMethods.cudaRuntime_cudaGetErrorString(error);
            return Marshal.PtrToStringAnsi(pStr);
        }
        #endregion
        #region 版本信息 / Version Information
        /// <summary>
        /// �回驱动程序版本。
        /// Returns the driver version.
        /// </summary>
        /// <returns>驱动程序版本号。/ The driver version number.</returns>
        public static int DriverGetVersion()
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaDriverGetVersion(out int driverVersion));
            return driverVersion;
        }
        /// <summary>
        /// 获取运行时 API 版本。
        /// Returns the runtime API version.
        /// </summary>
        /// <returns>运行时版本号。/ The runtime version number.</returns>
        public static int RuntimeGetVersion()
        {
            CudaHandleException.handler(
                NativeMethods.cudaRuntime_cudaRuntimeGetVersion(out int runtimeVersion));
            return runtimeVersion;
        }
        #endregion
    }
}

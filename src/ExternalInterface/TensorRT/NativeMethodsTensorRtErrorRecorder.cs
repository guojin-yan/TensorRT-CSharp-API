using JYPPX.TensorRtSharp.Exceptions;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.ExternalInterface
{
    public static partial class NativeMethods
    {
        #region Error Recorder Management

        /// <summary>
        /// 获取全局错误记录器实例
        /// Gets the global error recorder instance
        /// </summary>
        /// <param name="errorRecorder">输出参数，返回错误记录器指针 / Output parameter, returns error recorder pointer</param>
        /// <returns>初始化状态码 / Initialization status code</returns>
        [Pure, DllImport(dllExtern, EntryPoint = "trtErrorRecorder_getTrtErrorRecorder",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static InitExceptionStatus trtErrorRecorder_getTrtErrorRecorder(
            out IntPtr errorRecorder);

        #endregion

        #region Error Information Access

        /// <summary>
        /// 获取当前记录的错误数量
        /// Gets the number of errors currently recorded
        /// </summary>
        /// <returns>错误数量 / Number of errors</returns>
        [Pure, DllImport(dllExtern, EntryPoint = "trtErrorRecorder_getNbErrors",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static int trtErrorRecorder_getNbErrors();

        /// <summary>
        /// 获取指定索引的错误代码
        /// Gets the error code for the specified index
        /// </summary>
        /// <param name="errorIdx">错误索引（0-based）/ Error index (0-based)</param>
        /// <returns>错误状态枚举 / Error status enum</returns>
        [Pure, DllImport(dllExtern, EntryPoint = "trtErrorRecorder_getErrorCode",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static TrtExceptionStatus trtErrorRecorder_getErrorCode(
            int errorIdx);

        /// <summary>
        /// 获取指定索引的错误类型
        /// Gets the error type for the specified index
        /// </summary>
        /// <param name="errorIdx">错误索引（0-based）/ Error index (0-based)</param>
        /// <returns>错误类型代码（对应ErrorCode枚举）/ Error type code (corresponds to ErrorCode enum)</returns>
        [Pure, DllImport(dllExtern, EntryPoint = "trtErrorRecorder_getErrorType",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static int trtErrorRecorder_getErrorType(
            int errorIdx);

        /// <summary>
        /// 获取指定索引的错误描述
        /// Gets the error description for the specified index
        /// </summary>
        /// <param name="errorIdx">错误索引（0-based）/ Error index (0-based)</param>
        /// <returns>错误描述字符串 / Error description string</returns>
        [Pure, DllImport(dllExtern, EntryPoint = "trtErrorRecorder_getErrorDesc",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: MarshalAs(StringUnmanagedTypeNotWindows)]
        public extern static string trtErrorRecorder_getErrorDesc(
            int errorIdx);

        #endregion

        #region Status Checks

        /// <summary>
        /// 检查错误记录器是否溢出
        /// Checks if the error recorder has overflowed
        /// </summary>
        /// <returns>如果溢出返回非零值，否则返回0 / Returns non-zero if overflowed, 0 otherwise</returns>
        [Pure, DllImport(dllExtern, EntryPoint = "trtErrorRecorder_hasOverflowed",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static int trtErrorRecorder_hasOverflowed();

        /// <summary>
        /// 检查错误记录器是否为空
        /// Checks if the error recorder is empty
        /// </summary>
        /// <returns>如果为空返回非零值，否则返回0 / Returns non-zero if empty, 0 otherwise</returns>
        [Pure, DllImport(dllExtern, EntryPoint = "trtErrorRecorder_empty",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static int trtErrorRecorder_empty();

        /// <summary>
        /// 获取错误记录器的最大描述长度
        /// Gets the maximum description length for the error recorder
        /// </summary>
        /// <returns>最大描述长度（字节）/ Maximum description length in bytes</returns>
        /// <remarks>根据TensorRT规范，安全运行时中描述长度限制为127字节（不含null终止符）</remarks>
        [Pure, DllImport(dllExtern, EntryPoint = "trtErrorRecorder_getMaxDescLength",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static UIntPtr trtErrorRecorder_getMaxDescLength();

        #endregion

        #region Maintenance Operations

        /// <summary>
        /// 清除所有错误记录
        /// Clears all error records
        /// </summary>
        [DllImport(dllExtern, EntryPoint = "trtErrorRecorder_clear",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static void trtErrorRecorder_clear();

        /// <summary>
        /// 报告一个错误
        /// Reports an error
        /// </summary>
        /// <param name="val">异常状态枚举 / Exception status enum</param>
        /// <param name="desc">错误描述字符串 / Error description string</param>
        /// <returns>如果错误被确定为致命错误返回非零值，否则返回0 / Returns non-zero if error is fatal, 0 otherwise</returns>
        [DllImport(dllExtern, EntryPoint = "trtErrorRecorder_reportError",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static int trtErrorRecorder_reportError(
            TrtExceptionStatus val,
            [MarshalAs(StringUnmanagedTypeNotWindows)] string desc);

        #endregion

        #region Reference Counting

        /// <summary>
        /// 增加引用计数
        /// Increments the reference count
        /// </summary>
        /// <returns>增加后的引用计数值 / Reference count after increment</returns>
        [Pure, DllImport(dllExtern, EntryPoint = "trtErrorRecorder_incRefCount",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static int trtErrorRecorder_incRefCount();

        /// <summary>
        /// 减少引用计数
        /// Decrements the reference count
        /// </summary>
        /// <returns>减少后的引用计数值 / Reference count after decrement</returns>
        [DllImport(dllExtern, EntryPoint = "trtErrorRecorder_decRefCount",
            CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public extern static int trtErrorRecorder_decRefCount();

        #endregion
    }
}

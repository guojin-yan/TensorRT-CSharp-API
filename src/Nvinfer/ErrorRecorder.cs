using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
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
    /// 错误记录器类，用于记录和处理TensorRT运行时错误
    /// Error recorder class for recording and handling TensorRT runtime errors
    /// </summary>
    /// <remarks>
    /// 实现了单例模式，确保全局只有一个错误记录器实例
    /// Implements singleton pattern to ensure only one error recorder instance exists globally
    /// </remarks>
    public class ErrorRecorder
    {
        private IntPtr ptr;
        // 私有静态实例（确保唯一）
        // Private static instance (ensures uniqueness)
        private static ErrorRecorder _instance;

        // 私有构造函数（防止外部实例化）
        // Private constructor (prevents external instantiation)
        private ErrorRecorder()
        {
            InitHandleException.handler(
                NativeMethods.trtErrorRecorder_getTrtErrorRecorder(out ptr));
        }

        // 公共静态方法，获取唯一实例
        // Public static method to get the unique instance
        public static ErrorRecorder Instance
        {
            get
            {
                if (_instance == null)
                {
                    _instance = new ErrorRecorder();
                }
                return _instance;
            }
        }

        /// <summary>
        /// 获取错误记录器的句柄
        /// Gets the handle of the error recorder
        /// </summary>
        /// <returns>错误记录器的原生指针，Native pointer of the error recorder</returns>
        public IntPtr getHandle()
        {
            return ptr;
        }

        /// <summary>
        /// 获取错误数量
        /// Gets the number of errors
        /// </summary>
        /// <returns>错误数量，Number of errors</returns>
        public int getNbErrors()
        {
            return NativeMethods.trtErrorRecorder_getNbErrors();
        }

        /// <summary>
        /// 获取指定索引的错误代码
        /// Gets the error code for the specified index
        /// </summary>
        /// <param name="errorIdx">错误索引，Error index</param>
        /// <returns>异常状态枚举，Exception status enum</returns>
        public TrtExceptionStatus getErrorCode(int errorIdx)
        {
            return NativeMethods.trtErrorRecorder_getErrorCode(errorIdx);
        }

        /// <summary>
        /// 获取指定索引的错误描述
        /// Gets the error description for the specified index
        /// </summary>
        /// <param name="errorIdx">错误索引，Error index</param>
        /// <returns>错误描述字符串，Error description string</returns>
        public string getErrorDesc(int errorIdx)
        {
            return NativeMethods.trtErrorRecorder_getErrorDesc(errorIdx);
        }

        /// <summary>
        /// 检查错误记录器是否溢出
        /// Checks if the error recorder has overflowed
        /// </summary>
        /// <returns>如果溢出返回true，否则返回false，Returns true if overflowed, false otherwise</returns>
        public bool hasOverflowed()
        {
            int flag = NativeMethods.trtErrorRecorder_hasOverflowed();
            return flag != 0;
        }

        /// <summary>
        /// 检查错误记录器是否为空
        /// Checks if the error recorder is empty
        /// </summary>
        /// <returns>如果为空返回true，否则返回false，Returns true if empty, false otherwise</returns>
        public bool isEmpty()
        {
            int flag = NativeMethods.trtErrorRecorder_empty();
            return flag != 0;
        }

        /// <summary>
        /// 清除所有错误记录
        /// Clears all error records
        /// </summary>
        public void clear()
        {
            NativeMethods.trtErrorRecorder_clear();
        }

        /// <summary>
        /// 报告错误
        /// Reports an error
        /// </summary>
        /// <param name="val">异常状态枚举，Exception status enum</param>
        /// <param name="desc">错误描述字符串，Error description string</param>
        /// <returns>报告成功返回true，否则返回false，Returns true if reported successfully, false otherwise</returns>
        public bool reportError(
            TrtExceptionStatus val,
            string desc)
        {
            int result = NativeMethods.trtErrorRecorder_reportError(val, desc);
            return result != 0;
        }

        /// <summary>
        /// 增加引用计数
        /// Increments the reference count
        /// </summary>
        /// <returns>增加后的引用计数值，Reference count after increment</returns>
        public int incRefCount()
        {
            return NativeMethods.trtErrorRecorder_incRefCount();
        }

        /// <summary>
        /// 减少引用计数
        /// Decrements the reference count
        /// </summary>
        /// <returns>减少后的引用计数值，Reference count after decrement</returns>
        public int decRefCount()
        {
            return NativeMethods.trtErrorRecorder_decRefCount();
        }
    }

}

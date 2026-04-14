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
    /// 
    /// IErrorRecorder是TensorRT的错误报告接口，提供了线程安全的错误记录机制。
    /// 该接口的实现类用于记录TensorRT运行期间发生的错误，包括错误代码和描述信息。
    /// </remarks>
    public class ErrorRecorder
    {
        private IntPtr ptr;
        // 私有静态实例（确保唯一）
        // Private static instance (ensures uniqueness)
        private static ErrorRecorder _instance;
        private static readonly object _lock = new object();

        // 自定义错误事件委托
        // Custom error event delegate
        public delegate void ErrorReportedHandler(TrtExceptionStatus errorCode, string errorDesc);
        
        /// <summary>
        /// 当有新错误被报告时触发的事件
        /// Event triggered when a new error is reported
        /// </summary>
        public event ErrorReportedHandler OnErrorReported;

        // 私有构造函数（防止外部实例化）
        // Private constructor (prevents external instantiation)
        private ErrorRecorder()
        {
            InitHandleException.handler(
                NativeMethods.trtErrorRecorder_getTrtErrorRecorder(out ptr));
        }

        // 公共静态方法，获取唯一实例
        // Public static method to get the unique instance
        /// <summary>
        /// 获取错误记录器的单例实例
        /// Gets the singleton instance of the error recorder
        /// </summary>
        public static ErrorRecorder Instance
        {
            get
            {
                if (_instance == null)
                {
                    lock (_lock)
                    {
                        if (_instance == null)
                        {
                            _instance = new ErrorRecorder();
                        }
                    }
                }
                return _instance;
            }
        }

        /// <summary>
        /// 获取错误记录器的句柄
        /// Gets the handle of the error recorder
        /// </summary>
        /// <returns>错误记录器的原生指针 / Native pointer of the error recorder</returns>
        public IntPtr getHandle()
        {
            return ptr;
        }

        /// <summary>
        /// 获取错误数量
        /// Gets the number of errors
        /// </summary>
        /// <returns>错误数量 / Number of errors</returns>
        public int getNbErrors()
        {
            return NativeMethods.trtErrorRecorder_getNbErrors();
        }

        /// <summary>
        /// 获取指定索引的错误代码
        /// Gets the error code for the specified index
        /// </summary>
        /// <param name="errorIdx">错误索引 / Error index</param>
        /// <returns>异常状态枚举 / Exception status enum</returns>
        public TrtExceptionStatus getErrorCode(int errorIdx)
        {
            return NativeMethods.trtErrorRecorder_getErrorCode(errorIdx);
        }

        /// <summary>
        /// 获取指定索引的错误类型（原始整数值）
        /// Gets the error type for the specified index (raw integer value)
        /// </summary>
        /// <param name="errorIdx">错误索引 / Error index</param>
        /// <returns>ErrorCode枚举整数值 / ErrorCode enum integer value</returns>
        public int getErrorType(int errorIdx)
        {
            return NativeMethods.trtErrorRecorder_getErrorType(errorIdx);
        }

        /// <summary>
        /// 获取指定索引的错误类型（枚举形式）
        /// Gets the error type for the specified index (enum form)
        /// </summary>
        /// <param name="errorIdx">错误索引 / Error index</param>
        /// <returns>ErrorCode枚举值 / ErrorCode enum value</returns>
        public TrtErrorCode getErrorTypeEnum(int errorIdx)
        {
            int typeCode = NativeMethods.trtErrorRecorder_getErrorType(errorIdx);
            // 确保值在有效范围内
            // Ensure value is within valid range
            if (typeCode >= 0 && typeCode <= 10)
            {
                return (TrtErrorCode)typeCode;
            }
            return TrtErrorCode.kUNSPECIFIED_ERROR;
        }

        /// <summary>
        /// 获取指定索引的错误描述
        /// Gets the error description for the specified index
        /// </summary>
        /// <param name="errorIdx">错误索引 / Error index</param>
        /// <returns>错误描述字符串 / Error description string</returns>
        public string getErrorDesc(int errorIdx)
        {
            return NativeMethods.trtErrorRecorder_getErrorDesc(errorIdx);
        }

        /// <summary>
        /// 检查错误记录器是否溢出
        /// Checks if the error recorder has overflowed
        /// </summary>
        /// <returns>如果溢出返回true，否则返回false / Returns true if overflowed, false otherwise</returns>
        public bool hasOverflowed()
        {
            int flag = NativeMethods.trtErrorRecorder_hasOverflowed();
            return flag != 0;
        }

        /// <summary>
        /// 检查错误记录器是否为空
        /// Checks if the error recorder is empty
        /// </summary>
        /// <returns>如果为空返回true，否则返回false / Returns true if empty, false otherwise</returns>
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
        /// <param name="val">异常状态枚举 / Exception status enum</param>
        /// <param name="desc">错误描述字符串 / Error description string</param>
        /// <returns>如果错误被确定为致命错误返回true，否则返回false / Returns true if error is fatal, false otherwise</returns>
        public bool reportError(
            TrtExceptionStatus val,
            string desc)
        {
            int result = NativeMethods.trtErrorRecorder_reportError(val, desc);
            
            // 触发错误报告事件
            // Trigger error reported event
            OnErrorReported?.Invoke(val, desc);
            
            return result != 0;
        }

        /// <summary>
        /// 增加引用计数
        /// Increments the reference count
        /// </summary>
        /// <returns>增加后的引用计数值 / Reference count after increment</returns>
        public int incRefCount()
        {
            return NativeMethods.trtErrorRecorder_incRefCount();
        }

        /// <summary>
        /// 减少引用计数
        /// Decrements the reference count
        /// </summary>
        /// <returns>减少后的引用计数值 / Reference count after decrement</returns>
        public int decRefCount()
        {
            return NativeMethods.trtErrorRecorder_decRefCount();
        }

        /// <summary>
        /// 获取错误记录器的最大描述长度
        /// Gets the maximum description length for the error recorder
        /// </summary>
        /// <returns>最大描述长度（字节）/ Maximum description length in bytes</returns>
        /// <remarks>
        /// 根据TensorRT规范，安全运行时中描述长度限制为127字节（不含null终止符）
        /// Per TensorRT specification, description length is limited to 127 bytes (excluding null terminator) in safety runtime
        /// </remarks>
        public long getMaxDescLength()
        {
            return (long)NativeMethods.trtErrorRecorder_getMaxDescLength();
        }

        /// <summary>
        /// 获取所有错误的列表
        /// Gets the list of all errors
        /// </summary>
        /// <returns>错误信息列表 / List of error information</returns>
        public List<ErrorInfo> getAllErrors()
        {
            var errors = new List<ErrorInfo>();
            int count = getNbErrors();
            
            for (int i = 0; i < count; i++)
            {
                errors.Add(new ErrorInfo
                {
                    Index = i,
                    Code = getErrorCode(i),
                    Type = getErrorType(i),
                    Description = getErrorDesc(i)
                });
            }
            
            return errors;
        }

        /// <summary>
        /// 获取最后一个错误
        /// Gets the last error
        /// </summary>
        /// <returns>错误信息，如果没有错误则返回null / Error info, or null if no errors</returns>
        public ErrorInfo getLastError()
        {
            int count = getNbErrors();
            if (count == 0)
            {
                return null;
            }
            
            return new ErrorInfo
            {
                Index = count - 1,
                Code = getErrorCode(count - 1),
                Type = getErrorType(count - 1),
                Description = getErrorDesc(count - 1)
            };
        }
    }

    /// <summary>
    /// 错误信息结构体
    /// Error information structure
    /// </summary>
    public class ErrorInfo
    {
        /// <summary>
        /// 错误索引
        /// Error index
        /// </summary>
        public int Index { get; set; }

        /// <summary>
        /// 错误状态码
        /// Error status code
        /// </summary>
        public TrtExceptionStatus Code { get; set; }

        /// <summary>
        /// 错误类型代码（对应ErrorCode枚举的整数值）
        /// Error type code (integer value corresponding to ErrorCode enum)
        /// </summary>
        public int Type { get; set; }

        /// <summary>
        /// 错误类型（枚举形式）
        /// Error type (enum form)
        /// </summary>
        public TrtErrorCode TypeEnum => (Type >= 0 && Type <= 10) ? (TrtErrorCode)Type : TrtErrorCode.kUNSPECIFIED_ERROR;

        /// <summary>
        /// 错误描述
        /// Error description
        /// </summary>
        public string Description { get; set; }

        /// <summary>
        /// 判断此错误是否为致命错误
        /// Determines if this error is fatal
        /// </summary>
        public bool IsFatal => TypeEnum == TrtErrorCode.kINTERNAL_ERROR;

        /// <summary>
        /// 判断此错误是否可恢复
        /// Determines if this error is recoverable
        /// </summary>
        public bool IsRecoverable => TypeEnum == TrtErrorCode.kINVALID_STATE || 
                                     TypeEnum == TrtErrorCode.kUNSUPPORTED_STATE;

        /// <summary>
        /// 返回错误信息的字符串表示
        /// Returns string representation of error information
        /// </summary>
        public override string ToString()
        {
            return $"[{Index}] {Code} (Type: {TypeEnum}): {Description}";
        }
    }
}

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
    public class ErrorRecorder
    {
        private IntPtr ptr;
        // 私有静态实例（确保唯一）
        private static ErrorRecorder _instance;
        // 私有构造函数（防止外部实例化）
        private ErrorRecorder()
        {
            InitHandleException.handler(
                NativeMethods.trtErrorRecorder_getTrtErrorRecorder(out ptr));
        }
        // 公共静态方法，获取唯一实例
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

        public IntPtr getHandle()
        {
            return ptr;
        }
        public int getNbErrors()
        {
            return NativeMethods.trtErrorRecorder_getNbErrors();
        }

        public TrtExceptionStatus getErrorCode(int errorIdx)
        {
            return NativeMethods.trtErrorRecorder_getErrorCode(errorIdx);
        }

        public string getErrorDesc(int errorIdx)
        {
            return NativeMethods.trtErrorRecorder_getErrorDesc(errorIdx);
        }

        public bool hasOverflowed()
        {
            int flag = NativeMethods.trtErrorRecorder_hasOverflowed();
            return flag != 0;
        }
        
        public bool isEmpty()
        {
            int flag = NativeMethods.trtErrorRecorder_empty();
            return flag != 0;
        }
       

        public void clear()
        {
            NativeMethods.trtErrorRecorder_clear();
        }
       
        public bool reportError(
            TrtExceptionStatus val,
            string desc)
        {
            int result = NativeMethods.trtErrorRecorder_reportError(val, desc);
            return result != 0;
        }
        

        public int incRefCount()
        {
            return NativeMethods.trtErrorRecorder_incRefCount();
        }
        
        public int decRefCount()
        {
            return NativeMethods.trtErrorRecorder_decRefCount();
        }

    }
}

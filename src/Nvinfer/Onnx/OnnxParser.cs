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
    /// OnnxParser类，用于解析ONNX模型并将其转换为TensorRT网络定义
    /// OnnxParser class for parsing ONNX models and converting them to TensorRT network definitions
    /// </summary>
    /// <remarks>
    /// 此类继承自DisposableTrtObject，提供了解析ONNX模型的API，支持从内存中解析或从文件解析
    /// This class inherits from DisposableTrtObject and provides APIs for parsing ONNX models, 
    /// supporting parsing from memory or from files
    /// </remarks>
    public class OnnxParser : DisposableTrtObject
    {
        /// <summary>
        /// 构造函数，初始化ONNX解析器
        /// Constructor, initializes the ONNX parser
        /// </summary>
        /// <param name="networkDefinition">
        /// TensorRT网络定义对象，用于创建解析器
        /// TensorRT network definition object used to create the parser
        /// </param>
        /// <exception cref="Exception">
        /// 当创建解析器失败时抛出异常
        /// Thrown when the parser creation fails
        /// </exception>
        /// <summary>
        /// 创建构建器
        /// Creates Build
        /// </summary>
        public OnnxParser(NetworkDefinition networkDefinition)
        {
            InitHandleException.handler(
                NativeMethods.trtOnnxParser_createOnnxParser(networkDefinition.TrtPtr, out ptr));
        }

        /// <summary>
        /// 释放资源
        /// Releases the resources
        /// </summary>
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
            if (ptr != IntPtr.Zero && IsEnabledDispose)
                NativeMethods.trtBuild_free(ptr);
            base.DisposeUnmanaged();
        }

        /// <summary>
        /// 从内存中的序列化ONNX模型解析
        /// Parse from a serialized ONNX model in memory
        /// </summary>
        /// <param name="serialized_onnx_model">
        /// 序列化ONNX模型的内存地址
        /// Memory address of the serialized ONNX model
        /// </param>
        /// <param name="serialized_onnx_model_size">
        /// 序列化ONNX模型的大小（字节）
        /// Size of the serialized ONNX model in bytes
        /// </param>
        /// <param name="model_path">
        /// 模型文件路径，用于解析过程中可能需要的相对路径引用
        /// Model file path, used for relative path references that may be needed during parsing
        /// </param>
        /// <returns>
        /// 如果解析成功返回true，否则返回false
        /// Returns true if parsing is successful, otherwise false
        /// </returns>
        public bool parse(IntPtr serialized_onnx_model, long serialized_onnx_model_size, string model_path)
        {
            TrtHandleException.handler(NativeMethods.trtONNXParser_parse(ptr,
                serialized_onnx_model,
                serialized_onnx_model_size,
                model_path,
                out int flag));
            return flag != 0;
        }

        /// <summary>
        /// 从ONNX模型文件解析
        /// Parse from an ONNX model file
        /// </summary>
        /// <param name="onnxModelFile">
        /// ONNX模型文件的路径
        /// Path to the ONNX model file
        /// </param>
        /// <param name="verbosity">
        /// 详细级别，默认为0
        /// Verbosity level, default is 0
        /// </param>
        /// <returns>
        /// 如果解析成功返回true，否则返回false
        /// Returns true if parsing is successful, otherwise false
        /// </returns>
        public bool parseFromFile(string onnxModelFile, int verbosity = 0)
        {
            //sbyte[] c_onnxModelFile = (sbyte[])((Array)System.Text.Encoding.Default.GetBytes(onnxModelFile));
            TrtHandleException.handler(NativeMethods.trtONNXParser_parseFromFile(ptr,
                onnxModelFile,
                verbosity,
                out int flag));
            return flag != 0;
        }
    }


}

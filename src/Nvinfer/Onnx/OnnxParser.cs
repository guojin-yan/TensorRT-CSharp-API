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

    public class OnnxParser : DisposableTrtObject
    {


        /// <summary>
        /// Creates Build
        /// </summary>
        public OnnxParser(NetworkDefinition networkDefinition)
        {
            InitHandleException.handler(
                NativeMethods.trtOnnxParser_createOnnxParser(networkDefinition.TrtPtr, out ptr));
        }
        /// <summary>
        /// Releases the resources
        /// </summary>
        public void Release()
        {
            Dispose();
        }

        /// <inheritdoc />
        /// <summary>
        /// Releases unmanaged resources
        /// </summary>
        protected override void DisposeUnmanaged()
        {
            if (ptr != IntPtr.Zero && IsEnabledDispose)
                NativeMethods.trtBuild_free(ptr);
            base.DisposeUnmanaged();
        }

        public bool parse(IntPtr serialized_onnx_model, long serialized_onnx_model_size, string model_path)
        {
            TrtHandleException.handler(NativeMethods.trtONNXParser_parse(ptr, 
                serialized_onnx_model, 
                serialized_onnx_model_size, 
                model_path, 
                out int flag));
            return flag != 0;
        }

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

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using TensorRtSharp.Internal;

namespace TensorRtSharp.Custom
{
    public class Nvinfer : DisposableTrtObject
    {

        public static void OnnxToEngine(string modelPath) 
        {
            sbyte[] modelPathSbyte = (sbyte[])((Array)System.Text.Encoding.Default.GetBytes(modelPath));
            HandleException.Handler(NativeMethods.onnxToEngine(ref modelPathSbyte[0]));
        }

        public Nvinfer() { }

        public Nvinfer(string modelPath) 
        {
            sbyte[] modelPathSbyte = (sbyte[])((Array)System.Text.Encoding.Default.GetBytes(modelPath));
            HandleException.Handler(NativeMethods.nvinferInit(ref modelPathSbyte[0], out ptr));
        }

        public Dims GetBindingDimensions(int index) 
        {
            int l = 0;
            int[] d = new int[8];
            HandleException.Handler(NativeMethods.getBindingDimensionsByIndex(ptr, index, out l, ref d[0]));
            return new Dims(l, d);
        }

        public Dims GetBindingDimensions(string nodeName)
        {
            sbyte[] nodeNameSbyte = (sbyte[])((Array)System.Text.Encoding.Default.GetBytes(nodeName));
            int l = 0;
            int[] d = new int[8];
            HandleException.Handler(NativeMethods.getBindingDimensionsByName(ptr, ref nodeNameSbyte[0], out l, ref d[0]));
            return new Dims(l, d);
        }

        public void LoadInferenceData(string nodeName, float[] data) 
        {
            sbyte[] nodeNameSbyte = (sbyte[])((Array)System.Text.Encoding.Default.GetBytes(nodeName));
            HandleException.Handler(NativeMethods.copyFloatHostToDeviceByName(ptr, ref nodeNameSbyte[0], ref data[0]));
        }

        public void LoadInferenceData(int nodeIndex, float[] data)
        {
            HandleException.Handler(NativeMethods.copyFloatHostToDeviceByIndex(ptr, nodeIndex, ref data[0]));
        }

        public void infer() 
        {
            HandleException.Handler(NativeMethods.tensorRtInfer(ptr));
        }

        public float[] GetInferenceResult(string nodeName)
        {
            sbyte[] nodeNameSbyte = (sbyte[])((Array)System.Text.Encoding.Default.GetBytes(nodeName));
            int l = 0;
            int[] d = new int[8];
            HandleException.Handler(NativeMethods.getBindingDimensionsByName(ptr, ref nodeNameSbyte[0], out l, ref d[0]));
            Dims dims = new Dims(l, d);
            float[] data = new float[dims.prod()];
            HandleException.Handler(NativeMethods.copyFloatDeviceToHostByName(ptr, ref nodeNameSbyte[0], ref data[0]));
            return data;
        }

        public float[] GetInferenceResult(int nodeIndex)
        {
            int l = 0;
            int[] d = new int[8];
            HandleException.Handler(NativeMethods.getBindingDimensionsByIndex(ptr, nodeIndex, out l, ref d[0]));
            Dims dims = new Dims(l, d);
            float[] data = new float[dims.prod()];
            HandleException.Handler(NativeMethods.copyFloatDeviceToHostByIndex(ptr, nodeIndex, ref data[0]));
            return data;
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
                HandleException.Handler(NativeMethods.nvinferDelete(TrtPtr));
            base.DisposeUnmanaged();
        }
    }
}

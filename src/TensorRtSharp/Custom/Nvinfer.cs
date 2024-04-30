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

        public static void OnnxToEngine(string modelPath, int memorySize) 
        {
            sbyte[] modelPathSbyte = (sbyte[])((Array)System.Text.Encoding.Default.GetBytes(modelPath));
            HandleException.Handler(NativeMethods.onnxToEngine(ref modelPathSbyte[0], memorySize));
        }

        public static void OnnxToEngine(string modelPath, int memorySize, string nodeName, Dims minShapes, Dims optShapes, Dims maxShapes)
        {
            sbyte[] modelPathSbyte = (sbyte[])((Array)System.Text.Encoding.Default.GetBytes(modelPath));
            sbyte[] nodeNameSbyte = (sbyte[])((Array)System.Text.Encoding.Default.GetBytes(nodeName));
            int[] min = minShapes.d.ToArray();
            int[] opt = optShapes.d.ToArray();
            int[] max = maxShapes.d.ToArray();
            HandleException.Handler(NativeMethods.onnxToEngineDynamicShape(ref modelPathSbyte[0], memorySize, ref nodeNameSbyte[0], ref min[0], ref opt[0], ref max[0]));
        }

        public Nvinfer() { }

        public Nvinfer(string modelPath) 
        {
            sbyte[] modelPathSbyte = (sbyte[])((Array)System.Text.Encoding.Default.GetBytes(modelPath));
            HandleException.Handler(NativeMethods.nvinferInit(ref modelPathSbyte[0], out ptr));
        }
        public Nvinfer(string modelPath, int maxBatahSize)
        {
            sbyte[] modelPathSbyte = (sbyte[])((Array)System.Text.Encoding.Default.GetBytes(modelPath));
            HandleException.Handler(NativeMethods.nvinferInitDynamicShape(ref modelPathSbyte[0], maxBatahSize, out ptr));
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

        public void SetBindingDimensions(int index, Dims dims)
        {
            int l = dims.nbDims;
            int[] d = dims.d.ToArray();
            HandleException.Handler(NativeMethods.setBindingDimensionsByIndex(ptr, index, l, ref d[0]));
        }

        public void SetBindingDimensions(string nodeName, Dims dims)
        {
            sbyte[] nodeNameSbyte = (sbyte[])((Array)System.Text.Encoding.Default.GetBytes(nodeName));
            int l = dims.nbDims;
            int[] d = dims.d.ToArray();
            HandleException.Handler(NativeMethods.setBindingDimensionsByName(ptr, ref nodeNameSbyte[0], l, ref d[0]));
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
                HandleException.Handler(NativeMethods.nvinferDelete(ptr));
            base.DisposeUnmanaged();
        }
    }
}

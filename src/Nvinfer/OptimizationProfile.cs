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
    /// 用于动态输入维度和形状张量的优化配置。
    /// Optimization profile for dynamic input dimensions and shape tensors.
    /// </summary>
    /// <remarks>
    /// 当从具有动态可调整输入（至少有一个输入张量具有一个或多个指定为-1的维度）或形状输入张量的
    /// INetworkDefinition 构建 ICudaEngine 时，用户需要指定至少一个优化配置。
    /// 优化配置编号为 0、1、... 第一个定义的优化配置（索引为0）将在没有显式选择优化配置时
    /// 被 ICudaEngine 使用。
    ///
    /// When building an ICudaEngine from an INetworkDefinition that has dynamically resizable inputs
    /// (at least one input tensor has one or more of its dimensions specified as -1) or shape input tensors,
    /// users need to specify at least one optimization profile. Optimization profiles are numbered 0, 1, ...
    /// The first optimization profile that has been defined (with index 0) will be used by the ICudaEngine
    /// whenever no optimization profile has been selected explicitly.
    /// </remarks>
    public class OptimizationProfile : DisposableTrtObject
    {
        /// <summary>
        /// 创建一个空的 OptimizationProfile。
        /// Creates an empty OptimizationProfile.
        /// </summary>
        public OptimizationProfile()
        {
        }

        /// <summary>
        /// 从原生指针创建 OptimizationProfile。
        /// Creates an OptimizationProfile from a native pointer.
        /// </summary>
        /// <param name="ptr">指向原生对象的非托管指针。/ The unmanaged pointer to the native object.</param>
        internal OptimizationProfile(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
                throw new TrtException("Native object address is NULL");
            this.ptr = ptr;
        }

        /// <summary>
        /// 释放当前对象持有的所有资源。
        /// Releases all resources held by the current object.
        /// </summary>
        public void Release()
        {
            Dispose();
        }

        /// <inheritdoc />
        /// <summary>
        /// 释放所有非托管资源。
        /// Releases all unmanaged resources.
        /// </summary>
        protected override void DisposeUnmanaged()
        {
            base.DisposeUnmanaged();
        }

        /// <summary>
        /// 为动态输入张量设置最小/最佳/最大维度。
        /// Sets the minimum / optimum / maximum dimensions for a dynamic input tensor.
        /// </summary>
        public bool SetDimensions(string inputName, TrtOptProfileSelector select, Dims dims)
        {
            TrtHandleException.handler(NativeMethods.trtOptimizationProfile_setDimensions(
                ptr, inputName, select, dims, out int result));
            return result != 0;
        }

        /// <summary>
        /// 获取动态输入张量的最小/最佳/最大维度。
        /// Gets the minimum / optimum / maximum dimensions for a dynamic input tensor.
        /// </summary>
        public Dims GetDimensions(string inputName, TrtOptProfileSelector select)
        {
            TrtHandleException.handler(NativeMethods.trtOptimizationProfile_getDimensions(
                ptr, inputName, select, out Dims dims));
            return dims;
        }

        /// <summary>
        /// 为输入形状张量设置最小/最佳/最大值。
        /// Sets the minimum / optimum / maximum values for an input shape tensor.
        /// </summary>
        [Obsolete("Deprecated in TensorRT 10.11. Use SetShapeValuesV2() instead.")]
        public bool SetShapeValues(string inputName, TrtOptProfileSelector select, int[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            TrtHandleException.handler(NativeMethods.trtOptimizationProfile_setShapeValues(
                ptr, inputName, select, values, values.Length, out int result));
            return result != 0;
        }

        /// <summary>
        /// 获取输入形状张量的值数量。
        /// Gets the number of values for an input shape tensor.
        /// </summary>
        public int GetNbShapeValues(string inputName)
        {
            TrtHandleException.handler(NativeMethods.trtOptimizationProfile_getNbShapeValues(
                ptr, inputName, out int nbValues));
            return nbValues;
        }

        /// <summary>
        /// 获取输入形状张量的最小/最佳/最大值。
        /// Gets the minimum / optimum / maximum values for an input shape tensor.
        /// </summary>
        [Obsolete("Deprecated in TensorRT 10.11. Use GetShapeValuesV2() instead.")]
        public int[] GetShapeValues(string inputName, TrtOptProfileSelector select)
        {
            TrtHandleException.handler(NativeMethods.trtOptimizationProfile_getShapeValues(
                ptr, inputName, select, out IntPtr valuesPtr));

            if (valuesPtr == IntPtr.Zero)
                return null;

            int nbValues = GetNbShapeValues(inputName);
            if (nbValues <= 0)
                return null;

            int[] result = new int[nbValues];
            Marshal.Copy(valuesPtr, result, 0, nbValues);
            return result;
        }

        /// <summary>
        /// 设置此配置的目标额外 GPU 内存。
        /// Sets a target for extra GPU memory that may be used by this profile.
        /// </summary>
        public bool SetExtraMemoryTarget(float target)
        {
            TrtHandleException.handler(NativeMethods.trtOptimizationProfile_setExtraMemoryTarget(
                ptr, target, out int result));
            return result != 0;
        }

        /// <summary>
        /// 获取为此配置定义的额外内存目标。
        /// Gets the extra memory target that has been defined for this profile.
        /// </summary>
        public float GetExtraMemoryTarget()
        {
            TrtHandleException.handler(NativeMethods.trtOptimizationProfile_getExtraMemoryTarget(
                ptr, out float target));
            return target;
        }

        /// <summary>
        /// 检查优化配置是否可以传递给 IBuilderConfig 对象。
        /// Check whether the optimization profile can be passed to an IBuilderConfig object.
        /// </summary>
        public bool IsValid()
        {
            TrtHandleException.handler(NativeMethods.trtOptimizationProfile_isValid(
                ptr, out int result));
            return result != 0;
        }

        /// <summary>
        /// 为输入形状张量设置最小/最佳/最大值（V2 版本，支持 int64）。
        /// Sets the minimum / optimum / maximum values for an input shape tensor (V2 version with int64 support).
        /// </summary>
        public bool SetShapeValuesV2(string inputName, TrtOptProfileSelector select, long[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            TrtHandleException.handler(NativeMethods.trtOptimizationProfile_setShapeValuesV2(
                ptr, inputName, select, values, values.Length, out int result));
            return result != 0;
        }

        /// <summary>
        /// 获取输入形状张量的最小/最佳/最大值（V2 版本，支持 int64）。
        /// Gets the minimum / optimum / maximum values for an input shape tensor (V2 version with int64 support).
        /// </summary>
        public long[] GetShapeValuesV2(string inputName, TrtOptProfileSelector select)
        {
            TrtHandleException.handler(NativeMethods.trtOptimizationProfile_getShapeValuesV2(
                ptr, inputName, select, out IntPtr valuesPtr));

            if (valuesPtr == IntPtr.Zero)
                return null;

            int nbValues = GetNbShapeValues(inputName);
            if (nbValues <= 0)
                return null;

            long[] result = new long[nbValues];
            Marshal.Copy(valuesPtr, result, 0, nbValues);
            return result;
        }
    }
}

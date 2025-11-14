using JYPPX.TensorRtSharp.Cuda;
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
    /// 用于在不重建引擎的情况下更新现有引擎权重的类。继承自DisposableTrtObject。
    /// A class used to update weights in an existing engine without rebuilding it. Inherits from DisposableTrtObject.
    /// </summary>
    public class Refitter : DisposableTrtObject
    {

        /// <summary>
        /// 使用现有的 CudaEngine 来创建一个新的 Refitter 实例。
        /// Creates a new Refitter instance using an existing CudaEngine.
        /// </summary>
        /// <param name="cudaEngine">需要修改权重的 CudaEngine 实例。/ The CudaEngine instance whose weights are to be modified.</param>
        /// <exception cref="TrtException">如果创建 Refitter 失败。/ If the creation of the Refitter fails.</exception>
        public Refitter(CudaEngine cudaEngine)
        {
            InitHandleException.handler(
                NativeMethods.trtRefitter_createInferRefitter(cudaEngine.TrtPtr, out ptr));
        }

        /// <summary>
        /// 使用一个原生指针来初始化 Refitter 实例。主要用于内部封装。
        /// Initializes a Refitter instance from a native pointer. Primarily used for internal wrapping.
        /// </summary>
        /// <param name="ptr">指向原生对象的非托管指针。/ The unmanaged pointer to the native object.</param>
        internal Refitter(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
                throw new TrtException("Native object address is NULL");
            this.ptr = ptr;
        }

        /// <summary>
        /// 释放当前对象持有的所有资源。此方法为 Dispose 的显式别名。
        /// Releases all resources held by the current object. This method is an explicit alias for Dispose.
        /// </summary>
        public void Release()
        {
            Dispose();
        }

        /// <inheritdoc />
        /// <summary>
        /// 释放所有非托管资源。此方法由 Dispose 模式调用，不应直接调用。
        /// Releases all unmanaged resources. This method is called by the Dispose pattern and should not be called directly.
        /// </summary>
        protected override void DisposeUnmanaged()
        {
            if (ptr != IntPtr.Zero && IsEnabledDispose)
                NativeMethods.trtRefitter_free(ptr);
            base.DisposeUnmanaged();
        }

        /// <summary>
        /// 为指定层和权重角色设置新的权重。
        /// Sets new weights for a specified layer and weights role.
        /// </summary>
        /// <param name="layerName">层名。/ The name of the layer.</param>
        /// <param name="role">权重的角色（例如：kernel, bias）。/ The role of the weights (e.g., kernel, bias).</param>
        /// <param name="weights">包含新权重数据的 Weights 对象。/ A Weights object containing the new weight data.</param>
        /// <exception cref="TrtException">如果设置权重失败。/ If setting the weights fails.</exception>
        public void setWeights(string layerName, TrtWeightsRole role, Weights weights)
        {
            int success;
            TrtHandleException.handler(NativeMethods.trtRefitter_setWeights(
                ptr, layerName, role, weights.TrtPtr, out success));
            if (success == 0)
                throw new TrtException($"Failed to set weights for layer '{layerName}' with role '{role}'");
        }

        /// <summary>
        /// 执行重构操作，将所有已设置的权重应用到引擎中。
        /// Performs the refitting operation, applying all weights that have been set to the engine.
        /// </summary>
        /// <returns>如果重构成功，则为 true；否则为 false。/ True if the refitting was successful; otherwise, false.</returns>
        public bool refitCudaEngine()
        {
            int succ;
            TrtHandleException.handler(NativeMethods.trtRefitter_refitCudaEngine(
                ptr, out succ));
            return succ != 0;
        }

        /// <summary>
        /// 获取所有缺失的权重信息。这些权重是重构所需要但尚未提供的。
        /// Gets information about all missing weights. These are weights required for refitting that have not yet been provided.
        /// </summary>
        /// <param name="size">输出数组 `layerNames` 和 `roles` 的最大容量。/ The maximum capacity of the output arrays `layerNames` and `roles`.</param>
        /// <param name="layerNames">用于存储缺失权重所在层名的数组。/ An array to store the layer names of the missing weights.</param>
        /// <param name="roles">用于存储缺失权重角色的数组。/ An array to store the roles of the missing weights.</param>
        /// <param name="count">实际返回的缺失权重数量。如果 `size` 太小，则返回所需的总数。/ The actual number of missing weights returned. If `size` is too small, the total required number is returned.</param>
        public void getMissing(int size, string[] layerNames, TrtWeightsRole[] roles, out int count)
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_getMissing(
                ptr, size, layerNames, roles, out count));
        }

        /// <summary>
        /// 获取引擎中所有可重构的权重信息。
        /// Gets information for all refittable weights in the engine.
        /// </summary>
        /// <param name="size">输出数组 `layerNames` 和 `roles` 的最大容量。/ The maximum capacity of the output arrays `layerNames` and `roles`.</param>
        /// <param name="layerNames">用于存储层名的数组。/ An array to store the layer names.</param>
        /// <param name="roles">用于存储权重角色的数组。/ An array to store the weight roles.</param>
        /// <param name="count">实际返回的权重数量。如果 `size` 太小，则返回所需的总数。/ The actual number of weights returned. If `size` is too small, the total required number is returned.</param>
        public void getAll(int size, string[] layerNames, TrtWeightsRole[] roles, out int count)
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_getAll(
                ptr, size, layerNames, roles, out count));
        }

        /// <summary>
        /// 为指定的张量设置动态范围（用于INT8量化）。
        /// Sets the dynamic range for a specified tensor (for INT8 quantization).
        /// </summary>
        /// <param name="tensorName">张量名。/ The name of the tensor.</param>
        /// <param name="min">动态范围的最小值。/ The minimum value of the dynamic range.</param>
        /// <param name="max">动态范围的最大值。/ The maximum value of the dynamic range.</param>
        /// <returns>如果设置成功，则为 true；否则为 false。/ True if the operation was successful; otherwise, false.</returns>
        public bool setDynamicRange(string tensorName, float min, float max)
        {
            int success;
            TrtHandleException.handler(NativeMethods.trtRefitter_setDynamicRange(
                ptr, tensorName, min, max, out success));
            return success != 0;
        }

        /// <summary>
        /// 获取指定张量的动态范围最小值。
        /// Gets the minimum value of the dynamic range for a specified tensor.
        /// </summary>
        /// <param name="tensorName">张量名。/ The name of the tensor.</param>
        /// <returns>动态范围的最小值。/ The minimum value of the dynamic range.</returns>
        public float getDynamicRangeMin(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_getDynamicRangeMin(
                ptr, tensorName, out float min));
            return min;
        }

        /// <summary>
        /// 获取指定张量的动态范围最大值。
        /// Gets the maximum value of the dynamic range for a specified tensor.
        /// </summary>
        /// <param name="tensorName">张量名。/ The name of the tensor.</param>
        /// <returns>动态范围的最大值。/ The maximum value of the dynamic range.</returns>
        public float getDynamicRangeMax(string tensorName)
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_getDynamicRangeMax(
                ptr, tensorName, out float max));
            return max;
        }

        /// <summary>
        /// 获取所有设置了动态范围的张量的名称。
        /// Gets the names of all tensors that have a dynamic range set.
        /// </summary>
        /// <param name="size">输出数组 `tensorNames` 的最大容量。/ The maximum capacity of the output array `tensorNames`.</param>
        /// <param name="tensorNames">用于存储张量名的数组。/ An array to store the tensor names.</param>
        /// <returns>实际返回的张量数量。如果 `size` 太小，则返回所需的总数。/ The actual number of tensors returned. If `size` is too small, the total required number is returned.</returns>
        public int getTensorsWithDynamicRange(int size, string[] tensorNames)
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_getTensorsWithDynamicRange(
                ptr, size, tensorNames, out int count));
            return count;
        }

        /// <summary>
        /// 根据名称设置权重。
        /// Sets weights by name.
        /// </summary>
        /// <param name="name">权重的名称。/ The name of the weights.</param>
        /// <param name="weights">包含新权重数据的 Weights 对象。/ A Weights object containing the new weight data.</param>
        /// <returns>如果设置成功，则为 true；否则为 false。/ True if the operation was successful; otherwise, false.</returns>
        public bool setNamedWeights(string name, Weights weights)
        {
            int success;
            TrtHandleException.handler(NativeMethods.trtRefitter_setNamedWeights(
                ptr, name, weights.TrtPtr, out success));
            return success != 0;
        }

        /// <summary>
        /// 根据名称设置权重，并指定其存储位置（Device或Host）。
        /// Sets weights by name and specifies their storage location (Device or Host).
        /// </summary>
        /// <param name="name">权重的名称。/ The name of the weights.</param>
        /// <param name="weights">包含新权重数据的 Weights 对象。/ A Weights object containing the new weight data.</param>
        /// <param name="location">权重的存储位置。/ The storage location for the weights.</param>
        /// <returns>如果设置成功，则为 true；否则为 false。/ True if the operation was successful; otherwise, false.</returns>
        public bool setNamedWeightsWithLocation(string name, Weights weights, TrtTensorLocation location)
        {
            int success;
            TrtHandleException.handler(NativeMethods.trtRefitter_setNamedWeightsWithLocation(
                ptr, name, weights.TrtPtr, location, out success));
            return success != 0;
        }

        /// <summary>
        /// 获取所有缺失的命名权重的名称。
        /// Gets the names of all missing named weights.
        /// </summary>
        /// <param name="size">输出数组 `weightsNames` 的最大容量。/ The maximum capacity of the output array `weightsNames`.</param>
        /// <param name="weightsNames">用于存储权重名称的数组。/ An array to store the weight names.</param>
        /// <returns>实际返回的缺失权重数量。如果 `size` 太小，则返回所需的总数。/ The actual number of missing weights returned. If `size` is too small, the total required number is returned.</returns>
        public int getMissingWeights(int size, string[] weightsNames)
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_getMissingWeights(
                ptr, size, weightsNames, out int count));
            return count;
        }

        /// <summary>
        /// 获取引擎中所有可重构的命名权重的名称。
        /// Gets the names of all refittable named weights in the engine.
        /// </summary>
        /// <param name="size">输出数组 `weightsNames` 的最大容量。/ The maximum capacity of the output array `weightsNames`.</param>
        /// <param name="weightsNames">用于存储权重名称的数组。/ An array to store the weight names.</param>
        /// <returns>实际返回的权重数量。如果 `size` 太小，则返回所需的总数。/ The actual number of weights returned. If `size` is too small, the total required number is returned.</returns>
        public int getAllWeights(int size, string[] weightsNames)
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_getAllWeights(
                ptr, size, weightsNames, out int count));
            return count;
        }

        /// <summary>
        /// 设置重构操作可以使用的最大线程数。
        /// Sets the maximum number of threads that can be used for the refitting operation.
        /// </summary>
        /// <param name="maxThreads">最大线程数。/ The maximum number of threads.</param>
        /// <returns>如果设置成功，则为 true；否则为 false。/ True if the operation was successful; otherwise, false.</returns>
        public bool setMaxThreads(int maxThreads)
        {
            int success;
            TrtHandleException.handler(NativeMethods.trtRefitter_setMaxThreads(
                ptr, maxThreads, out success));
            return success != 0;
        }

        /// <summary>
        /// 获取重构操作可以使用的最大线程数。
        /// Gets the maximum number of threads that can be used for the refitting operation.
        /// </summary>
        /// <returns>最大线程数。/ The maximum number of threads.</returns>
        public int getMaxThreads()
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_getMaxThreads(
                ptr, out int maxThreads));
            return maxThreads;
        }

        /// <summary>
        /// 根据名称获取权重。
        /// Gets the weights by name.
        /// </summary>
        /// <param name="name">权重的名称。/ The name of the weights.</param>
        /// <returns>一个包含所请求权重的新 Weights 对象。/ A new Weights object containing the requested weights.</returns>
        public Weights getNamedWeights(string name)
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_getNamedWeights(
                ptr, name, out IntPtr weightsPtr));
            return new Weights(weightsPtr);
        }

        /// <summary>
        /// 获取指定名称权重的存储位置（Device或Host）。
        /// Gets the storage location (Device or Host) for the weights with the specified name.
        /// </summary>
        /// <param name="name">权重的名称。/ The name of the weights.</param>
        /// <returns>权重的存储位置。/ The storage location of the weights.</returns>
        public TrtTensorLocation getWeightsLocation(string name)
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_getWeightsLocation(
                ptr, name, out TrtTensorLocation location));
            return location;
        }

        /// <summary>
        /// 取消设置指定名称的权重，使其在重构时不再被应用。
        /// Unsets the weights with the specified name, so they are no longer applied during refitting.
        /// </summary>
        /// <param name="name">要取消设置的权重名称。/ The name of the weights to unset.</param>
        /// <returns>如果操作成功，则为 true；否则为 false。/ True if the operation was successful; otherwise, false.</returns>
        public bool unsetNamedWeights(string name)
        {
            int success;
            TrtHandleException.handler(NativeMethods.trtRefitter_unsetNamedWeights(
                ptr, name, out success));
            return success != 0;
        }

        /// <summary>
        /// 设置是否在设置权重时验证权重。
        /// Sets whether to validate weights when they are set.
        /// </summary>
        /// <param name="weightsValidation">1表示启用验证，0表示禁用。/ 1 to enable validation, 0 to disable.</param>
        public void setWeightsValidation(int weightsValidation)
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_setWeightsValidation(
                ptr, weightsValidation));
        }

        /// <summary>
        /// 获取当前权重验证的设置状态。
        /// Gets the current state of weight validation.
        /// </summary>
        /// <returns>如果启用了验证，则为 1；否则为 0。/ 1 if validation is enabled, otherwise 0.</returns>
        public int getWeightsValidation()
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_getWeightsValidation(
                ptr, out int weightsValidation));
            return weightsValidation;
        }

        /// <summary>
        /// 在指定的CUDA流上异步执行重构操作。
        /// Asynchronously performs the refitting operation on the specified CUDA stream.
        /// </summary>
        /// <param name="stream">用于执行操作的 CudaStream。/ The CudaStream to perform the operation on.</param>
        /// <returns>如果异步重构任务成功提交，则为 true；否则为 false。/ True if the async refit task was successfully submitted; otherwise, false.</returns>
        public bool refitCudaEngineAsync(CudaStream stream)
        {
            int success;
            TrtHandleException.handler(NativeMethods.trtRefitter_refitCudaEngineAsync(
                ptr, stream.TrtPtr, out success));
            return success != 0;
        }

        /// <summary>
        /// 获取指定名称权重的原型信息（如形状、数据类型等），但不包含实际权值数据。
        /// Gets the prototype information (e.g., shape, data type) for the weights of the specified name, without the actual weight values.
        /// </summary>
        /// <param name="weightsName">权重的名称。/ The name of the weights.</param>
        /// <returns>一个包含权重原型信息的新 Weights 对象。/ A new Weights object containing the weights prototype information.</returns>
        public Weights getWeightsPrototype(string weightsName)
        {
            TrtHandleException.handler(NativeMethods.trtRefitter_getWeightsPrototype(
                ptr, weightsName, out IntPtr weightsPtr));
            return new Weights(weightsPtr);
        }
    }

}

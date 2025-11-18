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
    /// 表示一个张量（Tensor）对象，是TensorRT网络中的基本数据单元。继承自DisposableTrtObject。
    /// Represents a Tensor object, which is the fundamental data unit in a TensorRT network. Inherits from DisposableTrtObject.
    /// </summary>
    public class Tensor : DisposableTrtObject
    {

        /// <summary>
        /// 创建一个空的、未初始化的 Tensor 对象。
        /// Creates an empty, uninitialized Tensor object.
        /// 注意：许多方法仅在指针已设置时才有效。/ Note: Many methods are only valid when the pointer has been set.
        public Tensor()
        {
        }

        /// <summary>
        /// 使用一个原生指针来初始化 Tensor 实例。主要用于内部封装。
        /// Initializes a Tensor instance from a native pointer. Primarily used for internal wrapping.
        /// </summary>
        /// <param name="ptr">指向原生对象的非托管指针。/ The unmanaged pointer to the native object.</param>
        public Tensor(IntPtr ptr)
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
                NativeMethods.trtTensor_free(ptr);
            base.DisposeUnmanaged();
        }

        /// <summary>
        /// 设置张量的名称。
        /// Sets the name of the tensor.
        /// </summary>
        /// <param name="name">要设置的张量名称。/ The name to be set for the tensor.</param>
        public void setName(string name)
        {
            TrtHandleException.handler(NativeMethods.trtTensor_setName(ptr, name));
        }

        /// <summary>
        /// 获取张量的名称。
        /// Gets the name of the tensor.
        /// </summary>
        /// <returns>张量名称。如果未设置，则返回空字符串。/ The tensor name. Returns an empty string if not set.</returns>
        public string getName()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_getName(ptr, out IntPtr namePtr));
            return Marshal.PtrToStringAnsi(namePtr) ?? string.Empty;
        }

        /// <summary>
        /// 设置张量的维度。
        /// Sets the dimensions of the tensor.
        /// </summary>
        /// <param name="dimensions">一个 Dims 对象，表示张量的维度。/ A Dims object representing the dimensions of the tensor.</param>
        public void setDimensions(Dims dimensions)
        {
            TrtHandleException.handler(NativeMethods.trtTensor_setDimensions(ptr, dimensions));
        }

        /// <summary>
        /// 获取张量的维度。
        /// Gets the dimensions of the tensor.
        /// </summary>
        /// <returns>一个 Dims 对象，表示张量的维度。/ A Dims object representing the dimensions of the tensor.</returns>
        public Dims getDimensions()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_getDimensions(ptr, out Dims dims));
            return dims;
        }

        /// <summary>
        /// 设置张量的数据类型。
        /// Sets the data type of the tensor.
        /// </summary>
        /// <param name="type">一个 TrtDataType 枚举值，表示数据类型。/ A TrtDataType enumeration value representing the data type.</param>
        public void setType(TrtDataType type)
        {
            TrtHandleException.handler(NativeMethods.trtTensor_setType(ptr, type));
        }

        /// <summary>
        /// 获取张量的数据类型。
        /// Gets the data type of the tensor.
        /// </summary>
        /// <returns>一个 TrtDataType 枚举值。/ A TrtDataType enumeration value.</returns>
        public TrtDataType getType()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_getType(ptr, out TrtDataType type));
            return type;
        }

        /// <summary>
        /// 设置张量的动态范围（用于INT8量化）。
        /// Sets the dynamic range of the tensor (for INT8 quantization).
        /// </summary>
        /// <param name="min">动态范围的最小值。/ The minimum value of the dynamic range.</param>
        /// <param name="max">动态范围的最大值。/ The maximum value of the dynamic range.</param>
        /// <param name="success">输出参数，1表示设置成功，0表示失败。/ Output parameter, 1 for success, 0 for failure.</param>
        public void setDynamicRange(float min, float max, out int success)
        {
            TrtHandleException.handler(NativeMethods.trtTensor_setDynamicRange(ptr, min, max, out success));
        }

        /// <summary>
        /// 检查张量是否为网络的输入。
        /// Checks if the tensor is an input to the network.
        /// </summary>
        /// <returns>如果是网络输入，则为 true；否则为 false。/ True if it is a network input, otherwise false.</returns>
        public bool isNetworkInput()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_isNetworkInput(ptr, out int isInput));
            return isInput != 0;
        }

        /// <summary>
        /// 检查张量是否为网络的输出。
        /// Checks if the tensor is an output of the network.
        /// </summary>
        /// <returns>如果是网络输出，则为 true；否则为 false。/ True if it is a network output, otherwise false.</returns>
        public bool isNetworkOutput()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_isNetworkOutput(ptr, out int isOutput));
            return isOutput != 0;
        }

        /// <summary>
        /// 设置张量是否跨批次广播。如果设置，张量的大小将不包含批次维度。
        /// Sets whether the tensor is broadcast across batch. If set, the tensor's size does not include the batch dimension.
        /// </summary>
        /// <param name="broadcast">非零值表示启用跨批次广播，0表示禁用。/ A non-zero value to enable broadcast across batch, 0 to disable.</param>
        public void setBroadcastAcrossBatch(int broadcast)
        {
            TrtHandleException.handler(NativeMethods.trtTensor_setBroadcastAcrossBatch(ptr, broadcast));
        }

        /// <summary>
        /// 获取张量是否跨批次广播的设置。
        /// Gets the setting of whether the tensor is broadcast across batch.
        /// </summary>
        /// <returns>非零值表示已启用，0表示禁用。/ A non-zero value if enabled, 0 if disabled.</returns>
        public int getBroadcastAcrossBatch()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_getBroadcastAcrossBatch(ptr, out int broadcast));
            return broadcast;
        }

        /// <summary>
        /// 获取张量的存储位置（设备或主机）。
        /// Gets the location of the tensor (device or host).
        /// </summary>
        /// <returns>一个 TrtTensorLocation 枚举值。/ A TrtTensorLocation enumeration value.</returns>
        public TrtTensorLocation getLocation()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_getLocation(ptr,
                out TrtTensorLocation location));
            return location;
        }

        /// <summary>
        /// 设置张量的存储位置（设备或主机）。
        /// Sets the location of the tensor (device or host).
        /// </summary>
        /// <param name="location">一个 TrtTensorLocation 枚举值。/ A TrtTensorLocation enumeration value.</param>
        public void setLocation(TrtTensorLocation location)
        {
            TrtHandleException.handler(NativeMethods.trtTensor_setLocation(ptr, location));
        }

        /// <summary>
        /// 检查是否为张量设置了动态范围。
        /// Checks if a dynamic range is set for the tensor.
        /// </summary>
        /// <returns>如果设置了动态范围，则为 true；否则为 false。/ True if a dynamic range is set, otherwise false.</returns>
        public bool dynamicRangeIsSet()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_dynamicRangeIsSet(ptr, out int isSet));
            return isSet != 0;
        }

        /// <summary>
        /// 重置（清除）张量的动态范围。
        /// Resets (clears) the dynamic range of the tensor.
        /// </summary>
        public void resetDynamicRange()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_resetDynamicRange(ptr));
        }

        /// <summary>
        /// 获取张量动态范围的最小值。
        /// Gets the minimum value of the tensor's dynamic range.
        /// </summary>
        /// <returns>动态范围的最小值。/ The minimum value of the dynamic range.</returns>
        public float getDynamicRangeMin()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_getDynamicRangeMin(ptr, out float min));
            return min;
        }

        /// <summary>
        /// 获取张量动态范围的最大值。
        /// Gets the maximum value of the tensor's dynamic range.
        /// </summary>
        /// <returns>动态范围的最大值。/ The maximum value of the dynamic range.</returns>
        public float getDynamicRangeMax()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_getDynamicRangeMax(ptr, out float max));
            return max;
        }

        /// <summary>
        /// 获取张量允许的格式。
        /// Gets the allowed formats for the tensor.
        /// </summary>
        /// <returns>一个 TrtTensorFormat 枚举值，是允许格式的位掩码。/ A TrtTensorFormat enumeration value, which is a bitmask of allowed formats.</returns>
        public TrtTensorFormat getAllowedFormats()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_getAllowedFormats(ptr, out TrtTensorFormat format));
            return format;
        }

        /// <summary>
        /// 检查张量是否为形状张量（shape tensor）。
        /// Checks if the tensor is a shape tensor.
        /// </summary>
        /// <returns>如果是形状张量，则为 true；否则为 false。/ True if it is a shape tensor, otherwise false.</returns>
        public bool isShapeTensor()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_isShapeTensor(ptr, out int isShape));
            return isShape != 0;
        }

        /// <summary>
        /// 检查张量是否为执行张量（execution tensor），即参与实际计算的数据张量。
        /// Checks if the tensor is an execution tensor, i.e., a data tensor participating in actual calculations.
        /// </summary>
        /// <returns>如果是执行张量，则为 true；否则为 false。/ True if it is an execution tensor, otherwise false.</returns>
        public bool isExecutionTensor()
        {
            TrtHandleException.handler(NativeMethods.trtTensor_isExecutionTensor(ptr, out int isExec));
            return isExec != 0;
        }

        /// <summary>
        /// 为张量的指定维度索引设置名称。
        /// Sets the name for a specific dimension index of the tensor.
        /// </summary>
        /// <param name="index">维度索引。/ The dimension index.</param>
        /// <param name="name">维度的名称。/ The name for the dimension.</param>
        public void setDimensionName(int index, string name)
        {
            TrtHandleException.handler(NativeMethods.trtTensor_setDimensionName(ptr, index, name));
        }

        /// <summary>
        /// 获取张量指定维度索引的名称。
        /// Gets the name for a specific dimension index of the tensor.
        /// </summary>
        /// <param name="index">维度索引。/ The dimension index.</param>
        /// <returns>维度的名称。如果未设置，则返回空字符串。/ The name of the dimension. Returns an empty string if not set.</returns>
        public string getDimensionName(int index)
        {
            TrtHandleException.handler(NativeMethods.trtTensor_getDimensionName(ptr, index, out IntPtr namePtr));
            return Marshal.PtrToStringAnsi(namePtr) ?? string.Empty;
        }
    }

}
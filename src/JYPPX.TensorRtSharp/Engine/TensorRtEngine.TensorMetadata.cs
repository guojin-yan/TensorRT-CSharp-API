using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Managed wrapper around a TensorRT engine.
/// TensorRT engine 的托管封装。
/// </summary>
public sealed partial class TensorRtEngine
{
    /// <summary>
    /// Gets high-level metadata for one engine I/O tensor.
    /// 获取一个 engine I/O tensor 的高层元数据。
    /// </summary>
    /// <param name="index">The zero-based tensor index. 从零开始的 tensor 索引。</param>
    /// <returns>The tensor metadata. tensor 元数据。</returns>
    public TensorRtTensorInfo GetIOTensorInfo(int index)
    {
        return BridgeInfoMapper.ToManaged(NativeBridgeApi.GetEngineIOTensorInfo(Line, _handle, index));
    }

    /// <summary>
    /// Gets the name of one engine I/O tensor.
    /// 获取一个 engine I/O tensor 的名称。
    /// </summary>
    /// <param name="index">The zero-based tensor index. 从零开始的 tensor 索引。</param>
    /// <returns>The TensorRT tensor name. TensorRT tensor 名称。</returns>
    public string GetIOTensorName(int index)
    {
        if (Line == TensorRtApiLine.TensorRt11)
        {
            return GetIOTensorInfo(index).Name;
        }

        return NativeBridgeApi.GetEngineIOTensorName(Line, _handle, index);
    }

    /// <summary>
    /// Gets the zero-based index for one named engine tensor.
    /// 获取一个已命名 engine tensor 的从零开始索引。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <returns>The zero-based tensor index. 从零开始的 tensor 索引。</returns>
    public int GetTensorIndex(string tensorName)
    {
        if (Line == TensorRtApiLine.TensorRt11)
        {
            if (string.IsNullOrWhiteSpace(tensorName))
            {
                throw new ArgumentException("Tensor name must not be null or empty.", nameof(tensorName));
            }

            int tensorCount = IOTensorCount;
            for (int index = 0; index < tensorCount; index++)
            {
                if (string.Equals(GetIOTensorInfo(index).Name, tensorName, StringComparison.Ordinal))
                {
                    return index;
                }
            }

            throw new TensorRtException(BridgeStatusCode.NotFound, BridgeErrorCategory.TensorRt, $"Tensor '{tensorName}' was not found in this TensorRT 11 engine.");
        }

        return NativeBridgeApi.GetEngineTensorIndex(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets the TensorRT data type for one engine tensor.
    /// 获取一个 engine tensor 的 TensorRT 数据类型。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <returns>The TensorRT tensor data type. TensorRT tensor 数据类型。</returns>
    public TensorRtDataType GetTensorDataType(string tensorName)
    {
        return NativeBridgeApi.GetEngineTensorDataType(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets the shape of one engine tensor.
    /// 获取一个 engine tensor 的形状。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <returns>The tensor shape. tensor 形状。</returns>
    public TensorRtDims GetTensorShape(string tensorName)
    {
        return NativeBridgeApi.GetEngineTensorShape(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets a TensorRT 11 engine tensor shape with 64-bit dimension extents.
    /// 获取 TensorRT 11 引擎张量形状，并保留 64 位维度 extent。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. 引擎张量名称。</param>
    /// <returns>The tensor shape reported by TensorRT. TensorRT 报告的张量形状。</returns>
    public TensorRtDims64 GetTensorShape64(string tensorName)
    {
        return NativeBridgeApi.GetEngineTensorShape64(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets one TensorRT 11 engine tensor dimension extent as a 64-bit value.
    /// 以 64 位整数获取 TensorRT 11 引擎张量的单个维度 extent。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. 引擎张量名称。</param>
    /// <param name="dimensionIndex">The zero-based dimension index. 从零开始的维度索引。</param>
    /// <returns>The dimension extent reported by TensorRT. TensorRT 报告的维度 extent。</returns>
    public long GetTensorDimensionExtent64(string tensorName, int dimensionIndex)
    {
        return NativeBridgeApi.GetEngineTensorDimensionExtent64(Line, _handle, tensorName, dimensionIndex);
    }

    /// <summary>
    /// Gets whether one engine tensor is an input or output tensor.
    /// 获取一个 engine tensor 是输入还是输出。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <returns>The TensorRT I/O mode. TensorRT I/O 模式。</returns>
    public TensorRtIOMode GetTensorIOMode(string tensorName)
    {
        return NativeBridgeApi.GetEngineTensorIOMode(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets the TensorRT tensor location for one engine tensor.
    /// 获取一个 engine tensor 的 TensorRT tensor location。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <returns>The TensorRT tensor location. TensorRT tensor 的内存位置。</returns>
    public TensorRtTensorLocation GetTensorLocation(string tensorName)
    {
        return NativeBridgeApi.GetEngineTensorLocation(Line, _handle, tensorName);
    }

    /// <summary>
    /// Returns whether one engine tensor participates in shape inference.
    /// 返回一个 engine tensor 是否参与 shape inference。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <returns><see langword="true"/> when the tensor is shape-inference I/O. 当该 tensor 是 shape inference I/O 时返回 <see langword="true"/>。</returns>
    public bool IsShapeInferenceIO(string tensorName)
    {
        return NativeBridgeApi.IsEngineShapeInferenceIO(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets the bytes-per-component value for one engine tensor.
    /// 获取一个 engine tensor 的每分量字节数。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <returns>The bytes-per-component value. 每分量字节数。</returns>
    public int GetTensorBytesPerComponent(string tensorName)
    {
        return NativeBridgeApi.GetEngineTensorBytesPerComponent(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets the profile-specific bytes-per-component value for one engine tensor.
    /// 获取一个 engine tensor 在指定 profile 下的每分量字节数。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <param name="profileIndex">The optimization profile index. optimization profile 索引。</param>
    /// <returns>The bytes-per-component value. 每分量字节数。</returns>
    public int GetTensorBytesPerComponent(string tensorName, int profileIndex)
    {
        return NativeBridgeApi.GetEngineTensorBytesPerComponent(Line, _handle, tensorName, profileIndex);
    }

    /// <summary>
    /// Gets the components-per-element value for one engine tensor.
    /// 获取一个 engine tensor 的每元素分量数。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <returns>The components-per-element value. 每元素分量数。</returns>
    public int GetTensorComponentsPerElement(string tensorName)
    {
        return NativeBridgeApi.GetEngineTensorComponentsPerElement(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets the profile-specific components-per-element value for one engine tensor.
    /// 获取一个 engine tensor 在指定 profile 下的每元素分量数。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <param name="profileIndex">The optimization profile index. optimization profile 索引。</param>
    /// <returns>The components-per-element value. 每元素分量数。</returns>
    public int GetTensorComponentsPerElement(string tensorName, int profileIndex)
    {
        return NativeBridgeApi.GetEngineTensorComponentsPerElement(Line, _handle, tensorName, profileIndex);
    }

    /// <summary>
    /// Gets the TensorRT tensor format for one engine tensor.
    /// 获取一个 engine tensor 的 TensorRT tensor format。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <returns>The TensorRT tensor format. TensorRT tensor 格式。</returns>
    public TensorRtTensorFormat GetTensorFormat(string tensorName)
    {
        return NativeBridgeApi.GetEngineTensorFormat(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets the profile-specific TensorRT tensor format for one engine tensor.
    /// 获取一个 engine tensor 在指定 profile 下的 TensorRT tensor format。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <param name="profileIndex">The optimization profile index. optimization profile 索引。</param>
    /// <returns>The TensorRT tensor format. TensorRT tensor 格式。</returns>
    public TensorRtTensorFormat GetTensorFormat(string tensorName, int profileIndex)
    {
        return NativeBridgeApi.GetEngineTensorFormat(Line, _handle, tensorName, profileIndex);
    }

    /// <summary>
    /// Gets TensorRT's human-readable tensor format description.
    /// 获取 TensorRT 返回的可读 tensor format 描述。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. Engine tensor 名称。</param>
    /// <returns>The TensorRT tensor format description. TensorRT tensor format 描述。</returns>
    public string GetTensorFormatDescription(string tensorName)
    {
        return NativeBridgeApi.GetEngineTensorFormatDescription(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets TensorRT's profile-specific human-readable tensor format description.
    /// 获取 TensorRT 针对指定 profile 返回的可读 tensor format 描述。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. Engine tensor 名称。</param>
    /// <param name="profileIndex">The optimization profile index. Optimization profile 索引。</param>
    /// <returns>The TensorRT tensor format description. TensorRT tensor format 描述。</returns>
    public string GetTensorFormatDescription(string tensorName, int profileIndex)
    {
        return NativeBridgeApi.GetEngineTensorFormatDescription(Line, _handle, tensorName, profileIndex);
    }

    /// <summary>
    /// Gets the vectorized dimension index for one engine tensor.
    /// 获取一个 engine tensor 的向量化维度索引。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <returns>The vectorized dimension index, or TensorRT's sentinel value. 向量化维度索引，或 TensorRT 的哨兵值。</returns>
    public int GetTensorVectorizedDimension(string tensorName)
    {
        return NativeBridgeApi.GetEngineTensorVectorizedDimension(Line, _handle, tensorName);
    }

    /// <summary>
    /// Gets the profile-specific vectorized dimension index for one engine tensor.
    /// 获取一个 engine tensor 在指定 profile 下的向量化维度索引。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <param name="profileIndex">The optimization profile index. optimization profile 索引。</param>
    /// <returns>The vectorized dimension index, or TensorRT's sentinel value. 向量化维度索引，或 TensorRT 的哨兵值。</returns>
    public int GetTensorVectorizedDimension(string tensorName, int profileIndex)
    {
        return NativeBridgeApi.GetEngineTensorVectorizedDimension(Line, _handle, tensorName, profileIndex);
    }

    /// <summary>
    /// Gets one optimization-profile shape for one engine tensor.
    /// 获取一个 engine tensor 在某个 optimization profile 中的形状。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. engine tensor 名称。</param>
    /// <param name="profileIndex">The optimization profile index. optimization profile 索引。</param>
    /// <param name="selector">The min/opt/max selector. min/opt/max 选择器。</param>
    /// <returns>The profile shape. profile 形状。</returns>
    public TensorRtDims GetProfileShape(string tensorName, int profileIndex, TensorRtOptimizationProfileSelector selector)
    {
        return NativeBridgeApi.GetEngineProfileShape(Line, _handle, tensorName, profileIndex, selector);
    }

    /// <summary>
    /// Gets TensorRT 8 legacy input shape-binding values for one optimization profile.
    /// 获取 TensorRT 8 legacy input shape binding 在某个 optimization profile 下的取值。
    /// </summary>
    /// <param name="bindingIndex">The legacy binding index. legacy binding 索引。</param>
    /// <param name="profileIndex">The optimization profile index. optimization profile 索引。</param>
    /// <param name="selector">The min/opt/max selector. min/opt/max 选择器。</param>
    /// <returns>Caller-owned copied shape-binding values. 调用方拥有的 shape-binding 值副本。</returns>
    public int[] GetProfileShapeValues(int bindingIndex, int profileIndex, TensorRtOptimizationProfileSelector selector)
    {
        return NativeBridgeApi.GetEngineProfileShapeValues(Line, _handle, bindingIndex, profileIndex, selector);
    }

    /// <summary>
    /// Gets one TensorRT 11 engine profile shape with 64-bit dimension extents.
    /// 获取 TensorRT 11 引擎中某个 profile selector 的形状，并保留 64 位维度 extent。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. 引擎张量名称。</param>
    /// <param name="profileIndex">The optimization profile index. 优化 profile 索引。</param>
    /// <param name="selector">The min/opt/max profile selector. min/opt/max profile 选择器。</param>
    /// <returns>The profile shape reported by TensorRT. TensorRT 报告的 profile 形状。</returns>
    public TensorRtDims64 GetProfileShape64(string tensorName, int profileIndex, TensorRtOptimizationProfileSelector selector)
    {
        return NativeBridgeApi.GetEngineProfileShape64(Line, _handle, tensorName, profileIndex, selector);
    }

    /// <summary>
    /// Gets one dimension extent from a TensorRT 11 engine profile shape as a 64-bit value.
    /// 以 64 位整数获取 TensorRT 11 引擎 profile shape 的单个维度 extent。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. 引擎张量名称。</param>
    /// <param name="profileIndex">The optimization profile index. 优化 profile 索引。</param>
    /// <param name="selector">The min/opt/max profile selector. min/opt/max profile 选择器。</param>
    /// <param name="dimensionIndex">The zero-based dimension index. 从零开始的维度索引。</param>
    /// <returns>The profile dimension extent reported by TensorRT. TensorRT 报告的 profile 维度 extent。</returns>
    public long GetProfileShapeDimensionExtent64(string tensorName, int profileIndex, TensorRtOptimizationProfileSelector selector, int dimensionIndex)
    {
        return NativeBridgeApi.GetEngineProfileShapeDimensionExtent64(Line, _handle, tensorName, profileIndex, selector, dimensionIndex);
    }

    /// <summary>
    /// Gets the device-memory requirement for a specific optimization profile.
    /// 获取指定 optimization profile 的设备内存需求。
    /// </summary>
    /// <param name="profileIndex">The optimization profile index. Optimization profile 索引。</param>
    /// <returns>The required bytes reported by TensorRT. TensorRT 报告的字节数。</returns>
    public ulong GetDeviceMemorySizeForProfile(int profileIndex)
    {
        return NativeBridgeApi.GetEngineDeviceMemorySizeForProfile(Line, _handle, profileIndex);
    }

    /// <summary>
    /// Gets the TensorRT 10 V2 device-memory requirement for a specific optimization profile.
    /// 获取指定 optimization profile 的 TensorRT 10 V2 设备内存需求。
    /// </summary>
    /// <param name="profileIndex">The optimization profile index. Optimization profile 索引。</param>
    /// <returns>The required bytes reported by TensorRT. TensorRT 报告的字节数。</returns>
    public ulong GetDeviceMemorySizeForProfileV2(int profileIndex)
    {
        return NativeBridgeApi.GetEngineDeviceMemorySizeForProfileV2(Line, _handle, profileIndex);
    }

    /// <summary>
    /// Gets whether TensorRT marks the named tensor as a debug tensor.
    /// 获取 TensorRT 是否将指定 tensor 标记为 debug tensor。
    /// </summary>
    /// <param name="tensorName">The engine tensor name. Engine tensor 名称。</param>
    /// <returns><c>true</c> when the tensor is a debug tensor. 如果该 tensor 是 debug tensor，则返回 <c>true</c>。</returns>
    public bool IsDebugTensor(string tensorName)
    {
        return NativeBridgeApi.IsEngineDebugTensor(Line, _handle, tensorName);
    }

}

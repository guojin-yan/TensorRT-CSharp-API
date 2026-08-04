using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Managed wrapper around a TensorRT ONNX parser configuration.
/// TensorRT ONNX parser 配置的托管封装。
/// </summary>
/// <remarks>
/// This wrapper owns the native ONNX config handle and exposes only copied scalar values.
/// 当前封装拥有 native ONNX config 句柄，并且只暴露复制后的标量值。
/// </remarks>
public sealed class TensorRtOnnxConfig : IDisposable
{
    private readonly SafeTensorRtObjectHandle _handle;
    private bool _disposed;

    /// <summary>
    /// Creates a TensorRT ONNX parser configuration for the selected API line.
    /// 为指定 TensorRT API line 创建 ONNX parser 配置。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    public TensorRtOnnxConfig(TensorRtApiLine line)
        : this(line, NativeBridgeApi.CreateOnnxConfig(line))
    {
    }

    internal TensorRtOnnxConfig(TensorRtApiLine line, SafeTensorRtObjectHandle handle)
    {
        Line = line;
        _handle = handle ?? throw new ArgumentNullException(nameof(handle));
    }

    internal SafeTensorRtObjectHandle Handle => _handle;

    /// <summary>
    /// Gets the TensorRT API line used by this ONNX config.
    /// 获取当前 ONNX config 使用的 TensorRT API 版本线。
    /// </summary>
    public TensorRtApiLine Line { get; }

    /// <summary>
    /// Gets or sets the model data type TensorRT should assume while parsing ONNX.
    /// 获取或设置 TensorRT 解析 ONNX 时采用的模型数据类型。
    /// </summary>
    /// <remarks>
    /// TensorRT ONNX config accepts only Float, Half, and Int8 for this scalar setting.
    /// TensorRT ONNX config 的此标量设置仅接受 Float、Half 和 Int8。
    /// </remarks>
    public TensorRtDataType ModelDataType
    {
        get => NativeBridgeApi.GetOnnxConfigModelDataType(Line, _handle);
        set => NativeBridgeApi.SetOnnxConfigModelDataType(Line, _handle, value);
    }

    /// <summary>
    /// Gets or sets the ONNX parser verbosity level.
    /// 获取或设置 ONNX parser 日志详细级别。
    /// </summary>
    public int VerbosityLevel
    {
        get => NativeBridgeApi.GetOnnxConfigVerbosityLevel(Line, _handle);
        set => NativeBridgeApi.SetOnnxConfigVerbosityLevel(Line, _handle, value);
    }

    /// <summary>
    /// Gets or sets the ONNX model file name stored in this parser configuration.
    /// 获取或设置当前 parser 配置中保存的 ONNX model 文件名。
    /// </summary>
    /// <remarks>
    /// The value is copied into a bridge-owned configuration object and copied back through a caller-provided buffer.
    /// 该值会复制到 bridge-owned 配置对象中，读取时通过调用方缓冲区复制回来。
    /// </remarks>
    public string ModelFileName
    {
        get => NativeBridgeApi.GetOnnxConfigModelFileName(Line, _handle);
        set => NativeBridgeApi.SetOnnxConfigModelFileName(Line, _handle, value);
    }

    /// <summary>
    /// Gets or sets the ONNX parser text output file name.
    /// 获取或设置 ONNX parser text 输出文件名。
    /// </summary>
    /// <remarks>
    /// This property stores metadata on the bridge-owned config. It does not open, create, or own the file.
    /// 该属性只在 bridge-owned config 中保存元数据；不会打开、创建或持有该文件。
    /// </remarks>
    public string TextFileName
    {
        get => NativeBridgeApi.GetOnnxConfigTextFileName(Line, _handle);
        set => NativeBridgeApi.SetOnnxConfigTextFileName(Line, _handle, value);
    }

    /// <summary>
    /// Gets or sets the ONNX parser full text output file name.
    /// 获取或设置 ONNX parser full text 输出文件名。
    /// </summary>
    /// <remarks>
    /// This property stores metadata on the bridge-owned config. It does not expose native string pointers.
    /// 该属性只在 bridge-owned config 中保存元数据；不会暴露 native 字符串指针。
    /// </remarks>
    public string FullTextFileName
    {
        get => NativeBridgeApi.GetOnnxConfigFullTextFileName(Line, _handle);
        set => NativeBridgeApi.SetOnnxConfigFullTextFileName(Line, _handle, value);
    }

    /// <summary>
    /// Gets or sets whether TensorRT should print layer information while parsing.
    /// 获取或设置 TensorRT 解析时是否打印 layer 信息。
    /// </summary>
    public bool PrintLayerInfo
    {
        get => NativeBridgeApi.GetOnnxConfigPrintLayerInfo(Line, _handle);
        set => NativeBridgeApi.SetOnnxConfigPrintLayerInfo(Line, _handle, value);
    }

    /// <summary>
    /// Increases the ONNX parser verbosity level by one when the native config supports it.
    /// 在 native config 支持时将 ONNX parser 日志详细级别增加一级。
    /// </summary>
    public void IncreaseVerbosity()
    {
        NativeBridgeApi.IncreaseOnnxConfigVerbosity(Line, _handle);
    }

    /// <summary>
    /// Decreases the ONNX parser verbosity level by one without going below zero.
    /// 将 ONNX parser 日志详细级别降低一级，且不会降到零以下。
    /// </summary>
    public void DecreaseVerbosity()
    {
        NativeBridgeApi.DecreaseOnnxConfigVerbosity(Line, _handle);
    }

    /// <summary>
    /// Copies the current ONNX parser configuration into a pointer-free managed snapshot.
    /// 将当前 ONNX parser 配置复制为不含指针的托管快照。
    /// </summary>
    /// <remarks>
    /// Strings are copied through caller-owned buffers by the native bridge before this snapshot is created.
    /// 字符串会先通过 native bridge 的调用方缓冲区复制回来，再进入该快照。
    /// </remarks>
    /// <returns>A copied ONNX config snapshot. 复制后的 ONNX config 快照。</returns>
    public TensorRtOnnxConfigSnapshot ToSnapshot()
    {
        return new TensorRtOnnxConfigSnapshot(
            Line,
            ModelDataType,
            VerbosityLevel,
            ModelFileName,
            TextFileName,
            FullTextFileName,
            PrintLayerInfo);
    }

    /// <summary>
    /// Releases the TensorRT ONNX config handle.
    /// 释放 TensorRT ONNX config 句柄。
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}

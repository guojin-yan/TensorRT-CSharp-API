using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Parses ONNX models into a TensorRT network definition.
/// 将 ONNX 模型解析到 TensorRT network definition。
/// </summary>
public sealed partial class TensorRtOnnxParser
{
    /// <summary>
    /// Loads serialized ONNX model-proto bytes into the TensorRT 11 parser without parsing immediately.
    /// 将已序列化 ONNX model proto 字节加载到 TensorRT 11 parser，但暂不立即解析。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model-proto bytes. 已序列化的 ONNX model proto 字节。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when TensorRT accepts the model proto. TensorRT 接受 model proto 时返回 <c>true</c>。</returns>
    /// <remarks>
    /// TensorRT 11 does not retain the model-proto byte buffer after this call returns; the managed buffer is pinned only for the native call.
    /// TensorRT 11 在该调用返回后不会继续持有 model-proto 字节缓冲区；托管缓冲区仅在 native 调用期间短期 pin。
    /// </remarks>
    public bool LoadModelProto(byte[] modelData, string? modelPath = null)
    {
        return NativeBridgeApi.LoadOnnxParserModelProto(Line, _handle, modelData, modelPath);
    }

    /// <summary>
    /// Loads serialized ONNX model-proto bytes from a byte-array segment into the TensorRT 11 parser.
    /// 从托管字节数组片段加载 ONNX model proto 到 TensorRT 11 parser。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model-proto byte segment. 已序列化 ONNX model proto 字节片段。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when TensorRT accepts the model proto. TensorRT 接受 model proto 时返回 <c>true</c>。</returns>
    public bool LoadModelProto(ArraySegment<byte> modelData, string? modelPath = null)
    {
        return LoadModelProto(CopyModelSegment(modelData, nameof(modelData)), modelPath);
    }

#if NETCOREAPP3_1_OR_GREATER || NET5_0_OR_GREATER || NET6_0_OR_GREATER || NET7_0_OR_GREATER || NET8_0_OR_GREATER || NET9_0_OR_GREATER || NET10_0_OR_GREATER
    /// <summary>
    /// Loads serialized ONNX model-proto bytes from a read-only span into the TensorRT 11 parser.
    /// 从只读 span 加载 ONNX model proto 到 TensorRT 11 parser。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model-proto bytes. 已序列化 ONNX model proto 字节。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when TensorRT accepts the model proto. TensorRT 接受 model proto 时返回 <c>true</c>。</returns>
    public bool LoadModelProto(ReadOnlySpan<byte> modelData, string? modelPath = null)
    {
        return LoadModelProto(modelData.ToArray(), modelPath);
    }

#endif
    /// <summary>
    /// Loads serialized ONNX model-proto bytes from a stream into the TensorRT 11 parser.
    /// 从 stream 加载 ONNX model proto 到 TensorRT 11 parser。
    /// </summary>
    /// <param name="modelStream">Readable stream containing serialized ONNX model-proto bytes. 包含已序列化 ONNX model proto 字节的可读 stream。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when TensorRT accepts the model proto. TensorRT 接受 model proto 时返回 <c>true</c>。</returns>
    public bool LoadModelProto(Stream modelStream, string? modelPath = null)
    {
        return LoadModelProto(CopyModelStream(modelStream, nameof(modelStream)), modelPath);
    }

    /// <summary>
    /// Loads an external ONNX initializer into the TensorRT 11 parser.
    /// 将外部 ONNX initializer 加载到 TensorRT 11 parser。
    /// </summary>
    /// <param name="name">Initializer name. initializer 名称。</param>
    /// <param name="data">Initializer data. initializer 数据。</param>
    /// <returns><c>true</c> when TensorRT accepts the initializer. TensorRT 接受 initializer 时返回 <c>true</c>。</returns>
    /// <remarks>
    /// The initializer data is copied into an owned managed array and pinned until this parser is disposed, matching TensorRT's lifetime requirement.
    /// initializer 数据会复制到当前 parser 拥有的托管数组，并 pin 到 parser 释放为止，以满足 TensorRT 生命周期要求。
    /// </remarks>
    public bool LoadInitializer(string name, byte[] data)
    {
        return _initializerPins.LoadOrReplace(
            name,
            data,
            (pointer, size) => NativeBridgeApi.LoadOnnxParserInitializer(Line, _handle, name, pointer, size));
    }

    /// <summary>
    /// Loads an external ONNX initializer from a byte-array segment into the TensorRT 11 parser.
    /// 从托管字节数组片段加载外部 ONNX initializer 到 TensorRT 11 parser。
    /// </summary>
    /// <param name="name">Initializer name. initializer 名称。</param>
    /// <param name="data">Initializer data segment. initializer 数据片段。</param>
    /// <returns><c>true</c> when TensorRT accepts the initializer. TensorRT 接受 initializer 时返回 <c>true</c>。</returns>
    public bool LoadInitializer(string name, ArraySegment<byte> data)
    {
        return LoadInitializer(name, CopyModelSegment(data, nameof(data)));
    }

#if NETCOREAPP3_1_OR_GREATER || NET5_0_OR_GREATER || NET6_0_OR_GREATER || NET7_0_OR_GREATER || NET8_0_OR_GREATER || NET9_0_OR_GREATER || NET10_0_OR_GREATER
    /// <summary>
    /// Loads an external ONNX initializer from a read-only span into the TensorRT 11 parser.
    /// 从只读 span 加载外部 ONNX initializer 到 TensorRT 11 parser。
    /// </summary>
    /// <param name="name">Initializer name. initializer 名称。</param>
    /// <param name="data">Initializer data. initializer 数据。</param>
    /// <returns><c>true</c> when TensorRT accepts the initializer. TensorRT 接受 initializer 时返回 <c>true</c>。</returns>
    public bool LoadInitializer(string name, ReadOnlySpan<byte> data)
    {
        return LoadInitializer(name, data.ToArray());
    }

#endif
    /// <summary>
    /// Loads an external ONNX initializer from a stream into the TensorRT 11 parser.
    /// 从 stream 加载外部 ONNX initializer 到 TensorRT 11 parser。
    /// </summary>
    /// <param name="name">Initializer name. initializer 名称。</param>
    /// <param name="dataStream">Readable stream containing initializer data. 包含 initializer 数据的可读 stream。</param>
    /// <returns><c>true</c> when TensorRT accepts the initializer. TensorRT 接受 initializer 时返回 <c>true</c>。</returns>
    public bool LoadInitializer(string name, Stream dataStream)
    {
        return LoadInitializer(name, CopyModelStream(dataStream, nameof(dataStream)));
    }

    /// <summary>
    /// Parses the model proto previously loaded through <see cref="LoadModelProto(byte[], string?)"/>.
    /// 解析先前通过 <see cref="LoadModelProto(byte[], string?)"/> 加载的 model proto。
    /// </summary>
    /// <returns><c>true</c> when parsing succeeds. 解析成功时返回 <c>true</c>。</returns>
    public bool ParseLoadedModel()
    {
        return NativeBridgeApi.ParseOnnxLoadedModelProto(Line, _handle);
    }

}

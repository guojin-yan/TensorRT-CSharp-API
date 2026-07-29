using System;
using System.IO;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtOnnxParserRefitter
{
    /// <summary>
    /// Loads serialized ONNX model-proto bytes into the TensorRT 11 parser-refitter without refitting immediately.
    /// 将已序列化 ONNX model proto 字节加载到 TensorRT 11 parser-refitter，但暂不立即重整。
    /// </summary>
    /// <param name="modelData">Serialized ONNX model-proto bytes. 已序列化 ONNX model proto 字节。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when TensorRT accepts the model proto. TensorRT 接受 model proto 时返回 <c>true</c>。</returns>
    public bool LoadModelProto(byte[] modelData, string? modelPath = null)
    {
        return NativeBridgeApi.LoadOnnxParserRefitterModelProto(Line, _handle, modelData, modelPath);
    }

    /// <summary>
    /// Loads serialized ONNX model-proto bytes from a byte-array segment into the TensorRT 11 parser-refitter.
    /// 从托管字节数组片段加载 ONNX model proto 到 TensorRT 11 parser-refitter。
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
    /// Loads serialized ONNX model-proto bytes from a read-only span into the TensorRT 11 parser-refitter.
    /// 从只读 span 加载 ONNX model proto 到 TensorRT 11 parser-refitter。
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
    /// Loads serialized ONNX model-proto bytes from a stream into the TensorRT 11 parser-refitter.
    /// 从 stream 加载 ONNX model proto 到 TensorRT 11 parser-refitter。
    /// </summary>
    /// <param name="modelStream">Readable stream containing serialized ONNX model-proto bytes. 包含已序列化 ONNX model proto 字节的可读 stream。</param>
    /// <param name="modelPath">Optional model path used by TensorRT diagnostics. TensorRT 诊断信息使用的可选模型路径。</param>
    /// <returns><c>true</c> when TensorRT accepts the model proto. TensorRT 接受 model proto 时返回 <c>true</c>。</returns>
    public bool LoadModelProto(Stream modelStream, string? modelPath = null)
    {
        return LoadModelProto(CopyModelStream(modelStream, nameof(modelStream)), modelPath);
    }

    /// <summary>
    /// Refits the model proto previously loaded through <see cref="LoadModelProto(byte[], string?)"/>.
    /// 重整先前通过 <see cref="LoadModelProto(byte[], string?)"/> 加载的 model proto。
    /// </summary>
    /// <returns><c>true</c> when TensorRT reports the refit succeeded. TensorRT 报告重整成功时返回 <c>true</c>。</returns>
    public bool RefitLoadedModel()
    {
        return NativeBridgeApi.RefitOnnxParserRefitterLoadedModel(Line, _handle);
    }
}

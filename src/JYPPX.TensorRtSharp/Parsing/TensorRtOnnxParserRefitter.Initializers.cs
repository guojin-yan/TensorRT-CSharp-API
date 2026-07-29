using System;
using System.IO;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtOnnxParserRefitter
{
    /// <summary>
    /// Loads an external ONNX initializer into the TensorRT 11 parser-refitter.
    /// 将外部 ONNX initializer 加载到 TensorRT 11 parser-refitter。
    /// </summary>
    /// <param name="name">Initializer name. initializer 名称。</param>
    /// <param name="data">Initializer data. initializer 数据。</param>
    /// <returns><c>true</c> when TensorRT accepts the initializer. TensorRT 接受 initializer 时返回 <c>true</c>。</returns>
    /// <remarks>
    /// The initializer data is copied into an owned managed array and pinned until this parser-refitter is disposed.
    /// initializer 数据会复制到当前 parser-refitter 拥有的托管数组，并 pin 到 parser-refitter 释放为止。
    /// </remarks>
    public bool LoadInitializer(string name, byte[] data)
    {
        return _initializerPins.LoadOrReplace(
            name,
            data,
            (pointer, size) => NativeBridgeApi.LoadOnnxParserRefitterInitializer(Line, _handle, name, pointer, size));
    }

    /// <summary>
    /// Loads an external ONNX initializer from a byte-array segment into the TensorRT 11 parser-refitter.
    /// 从托管字节数组片段加载外部 ONNX initializer 到 TensorRT 11 parser-refitter。
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
    /// Loads an external ONNX initializer from a read-only span into the TensorRT 11 parser-refitter.
    /// 从只读 span 加载外部 ONNX initializer 到 TensorRT 11 parser-refitter。
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
    /// Loads an external ONNX initializer from a stream into the TensorRT 11 parser-refitter.
    /// 从 stream 加载外部 ONNX initializer 到 TensorRT 11 parser-refitter。
    /// </summary>
    /// <param name="name">Initializer name. initializer 名称。</param>
    /// <param name="dataStream">Readable stream containing initializer data. 包含 initializer 数据的可读 stream。</param>
    /// <returns><c>true</c> when TensorRT accepts the initializer. TensorRT 接受 initializer 时返回 <c>true</c>。</returns>
    public bool LoadInitializer(string name, Stream dataStream)
    {
        return LoadInitializer(name, CopyModelStream(dataStream, nameof(dataStream)));
    }
}

using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtOnnxParser
{
    private readonly object _builderConfigAttachmentLock = new object();
    private SafeTensorRtObjectHandleLease? _builderConfigLease;

    /// <summary>
    /// Gets whether this parser currently retains a successful TensorRT 11 builder-config attachment.
    /// 获取当前 parser 是否保留了成功建立的 TensorRT 11 builder-config 关联。
    /// </summary>
    public bool HasBuilderConfigAttached
    {
        get
        {
            lock (_builderConfigAttachmentLock)
            {
                return !_disposed && _builderConfigLease != null;
            }
        }
    }

    /// <summary>
    /// Attaches a TensorRT 11 builder configuration to this ONNX parser.
    /// 将 TensorRT 11 builder 配置关联到当前 ONNX parser。
    /// </summary>
    /// <param name="config">The builder configuration borrowed by TensorRT. TensorRT 借用的 builder 配置。</param>
    /// <returns><c>true</c> when TensorRT accepts the configuration. TensorRT 接受该配置时返回 <c>true</c>。</returns>
    /// <remarks>
    /// The parser keeps the configuration's native owner alive until another configuration is accepted or the parser is disposed.
    /// 在另一个配置被成功接受或 parser 被释放前，当前 parser 会保持该配置的 native owner 有效。
    /// </remarks>
    public bool SetBuilderConfig(TensorRtBuilderConfig config)
    {
        if (config == null)
        {
            throw new ArgumentNullException(nameof(config));
        }

        if (Line != TensorRtApiLine.TensorRt11)
        {
            throw new NotSupportedException("ONNX parser builder-config attachment requires TensorRT 11.");
        }

        if (config.Line != Line)
        {
            throw new ArgumentException("Parser and builder configuration must belong to the same TensorRT API line.", nameof(config));
        }

        SafeTensorRtObjectHandleLease? pendingLease = SafeTensorRtObjectHandleLease.Create(config.Handle);
        SafeTensorRtObjectHandleLease? previousLease = null;
        try
        {
            lock (_builderConfigAttachmentLock)
            {
                if (_disposed)
                {
                    throw new ObjectDisposedException(nameof(TensorRtOnnxParser));
                }

                if (!NativeBridgeApi.SetOnnxParserBuilderConfig(Line, _handle, config.Handle))
                {
                    return false;
                }

                previousLease = _builderConfigLease;
                _builderConfigLease = pendingLease;
                pendingLease = null;
            }

            previousLease?.Dispose();
            return true;
        }
        finally
        {
            pendingLease?.Dispose();
        }
    }
}

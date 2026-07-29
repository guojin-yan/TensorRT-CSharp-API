using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;
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
    /// Creates a refitter for updating refittable weights in this engine.
    /// 为当前 engine 创建用于更新可 refit 权重的 refitter。
    /// </summary>
    /// <param name="logger">The TensorRT logger used by the refitter. Refitter 使用的 TensorRT logger。</param>
    /// <returns>A managed refitter wrapper. 托管 refitter 封装对象。</returns>
    public TensorRtRefitter CreateRefitter(TensorRtLogger logger)
    {
        if (logger == null)
        {
            throw new ArgumentNullException(nameof(logger));
        }

        if (logger.Line != Line)
        {
            throw new ArgumentException("Logger and engine must belong to the same TensorRT API line.", nameof(logger));
        }

        logger.AttachBorrower(Line);
        try
        {
            return new TensorRtRefitter(
                Line,
                NativeBridgeApi.CreateRefitter(Line, _handle, logger.Handle),
                logger,
                loggerBorrowAttached: true);
        }
        catch
        {
            logger.DetachBorrower();
            throw;
        }
    }

}

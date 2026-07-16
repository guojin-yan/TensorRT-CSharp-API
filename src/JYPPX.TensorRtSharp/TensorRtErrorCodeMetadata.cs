using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>Provides pointer-free TensorRT error-code enum metadata. 提供无指针的 TensorRT error-code 枚举元数据。</summary>
public static class TensorRtErrorCodeMetadata
{
    /// <summary>
    /// Gets one greater than the largest TensorRT <c>ErrorCode</c> value for an API line.
    /// 获取指定 API line 中 TensorRT <c>ErrorCode</c> 最大值加一后的独占上界。
    /// </summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <returns>The exclusive error-code upper bound. ErrorCode 的独占上界。</returns>
    public static int GetExclusiveUpperBound(TensorRtApiLine line) =>
        NativeBridgeApi.GetErrorCodeExclusiveUpperBound(line);

    /// <summary>Checks whether a raw error code is within the line's declared enum range. 检查原始错误码是否位于版本线声明的枚举范围内。</summary>
    /// <param name="line">The TensorRT API line. TensorRT API 版本线。</param>
    /// <param name="errorCode">The raw error code. 原始错误码。</param>
    /// <returns><see langword="true"/> when the value is in range. 值位于范围内时返回 <see langword="true"/>。</returns>
    public static bool IsDefinedRangeValue(TensorRtApiLine line, int errorCode) =>
        errorCode >= 0 && errorCode < GetExclusiveUpperBound(line);
}

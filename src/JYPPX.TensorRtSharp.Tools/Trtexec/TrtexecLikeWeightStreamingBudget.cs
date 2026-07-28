using System;
using System.Globalization;

namespace JYPPX.TensorRtSharp.Tools;

/// <summary>
/// Represents the normalized trtexec weight-streaming budget grammar.
/// 表示规范化后的 trtexec 权重流式加载预算语法。
/// </summary>
public sealed class TrtexecLikeWeightStreamingBudget
{
    private const string UnspecifiedKind = "unspecified";
    private const string DisabledKind = "disabled";
    private const string AutomaticKind = "automatic";
    private const string PercentageKind = "percentage";
    private const string BytesKind = "bytes";

    private TrtexecLikeWeightStreamingBudget(string kind, ulong? bytes, decimal? percentage, string argumentValue)
    {
        Kind = kind;
        Bytes = bytes;
        Percentage = percentage;
        ArgumentValue = argumentValue;
    }

    /// <summary>
    /// Gets an unspecified budget.
    /// 获取未指定的预算。
    /// </summary>
    public static TrtexecLikeWeightStreamingBudget Unspecified { get; } =
        new TrtexecLikeWeightStreamingBudget(UnspecifiedKind, null, null, string.Empty);

    /// <summary>
    /// Gets the normalized budget kind: unspecified, disabled, automatic, percentage, or bytes.
    /// 获取规范化预算类型：unspecified、disabled、automatic、percentage 或 bytes。
    /// </summary>
    public string Kind { get; }

    /// <summary>
    /// Gets whether the command explicitly specified a budget.
    /// 获取命令是否显式指定了预算。
    /// </summary>
    public bool IsSpecified => !string.Equals(Kind, UnspecifiedKind, StringComparison.Ordinal);

    /// <summary>
    /// Gets the exact byte budget when <see cref="Kind"/> is bytes.
    /// 当 <see cref="Kind"/> 为 bytes 时获取精确字节预算。
    /// </summary>
    public ulong? Bytes { get; }

    /// <summary>
    /// Gets the percentage of streamable weights that should remain on the GPU.
    /// 获取应驻留在 GPU 上的可流式加载权重百分比。
    /// </summary>
    public decimal? Percentage { get; }

    /// <summary>
    /// Gets the normalized command-line argument value.
    /// 获取规范化后的命令行参数值。
    /// </summary>
    public string ArgumentValue { get; }

    /// <summary>
    /// Parses the official trtexec budget forms: -2, -1, 0..100%, or a non-negative byte size.
    /// 解析官方 trtexec 预算形式：-2、-1、0..100% 或非负字节大小。
    /// </summary>
    /// <param name="value">The command-line value. / 命令行参数值。</param>
    /// <returns>The normalized budget. / 规范化后的预算。</returns>
    public static TrtexecLikeWeightStreamingBudget Parse(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return Unspecified;
        }

        string trimmed = value.Trim();
        if (string.Equals(trimmed, "-2", StringComparison.Ordinal))
        {
            return new TrtexecLikeWeightStreamingBudget(DisabledKind, null, null, "-2");
        }

        if (string.Equals(trimmed, "-1", StringComparison.Ordinal))
        {
            return new TrtexecLikeWeightStreamingBudget(AutomaticKind, null, null, "-1");
        }

        if (trimmed.EndsWith("%", StringComparison.Ordinal))
        {
            string numberText = trimmed.Substring(0, trimmed.Length - 1).Trim();
            if (!decimal.TryParse(numberText, NumberStyles.Float, CultureInfo.InvariantCulture, out decimal percentage) ||
                percentage < 0m || percentage > 100m)
            {
                throw new ArgumentException("--weightStreamingBudget percentage must be in the range [0, 100].", nameof(value));
            }

            return new TrtexecLikeWeightStreamingBudget(
                PercentageKind,
                null,
                percentage,
                percentage.ToString(CultureInfo.InvariantCulture) + "%");
        }

        ulong bytes = ParseBytes(trimmed);
        if (bytes > long.MaxValue)
        {
            throw new ArgumentException($"--weightStreamingBudget must not exceed {long.MaxValue} bytes.", nameof(value));
        }

        const ulong mib = 1024UL * 1024UL;
        string argumentValue = bytes % mib == 0
            ? (bytes / mib).ToString(CultureInfo.InvariantCulture)
            : bytes.ToString(CultureInfo.InvariantCulture) + "B";
        return new TrtexecLikeWeightStreamingBudget(BytesKind, bytes, null, argumentValue);
    }

    /// <summary>
    /// Creates an exact byte budget while preserving the existing managed constructor contract.
    /// 创建精确字节预算，同时保留现有托管构造函数契约。
    /// </summary>
    /// <param name="bytes">Budget in bytes. / 预算字节数。</param>
    /// <returns>The normalized byte budget. / 规范化后的字节预算。</returns>
    public static TrtexecLikeWeightStreamingBudget FromBytes(ulong bytes)
    {
        return Parse(bytes.ToString(CultureInfo.InvariantCulture) + "B");
    }

    internal long ResolveBytes(long streamableWeightsSize, long automaticBudget)
    {
        if (streamableWeightsSize < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(streamableWeightsSize));
        }

        if (string.Equals(Kind, AutomaticKind, StringComparison.Ordinal))
        {
            return automaticBudget;
        }

        if (string.Equals(Kind, PercentageKind, StringComparison.Ordinal))
        {
            decimal resolved = decimal.Truncate(streamableWeightsSize * Percentage.GetValueOrDefault() / 100m);
            return checked((long)resolved);
        }

        if (string.Equals(Kind, BytesKind, StringComparison.Ordinal))
        {
            return checked((long)Bytes.GetValueOrDefault());
        }

        return streamableWeightsSize;
    }

    private static ulong ParseBytes(string value)
    {
        string numberText = value;
        decimal multiplier = 1024m * 1024m;
        if (TryTrimSuffix(value, "gib", out numberText) || TryTrimSuffix(value, "gb", out numberText) || TryTrimSuffix(value, "g", out numberText))
        {
            multiplier = 1024m * 1024m * 1024m;
        }
        else if (TryTrimSuffix(value, "mib", out numberText) || TryTrimSuffix(value, "mb", out numberText) || TryTrimSuffix(value, "m", out numberText))
        {
            multiplier = 1024m * 1024m;
        }
        else if (TryTrimSuffix(value, "kib", out numberText) || TryTrimSuffix(value, "kb", out numberText) || TryTrimSuffix(value, "k", out numberText))
        {
            multiplier = 1024m;
        }
        else if (TryTrimSuffix(value, "b", out numberText))
        {
            multiplier = 1m;
        }

        if (!decimal.TryParse(numberText.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out decimal parsed) || parsed < 0m)
        {
            throw new ArgumentException("--weightStreamingBudget must be -2, -1, 0..100%, or a non-negative memory size.", nameof(value));
        }

        decimal bytes = parsed * multiplier;
        if (bytes != decimal.Truncate(bytes) || bytes > ulong.MaxValue)
        {
            throw new ArgumentException("--weightStreamingBudget must resolve to a whole number of bytes.", nameof(value));
        }

        return checked((ulong)bytes);
    }

    private static bool TryTrimSuffix(string value, string suffix, out string numberText)
    {
        if (value.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
        {
            numberText = value.Substring(0, value.Length - suffix.Length);
            return true;
        }

        numberText = value;
        return false;
    }
}

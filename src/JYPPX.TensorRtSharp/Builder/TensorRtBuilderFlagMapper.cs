using System;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

internal static class TensorRtBuilderFlagMapper
{
    private const int FlagCount = (int)TensorRtBuilderFlag.WeightStreaming + 1;
    private const uint KnownLogicalMask = (1u << FlagCount) - 1u;

    public static int ToNativeFlag(TensorRtApiLine line, TensorRtBuilderFlag flag)
    {
        if (!TryToNativeFlag(line, flag, out int nativeFlag))
        {
            if ((int)flag < 0 || (int)flag >= FlagCount)
            {
                throw new ArgumentOutOfRangeException(nameof(flag), flag, "Unknown TensorRT builder flag.");
            }

            throw new NotSupportedException($"Builder flag {flag} is not available on {line}.");
        }

        return nativeFlag;
    }

    public static uint ToNativeFlags(TensorRtApiLine line, TensorRtBuilderFlags flags)
    {
        uint logicalFlags = (uint)flags;
        if ((logicalFlags & ~KnownLogicalMask) != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(flags), flags, "The builder flag mask contains unknown logical bits.");
        }

        uint nativeFlags = 0;
        for (int logicalIndex = 0; logicalIndex < FlagCount; logicalIndex++)
        {
            uint logicalBit = 1u << logicalIndex;
            if ((logicalFlags & logicalBit) == 0)
            {
                continue;
            }

            int nativeIndex = ToNativeFlag(line, (TensorRtBuilderFlag)logicalIndex);
            nativeFlags |= 1u << nativeIndex;
        }

        return nativeFlags;
    }

    public static TensorRtBuilderFlags FromNativeFlags(TensorRtApiLine line, uint nativeFlags)
    {
        uint logicalFlags = 0;
        for (int logicalIndex = 0; logicalIndex < FlagCount; logicalIndex++)
        {
            if (!TryToNativeFlag(line, (TensorRtBuilderFlag)logicalIndex, out int nativeIndex))
            {
                continue;
            }

            if ((nativeFlags & (1u << nativeIndex)) != 0)
            {
                logicalFlags |= 1u << logicalIndex;
            }
        }

        return (TensorRtBuilderFlags)logicalFlags;
    }

    private static bool TryToNativeFlag(TensorRtApiLine line, TensorRtBuilderFlag flag, out int nativeFlag)
    {
        nativeFlag = line switch
        {
            TensorRtApiLine.TensorRt8 => ToTensorRt8Flag(flag),
            TensorRtApiLine.TensorRt10 => ToTensorRt10Flag(flag),
            TensorRtApiLine.TensorRt11 => ToTensorRt11Flag(flag),
            _ => -1
        };

        return nativeFlag >= 0;
    }

    private static int ToTensorRt8Flag(TensorRtBuilderFlag flag) => flag switch
    {
        TensorRtBuilderFlag.Fp16 => 0,
        TensorRtBuilderFlag.Int8 => 1,
        TensorRtBuilderFlag.Debug => 2,
        TensorRtBuilderFlag.GpuFallback => 3,
        TensorRtBuilderFlag.Refit => 5,
        TensorRtBuilderFlag.DisableTimingCache => 6,
        TensorRtBuilderFlag.Tf32 => 7,
        TensorRtBuilderFlag.SparseWeights => 8,
        TensorRtBuilderFlag.SafetyScope => 9,
        TensorRtBuilderFlag.ObeyPrecisionConstraints => 10,
        TensorRtBuilderFlag.PreferPrecisionConstraints => 11,
        TensorRtBuilderFlag.DirectIO => 12,
        TensorRtBuilderFlag.RejectEmptyAlgorithms => 13,
        TensorRtBuilderFlag.VersionCompatible => 15,
        TensorRtBuilderFlag.ExcludeLeanRuntime => 16,
        TensorRtBuilderFlag.Fp8 => 17,
        _ => -1
    };

    private static int ToTensorRt10Flag(TensorRtBuilderFlag flag) => flag switch
    {
        TensorRtBuilderFlag.Fp16 => 0,
        TensorRtBuilderFlag.Int8 => 1,
        TensorRtBuilderFlag.Debug => 2,
        TensorRtBuilderFlag.GpuFallback => 3,
        TensorRtBuilderFlag.Refit => 4,
        TensorRtBuilderFlag.DisableTimingCache => 5,
        TensorRtBuilderFlag.Tf32 => 6,
        TensorRtBuilderFlag.SparseWeights => 7,
        TensorRtBuilderFlag.SafetyScope => 8,
        TensorRtBuilderFlag.ObeyPrecisionConstraints => 9,
        TensorRtBuilderFlag.PreferPrecisionConstraints => 10,
        TensorRtBuilderFlag.DirectIO => 11,
        TensorRtBuilderFlag.RejectEmptyAlgorithms => 12,
        TensorRtBuilderFlag.VersionCompatible => 13,
        TensorRtBuilderFlag.ExcludeLeanRuntime => 14,
        TensorRtBuilderFlag.Fp8 => 15,
        TensorRtBuilderFlag.ErrorOnTimingCacheMiss => 16,
        TensorRtBuilderFlag.Bf16 => 17,
        TensorRtBuilderFlag.DisableCompilationCache => 18,
        TensorRtBuilderFlag.StripPlan => 19,
        TensorRtBuilderFlag.RefitIdentical => 20,
        TensorRtBuilderFlag.WeightStreaming => 21,
        _ => -1
    };

    private static int ToTensorRt11Flag(TensorRtBuilderFlag flag) => flag switch
    {
        TensorRtBuilderFlag.Debug => 2,
        TensorRtBuilderFlag.GpuFallback => 3,
        TensorRtBuilderFlag.Refit => 4,
        TensorRtBuilderFlag.DisableTimingCache => 5,
        TensorRtBuilderFlag.Tf32 => 6,
        TensorRtBuilderFlag.SparseWeights => 7,
        TensorRtBuilderFlag.SafetyScope => 8,
        TensorRtBuilderFlag.DirectIO => 11,
        TensorRtBuilderFlag.VersionCompatible => 12,
        TensorRtBuilderFlag.ExcludeLeanRuntime => 13,
        TensorRtBuilderFlag.ErrorOnTimingCacheMiss => 15,
        TensorRtBuilderFlag.DisableCompilationCache => 17,
        TensorRtBuilderFlag.StripPlan => 18,
        TensorRtBuilderFlag.RefitIdentical => 19,
        TensorRtBuilderFlag.WeightStreaming => 20,
        _ => -1
    };
}

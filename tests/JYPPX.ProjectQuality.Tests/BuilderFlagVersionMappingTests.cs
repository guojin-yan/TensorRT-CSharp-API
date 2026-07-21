using System.IO;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class BuilderFlagVersionMappingTests
{
    public static TheoryData<TensorRtApiLine, TensorRtBuilderFlag, int> SupportedFlagMappings => new()
    {
        { TensorRtApiLine.TensorRt8, TensorRtBuilderFlag.Fp16, 0 },
        { TensorRtApiLine.TensorRt8, TensorRtBuilderFlag.Int8, 1 },
        { TensorRtApiLine.TensorRt8, TensorRtBuilderFlag.Debug, 2 },
        { TensorRtApiLine.TensorRt8, TensorRtBuilderFlag.GpuFallback, 3 },
        { TensorRtApiLine.TensorRt8, TensorRtBuilderFlag.Refit, 5 },
        { TensorRtApiLine.TensorRt8, TensorRtBuilderFlag.DisableTimingCache, 6 },
        { TensorRtApiLine.TensorRt8, TensorRtBuilderFlag.Tf32, 7 },
        { TensorRtApiLine.TensorRt8, TensorRtBuilderFlag.SparseWeights, 8 },
        { TensorRtApiLine.TensorRt8, TensorRtBuilderFlag.SafetyScope, 9 },
        { TensorRtApiLine.TensorRt8, TensorRtBuilderFlag.ObeyPrecisionConstraints, 10 },
        { TensorRtApiLine.TensorRt8, TensorRtBuilderFlag.PreferPrecisionConstraints, 11 },
        { TensorRtApiLine.TensorRt8, TensorRtBuilderFlag.DirectIO, 12 },
        { TensorRtApiLine.TensorRt8, TensorRtBuilderFlag.RejectEmptyAlgorithms, 13 },
        { TensorRtApiLine.TensorRt8, TensorRtBuilderFlag.VersionCompatible, 15 },
        { TensorRtApiLine.TensorRt8, TensorRtBuilderFlag.ExcludeLeanRuntime, 16 },
        { TensorRtApiLine.TensorRt8, TensorRtBuilderFlag.Fp8, 17 },
        { TensorRtApiLine.TensorRt10, TensorRtBuilderFlag.Fp16, 0 },
        { TensorRtApiLine.TensorRt10, TensorRtBuilderFlag.Int8, 1 },
        { TensorRtApiLine.TensorRt10, TensorRtBuilderFlag.Debug, 2 },
        { TensorRtApiLine.TensorRt10, TensorRtBuilderFlag.GpuFallback, 3 },
        { TensorRtApiLine.TensorRt10, TensorRtBuilderFlag.Refit, 4 },
        { TensorRtApiLine.TensorRt10, TensorRtBuilderFlag.DisableTimingCache, 5 },
        { TensorRtApiLine.TensorRt10, TensorRtBuilderFlag.Tf32, 6 },
        { TensorRtApiLine.TensorRt10, TensorRtBuilderFlag.SparseWeights, 7 },
        { TensorRtApiLine.TensorRt10, TensorRtBuilderFlag.SafetyScope, 8 },
        { TensorRtApiLine.TensorRt10, TensorRtBuilderFlag.ObeyPrecisionConstraints, 9 },
        { TensorRtApiLine.TensorRt10, TensorRtBuilderFlag.PreferPrecisionConstraints, 10 },
        { TensorRtApiLine.TensorRt10, TensorRtBuilderFlag.DirectIO, 11 },
        { TensorRtApiLine.TensorRt10, TensorRtBuilderFlag.RejectEmptyAlgorithms, 12 },
        { TensorRtApiLine.TensorRt10, TensorRtBuilderFlag.VersionCompatible, 13 },
        { TensorRtApiLine.TensorRt10, TensorRtBuilderFlag.ExcludeLeanRuntime, 14 },
        { TensorRtApiLine.TensorRt10, TensorRtBuilderFlag.Fp8, 15 },
        { TensorRtApiLine.TensorRt10, TensorRtBuilderFlag.ErrorOnTimingCacheMiss, 16 },
        { TensorRtApiLine.TensorRt10, TensorRtBuilderFlag.Bf16, 17 },
        { TensorRtApiLine.TensorRt10, TensorRtBuilderFlag.DisableCompilationCache, 18 },
        { TensorRtApiLine.TensorRt10, TensorRtBuilderFlag.StripPlan, 19 },
        { TensorRtApiLine.TensorRt10, TensorRtBuilderFlag.RefitIdentical, 20 },
        { TensorRtApiLine.TensorRt10, TensorRtBuilderFlag.WeightStreaming, 21 },
        { TensorRtApiLine.TensorRt11, TensorRtBuilderFlag.Debug, 2 },
        { TensorRtApiLine.TensorRt11, TensorRtBuilderFlag.GpuFallback, 3 },
        { TensorRtApiLine.TensorRt11, TensorRtBuilderFlag.Refit, 4 },
        { TensorRtApiLine.TensorRt11, TensorRtBuilderFlag.DisableTimingCache, 5 },
        { TensorRtApiLine.TensorRt11, TensorRtBuilderFlag.Tf32, 6 },
        { TensorRtApiLine.TensorRt11, TensorRtBuilderFlag.SparseWeights, 7 },
        { TensorRtApiLine.TensorRt11, TensorRtBuilderFlag.SafetyScope, 8 },
        { TensorRtApiLine.TensorRt11, TensorRtBuilderFlag.DirectIO, 11 },
        { TensorRtApiLine.TensorRt11, TensorRtBuilderFlag.VersionCompatible, 12 },
        { TensorRtApiLine.TensorRt11, TensorRtBuilderFlag.ExcludeLeanRuntime, 13 },
        { TensorRtApiLine.TensorRt11, TensorRtBuilderFlag.ErrorOnTimingCacheMiss, 15 },
        { TensorRtApiLine.TensorRt11, TensorRtBuilderFlag.DisableCompilationCache, 17 },
        { TensorRtApiLine.TensorRt11, TensorRtBuilderFlag.StripPlan, 18 },
        { TensorRtApiLine.TensorRt11, TensorRtBuilderFlag.RefitIdentical, 19 },
        { TensorRtApiLine.TensorRt11, TensorRtBuilderFlag.WeightStreaming, 20 }
    };

    [Theory]
    [MemberData(nameof(SupportedFlagMappings))]
    internal void SingleFlagMappingUsesVendorIndexForEachApiLine(
        TensorRtApiLine line,
        TensorRtBuilderFlag logicalFlag,
        int expectedNativeFlag)
    {
        Assert.Equal(expectedNativeFlag, TensorRtBuilderFlagMapper.ToNativeFlag(line, logicalFlag));
    }

    [Theory]
    [InlineData(TensorRtApiLine.TensorRt8, TensorRtBuilderFlag.Bf16)]
    [InlineData(TensorRtApiLine.TensorRt8, TensorRtBuilderFlag.ErrorOnTimingCacheMiss)]
    [InlineData(TensorRtApiLine.TensorRt11, TensorRtBuilderFlag.Fp16)]
    [InlineData(TensorRtApiLine.TensorRt11, TensorRtBuilderFlag.Int8)]
    [InlineData(TensorRtApiLine.TensorRt11, TensorRtBuilderFlag.ObeyPrecisionConstraints)]
    [InlineData(TensorRtApiLine.TensorRt11, TensorRtBuilderFlag.PreferPrecisionConstraints)]
    [InlineData(TensorRtApiLine.TensorRt11, TensorRtBuilderFlag.Bf16)]
    internal void RemovedOrUnavailableFlagsFailClosed(TensorRtApiLine line, TensorRtBuilderFlag logicalFlag)
    {
        NotSupportedException exception = Assert.Throws<NotSupportedException>(
            () => TensorRtBuilderFlagMapper.ToNativeFlag(line, logicalFlag));

        Assert.Contains(logicalFlag.ToString(), exception.Message);
        Assert.Contains(line.ToString(), exception.Message);
    }

    [Fact]
    internal void TensorRt8MaskTranslationDoesNotConfuseVendorHolesWithLogicalFlags()
    {
        TensorRtBuilderFlags logical = TensorRtBuilderFlags.Refit |
            TensorRtBuilderFlags.DirectIO |
            TensorRtBuilderFlags.Fp8;
        uint expectedNative = (1u << 5) | (1u << 12) | (1u << 17);

        Assert.Equal(expectedNative, TensorRtBuilderFlagMapper.ToNativeFlags(TensorRtApiLine.TensorRt8, logical));
        Assert.Equal(logical, TensorRtBuilderFlagMapper.FromNativeFlags(TensorRtApiLine.TensorRt8, expectedNative));

        uint vendorOnlyHoles = (1u << 4) | (1u << 14);
        Assert.Equal(TensorRtBuilderFlags.None, TensorRtBuilderFlagMapper.FromNativeFlags(TensorRtApiLine.TensorRt8, vendorOnlyHoles));
    }

    [Fact]
    internal void TensorRt11MaskTranslationIgnoresUnmodelledVendorFlags()
    {
        TensorRtBuilderFlags logical = TensorRtBuilderFlags.Debug |
            TensorRtBuilderFlags.VersionCompatible |
            TensorRtBuilderFlags.WeightStreaming;
        uint expectedNative = (1u << 2) | (1u << 12) | (1u << 20);

        Assert.Equal(expectedNative, TensorRtBuilderFlagMapper.ToNativeFlags(TensorRtApiLine.TensorRt11, logical));
        Assert.Equal(logical, TensorRtBuilderFlagMapper.FromNativeFlags(TensorRtApiLine.TensorRt11, expectedNative));

        uint vendorOnlyFlags = (1u << 22) | (1u << 23) | (1u << 24) | (1u << 26) | (1u << 27);
        Assert.Equal(TensorRtBuilderFlags.None, TensorRtBuilderFlagMapper.FromNativeFlags(TensorRtApiLine.TensorRt11, vendorOnlyFlags));
    }

    [Fact]
    internal void LogicalMasksRejectUnknownOrUnsupportedBits()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            TensorRtBuilderFlagMapper.ToNativeFlags(TensorRtApiLine.TensorRt10, (TensorRtBuilderFlags)(1u << 31)));
        Assert.Throws<NotSupportedException>(() =>
            TensorRtBuilderFlagMapper.ToNativeFlags(TensorRtApiLine.TensorRt11, TensorRtBuilderFlags.Fp16));
    }

    [Fact]
    public void NetworkBuilderSmokeSeparatesTensorRt8DirectIoFromPrecisionPreference()
    {
        string program = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "smoke",
            "NetworkBuilderSmokeRunner",
            "Program.cs"));

        Assert.Contains("ProbeVersionedBuilderFlagMapping(config)", program);
        Assert.Contains("TRT8DirectIORaw12PreferRaw11:Isolated", program);
        Assert.Contains("config.SetFlag(TensorRtBuilderFlag.DirectIO, true)", program);
        Assert.Contains("config.GetFlag(TensorRtBuilderFlag.PreferPrecisionConstraints)", program);
        Assert.Contains("config.SetFlag(TensorRtBuilderFlag.DirectIO, originalDirectIo)", program);
    }
}

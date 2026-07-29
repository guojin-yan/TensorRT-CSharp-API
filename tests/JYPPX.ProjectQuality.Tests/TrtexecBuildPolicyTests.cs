using System;
using System.IO;
using JYPPX.TensorRtSharp;
using JYPPX.TensorRtSharp.Tools;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TrtexecBuildPolicyTests
{
    [Fact]
    public void ParserNormalizesIoAndLayerPolicyGrammar()
    {
        TrtexecLikeOptions options = TrtexecLikeParser.Parse(new[]
        {
            "--inputIOFormats", " FP16:CHW+chw2, INT8:DLA_HWC4 ",
            "--outputIOFormats", "fp32:HWC",
            "--precisionConstraints", "PREFER",
            "--layerPrecisions", "conv*:FP16,conv1:fp32,conv1:INT8",
            "--layerOutputTypes", "head*:fp32+INT32,head0:fp16",
            "--buildOnly"
        });

        Assert.Equal("fp16:chw+chw2,int8:dla_hwc4", options.DeploymentOptions.InputIOFormats);
        Assert.Equal("fp32:hwc", options.DeploymentOptions.OutputIOFormats);
        Assert.Equal("prefer", options.DeploymentOptions.PrecisionConstraints);
        Assert.Equal("conv*:fp16,conv1:fp32,conv1:int8", options.DeploymentOptions.LayerPrecisions);
        Assert.Equal("head*:fp32+int32,head0:fp16", options.DeploymentOptions.LayerOutputTypes);
    }

    [Fact]
    internal void ExactLayerRuleWinsAndLaterDuplicateOverridesEarlierRule()
    {
        var rules = TrtexecLikeBuildPolicy.ParseLayerRules(
            "conv*:fp16,conv1:fp32,conv*:int8,conv1:bf16",
            "--layerPrecisions",
            allowMultipleTypes: false);

        Assert.Equal(TensorRtDataType.BFloat16, TrtexecLikeBuildPolicy.ResolveLayerRule("conv1", rules)!.DataTypes[0]);
        Assert.Equal(TensorRtDataType.Int8, TrtexecLikeBuildPolicy.ResolveLayerRule("conv2", rules)!.DataTypes[0]);
        Assert.Null(TrtexecLikeBuildPolicy.ResolveLayerRule("head", rules));
    }

    [Theory]
    [InlineData("--inputIOFormats", "fp16")]
    [InlineData("--inputIOFormats", "fp16:unknown")]
    [InlineData("--inputIOFormats", "fp16:chw+")]
    [InlineData("--outputIOFormats", "fp64:chw")]
    [InlineData("--precisionConstraints", "strict")]
    [InlineData("--layerPrecisions", "conv**:fp16")]
    [InlineData("--layerPrecisions", "conv:fp16+fp32")]
    [InlineData("--layerOutputTypes", "head:")]
    public void ParserRejectsMalformedBuildPolicies(string option, string value)
    {
        Assert.Throws<ArgumentException>(() => TrtexecLikeParser.Parse(new[] { option, value, "--buildOnly" }));
    }

    [Theory]
    [InlineData("--layerPrecisions", "conv*:fp16")]
    [InlineData("--layerOutputTypes", "head*:fp32")]
    public void LayerRulesRequireActivePrecisionConstraintMode(string option, string value)
    {
        Assert.Throws<ArgumentException>(() => TrtexecLikeParser.Parse(new[] { option, value, "--buildOnly" }));
        Assert.Throws<ArgumentException>(() => TrtexecLikeParser.Parse(new[]
        {
            "--precisionConstraints", "none",
            option, value,
            "--buildOnly"
        }));
    }

    [Fact]
    public void BuildServiceAppliesPoliciesAfterOnnxParseWithReadbackAndVersionGuards()
    {
        string service = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp.Tools", "Build", "OnnxEngineBuildService.cs"));
        string policy = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp.Tools", "Trtexec", "TrtexecLikeBuildPolicy.cs"));
        string diagnostics = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp.Tools", "Build", "OnnxEngineBuildDiagnostics.cs"));

        Assert.Contains("TrtexecLikeBuildPolicy.Apply(config, network, options.DeploymentOptions, log)", service, StringComparison.Ordinal);
        Assert.Contains("tensor.AllowedFormats = spec.Formats", policy, StringComparison.Ordinal);
        Assert.Contains("tensor.DataType = spec.DataType", policy, StringComparison.Ordinal);
        Assert.Contains("config.SetFlag(TensorRtBuilderFlag.PreferPrecisionConstraints", policy, StringComparison.Ordinal);
        Assert.Contains("config.SetFlag(TensorRtBuilderFlag.ObeyPrecisionConstraints", policy, StringComparison.Ordinal);
        Assert.Contains("layer.Precision = requested", policy, StringComparison.Ordinal);
        Assert.Contains("layer.SetOutputType(outputIndex, requested)", policy, StringComparison.Ordinal);
        Assert.Contains("exact ?? wildcard", policy, StringComparison.Ordinal);
        Assert.Contains("tensor-set-type-removed", policy, StringComparison.Ordinal);
        Assert.Contains("layer-set-precision-removed", policy, StringComparison.Ordinal);
        Assert.Contains("layer-set-output-type-removed", policy, StringComparison.Ordinal);
        Assert.Contains("HasAppliedBuildPolicy", diagnostics, StringComparison.Ordinal);
        Assert.Contains("HasNormalizedOption", diagnostics, StringComparison.Ordinal);

        string parity = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "applications",
            "TensorRtExec",
            "tensor-rt-exec-trtexec-parity-matrix.json"));
        string article = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "tensorrtexec-io-layer-precision-policies.md"));
        Assert.Contains("io-layer-precision-policies", parity, StringComparison.Ordinal);
        Assert.Contains("implemented-build-readback-with-version-guards", parity, StringComparison.Ordinal);
        Assert.Contains("TRT8DirectIORaw12PreferRaw11", article, StringComparison.Ordinal);
    }

    [Fact]
    public void EnginePackagingAndWeightStreamingUseTypedReadbackBeforeContextCreation()
    {
        string service = string.Concat(
            File.ReadAllText(Path.Combine(
                RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "Build", "OnnxEngineBuildService.cs")),
            File.ReadAllText(Path.Combine(
                RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "Build", "OnnxEngineBuildService.DeploymentConfiguration.cs")),
            File.ReadAllText(Path.Combine(
                RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "Build", "OnnxEngineBuildService.RuntimeExecution.cs")),
            File.ReadAllText(Path.Combine(
                RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "Build", "OnnxEngineBuildService.Benchmarking.cs")));
        string diagnostics = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp.Tools", "Build", "OnnxEngineBuildDiagnostics.cs"));
        string parser = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp.Tools", "Trtexec", "TrtexecLikeParser.cs"));

        Assert.Contains("ApplyEnginePackagingOptions(config, options, log)", service, StringComparison.Ordinal);
        Assert.Contains("TensorRtBuilderFlag.VersionCompatible", service, StringComparison.Ordinal);
        Assert.Contains("TensorRtBuilderFlag.ExcludeLeanRuntime", service, StringComparison.Ordinal);
        Assert.Contains("TensorRtBuilderFlag.StripPlan", service, StringComparison.Ordinal);
        Assert.Contains("TensorRtBuilderFlag.RefitIdentical", service, StringComparison.Ordinal);
        Assert.Contains("TensorRtBuilderFlag.WeightStreaming", service, StringComparison.Ordinal);
        Assert.Contains("version-compatible-refit-vendor-readback-conflict", service, StringComparison.Ordinal);
        Assert.Contains("ApplyEngineRuntimePolicies(engine, options, log, validateRefittableState);", service, StringComparison.Ordinal);
        Assert.Contains("validateRefittableState: !refitPersistenceSnapshot.Succeeded", service, StringComparison.Ordinal);
        Assert.Contains("full-weight-reload-does-not-require-refittable-state", service, StringComparison.Ordinal);
        Assert.Contains("engine.SetWeightStreamingBudgetV2(requestedBudget)", service, StringComparison.Ordinal);
        Assert.Contains("engine.WeightStreamingBudgetV2InBytes", service, StringComparison.Ordinal);
        Assert.Contains("engine.WeightStreamingScratchMemorySizeInBytes", service, StringComparison.Ordinal);
        Assert.True(
            service.IndexOf("ApplyEngineRuntimePolicies(engine, options, log, validateRefittableState);", StringComparison.Ordinal) <
            service.IndexOf("new OnnxEngineBenchmarkWorker", StringComparison.Ordinal));

        Assert.Contains("--excludeLeanRuntime requires --versionCompatible", parser, StringComparison.Ordinal);
        Assert.Contains("--allowWeightStreaming requires --stronglyTyped", parser, StringComparison.Ordinal);
        Assert.Contains("--stripWeights requires --buildOnly or --skipInference", parser, StringComparison.Ordinal);
        Assert.Contains("HasAppliedDeploymentControl(result, \"VersionCompatible\")", diagnostics, StringComparison.Ordinal);
        Assert.Contains("HasAppliedDeploymentControl(result, \"WeightStreamingBudget\")", diagnostics, StringComparison.Ordinal);
    }
}

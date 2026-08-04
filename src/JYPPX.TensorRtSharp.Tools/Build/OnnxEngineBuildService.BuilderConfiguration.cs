using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.ExceptionServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed partial class OnnxEngineBuildService
{
    private static int AddOptimizationProfile(TensorRtBuilder builder, TensorRtBuilderConfig config, OnnxEngineBuildOptions options)
    {
        if (options.ShapeProfile.IsEmpty && options.UsesExternalOnnx)
        {
            return -1;
        }

        using TensorRtOptimizationProfile profile = builder.CreateOptimizationProfile();
        if (!options.ShapeProfile.IsEmpty)
        {
            foreach (EngineBuildShape minShape in options.ShapeProfile.MinShapes)
            {
                if (!options.ShapeProfile.TryGetShapeTriple(minShape.TensorName, out EngineBuildShape min, out EngineBuildShape opt, out EngineBuildShape max))
                {
                    throw new ArgumentException($"Missing complete shape profile for tensor '{minShape.TensorName}'.");
                }

                profile.SetShape(min.TensorName, new TensorRtDims(min.Dimensions), new TensorRtDims(opt.Dimensions), new TensorRtDims(max.Dimensions));
            }

            return config.AddOptimizationProfile(profile);
        }

        profile.SetShape(
            "input",
            new TensorRtDims(new[] { 1, 4 }),
            new TensorRtDims(new[] { 2, 4 }),
            new TensorRtDims(new[] { 4, 4 }));
        return config.AddOptimizationProfile(profile);
    }

    private static string DryRunModelSource(OnnxEngineBuildOptions options)
    {
        if (options.UsesExternalOnnx)
        {
            return options.OnnxPath;
        }

        if (options.LoadsExistingEngine)
        {
            return options.LoadEnginePath;
        }

        return "embedded-dynamic-identity";
    }

    private static void ApplyPrecisionFlags(TensorRtBuilderConfig config, OnnxEngineBuildOptions options, List<string> log)
    {
        if (options.Fp16)
        {
            ApplyGlobalPrecisionFlag(config, TensorRtBuilderFlag.Fp16, "Fp16", options.TensorRtLine != TensorRtApiLine.TensorRt11, log);
        }

        if (options.Int8)
        {
            ApplyGlobalPrecisionFlag(config, TensorRtBuilderFlag.Int8, "Int8", options.TensorRtLine != TensorRtApiLine.TensorRt11, log);
        }

        if (options.Bf16)
        {
            ApplyGlobalPrecisionFlag(config, TensorRtBuilderFlag.Bf16, "Bf16", options.TensorRtLine == TensorRtApiLine.TensorRt10, log);
        }

        config.SetFlag(TensorRtBuilderFlag.Tf32, options.Tf32);
        bool tf32Readback = config.GetFlag(TensorRtBuilderFlag.Tf32);
        bool tf32Match = tf32Readback == options.Tf32;
        log.Add($"TrtexecBuildPolicy Name=Tf32 Applied={tf32Match} Requested={options.Tf32} Readback={tf32Readback} ReadbackMatch={tf32Match}");
        if (!tf32Match)
        {
            throw new InvalidOperationException("TF32 builder flag did not match TensorRT readback.");
        }
    }

    private static void ApplyGlobalPrecisionFlag(
        TensorRtBuilderConfig config,
        TensorRtBuilderFlag flag,
        string name,
        bool supported,
        List<string> log)
    {
        if (!supported)
        {
            log.Add($"TrtexecBuildPolicy Name={name} Applied=False Requested=True VersionGuard={config.Line} Reason=builder-precision-flag-not-supported ReadbackMatch=False");
            return;
        }

        config.SetFlag(flag, true);
        bool readback = config.GetFlag(flag);
        log.Add($"TrtexecBuildPolicy Name={name} Applied={readback} Requested=True Readback={readback} ReadbackMatch={readback}");
        if (!readback)
        {
            throw new InvalidOperationException(name + " builder flag did not match TensorRT readback.");
        }
    }

}

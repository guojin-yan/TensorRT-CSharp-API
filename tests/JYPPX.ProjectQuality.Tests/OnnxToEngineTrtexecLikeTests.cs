using JYPPX.TensorRtSharp.Tools;
using System.Security.Cryptography;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OnnxToEngineTrtexecLikeTests
{
    [Fact]
    public void ShapeProfileParserReadsNamedMinOptMaxShapes()
    {
        EngineBuildProfile profile = EngineBuildProfile.Parse(
            "input:1x3x640x640",
            "input:2x3x640x640",
            "input:4x3x640x640");

        Assert.True(profile.TryGetShapeTriple("input", out EngineBuildShape min, out EngineBuildShape opt, out EngineBuildShape max));
        Assert.Equal(new[] { 1, 3, 640, 640 }, min.Dimensions);
        Assert.Equal(new[] { 2, 3, 640, 640 }, opt.Dimensions);
        Assert.Equal(new[] { 4, 3, 640, 640 }, max.Dimensions);
    }

    [Fact]
    public void TrtexecParserAcceptsIdentityDefaultsAndPrecisionFlags()
    {
        string sidecarPath = Path.Combine(Path.GetTempPath(), "onnx-evidence.sidecar.json");
        TrtexecLikeOptions options = TrtexecLikeParser.Parse(new[]
        {
            "--tensor-rt-line", "11",
            "--fp16",
            "--bf16",
            "--workspace", "256",
            "--minShapes", "input:1x4",
            "--optShapes", "input:2x4",
            "--maxShapes", "input:4x4",
            "--exportReport", Path.Combine(Path.GetTempPath(), "onnx-report.json"),
            "--evidenceSidecar", sidecarPath,
            "--buildOnly"
        });

        Assert.Equal(11, (int)options.TensorRtLine);
        Assert.True(options.Fp16);
        Assert.True(options.Bf16);
        Assert.Equal(256UL * 1024UL * 1024UL, options.WorkspaceBytes);
        Assert.True(options.BuildOnly);
        Assert.EndsWith("onnx-report.json", options.ExportReportPath, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(Path.GetFullPath(sidecarPath), options.EvidenceSidecarPath);
        Assert.False(options.UsesExternalOnnx);
        Assert.True(options.ShapeProfile.TryGetShapeTriple("input", out _, out _, out _));
    }

    [Fact]
    public void TrtexecParserAcceptsReportAliasAndNormalizesToExportReport()
    {
        string reportPath = Path.Combine(Path.GetTempPath(), "onnx-report-alias.json");
        TrtexecLikeOptions options = TrtexecLikeParser.Parse(new[]
        {
            "--dryRun",
            "--onnx", Path.Combine(Path.GetTempPath(), "missing-owner-model.onnx"),
            "--report", reportPath,
            "--buildOnly"
        });

        string argumentLine = options.ToArgumentLine();
        Assert.EndsWith("onnx-report-alias.json", options.ExportReportPath, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("--exportReport", argumentLine, StringComparison.Ordinal);
        Assert.DoesNotContain("--report ", argumentLine, StringComparison.Ordinal);
        Assert.True(options.DryRun);
    }

    [Fact]
    public void TrtexecParserCapturesRuntimeBenchmarkAndOutputDiagnostics()
    {
        string tempPath = Path.GetTempPath();
        TrtexecLikeOptions options = TrtexecLikeParser.Parse(new[]
        {
            "--tensor-rt-line", "11",
            "--shapes", "images:1x3x640x640",
            "--noDataTransfers",
            "--useSpinWait",
            "--threads", "3",
            "--avgRuns", "20",
            "--percentile", "99",
            "--sleepTime", "5",
            "--idleTime", "7",
            "--infStreams", "2",
            "--loadInputs", "images:input.bin",
            "--dumpOutput",
            "--dumpRawBindingsToFile", Path.Combine(tempPath, "bindings.raw"),
            "--exportOutput", Path.Combine(tempPath, "output.json"),
            "--exportTimes", Path.Combine(tempPath, "times.json"),
            "--exportProfile", Path.Combine(tempPath, "profile.json"),
            "--saveProfile", Path.Combine(tempPath, "profile.txt"),
            "--minTiming", "2",
            "--avgTiming", "9",
            "--precisionConstraints", "prefer",
            "--layerPrecisions", "conv1:fp16,head:fp32",
            "--layerOutputTypes", "head:fp32",
            "--fp8",
            "--best",
            "--dumpRefit",
            "--allowWeightStreaming",
            "--markDebug", "features,boxes",
            "--dumpDebugTensors",
            "--versionCompatible",
            "--excludeLeanRuntime",
            "--stripWeights",
            "--refit",
            "--weightStreamingBudget", "512MiB",
            "--exportTimingCache", Path.Combine(tempPath, "exported.cache"),
            "--safe",
            "--consistency",
            "--builderCache",
            "--maxNbTactics", "64",
            "--tilingOptimizationLevel", "moderate",
            "--l2LimitForTiling", "256MiB",
            "--quantizationFlags", "calibrateBeforeFusion",
            "--buildOnly",
            "--skipInference"
        });

        Assert.True(options.ShapeProfile.TryGetShapeTriple("images", out EngineBuildShape min, out EngineBuildShape opt, out EngineBuildShape max));
        Assert.Equal(new[] { 1, 3, 640, 640 }, min.Dimensions);
        Assert.Equal(min.Dimensions, opt.Dimensions);
        Assert.Equal(min.Dimensions, max.Dimensions);
        Assert.True(options.RuntimeOptions.NoDataTransfers);
        Assert.True(options.RuntimeOptions.UseSpinWait);
        Assert.Equal(3, options.RuntimeOptions.Threads);
        Assert.Equal(20, options.RuntimeOptions.AvgRuns);
        Assert.Equal("99", options.RuntimeOptions.Percentile?.ToString());
        Assert.Equal(5, options.RuntimeOptions.SleepTimeMilliseconds);
        Assert.Equal(7, options.RuntimeOptions.IdleTimeMilliseconds);
        Assert.Equal(2, options.RuntimeOptions.InfStreams);
        Assert.Equal("images:input.bin", options.RuntimeOptions.LoadInputs);
        Assert.True(options.RuntimeOptions.DumpOutput);
        Assert.EndsWith("bindings.raw", options.RuntimeOptions.DumpRawBindingsToFile, StringComparison.OrdinalIgnoreCase);
        Assert.EndsWith("output.json", options.RuntimeOptions.ExportOutputPath, StringComparison.OrdinalIgnoreCase);
        Assert.EndsWith("times.json", options.RuntimeOptions.ExportTimesPath, StringComparison.OrdinalIgnoreCase);
        Assert.EndsWith("profile.json", options.RuntimeOptions.ExportProfilePath, StringComparison.OrdinalIgnoreCase);
        Assert.EndsWith("profile.txt", options.RuntimeOptions.SaveProfilePath, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(2, options.DeploymentOptions.MinTiming);
        Assert.Equal(9, options.DeploymentOptions.AvgTiming);
        Assert.Equal("prefer", options.DeploymentOptions.PrecisionConstraints);
        Assert.Equal("conv1:fp16,head:fp32", options.DeploymentOptions.LayerPrecisions);
        Assert.Equal("head:fp32", options.DeploymentOptions.LayerOutputTypes);
        Assert.True(options.DeploymentOptions.Fp8);
        Assert.True(options.DeploymentOptions.Best);
        Assert.True(options.DeploymentOptions.DumpRefit);
        Assert.True(options.DeploymentOptions.AllowWeightStreaming);
        Assert.Equal("features,boxes", options.DeploymentOptions.MarkDebug);
        Assert.True(options.DeploymentOptions.DumpDebugTensors);
        Assert.True(options.DeploymentOptions.VersionCompatible);
        Assert.True(options.DeploymentOptions.ExcludeLeanRuntime);
        Assert.True(options.DeploymentOptions.StripWeights);
        Assert.True(options.DeploymentOptions.Refit);
        Assert.Equal(512UL * 1024UL * 1024UL, options.DeploymentOptions.WeightStreamingBudgetBytes);
        Assert.EndsWith("exported.cache", options.DeploymentOptions.ExportTimingCachePath, StringComparison.OrdinalIgnoreCase);
        Assert.True(options.DeploymentOptions.Safe);
        Assert.True(options.DeploymentOptions.Consistency);
        Assert.True(options.DeploymentOptions.BuilderCache);
        Assert.False(options.DeploymentOptions.NoBuilderCache);
        Assert.Equal(64, options.DeploymentOptions.MaxNbTactics);
        Assert.Equal(JYPPX.TensorRtSharp.TensorRtTilingOptimizationLevel.Moderate, options.DeploymentOptions.TilingOptimizationLevel);
        Assert.Equal(256L * 1024L * 1024L, options.DeploymentOptions.L2LimitForTilingBytes);
        Assert.Equal(JYPPX.TensorRtSharp.TensorRtQuantizationFlags.CalibrateBeforeFusion, options.DeploymentOptions.QuantizationFlags);

        string argumentLine = options.ToArgumentLine();
        OnnxEngineBuildOptions buildOptions = OnnxEngineBuildOptions.FromTrtexecLikeOptions(options);

        Assert.Equal(options.TimingCacheFile, buildOptions.TimingCacheFile);
        Assert.Contains("--infStreams 2", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--exportTimes", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--exportProfile", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--minTiming 2", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--avgTiming 9", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--precisionConstraints prefer", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--layerPrecisions conv1:fp16,head:fp32", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--layerOutputTypes head:fp32", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--fp8", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--best", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--dumpRefit", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--allowWeightStreaming", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--markDebug features,boxes", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--dumpDebugTensors", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--versionCompatible", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--excludeLeanRuntime", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--stripWeights", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--refit", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--weightStreamingBudget 512", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--exportTimingCache", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--safe", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--consistency", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--builderCache", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--maxNbTactics 64", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--tilingOptimizationLevel Moderate", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--l2LimitForTiling 268435456B", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--quantizationFlags CalibrateBeforeFusion", argumentLine, StringComparison.Ordinal);
        Assert.Contains("Runtime benchmark/output options", string.Join("\n", buildOptions.Diagnostics), StringComparison.Ordinal);
        Assert.Contains("shortcut precision arguments are parse/report-only", string.Join("\n", buildOptions.Diagnostics), StringComparison.Ordinal);
        Assert.Contains("debug tensor diagnostic arguments are parse/report-only", string.Join("\n", buildOptions.Diagnostics), StringComparison.Ordinal);
        Assert.Contains("Safety/consistency arguments are parse/report-only", string.Join("\n", buildOptions.Diagnostics), StringComparison.Ordinal);
        Assert.Contains("Builder cache policy arguments are parse/report-only", string.Join("\n", buildOptions.Diagnostics), StringComparison.Ordinal);
        Assert.Contains("TrtexecAlignmentStatus=parse-only", string.Join("\n", buildOptions.Diagnostics), StringComparison.Ordinal);
    }

    [Fact]
    public void TrtexecParserAcceptsInputShapesAliasAndValidatesRuntimeNumbers()
    {
        TrtexecLikeOptions options = TrtexecLikeParser.Parse(new[]
        {
            "--inputShapes", "tokens:2x16",
            "--percentile", "99.5",
            "--threads", "1",
            "--avgRuns", "2",
            "--sleepTime", "0",
            "--idleTime", "3"
        });

        Assert.True(options.ShapeProfile.TryGetShapeTriple("tokens", out EngineBuildShape min, out EngineBuildShape opt, out EngineBuildShape max));
        Assert.Equal(new[] { 2, 16 }, min.Dimensions);
        Assert.Equal(min.Dimensions, opt.Dimensions);
        Assert.Equal(min.Dimensions, max.Dimensions);
        Assert.Equal("99.5", options.RuntimeOptions.Percentile?.ToString(System.Globalization.CultureInfo.InvariantCulture));
        Assert.Throws<ArgumentException>(() => TrtexecLikeParser.Parse(new[] { "--threads", "0" }));
        Assert.Throws<ArgumentException>(() => TrtexecLikeParser.Parse(new[] { "--avgRuns", "-1" }));
        Assert.Throws<ArgumentException>(() => TrtexecLikeParser.Parse(new[] { "--sleepTime", "-1" }));
        Assert.Throws<ArgumentException>(() => TrtexecLikeParser.Parse(new[] { "--idleTime", "-1" }));
        Assert.Throws<ArgumentException>(() => TrtexecLikeParser.Parse(new[] { "--infStreams", "0" }));
        Assert.Throws<ArgumentException>(() => TrtexecLikeParser.Parse(new[] { "--minTiming", "0" }));
        Assert.Throws<ArgumentException>(() => TrtexecLikeParser.Parse(new[] { "--avgTiming", "-1" }));
        Assert.Throws<ArgumentException>(() => TrtexecLikeParser.Parse(new[] { "--weightStreamingBudget", "-1MiB" }));
        Assert.Throws<ArgumentException>(() => TrtexecLikeParser.Parse(new[] { "--percentile", "101" }));
        Assert.Throws<ArgumentException>(() => TrtexecLikeParser.Parse(new[] { "--builderCache", "--noBuilderCache" }));
    }

    [Fact]
    public void TrtexecParserAcceptsTrtexecAliasesAndMemoryUnits()
    {
        string tempPath = Path.GetTempPath();
        TrtexecLikeOptions options = TrtexecLikeParser.Parse(new[]
        {
            "--save-engine", Path.Combine(tempPath, "alias.plan"),
            "--load-engine", Path.Combine(tempPath, "existing.plan"),
            "--dryRun",
            "--workspace", "1GiB",
            "--memPoolSize", "workspace:512MiB,tacticDram:1GiB",
            "--timingCache", Path.Combine(tempPath, "alias.cache"),
            "--verbose",
            "--iterations", "12",
            "--warmUp", "250",
            "--duration", "7",
            "--streams", "2",
            "--useCudaGraph"
        });

        Assert.Equal(1024UL * 1024UL * 1024UL, options.WorkspaceBytes);
        Assert.EndsWith("alias.plan", options.SaveEnginePath, StringComparison.OrdinalIgnoreCase);
        Assert.EndsWith("existing.plan", options.LoadEnginePath, StringComparison.OrdinalIgnoreCase);
        Assert.EndsWith("alias.cache", options.TimingCacheFile, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("detailed", options.ProfilingVerbosity);
        Assert.Equal(12, options.Iterations);
        Assert.Equal(250, options.WarmUpMilliseconds);
        Assert.Equal(7, options.DurationSeconds);
        Assert.Equal(2, options.Streams);
        Assert.True(options.UseCudaGraph);
        Assert.Collection(
            options.DeploymentOptions.MemoryPoolSizes,
            item =>
            {
                Assert.Equal("workspace", item.Name);
                Assert.Equal(512UL, item.SizeMiB);
                Assert.Equal(512UL * 1024UL * 1024UL, item.SizeBytes);
            },
            item =>
            {
                Assert.Equal("tacticDram", item.Name);
                Assert.Equal(1024UL, item.SizeMiB);
                Assert.Equal(1024UL * 1024UL * 1024UL, item.SizeBytes);
            });
        Assert.Equal(JYPPX.TensorRtSharp.TensorRtMemoryPoolType.Workspace, options.DeploymentOptions.MemoryPoolSizes[0].ToTensorRtMemoryPoolType());
        Assert.Equal(JYPPX.TensorRtSharp.TensorRtMemoryPoolType.TacticDram, options.DeploymentOptions.MemoryPoolSizes[1].ToTensorRtMemoryPoolType());
        Assert.Equal(
            JYPPX.TensorRtSharp.TensorRtMemoryPoolType.DlaManagedSram,
            new TrtexecLikeMemoryPoolSize("dlaSRAM", 1).ToTensorRtMemoryPoolType());
        Assert.Throws<ArgumentException>(() => new TrtexecLikeMemoryPoolSize("unknownPool", 1).ToTensorRtMemoryPoolType());

        string argumentLine = options.ToArgumentLine();
        Assert.Contains("--saveEngine", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--loadEngine", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--timingCacheFile", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--profilingVerbosity detailed", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--memPoolSize workspace:512,tacticDram:1024", argumentLine, StringComparison.Ordinal);
    }

    [Fact]
    public void TrtexecParserRejectsUnsupportedProfilingVerbosityAndFractionalMiBPool()
    {
        Assert.Throws<ArgumentException>(() => TrtexecLikeParser.Parse(new[]
        {
            "--profilingVerbosity", "layer_details"
        }));
        Assert.Throws<ArgumentException>(() => TrtexecLikeParser.Parse(new[]
        {
            "--memPoolSize", "workspace:1KB"
        }));
        Assert.Throws<ArgumentException>(() => TrtexecLikeParser.Parse(new[]
        {
            "--workspace", "-1MiB"
        }));
    }

    [Fact]
    public void RuntimeArtifactWriterWritesBoundaryArtifactsWithoutPromotingBuildOnlyProof()
    {
        string artifactRoot = Path.Combine(Path.GetTempPath(), "jyppx-runtime-artifacts-" + Guid.NewGuid().ToString("N"));
        try
        {
            TrtexecLikeRuntimeOptions runtimeOptions = new TrtexecLikeRuntimeOptions(
                noDataTransfers: false,
                useSpinWait: false,
                threads: null,
                avgRuns: null,
                percentile: null,
                sleepTimeMilliseconds: null,
                idleTimeMilliseconds: null,
                infStreams: null,
                loadInputs: string.Empty,
                dumpOutput: false,
                dumpRawBindingsToFile: Path.Combine(artifactRoot, "bindings.raw"),
                exportOutputPath: Path.Combine(artifactRoot, "output.json"),
                exportTimesPath: Path.Combine(artifactRoot, "times.json"),
                exportProfilePath: Path.Combine(artifactRoot, "profile.json"),
                saveProfilePath: Path.Combine(artifactRoot, "profile.txt"));
            OnnxEngineBuildResult result = new OnnxEngineBuildResult(
                success: true,
                skipped: false,
                state: "build-only",
                tensorRtLine: JYPPX.Shared.Interop.TensorRtApiLine.TensorRt10,
                modelSource: "embedded-dynamic-identity",
                enginePath: "model.plan",
                parsed: true,
                engineSaved: true,
                engineFileRoundTrip: false,
                inferenceRan: false,
                outputMatch: false,
                profileIndex: 0,
                elapsedMilliseconds: null,
                skipReason: string.Empty,
                normalizedCommandLine: "--buildOnly --exportTimes times.json",
                diagnostics: Array.Empty<string>(),
                logLines: new[] { "OnnxToEngine BuildOnly=True" },
                runtimeOptions: runtimeOptions);

            OnnxEngineRuntimeArtifactWriter.WriteArtifacts(result);

            string times = File.ReadAllText(Path.Combine(artifactRoot, "times.json"));
            string output = File.ReadAllText(Path.Combine(artifactRoot, "output.json"));
            string profile = File.ReadAllText(Path.Combine(artifactRoot, "profile.json"));
            string engineReadback = File.ReadAllText(Path.Combine(artifactRoot, "profile.engine-readback.json"));
            string rawBoundary = File.ReadAllText(Path.Combine(artifactRoot, "bindings.raw"));
            using JsonDocument timesDocument = JsonDocument.Parse(times);
            using JsonDocument outputDocument = JsonDocument.Parse(output);
            using JsonDocument engineReadbackDocument = JsonDocument.Parse(engineReadback);
            using JsonDocument rawBoundaryDocument = JsonDocument.Parse(rawBoundary);
            JsonElement timesRoot = timesDocument.RootElement;
            JsonElement outputRoot = outputDocument.RootElement;
            JsonElement engineReadbackRoot = engineReadbackDocument.RootElement;
            JsonElement rawRoot = rawBoundaryDocument.RootElement;

            Assert.Contains("trtexec-like-times", times, StringComparison.Ordinal);
            Assert.Contains("timing-skipped-build-only", times, StringComparison.Ordinal);
            Assert.Contains("\"IsRuntimeExecutionProof\": false", times, StringComparison.Ordinal);
            Assert.Contains("output-skipped-build-only", output, StringComparison.Ordinal);
            Assert.Contains("profile-skipped-build-only", profile, StringComparison.Ordinal);
            Assert.Contains("trtexec-like-raw-bindings-skipped", rawBoundary, StringComparison.Ordinal);
            Assert.Equal("build-only", timesRoot.GetProperty("RuntimeProofClass").GetString());
            Assert.Equal("timing-skipped-build-only; engine build evidence does not include inference output, timing, or package-consumer runtime proof.", timesRoot.GetProperty("ArtifactProofBoundary").GetString());
            Assert.False(timesRoot.GetProperty("HasTensorOutputProof").GetBoolean());
            Assert.False(timesRoot.GetProperty("HasRawBindingProof").GetBoolean());
            Assert.True(timesRoot.GetProperty("IsBuildOnlyEvidence").GetBoolean());
            Assert.False(timesRoot.GetProperty("IsDependencyProbeOnly").GetBoolean());
            Assert.False(timesRoot.GetProperty("IsSyntheticRuntime").GetBoolean());
            Assert.Equal("embedded-dynamic-identity", timesRoot.GetProperty("ModelSource").GetString());
            Assert.Equal("model.plan", timesRoot.GetProperty("EnginePath").GetString());
            Assert.Equal("trtexec-like-engine-readback-skipped", engineReadbackRoot.GetProperty("ArtifactKind").GetString());
            Assert.Equal("build-only", engineReadbackRoot.GetProperty("RuntimeProofClass").GetString());
            Assert.False(engineReadbackRoot.GetProperty("ReadbackAvailable").GetBoolean());
            Assert.False(engineReadbackRoot.GetProperty("IsRuntimeExecutionProof").GetBoolean());
            Assert.False(engineReadbackRoot.GetProperty("IsRealModelRuntimeProof").GetBoolean());
            Assert.False(engineReadbackRoot.GetProperty("IsPackageConsumerRuntimeProof").GetBoolean());
            Assert.Contains("readonly engine diagnostics", engineReadbackRoot.GetProperty("SkippedReason").GetString(), StringComparison.Ordinal);
            Assert.Equal("build-only", outputRoot.GetProperty("RuntimeProofClass").GetString());
            Assert.False(outputRoot.GetProperty("HasTensorOutputProof").GetBoolean());
            Assert.False(outputRoot.GetProperty("HasRawBindingProof").GetBoolean());
            Assert.Equal("build-only", rawRoot.GetProperty("RuntimeProofClass").GetString());
            Assert.False(rawRoot.GetProperty("HasRawBindingProof").GetBoolean());
            Assert.Contains("LayerProfileAvailable: False", File.ReadAllText(Path.Combine(artifactRoot, "profile.txt")), StringComparison.Ordinal);
        }
        finally
        {
            if (Directory.Exists(artifactRoot))
            {
                Directory.Delete(artifactRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void RuntimeArtifactWriterWritesSyntheticRuntimeArtifactsAndRawBindings()
    {
        string artifactRoot = Path.Combine(Path.GetTempPath(), "jyppx-runtime-artifacts-" + Guid.NewGuid().ToString("N"));
        try
        {
            TrtexecLikeRuntimeOptions runtimeOptions = new TrtexecLikeRuntimeOptions(
                noDataTransfers: false,
                useSpinWait: false,
                threads: null,
                avgRuns: null,
                percentile: null,
                sleepTimeMilliseconds: null,
                idleTimeMilliseconds: null,
                infStreams: null,
                loadInputs: string.Empty,
                dumpOutput: true,
                dumpRawBindingsToFile: Path.Combine(artifactRoot, "bindings.raw"),
                exportOutputPath: Path.Combine(artifactRoot, "output.json"),
                exportTimesPath: Path.Combine(artifactRoot, "times.json"),
                exportProfilePath: Path.Combine(artifactRoot, "profile.json"),
                saveProfilePath: Path.Combine(artifactRoot, "profile.txt"));
            OnnxEngineBuildResult result = new OnnxEngineBuildResult(
                success: true,
                skipped: false,
                state: "identity-roundtrip",
                tensorRtLine: JYPPX.Shared.Interop.TensorRtApiLine.TensorRt10,
                modelSource: "embedded-dynamic-identity",
                enginePath: "model.plan",
                parsed: true,
                engineSaved: true,
                engineFileRoundTrip: true,
                inferenceRan: true,
                outputMatch: true,
                profileIndex: 0,
                elapsedMilliseconds: 1.25f,
                skipReason: string.Empty,
                normalizedCommandLine: "--exportTimes times.json",
                diagnostics: Array.Empty<string>(),
                logLines: new[] { "Execution ElapsedMs=1.25 OutputMatch=True" },
                runtimeOptions: runtimeOptions);
            OnnxEngineRuntimeArtifactData artifactData = OnnxEngineRuntimeArtifactData.CreateIdentityOutput(
                "output",
                new[] { 2, 4 },
                new[] { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f },
                new[] { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f },
                "Execution ElapsedMs=1.25 OutputMatch=True");

            OnnxEngineRuntimeArtifactWriter.WriteArtifacts(result, artifactData);

            string times = File.ReadAllText(Path.Combine(artifactRoot, "times.json"));
            string output = File.ReadAllText(Path.Combine(artifactRoot, "output.json"));
            string profile = File.ReadAllText(Path.Combine(artifactRoot, "profile.json"));
            byte[] raw = File.ReadAllBytes(Path.Combine(artifactRoot, "bindings.raw"));
            using JsonDocument timesDocument = JsonDocument.Parse(times);
            using JsonDocument outputDocument = JsonDocument.Parse(output);
            JsonElement timesRoot = timesDocument.RootElement;
            JsonElement outputRoot = outputDocument.RootElement;

            Assert.Contains("runtime-executed-synthetic-input", times, StringComparison.Ordinal);
            Assert.Contains("\"IsRuntimeExecutionProof\": true", times, StringComparison.Ordinal);
            Assert.Contains("\"ProofClassification\": \"synthetic-input-runtime\"", times, StringComparison.Ordinal);
            Assert.Contains("\"OutputMatch\": true", output, StringComparison.Ordinal);
            Assert.Contains("\"OutputElementCount\": 8", output, StringComparison.Ordinal);
            Assert.Contains("LayerProfileAvailable", profile, StringComparison.Ordinal);
            Assert.Equal("synthetic-input-runtime", timesRoot.GetProperty("RuntimeProofClass").GetString());
            Assert.Equal("runtime-executed-synthetic-input; this is sample runtime evidence only, not real-model or package-consumer proof.", timesRoot.GetProperty("ArtifactProofBoundary").GetString());
            Assert.True(timesRoot.GetProperty("HasTensorOutputProof").GetBoolean());
            Assert.True(timesRoot.GetProperty("HasRawBindingProof").GetBoolean());
            Assert.False(timesRoot.GetProperty("IsBuildOnlyEvidence").GetBoolean());
            Assert.False(timesRoot.GetProperty("IsDependencyProbeOnly").GetBoolean());
            Assert.True(timesRoot.GetProperty("IsSyntheticRuntime").GetBoolean());
            Assert.Equal("synthetic-input-runtime", outputRoot.GetProperty("RuntimeProofClass").GetString());
            Assert.True(outputRoot.GetProperty("HasTensorOutputProof").GetBoolean());
            Assert.True(outputRoot.GetProperty("HasRawBindingProof").GetBoolean());
            Assert.Equal(8 * sizeof(float), raw.Length);
        }
        finally
        {
            if (Directory.Exists(artifactRoot))
            {
                Directory.Delete(artifactRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void RuntimeArtifactWriterCapturesUnverifiedRuntimeOutputWithoutRawBindingPromotion()
    {
        string artifactRoot = Path.Combine(Path.GetTempPath(), "jyppx-runtime-artifacts-" + Guid.NewGuid().ToString("N"));
        try
        {
            TrtexecLikeRuntimeOptions runtimeOptions = new TrtexecLikeRuntimeOptions(
                noDataTransfers: false,
                useSpinWait: false,
                threads: null,
                avgRuns: 2,
                percentile: null,
                sleepTimeMilliseconds: null,
                idleTimeMilliseconds: null,
                infStreams: null,
                loadInputs: "images:input.bin",
                dumpOutput: true,
                dumpRawBindingsToFile: Path.Combine(artifactRoot, "bindings.raw"),
                exportOutputPath: Path.Combine(artifactRoot, "output.json"),
                exportTimesPath: Path.Combine(artifactRoot, "times.json"),
                exportProfilePath: Path.Combine(artifactRoot, "profile.json"),
                saveProfilePath: Path.Combine(artifactRoot, "profile.txt"));
            OnnxEngineBuildResult result = new OnnxEngineBuildResult(
                success: true,
                skipped: false,
                state: "load-engine-runtime-output-unverified",
                tensorRtLine: JYPPX.Shared.Interop.TensorRtApiLine.TensorRt10,
                modelSource: Path.Combine(artifactRoot, "model.plan"),
                enginePath: Path.Combine(artifactRoot, "model.plan"),
                parsed: false,
                engineSaved: false,
                engineFileRoundTrip: true,
                inferenceRan: true,
                outputMatch: false,
                profileIndex: 0,
                elapsedMilliseconds: 2.5f,
                skipReason: string.Empty,
                normalizedCommandLine: "--loadEngine model.plan --loadInputs images:input.bin",
                diagnostics: Array.Empty<string>(),
                logLines: new[] { "LoadEngineBoundedRuntime Attempted=True Succeeded=True" },
                runtimeOptions: runtimeOptions);
            OnnxEngineRuntimeArtifactData artifactData = OnnxEngineRuntimeArtifactData.CreateOutputSummary(
                "scores",
                new[] { 1, 3 },
                inputElementCount: 12,
                outputValues: new[] { 0.1f, 0.2f, 0.3f },
                executionSummary: "profile=0 bound=2 synchronized=False ready=True",
                timingSamplesMilliseconds: new[] { 2.5f, 2.25f });

            OnnxEngineRuntimeArtifactWriter.WriteArtifacts(result, artifactData);

            string times = File.ReadAllText(Path.Combine(artifactRoot, "times.json"));
            string output = File.ReadAllText(Path.Combine(artifactRoot, "output.json"));
            string rawBoundary = File.ReadAllText(Path.Combine(artifactRoot, "bindings.raw"));
            using JsonDocument outputDocument = JsonDocument.Parse(output);
            using JsonDocument rawDocument = JsonDocument.Parse(rawBoundary);
            JsonElement outputRoot = outputDocument.RootElement;
            JsonElement rawRoot = rawDocument.RootElement;

            Assert.Contains("runtime-output-captured-unverified", times, StringComparison.Ordinal);
            Assert.Contains("\"InferenceRan\": true", output, StringComparison.Ordinal);
            Assert.Contains("\"OutputMatch\": false", output, StringComparison.Ordinal);
            Assert.Equal("build-only", outputRoot.GetProperty("RuntimeProofClass").GetString());
            Assert.False(outputRoot.GetProperty("HasTensorOutputProof").GetBoolean());
            Assert.False(outputRoot.GetProperty("HasRawBindingProof").GetBoolean());
            Assert.Equal(12, outputRoot.GetProperty("InputElementCount").GetInt32());
            Assert.Equal(3, outputRoot.GetProperty("OutputElementCount").GetInt32());
            Assert.Equal("trtexec-like-raw-bindings-skipped", rawRoot.GetProperty("ArtifactKind").GetString());
            Assert.False(rawRoot.GetProperty("HasRawBindingProof").GetBoolean());
        }
        finally
        {
            if (Directory.Exists(artifactRoot))
            {
                Directory.Delete(artifactRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void RuntimeArtifactWriterCapturesBenchmarkSamplesAndBoundaries()
    {
        string artifactRoot = Path.Combine(Path.GetTempPath(), "jyppx-runtime-artifacts-" + Guid.NewGuid().ToString("N"));
        try
        {
            TrtexecLikeRuntimeOptions runtimeOptions = new TrtexecLikeRuntimeOptions(
                noDataTransfers: true,
                useSpinWait: true,
                threads: 4,
                avgRuns: 4,
                percentile: 75.0f,
                sleepTimeMilliseconds: 2,
                idleTimeMilliseconds: 3,
                infStreams: null,
                loadInputs: string.Empty,
                dumpOutput: false,
                dumpRawBindingsToFile: string.Empty,
                exportOutputPath: string.Empty,
                exportTimesPath: Path.Combine(artifactRoot, "times.json"),
                exportProfilePath: Path.Combine(artifactRoot, "profile.json"),
                saveProfilePath: Path.Combine(artifactRoot, "profile.txt"));
            OnnxEngineBuildResult result = new OnnxEngineBuildResult(
                success: true,
                skipped: false,
                state: "identity-roundtrip",
                tensorRtLine: JYPPX.Shared.Interop.TensorRtApiLine.TensorRt10,
                modelSource: "embedded-dynamic-identity",
                enginePath: "model.plan",
                parsed: true,
                engineSaved: true,
                engineFileRoundTrip: true,
                inferenceRan: true,
                outputMatch: true,
                profileIndex: 0,
                elapsedMilliseconds: 1.0f,
                skipReason: string.Empty,
                normalizedCommandLine: "--avgRuns 4 --percentile 75 --threads 4 --noDataTransfers",
                diagnostics: Array.Empty<string>(),
                logLines: new[] { "Execution ElapsedMs=1 OutputMatch=True" },
                runtimeOptions: runtimeOptions,
                benchmarkSummary: OnnxEngineBenchmarkSummary.Create(
                    new[] { 1.0f, 3.0f, 2.0f, 4.0f },
                    runtimeOptions,
                    inferenceRan: true,
                    outputMatch: true));
            OnnxEngineRuntimeArtifactData artifactData = OnnxEngineRuntimeArtifactData.CreateIdentityOutput(
                "output",
                new[] { 1, 4 },
                new[] { 0.5f, 1.5f, 2.5f, 3.5f },
                new[] { 0.5f, 1.5f, 2.5f, 3.5f },
                "Execution ElapsedMs=1 OutputMatch=True",
                new[] { 1.0f, 3.0f, 2.0f, 4.0f });

            OnnxEngineRuntimeArtifactWriter.WriteArtifacts(result, artifactData);

            string times = File.ReadAllText(Path.Combine(artifactRoot, "times.json"));
            string profile = File.ReadAllText(Path.Combine(artifactRoot, "profile.json"));
            string profileText = File.ReadAllText(Path.Combine(artifactRoot, "profile.txt"));

            Assert.Contains("\"TimingSampleCount\": 4", times, StringComparison.Ordinal);
            Assert.Contains("\"AverageElapsedMilliseconds\": 2.5", times, StringComparison.Ordinal);
            Assert.Contains("\"PercentileRequested\": 75", times, StringComparison.Ordinal);
            Assert.Contains("\"PercentileElapsedMilliseconds\": 3", times, StringComparison.Ordinal);
            Assert.Contains("\"ThreadsRequested\": 4", times, StringComparison.Ordinal);
            Assert.Contains("\"ThreadsExecuted\": 1", times, StringComparison.Ordinal);
            Assert.Contains("\"NoDataTransfersRequested\": true", times, StringComparison.Ordinal);
            Assert.Contains("\"NoDataTransfersApplied\": false", times, StringComparison.Ordinal);
            Assert.Contains("benchmark-executed-synthetic-input", times, StringComparison.Ordinal);
            Assert.Contains("\"TimingSampleCount\": 4", profile, StringComparison.Ordinal);
            Assert.Contains("TimingSampleCount: 4", profileText, StringComparison.Ordinal);
            Assert.Contains("AverageElapsedMilliseconds: 2.5", profileText, StringComparison.Ordinal);
        }
        finally
        {
            if (Directory.Exists(artifactRoot))
            {
                Directory.Delete(artifactRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void BuildDiagnosticsIncludesBenchmarkSummaryWithoutPromotingProof()
    {
        TrtexecLikeRuntimeOptions runtimeOptions = new TrtexecLikeRuntimeOptions(
            noDataTransfers: true,
            useSpinWait: false,
            threads: 2,
            avgRuns: 3,
            percentile: 90.0f,
            sleepTimeMilliseconds: null,
            idleTimeMilliseconds: null,
            infStreams: null,
            loadInputs: string.Empty,
            dumpOutput: false,
            dumpRawBindingsToFile: string.Empty,
            exportOutputPath: string.Empty,
            exportTimesPath: string.Empty,
            exportProfilePath: string.Empty,
            saveProfilePath: string.Empty);
        OnnxEngineBuildResult result = new OnnxEngineBuildResult(
            success: true,
            skipped: false,
            state: "identity-roundtrip",
            tensorRtLine: JYPPX.Shared.Interop.TensorRtApiLine.TensorRt11,
            modelSource: "embedded-dynamic-identity",
            enginePath: "model.plan",
            parsed: true,
            engineSaved: true,
            engineFileRoundTrip: true,
            inferenceRan: true,
            outputMatch: true,
            profileIndex: 0,
            elapsedMilliseconds: 1.0f,
            skipReason: string.Empty,
            normalizedCommandLine: "--avgRuns 3 --percentile 90 --threads 2 --noDataTransfers",
            diagnostics: Array.Empty<string>(),
            logLines: new[] { "RuntimeBenchmark Samples=3" },
            runtimeOptions: runtimeOptions,
            benchmarkSummary: OnnxEngineBenchmarkSummary.Create(
                new[] { 1.0f, 3.0f, 2.0f },
                runtimeOptions,
                inferenceRan: true,
                outputMatch: true));

        string json = OnnxEngineBuildDiagnostics.ToJson(result);
        string markdown = OnnxEngineBuildDiagnostics.ToMarkdown(result);

        Assert.Contains("\"BenchmarkSummary\"", json, StringComparison.Ordinal);
        Assert.Contains("\"TimingSampleCount\": 3", json, StringComparison.Ordinal);
        Assert.Contains("\"PercentileElapsedMilliseconds\": 3", json, StringComparison.Ordinal);
        Assert.Contains("\"ThreadsExecuted\": 1", json, StringComparison.Ordinal);
        Assert.Contains("\"NoDataTransfersApplied\": false", json, StringComparison.Ordinal);
        Assert.Contains("\"IsRuntimeExecutionProof\": true", json, StringComparison.Ordinal);
        Assert.Contains("\"IsRealModelRuntimeProof\": false", json, StringComparison.Ordinal);
        Assert.Contains("Benchmark sample count: `3`", markdown, StringComparison.Ordinal);
        Assert.Contains("Benchmark boundary: `benchmark-executed-synthetic-input", markdown, StringComparison.Ordinal);
        Assert.Contains("synthetic-input-runtime is not real model proof", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void TrtexecParserRecordsBuildConversionDiagnosticsAndArgumentPreview()
    {
        TrtexecLikeOptions options = TrtexecLikeParser.Parse(new[]
        {
            "--tensor-rt-line", "10",
            "--saveEngine", Path.Combine(Path.GetTempPath(), "model.plan"),
            "--plugins", "custom_a.dll;custom_b.dll",
            "--plugin", "custom_c.dll",
            "--dynamicPlugins=custom_d.dll,custom_a.dll",
            "--setPluginsToSerialize", "serialize_a.dll;serialize_b.dll",
            "--timingCacheFile", Path.Combine(Path.GetTempPath(), "model.cache"),
            "--profilingVerbosity", "detailed",
            "--builderOptimizationLevel=4",
            "--maxAuxStreams", "2",
            "--device", "0",
            "--useDLACore", "1",
            "--allowGPUFallback",
            "--tacticSources", "+CUBLAS,-CUDNN",
            "--memPoolSize", "workspace:512,tacticDram:1024",
            "--inputIOFormats", "fp16:chw",
            "--outputIOFormats", "fp32:chw",
            "--calib", Path.Combine(Path.GetTempPath(), "model.calib"),
            "--directIO",
            "--sparsity", "enable",
            "--stronglyTyped",
            "--dumpLayerInfo",
            "--exportLayerInfo", Path.Combine(Path.GetTempPath(), "layers.json"),
            "--dumpProfile",
            "--separateProfileRun",
            "--buildOnly",
            "--skipInference"
        });

        OnnxEngineBuildOptions buildOptions = OnnxEngineBuildOptions.FromTrtexecLikeOptions(options);
        string argumentLine = options.ToArgumentLine();

        Assert.Equal(6, options.Plugins.Count);
        Assert.Contains("custom_a.dll", options.Plugins);
        Assert.Contains("custom_b.dll", options.Plugins);
        Assert.Contains("custom_c.dll", options.Plugins);
        Assert.Contains("custom_d.dll", options.Plugins);
        Assert.Contains("serialize_a.dll", options.Plugins);
        Assert.Contains("serialize_b.dll", options.Plugins);
        Assert.Equal("detailed", options.ProfilingVerbosity);
        Assert.True(options.DumpLayerInfo);
        Assert.True(buildOptions.DumpLayerInfo);
        Assert.Equal(options.ExportLayerInfoPath, buildOptions.ExportLayerInfoPath);
        Assert.True(options.DumpProfile);
        Assert.True(options.SeparateProfileRun);
        Assert.Equal(4, options.DeploymentOptions.BuilderOptimizationLevel);
        Assert.Equal(2, options.DeploymentOptions.MaxAuxStreams);
        Assert.Equal(0, options.DeploymentOptions.DeviceOrdinal);
        Assert.Equal(1, options.DeploymentOptions.DlaCore);
        Assert.True(options.DeploymentOptions.AllowGpuFallback);
        Assert.Equal("+CUBLAS,-CUDNN", options.DeploymentOptions.TacticSources);
        Assert.Equal(2, options.DeploymentOptions.MemoryPoolSizes.Count);
        Assert.Equal("workspace", options.DeploymentOptions.MemoryPoolSizes[0].Name);
        Assert.Equal(512UL, options.DeploymentOptions.MemoryPoolSizes[0].SizeMiB);
        Assert.Equal("fp16:chw", options.DeploymentOptions.InputIOFormats);
        Assert.Equal("fp32:chw", options.DeploymentOptions.OutputIOFormats);
        Assert.True(options.DeploymentOptions.DirectIO);
        Assert.Equal("enable", options.DeploymentOptions.Sparsity);
        Assert.True(options.DeploymentOptions.StronglyTyped);
        Assert.Equal(4, buildOptions.DeploymentOptions.BuilderOptimizationLevel);
        Assert.Equal(2, buildOptions.DeploymentOptions.MaxAuxStreams);
        Assert.Contains("does not load plugin libraries", string.Join("\n", buildOptions.Diagnostics), StringComparison.Ordinal);
        Assert.Contains("Timing cache input is imported", string.Join("\n", buildOptions.Diagnostics), StringComparison.Ordinal);
        Assert.Contains("ProfilingVerbosity=detailed", string.Join("\n", buildOptions.Diagnostics), StringComparison.Ordinal);
        Assert.Contains("BuilderOptimizationLevel=4 is applied", string.Join("\n", buildOptions.Diagnostics), StringComparison.Ordinal);
        Assert.Contains("MaxAuxStreams=2 is applied", string.Join("\n", buildOptions.Diagnostics), StringComparison.Ordinal);
        Assert.Contains("DLA options are parsed", string.Join("\n", buildOptions.Diagnostics), StringComparison.Ordinal);
        Assert.Contains("MemPoolSize=workspace:512,tacticDram:1024", string.Join("\n", buildOptions.Diagnostics), StringComparison.Ordinal);
        Assert.Contains("Calibration cache path is recorded", string.Join("\n", buildOptions.Diagnostics), StringComparison.Ordinal);
        Assert.Contains("copied TensorRT engine-inspector readback", string.Join("\n", buildOptions.Diagnostics), StringComparison.Ordinal);
        Assert.Contains("--plugins", argumentLine, StringComparison.Ordinal);
        Assert.Contains("custom_a.dll;custom_b.dll;custom_c.dll;custom_d.dll;serialize_a.dll;serialize_b.dll", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--timingCacheFile", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--profilingVerbosity detailed", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--builderOptimizationLevel 4", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--maxAuxStreams 2", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--memPoolSize workspace:512,tacticDram:1024", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--calib", argumentLine, StringComparison.Ordinal);
        Assert.Contains("--dumpLayerInfo", argumentLine, StringComparison.Ordinal);
    }

    [Fact]
    public void TrtexecParserRejectsPluginAliasesWithoutValues()
    {
        Assert.Throws<ArgumentException>(() => TrtexecLikeParser.Parse(new[] { "--plugin" }));
        Assert.Throws<ArgumentException>(() => TrtexecLikeParser.Parse(new[] { "--dynamicPlugins", "--buildOnly" }));
        Assert.Throws<ArgumentException>(() => TrtexecLikeParser.Parse(new[] { "--setPluginsToSerialize=" }));
    }

    [Fact]
    public void MemoryPoolOptionsUseTypedBuilderSetAndReadbackOnlyDuringBuild()
    {
        string service = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "OnnxEngineBuildService.cs"));
        string diagnostics = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "OnnxEngineBuildDiagnostics.cs"));
        string deployment = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "TrtexecLikeDeploymentOptions.cs"));

        Assert.Contains("memoryPool.ToTensorRtMemoryPoolType()", service, StringComparison.Ordinal);
        Assert.Contains("config.SetMemoryPoolLimit(pool, memoryPool.SizeBytes)", service, StringComparison.Ordinal);
        Assert.Contains("config.GetMemoryPoolLimit(pool)", service, StringComparison.Ordinal);
        Assert.Contains("ToTensorRtMemoryPoolType", deployment, StringComparison.Ordinal);
        Assert.Contains("result.Parsed || result.EngineSaved", diagnostics, StringComparison.Ordinal);
        Assert.Contains("!(result.Parsed || result.EngineSaved)", diagnostics, StringComparison.Ordinal);
        Assert.DoesNotContain("result.InferenceRan));", diagnostics, StringComparison.Ordinal);
    }

    [Fact]
    public void TimingIterationOptionsUseVersionGuardedBuilderSettersAndReadback()
    {
        string service = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "OnnxEngineBuildService.cs"));
        string options = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "OnnxEngineBuildOptions.cs"));
        string diagnostics = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "OnnxEngineBuildDiagnostics.cs"));

        Assert.Contains("config.SetAverageTimingIterations(requestedIterations)", service, StringComparison.Ordinal);
        Assert.Contains("config.GetAverageTimingIterations()", service, StringComparison.Ordinal);
        Assert.Contains("config.SetMinTimingIterationsCompatibility(requestedIterations)", service, StringComparison.Ordinal);
        Assert.Contains("config.MinTimingIterationsCompatibility", service, StringComparison.Ordinal);
        Assert.Contains("options.TensorRtLine == TensorRtApiLine.TensorRt8", service, StringComparison.Ordinal);
        Assert.Contains("AverageApplied=True", service, StringComparison.Ordinal);
        Assert.Contains("ReadbackMatch={readbackIterations == requestedIterations}", service, StringComparison.Ordinal);
        Assert.Contains("TensorRT native bridge dependency is unavailable", service, StringComparison.Ordinal);
        Assert.Contains("exception is CudaException", service, StringComparison.Ordinal);
        Assert.Contains("state: \"dependency-probe-only\"", service, StringComparison.Ordinal);
        Assert.Contains("TensorRT 10/11 keep this option parse-only", options, StringComparison.Ordinal);
        Assert.Contains("--avgTiming", diagnostics, StringComparison.Ordinal);
        Assert.Contains("TensorRtApiLine.TensorRt8", diagnostics, StringComparison.Ordinal);
    }

    [Fact]
    public void LayerInfoOptionsUseCopiedInspectorReadbackAndEvidenceBoundary()
    {
        string service = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "OnnxEngineBuildService.cs"));
        string options = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "OnnxEngineBuildOptions.cs"));
        string diagnostics = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "OnnxEngineBuildDiagnostics.cs"));

        Assert.Contains("TryCollectLayerInformationFromSerializedEngine", service, StringComparison.Ordinal);
        Assert.Contains("inspector.GetLayerInformation(index, TensorRtLayerInformationFormat.Oneline)", service, StringComparison.Ordinal);
        Assert.Contains("LayerInfo Collected=True", service, StringComparison.Ordinal);
        Assert.Contains("LayerInfo ExportRequested=True Written=True", service, StringComparison.Ordinal);
        Assert.Contains("new UTF8Encoding(encoderShouldEmitUTF8Identifier: false)", service, StringComparison.Ordinal);
        Assert.Contains("copied-engine-inspector-diagnostics-only", service, StringComparison.Ordinal);
        Assert.Contains("public bool DumpLayerInfo", options, StringComparison.Ordinal);
        Assert.Contains("public string ExportLayerInfoPath", options, StringComparison.Ordinal);
        Assert.Contains("LayerInfo Collected=True", diagnostics, StringComparison.Ordinal);
        Assert.Contains("LayerInfo ExportRequested=True Written=True", diagnostics, StringComparison.Ordinal);
    }

    [Fact]
    public void BuildReportIncludesNormalizedCommandAndEvidenceBoundary()
    {
        TrtexecLikeOptions options = TrtexecLikeParser.Parse(new[]
        {
            "--tensor-rt-line", "10",
            "--workspace", "128",
            "--memPoolSize", "workspace:64,tacticDram:128",
            "--minTiming", "2",
            "--avgTiming", "4",
            "--precisionConstraints", "prefer",
            "--layerPrecisions", "conv1:fp16",
            "--fp8",
            "--best",
            "--dumpRefit",
            "--allowWeightStreaming",
            "--markDebug", "conv1,head",
            "--dumpDebugTensors",
            "--versionCompatible",
            "--weightStreamingBudget", "256MiB",
            "--safe",
            "--noBuilderCache",
            "--infStreams", "2",
            "--dumpLayerInfo",
            "--dumpProfile",
            "--separateProfileRun",
            "--buildOnly",
            "--skipInference"
        });
        OnnxEngineBuildOptions buildOptions = OnnxEngineBuildOptions.FromTrtexecLikeOptions(options);
        OnnxEngineBuildResult result = new OnnxEngineBuildResult(
            success: true,
            skipped: false,
            state: "build-only",
            tensorRtLine: buildOptions.TensorRtLine,
            modelSource: "embedded-dynamic-identity",
            enginePath: "model.plan",
            parsed: true,
            engineSaved: true,
            engineFileRoundTrip: false,
            inferenceRan: false,
            outputMatch: false,
            profileIndex: 0,
            elapsedMilliseconds: null,
            skipReason: string.Empty,
            normalizedCommandLine: buildOptions.NormalizedCommandLine,
            deploymentOptions: buildOptions.DeploymentOptions,
            diagnostics: buildOptions.Diagnostics,
            logLines: new[] { "OnnxToEngine BuildOnly=True" },
            runtimeOptions: buildOptions.RuntimeOptions,
            capabilityProbe: new OnnxEngineCapabilityProbe(
                attempted: true,
                probeState: "capability-probe-only",
                tensorRtLine: buildOptions.TensorRtLine,
                tensorRtVersion: "test-trt",
                cudaToolkitVersion: "test-cuda",
                runtimeAvailable: true,
                builderAvailable: true,
                builderConfigAvailable: true,
                engineInspectorApiAvailable: true,
                fp8FlagRequested: true,
                fp8FlagKnown: true,
                debugTensorOptionsRequested: true,
                debugTensorApiKnown: true,
                weightStreamingRequested: true,
                weightStreamingApiKnown: true,
                probeItems: new[]
                {
                    "runtime:create:true",
                    "builder:create:true",
                    "fp8-builder-flag:requested:True:known:True",
                    "debug-tensor-options:requested:True:known:True",
                    "weight-streaming-options:requested:True:known:True"
                },
                evidenceBoundary: "capability-probe-only records API availability; it does not enqueue inference or prove package-consumer-runtime."));

        string json = OnnxEngineBuildDiagnostics.ToJson(result);
        string markdown = OnnxEngineBuildDiagnostics.ToMarkdown(result);

        Assert.True(result.BuildEvidenceOnly);
        Assert.False(result.IsRuntimeExecutionProof);
        Assert.Equal("build-only", result.ProofClassification);
        Assert.False(result.IsRealModelRuntimeProof);
        Assert.False(result.IsPackageConsumerRuntimeProof);
        Assert.Contains("\"NormalizedCommandLine\"", json, StringComparison.Ordinal);
        Assert.Contains("\"NormalizedCommandSha256\"", json, StringComparison.Ordinal);
        Assert.Contains("\"DeploymentOptions\"", json, StringComparison.Ordinal);
        Assert.Contains("\"RuntimeOptions\"", json, StringComparison.Ordinal);
        Assert.Contains("\"OptionImplementationStatus\"", json, StringComparison.Ordinal);
        Assert.Contains("\"ParsedOptions\"", json, StringComparison.Ordinal);
        Assert.Contains("\"AppliedOptions\"", json, StringComparison.Ordinal);
        Assert.Contains("\"ParseOnlyOptions\"", json, StringComparison.Ordinal);
        Assert.Contains("\"--minTiming\"", json, StringComparison.Ordinal);
        Assert.Contains("\"--precisionConstraints\"", json, StringComparison.Ordinal);
        Assert.Contains("\"--fp8\"", json, StringComparison.Ordinal);
        Assert.Contains("\"--best\"", json, StringComparison.Ordinal);
        Assert.Contains("\"--dumpRefit\"", json, StringComparison.Ordinal);
        Assert.Contains("\"--allowWeightStreaming\"", json, StringComparison.Ordinal);
        Assert.Contains("\"--markDebug\"", json, StringComparison.Ordinal);
        Assert.Contains("\"--dumpDebugTensors\"", json, StringComparison.Ordinal);
        Assert.Contains("\"--weightStreamingBudget\"", json, StringComparison.Ordinal);
        Assert.Contains("\"--safe\"", json, StringComparison.Ordinal);
        Assert.Contains("\"--noBuilderCache\"", json, StringComparison.Ordinal);
        Assert.Contains("\"--infStreams\"", json, StringComparison.Ordinal);
        Assert.Contains("\"--dumpLayerInfo\"", json, StringComparison.Ordinal);
        Assert.Contains("\"--dumpProfile\"", json, StringComparison.Ordinal);
        Assert.Contains("\"--separateProfileRun\"", json, StringComparison.Ordinal);
        using JsonDocument report = JsonDocument.Parse(json);
        JsonElement optionStatus = report.RootElement.GetProperty("OptionImplementationStatus");
        Assert.Contains(optionStatus.GetProperty("ParsedOptions").EnumerateArray(), static item => item.GetString() == "--minTiming");
        Assert.Contains(optionStatus.GetProperty("ParsedOptions").EnumerateArray(), static item => item.GetString() == "--infStreams");
        Assert.Contains(optionStatus.GetProperty("ParsedOptions").EnumerateArray(), static item => item.GetString() == "--dumpLayerInfo");
        Assert.Contains(optionStatus.GetProperty("ParsedOptions").EnumerateArray(), static item => item.GetString() == "--dumpProfile");
        Assert.Contains(optionStatus.GetProperty("ParsedOptions").EnumerateArray(), static item => item.GetString() == "--separateProfileRun");
        Assert.Contains(optionStatus.GetProperty("AppliedOptions").EnumerateArray(), static item => item.GetString() == "--builderOptimizationLevel");
        Assert.Contains(optionStatus.GetProperty("AppliedOptions").EnumerateArray(), static item => item.GetString() == "--memPoolSize");
        Assert.Contains(optionStatus.GetProperty("ParseOnlyOptions").EnumerateArray(), static item => item.GetString() == "--minTiming");
        Assert.Contains(optionStatus.GetProperty("ParseOnlyOptions").EnumerateArray(), static item => item.GetString() == "--precisionConstraints");
        Assert.Contains(optionStatus.GetProperty("ParseOnlyOptions").EnumerateArray(), static item => item.GetString() == "--fp8");
        Assert.Contains(optionStatus.GetProperty("ParseOnlyOptions").EnumerateArray(), static item => item.GetString() == "--best");
        Assert.Contains(optionStatus.GetProperty("ParseOnlyOptions").EnumerateArray(), static item => item.GetString() == "--dumpRefit");
        Assert.Contains(optionStatus.GetProperty("ParseOnlyOptions").EnumerateArray(), static item => item.GetString() == "--allowWeightStreaming");
        Assert.Contains(optionStatus.GetProperty("ParseOnlyOptions").EnumerateArray(), static item => item.GetString() == "--markDebug");
        Assert.Contains(optionStatus.GetProperty("ParseOnlyOptions").EnumerateArray(), static item => item.GetString() == "--dumpDebugTensors");
        Assert.Contains(optionStatus.GetProperty("ParseOnlyOptions").EnumerateArray(), static item => item.GetString() == "--weightStreamingBudget");
        Assert.Contains(optionStatus.GetProperty("ParseOnlyOptions").EnumerateArray(), static item => item.GetString() == "--safe");
        Assert.Contains(optionStatus.GetProperty("ParseOnlyOptions").EnumerateArray(), static item => item.GetString() == "--noBuilderCache");
        Assert.Contains(optionStatus.GetProperty("ParseOnlyOptions").EnumerateArray(), static item => item.GetString() == "--infStreams");
        Assert.Contains(optionStatus.GetProperty("ParseOnlyOptions").EnumerateArray(), static item => item.GetString() == "--dumpLayerInfo");
        Assert.Contains(optionStatus.GetProperty("ParseOnlyOptions").EnumerateArray(), static item => item.GetString() == "--dumpProfile");
        Assert.Contains(optionStatus.GetProperty("ParseOnlyOptions").EnumerateArray(), static item => item.GetString() == "--separateProfileRun");
        Assert.Contains(optionStatus.GetProperty("ParsedOptions").EnumerateArray(), static item => item.GetString() == "--profilingVerbosity");
        Assert.Contains("parse-only/build-only/capability-probe-only evidence cannot promote", optionStatus.GetProperty("EvidenceBoundary").GetString(), StringComparison.Ordinal);
        Assert.Contains("capability-probe-only", optionStatus.GetProperty("ParseOnlyOptions").EnumerateArray().Select(static item => item.GetString()));
        Assert.True(report.RootElement.GetProperty("CapabilityProbe").GetProperty("Attempted").GetBoolean());
        Assert.Contains("capability-probe-only", report.RootElement.GetProperty("CapabilityProbe").GetProperty("EvidenceBoundary").GetString(), StringComparison.Ordinal);
        Assert.Contains("\"BuilderOptimizationLevel\": 3", json, StringComparison.Ordinal);
        Assert.Contains("\"ProofClassification\": \"build-only\"", json, StringComparison.Ordinal);
        Assert.Contains("\"EvidenceClassifications\"", json, StringComparison.Ordinal);
        Assert.Contains("\"BuildEvidenceOnly\": true", json, StringComparison.Ordinal);
        Assert.Contains("\"IsRealModelRuntimeProof\": false", json, StringComparison.Ordinal);
        Assert.Contains("\"IsPackageConsumerRuntimeProof\": false", json, StringComparison.Ordinal);
        Assert.Contains("\"StdoutSummary\"", json, StringComparison.Ordinal);
        Assert.Contains("\"StderrSummary\"", json, StringComparison.Ordinal);
        Assert.Contains("\"ModelEvidence\"", json, StringComparison.Ordinal);
        Assert.Contains("\"EvidenceSidecarPath\"", json, StringComparison.Ordinal);
        Assert.Contains("\"EvidenceSidecarDiagnostics\"", json, StringComparison.Ordinal);
        Assert.Contains("\"IsRuntimeExecutionProof\": false", json, StringComparison.Ordinal);
        Assert.Contains("\"ReportBoundary\"", json, StringComparison.Ordinal);
        Assert.Contains("\"IsRuntimeProof\": false", json, StringComparison.Ordinal);
        Assert.Contains("\"TensorRtExec report\"", json, StringComparison.Ordinal);
        Assert.Contains("--workspace 128", json, StringComparison.Ordinal);
        Assert.Contains(optionStatus.GetProperty("AppliedOptions").EnumerateArray(), static item => item.GetString() == "--avgTiming");
        Assert.Contains("Normalized command line", markdown, StringComparison.Ordinal);
        Assert.Contains("Normalized command SHA256", markdown, StringComparison.Ordinal);
        Assert.Contains("Option Implementation Status", markdown, StringComparison.Ordinal);
        Assert.Contains("Parse-only options", markdown, StringComparison.Ordinal);
        Assert.Contains("`--minTiming`", markdown, StringComparison.Ordinal);
        Assert.Contains("`--precisionConstraints`", markdown, StringComparison.Ordinal);
        Assert.Contains("`--fp8`", markdown, StringComparison.Ordinal);
        Assert.Contains("`--best`", markdown, StringComparison.Ordinal);
        Assert.Contains("`--dumpRefit`", markdown, StringComparison.Ordinal);
        Assert.Contains("`--allowWeightStreaming`", markdown, StringComparison.Ordinal);
        Assert.Contains("`--markDebug`", markdown, StringComparison.Ordinal);
        Assert.Contains("`--dumpDebugTensors`", markdown, StringComparison.Ordinal);
        Assert.Contains("`--weightStreamingBudget`", markdown, StringComparison.Ordinal);
        Assert.Contains("`--safe`", markdown, StringComparison.Ordinal);
        Assert.Contains("`--noBuilderCache`", markdown, StringComparison.Ordinal);
        Assert.Contains("`--infStreams`", markdown, StringComparison.Ordinal);
        Assert.Contains("Builder optimization level: `3`", markdown, StringComparison.Ordinal);
        Assert.Contains("Max aux streams", markdown, StringComparison.Ordinal);
        Assert.Contains("No data transfers", markdown, StringComparison.Ordinal);
        Assert.Contains("Export profile", markdown, StringComparison.Ordinal);
        Assert.Contains("Proof classification: `build-only`", markdown, StringComparison.Ordinal);
        Assert.Contains("Evidence Classifications", markdown, StringComparison.Ordinal);
        Assert.Contains("Evidence Sidecar Diagnostics", markdown, StringComparison.Ordinal);
        Assert.Contains("Build evidence only: `True`", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void DryRunReportDoesNotRequireModelFileOrPromoteRuntimeProof()
    {
        string artifactRoot = Path.Combine(RepositoryPaths.Root, "artifacts", "test-temp", $"jyppx-dry-run-{Guid.NewGuid():N}");
        string reportPath = Path.Combine(artifactRoot, "report.json");
        string timingCachePath = Path.Combine(artifactRoot, "input.cache");
        string exportedTimingCachePath = Path.Combine(artifactRoot, "output.cache");
        string missingOnnx = Path.Combine(artifactRoot, "missing.onnx");
        Directory.CreateDirectory(artifactRoot);
        File.WriteAllBytes(timingCachePath, new byte[] { 0x54, 0x49, 0x4D, 0x45 });
        TrtexecLikeOptions options = TrtexecLikeParser.Parse(new[]
        {
            "--tensor-rt-line", "10",
            "--onnx", missingOnnx,
            "--saveEngine", Path.Combine(Path.GetTempPath(), "model.plan"),
            "--minShapes", "images:1x3x640x640",
            "--optShapes", "images:1x3x640x640",
            "--maxShapes", "images:4x3x640x640",
            "--builderOptimizationLevel", "4",
            "--maxAuxStreams", "2",
            "--maxNbTactics", "64",
            "--tilingOptimizationLevel", "moderate",
            "--l2LimitForTiling", "256MiB",
            "--quantizationFlags", "calibrateBeforeFusion",
            "--timingCacheFile", timingCachePath,
            "--exportTimingCache", exportedTimingCachePath,
            "--exportReport", reportPath,
            "--previewOnly"
        });

        try
        {
            Assert.True(options.DryRun);
            Assert.Contains("--dryRun", options.ToArgumentLine(), StringComparison.Ordinal);
            OnnxEngineBuildOptions buildOptions = OnnxEngineBuildOptions.FromTrtexecLikeOptions(options);
            Assert.True(buildOptions.DryRun);
            Assert.Contains("Dry run requested", string.Join("\n", buildOptions.Diagnostics), StringComparison.Ordinal);

            OnnxEngineBuildResult result = new OnnxEngineBuildService().Execute(buildOptions);
            string json = File.ReadAllText(reportPath);
            using JsonDocument document = JsonDocument.Parse(json);
            JsonElement root = document.RootElement;

            Assert.True(result.Success);
            Assert.Equal("dry-run-precheck", result.State);
            Assert.Equal("precheck", result.ProofClassification);
            Assert.True(result.BuildEvidenceOnly);
            Assert.False(result.Parsed);
            Assert.False(result.EngineSaved);
            Assert.False(result.InferenceRan);
            Assert.False(result.IsRuntimeExecutionProof);
            Assert.False(result.IsPackageConsumerRuntimeProof);
            Assert.Equal(missingOnnx, result.ModelSource);
            Assert.Equal(64, result.NormalizedCommandSha256.Length);
            Assert.True(result.TimingCacheArtifact.InputRequested);
            Assert.False(result.TimingCacheArtifact.InputApplied);
            Assert.Equal(0, result.TimingCacheArtifact.InputLengthBytes);
            Assert.True(result.TimingCacheArtifact.OutputRequested);
            Assert.False(result.TimingCacheArtifact.OutputWritten);
            Assert.Equal("precheck-not-executed", result.TimingCacheArtifact.State);
            Assert.False(File.Exists(exportedTimingCachePath));
            Assert.True(root.GetProperty("DryRun").GetBoolean());
            Assert.Equal("precheck", root.GetProperty("ProofClassification").GetString());
            Assert.Equal(result.NormalizedCommandSha256, root.GetProperty("NormalizedCommandSha256").GetString());
            Assert.Contains("runtime probing, ONNX parsing, engine build", string.Join("\n", result.LogLines), StringComparison.Ordinal);
            JsonElement status = root.GetProperty("OptionImplementationStatus");
            foreach (string optionName in new[] { "--maxNbTactics", "--tilingOptimizationLevel", "--l2LimitForTiling", "--quantizationFlags" })
            {
                Assert.Contains(status.GetProperty("ParsedOptions").EnumerateArray(), item => item.GetString() == optionName);
                Assert.Contains(status.GetProperty("ParseOnlyOptions").EnumerateArray(), item => item.GetString() == optionName);
                Assert.DoesNotContain(status.GetProperty("AppliedOptions").EnumerateArray(), item => item.GetString() == optionName);
            }

            string service = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "OnnxEngineBuildService.cs"));
            int dryRunBranch = service.IndexOf("if (options.DryRun)", StringComparison.Ordinal);
            int runtimeProbe = service.IndexOf("TensorRtEnvironmentProbe.GetCurrent()", StringComparison.Ordinal);
            Assert.True(dryRunBranch >= 0);
            Assert.True(runtimeProbe > dryRunBranch);
        }
        finally
        {
            if (File.Exists(reportPath))
            {
                File.Delete(reportPath);
            }

            if (Directory.Exists(artifactRoot))
            {
                Directory.Delete(artifactRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void LoadEnginePreflightReportIncludesFileMetadataWithoutRuntimeProof()
    {
        string enginePath = Path.Combine(Path.GetTempPath(), $"jyppx-load-engine-{Guid.NewGuid():N}.plan");
        string reportPath = Path.Combine(Path.GetTempPath(), $"jyppx-load-engine-{Guid.NewGuid():N}.json");
        byte[] engineBytes = new byte[] { 0x54, 0x52, 0x54, 0x45, 0x58, 0x45, 0x43, 0x01 };
        File.WriteAllBytes(enginePath, engineBytes);

        try
        {
            TrtexecLikeOptions options = TrtexecLikeParser.Parse(new[]
            {
                "--tensor-rt-line", "10",
                "--loadEngine", enginePath,
                "--exportReport", reportPath,
                "--buildOnly"
            });

            OnnxEngineBuildResult result = new OnnxEngineBuildService().Execute(OnnxEngineBuildOptions.FromTrtexecLikeOptions(options));
            string json = File.ReadAllText(reportPath);
            using JsonDocument document = JsonDocument.Parse(json);
            JsonElement root = document.RootElement;
            JsonElement metadata = root.GetProperty("PreflightMetadata");
            JsonElement loadedEngineDiagnostics = root.GetProperty("LoadedEngineDiagnostics");

            Assert.True(result.Success);
            Assert.Contains("load-engine-", result.State, StringComparison.Ordinal);
            Assert.Equal("dependency-probe-only", result.ProofClassification);
            Assert.True(result.BuildEvidenceOnly);
            Assert.False(result.Parsed);
            Assert.False(result.EngineSaved);
            Assert.False(result.InferenceRan);
            Assert.False(result.IsRuntimeExecutionProof);
            Assert.False(result.IsRealModelRuntimeProof);
            Assert.False(result.IsPackageConsumerRuntimeProof);
            Assert.Equal(Path.GetFullPath(enginePath), result.PreflightMetadata.Path);
            Assert.True(result.PreflightMetadata.Exists);
            Assert.Equal(engineBytes.Length, result.PreflightMetadata.LengthBytes);
            Assert.Equal(Convert.ToHexString(SHA256.HashData(engineBytes)).ToLowerInvariant(), result.PreflightMetadata.Sha256);
            Assert.Equal("load-engine-preflight", metadata.GetProperty("Kind").GetString());
            Assert.True(metadata.GetProperty("Exists").GetBoolean());
            Assert.Equal(engineBytes.Length, metadata.GetProperty("LengthBytes").GetInt64());
            Assert.Equal(result.PreflightMetadata.Sha256, metadata.GetProperty("Sha256").GetString());
            Assert.Equal("dependency-probe-only", metadata.GetProperty("ProofClassification").GetString());
            Assert.Contains("does not deserialize, bind tensors, enqueue inference", metadata.GetProperty("EvidenceBoundary").GetString(), StringComparison.Ordinal);
            Assert.Equal(64UL * 1024UL * 1024UL, root.GetProperty("WorkspaceBytes").GetUInt64());
            Assert.Equal("dependency-probe-only", root.GetProperty("ProofClassification").GetString());
            Assert.False(root.GetProperty("InferenceRan").GetBoolean());
            Assert.False(root.GetProperty("IsRuntimeExecutionProof").GetBoolean());
            Assert.False(root.GetProperty("IsPackageConsumerRuntimeProof").GetBoolean());
            Assert.True(loadedEngineDiagnostics.TryGetProperty("DiagnosticsState", out _));
            Assert.True(loadedEngineDiagnostics.TryGetProperty("EvidenceBoundary", out _));
            Assert.True(loadedEngineDiagnostics.TryGetProperty("ReadbackFingerprint", out _));
            Assert.True(loadedEngineDiagnostics.TryGetProperty("ReadbackSha256", out _));
            Assert.Contains("does not create execution bindings", loadedEngineDiagnostics.GetProperty("EvidenceBoundary").GetString(), StringComparison.Ordinal);
            Assert.Contains("LoadEnginePreflight Exists=True", string.Join("\n", result.LogLines), StringComparison.Ordinal);
            Assert.Contains("LoadEngine=ReadonlyDiagnostics", string.Join("\n", result.LogLines), StringComparison.Ordinal);
            Assert.Contains(root.GetProperty("OptionImplementationStatus").GetProperty("ParsedOptions").EnumerateArray(), static item => item.GetString() == "--loadEngine");
            Assert.Contains(root.GetProperty("OptionImplementationStatus").GetProperty("AppliedOptions").EnumerateArray(), static item => item.GetString() == "--loadEngine");
        }
        finally
        {
            if (File.Exists(enginePath))
            {
                File.Delete(enginePath);
            }

            if (File.Exists(reportPath))
            {
                File.Delete(reportPath);
            }
        }
    }

    [Fact]
    public void RuntimeArtifactWriterIncludesLoadEnginePreflightMetadataWithoutPromotingProof()
    {
        string artifactRoot = Path.Combine(Path.GetTempPath(), "jyppx-runtime-artifacts-" + Guid.NewGuid().ToString("N"));
        string enginePath = Path.Combine(artifactRoot, "existing.plan");
        byte[] engineBytes = new byte[] { 0x54, 0x52, 0x54, 0x45, 0x58, 0x45, 0x43, 0x02 };
        try
        {
            Directory.CreateDirectory(artifactRoot);
            File.WriteAllBytes(enginePath, engineBytes);
            TrtexecLikeRuntimeOptions runtimeOptions = new TrtexecLikeRuntimeOptions(
                noDataTransfers: false,
                useSpinWait: false,
                threads: null,
                avgRuns: null,
                percentile: null,
                sleepTimeMilliseconds: null,
                idleTimeMilliseconds: null,
                infStreams: null,
                loadInputs: string.Empty,
                dumpOutput: false,
                dumpRawBindingsToFile: Path.Combine(artifactRoot, "bindings.raw"),
                exportOutputPath: Path.Combine(artifactRoot, "output.json"),
                exportTimesPath: Path.Combine(artifactRoot, "times.json"),
                exportProfilePath: Path.Combine(artifactRoot, "profile.json"),
                saveProfilePath: Path.Combine(artifactRoot, "profile.txt"));
            OnnxEnginePreflightMetadata preflightMetadata = OnnxEnginePreflightMetadata.FromExistingEngine(enginePath);
            OnnxLoadedEngineDiagnostics loadedEngineDiagnostics = new OnnxLoadedEngineDiagnostics(
                attempted: true,
                succeeded: true,
                diagnosticsState: "readonly-deserialize-succeeded",
                failureReason: string.Empty,
                engineName: "readonly-engine",
                ioTensorCount: 2,
                layerCount: 3,
                optimizationProfileCount: 1,
                deviceMemorySizeInBytes: 4096,
                auxiliaryStreamCount: 1,
                capability: "Standard",
                profilingVerbosity: "Detailed",
                inspectorInformationLength: 128,
                ioTensorSummaries: new[] { "0:images:Input:Float:1x3x640x640", "1:output:Output:Float:1x84x8400" },
                readbackFingerprint: "engine=readonly-engine|io=2|layers=3|profiles=1",
                readbackSha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                evidenceBoundary: "readonly diagnostics only");
            OnnxEngineBuildResult result = new OnnxEngineBuildResult(
                success: true,
                skipped: false,
                state: "load-engine-preflight",
                tensorRtLine: JYPPX.Shared.Interop.TensorRtApiLine.TensorRt10,
                modelSource: Path.GetFullPath(enginePath),
                enginePath: Path.GetFullPath(enginePath),
                parsed: false,
                engineSaved: false,
                engineFileRoundTrip: false,
                inferenceRan: false,
                outputMatch: false,
                profileIndex: 0,
                elapsedMilliseconds: null,
                skipReason: string.Empty,
                normalizedCommandLine: "--loadEngine " + enginePath + " --exportTimes times.json",
                diagnostics: Array.Empty<string>(),
                logLines: new[] { "LoadEnginePreflight Exists=True" },
                runtimeOptions: runtimeOptions,
                preflightMetadata: preflightMetadata,
                loadedEngineDiagnostics: loadedEngineDiagnostics);

            OnnxEngineRuntimeArtifactWriter.WriteArtifacts(result);

            string times = File.ReadAllText(Path.Combine(artifactRoot, "times.json"));
            string output = File.ReadAllText(Path.Combine(artifactRoot, "output.json"));
            string engineReadback = File.ReadAllText(Path.Combine(artifactRoot, "profile.engine-readback.json"));
            string rawBoundary = File.ReadAllText(Path.Combine(artifactRoot, "bindings.raw"));
            string profileText = File.ReadAllText(Path.Combine(artifactRoot, "profile.txt"));
            using JsonDocument timesDocument = JsonDocument.Parse(times);
            using JsonDocument outputDocument = JsonDocument.Parse(output);
            using JsonDocument engineReadbackDocument = JsonDocument.Parse(engineReadback);
            using JsonDocument rawBoundaryDocument = JsonDocument.Parse(rawBoundary);
            JsonElement timesRoot = timesDocument.RootElement;
            JsonElement metadata = timesRoot.GetProperty("PreflightMetadata");
            JsonElement engineReadbackRoot = engineReadbackDocument.RootElement;
            JsonElement readbackDiagnostics = engineReadbackRoot.GetProperty("LoadedEngineDiagnostics");

            Assert.Equal("dependency-probe-only", timesRoot.GetProperty("RuntimeProofClass").GetString());
            Assert.True(timesRoot.GetProperty("IsBuildOnlyEvidence").GetBoolean());
            Assert.True(timesRoot.GetProperty("IsDependencyProbeOnly").GetBoolean());
            Assert.False(timesRoot.GetProperty("HasTensorOutputProof").GetBoolean());
            Assert.False(timesRoot.GetProperty("HasRawBindingProof").GetBoolean());
            Assert.Equal("load-engine-preflight", metadata.GetProperty("Kind").GetString());
            Assert.True(metadata.GetProperty("Exists").GetBoolean());
            Assert.Equal(engineBytes.Length, metadata.GetProperty("LengthBytes").GetInt64());
            Assert.Equal(preflightMetadata.Sha256, metadata.GetProperty("Sha256").GetString());
            Assert.Contains("does not deserialize, bind tensors, enqueue inference", metadata.GetProperty("EvidenceBoundary").GetString(), StringComparison.Ordinal);
            Assert.Equal("dependency-probe-only", outputDocument.RootElement.GetProperty("RuntimeProofClass").GetString());
            Assert.False(outputDocument.RootElement.GetProperty("HasTensorOutputProof").GetBoolean());
            Assert.Equal("trtexec-like-engine-readback", engineReadbackRoot.GetProperty("ArtifactKind").GetString());
            Assert.Equal("dependency-probe-only", engineReadbackRoot.GetProperty("RuntimeProofClass").GetString());
            Assert.True(engineReadbackRoot.GetProperty("ReadbackAvailable").GetBoolean());
            Assert.Equal(loadedEngineDiagnostics.ReadbackFingerprint, engineReadbackRoot.GetProperty("ReadbackFingerprint").GetString());
            Assert.Equal(loadedEngineDiagnostics.ReadbackSha256, engineReadbackRoot.GetProperty("ReadbackSha256").GetString());
            Assert.False(engineReadbackRoot.GetProperty("IsRuntimeExecutionProof").GetBoolean());
            Assert.False(engineReadbackRoot.GetProperty("IsRealModelRuntimeProof").GetBoolean());
            Assert.False(engineReadbackRoot.GetProperty("IsPackageConsumerRuntimeProof").GetBoolean());
            Assert.Equal("readonly-engine", readbackDiagnostics.GetProperty("EngineName").GetString());
            Assert.Equal(2, readbackDiagnostics.GetProperty("IOTensorCount").GetInt32());
            Assert.Equal(128, readbackDiagnostics.GetProperty("InspectorInformationLength").GetInt32());
            Assert.Contains("does not bind tensors, enqueue inference, validate outputs", engineReadbackRoot.GetProperty("Note").GetString(), StringComparison.Ordinal);
            Assert.Equal("dependency-probe-only", rawBoundaryDocument.RootElement.GetProperty("RuntimeProofClass").GetString());
            Assert.False(rawBoundaryDocument.RootElement.GetProperty("HasRawBindingProof").GetBoolean());
            Assert.Contains("PreflightKind: load-engine-preflight", profileText, StringComparison.Ordinal);
            Assert.Contains("PreflightEvidenceBoundary: load-engine preflight records file metadata only", profileText, StringComparison.Ordinal);
            Assert.Contains("LoadedEngineDiagnostics", OnnxEngineBuildDiagnostics.ToJson(result), StringComparison.Ordinal);
            Assert.Contains("ReadbackFingerprint", OnnxEngineBuildDiagnostics.ToJson(result), StringComparison.Ordinal);
            Assert.Contains("ReadbackSha256", OnnxEngineBuildDiagnostics.ToJson(result), StringComparison.Ordinal);
            Assert.Contains("WorkspaceBytes", OnnxEngineBuildDiagnostics.ToJson(result), StringComparison.Ordinal);
            Assert.Contains("BuilderConfigDeploymentSnapshot", OnnxEngineBuildDiagnostics.ToJson(result), StringComparison.Ordinal);
        }
        finally
        {
            if (Directory.Exists(artifactRoot))
            {
                Directory.Delete(artifactRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void BuildReportCarriesCopiedBuilderConfigReadbackWithoutPromotingProof()
    {
        OnnxEngineBuildResult result = new OnnxEngineBuildResult(
            success: true,
            skipped: false,
            state: "build-only",
            tensorRtLine: JYPPX.Shared.Interop.TensorRtApiLine.TensorRt10,
            modelSource: "embedded-dynamic-identity",
            enginePath: "model.plan",
            parsed: true,
            engineSaved: true,
            engineFileRoundTrip: false,
            inferenceRan: false,
            outputMatch: false,
            profileIndex: 0,
            elapsedMilliseconds: null,
            skipReason: string.Empty,
            normalizedCommandLine: "--buildOnly --avgTiming 2",
            deploymentOptions: TrtexecLikeDeploymentOptions.Default,
            diagnostics: Array.Empty<string>(),
            logLines: new[] { "BuilderConfigDeploymentSnapshot State=copied-readback" });

        string json = OnnxEngineBuildDiagnostics.ToJson(result);
        string markdown = OnnxEngineBuildDiagnostics.ToMarkdown(result);

        using JsonDocument document = JsonDocument.Parse(json);
        Assert.Equal(JsonValueKind.Null, document.RootElement.GetProperty("BuilderConfigDeploymentSnapshot").ValueKind);
        Assert.Contains("Builder config deployment snapshot: `unavailable`", markdown, StringComparison.Ordinal);
        Assert.False(result.IsRuntimeExecutionProof);
        Assert.False(result.IsPackageConsumerRuntimeProof);

        string service = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "OnnxEngineBuildService.cs"));
        Assert.Contains("config.GetDeploymentSnapshot()", service, StringComparison.Ordinal);
        Assert.Contains("BuilderConfigDeploymentSnapshot State=copied-readback", service, StringComparison.Ordinal);
        Assert.Contains("builderConfigDeploymentSnapshot: builderConfigDeploymentSnapshot", service, StringComparison.Ordinal);
    }

    [Fact]
    public void EvidenceSidecarFeedsModelEvidenceAndSummariesWithoutPromotingPackageConsumerProof()
    {
        string sidecarPath = Path.Combine(Path.GetTempPath(), $"jyppx-evidence-{Guid.NewGuid():N}.json");
        File.WriteAllText(sidecarPath, """
{
  "proofClassification": "package-consumer-runtime",
  "stdoutSummary": "Build report stdout summary",
  "stderrSummary": "Build report stderr summary",
  "modelEvidence": {
    "modelSource": "models/model.onnx",
    "modelSha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "modelLicense": "Apache-2.0",
    "inputAssetName": "image.jpg",
    "inputAssetSha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
  }
}
""");
        try
        {
            OnnxEngineBuildEvidenceSidecar sidecar = OnnxEngineBuildEvidenceSidecarReader.Read(sidecarPath);
            OnnxEngineBuildResult result = new OnnxEngineBuildResult(
                success: true,
                skipped: false,
                state: "external-onnx-build-only",
                tensorRtLine: JYPPX.Shared.Interop.TensorRtApiLine.TensorRt10,
                modelSource: "fallback.onnx",
                enginePath: "model.plan",
                parsed: true,
                engineSaved: true,
                engineFileRoundTrip: false,
                inferenceRan: false,
                outputMatch: false,
                profileIndex: 0,
                elapsedMilliseconds: null,
                skipReason: string.Empty,
                normalizedCommandLine: "--buildOnly --evidenceSidecar " + sidecarPath,
                diagnostics: Array.Empty<string>(),
                logLines: new[] { "log summary" },
                evidenceSidecar: sidecar);

            string json = OnnxEngineBuildDiagnostics.ToJson(result);
            using JsonDocument document = JsonDocument.Parse(json);
            JsonElement root = document.RootElement;

            Assert.Equal("build-only", result.ProofClassification);
            Assert.False(result.IsPackageConsumerRuntimeProof);
            Assert.True(result.BuildEvidenceOnly);
            Assert.Equal("Build report stdout summary", result.StdoutSummary);
            Assert.Equal("Build report stderr summary", result.StderrSummary);
            Assert.Equal("models/model.onnx", result.ModelEvidence.ModelSource);
            Assert.Equal("Apache-2.0", result.ModelEvidence.ModelLicense);
            Assert.Equal("package-consumer-runtime", root.GetProperty("EvidenceSidecarProofClassification").GetString());
            Assert.Contains("package-consumer-runtime is ignored", string.Join("\n", result.EvidenceSidecar.Diagnostics), StringComparison.Ordinal);
        }
        finally
        {
            File.Delete(sidecarPath);
        }
    }

    [Fact]
    public void RealModelRuntimeSidecarRemainsBuildReportEvidenceUntilSampleRunnerPromotesIt()
    {
        OnnxEngineBuildEvidenceSidecar sidecar = new OnnxEngineBuildEvidenceSidecar(
            "models/model.sidecar.json",
            "models/model.onnx",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "BSD-3-Clause",
            "image.jpg",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "YoloVision Passed=True",
            string.Empty,
            "real-model-runtime",
            Array.Empty<string>());

        OnnxEngineBuildResult result = new OnnxEngineBuildResult(
            success: true,
            skipped: false,
            state: "external-onnx-build-only",
            tensorRtLine: JYPPX.Shared.Interop.TensorRtApiLine.TensorRt10,
            modelSource: "models/model.onnx",
            enginePath: "model.plan",
            parsed: true,
            engineSaved: true,
            engineFileRoundTrip: false,
            inferenceRan: false,
            outputMatch: false,
            profileIndex: 0,
            elapsedMilliseconds: null,
            skipReason: string.Empty,
            normalizedCommandLine: "--buildOnly",
            diagnostics: Array.Empty<string>(),
            logLines: new[] { "Parsed=True" },
            evidenceSidecar: sidecar);

        Assert.Equal("build-only", result.ProofClassification);
        Assert.False(result.IsRealModelRuntimeProof);
        Assert.Equal("YoloVision Passed=True", result.StdoutSummary);
        Assert.Equal("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", result.ModelEvidence.ModelSha256);
    }

    [Fact]
    public void BuildResultClassifiesSkippedPreflightAndSyntheticRuntimeEvidence()
    {
        OnnxEngineBuildResult skipped = CreateResult(
            skipped: true,
            state: "missing-runtime",
            inferenceRan: false,
            outputMatch: false);
        OnnxEngineBuildResult preflight = CreateResult(
            skipped: false,
            state: "load-engine-preflight",
            inferenceRan: false,
            outputMatch: false);
        OnnxEngineBuildResult syntheticRuntime = CreateResult(
            skipped: false,
            state: "identity-roundtrip",
            inferenceRan: true,
            outputMatch: true);

        Assert.Equal("dependency-probe-only", skipped.ProofClassification);
        Assert.True(skipped.BuildEvidenceOnly);
        Assert.Equal("dependency-probe-only", preflight.ProofClassification);
        Assert.True(preflight.BuildEvidenceOnly);
        Assert.Equal("synthetic-input-runtime", syntheticRuntime.ProofClassification);
        Assert.False(syntheticRuntime.BuildEvidenceOnly);
        Assert.False(syntheticRuntime.IsRealModelRuntimeProof);
        Assert.False(syntheticRuntime.IsPackageConsumerRuntimeProof);
        Assert.Contains(syntheticRuntime.EvidenceClassifications, static item => item == "real-model-runtime");
        Assert.Contains(syntheticRuntime.EvidenceClassifications, static item => item == "package-consumer-runtime");
    }

    [Fact]
    public void OnnxEngineBuildServiceContainsBoundedLoadEngineRuntimeWithoutChangingProofBoundary()
    {
        string service = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "OnnxEngineBuildService.cs"));
        string artifactWriter = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "OnnxEngineRuntimeArtifactWriter.cs"));

        Assert.Contains("TryRunGenericFloatEngineFromFile", service, StringComparison.Ordinal);
        Assert.Contains("load-engine-identity-runtime", service, StringComparison.Ordinal);
        Assert.Contains("load-engine-runtime-output-unverified", service, StringComparison.Ordinal);
        Assert.Contains("external-onnx-runtime-output-unverified", service, StringComparison.Ordinal);
        Assert.Contains("Generic bounded runtime supports exactly one input tensor", service, StringComparison.Ordinal);
        Assert.Contains("Generic bounded runtime supports float input tensors only", service, StringComparison.Ordinal);
        Assert.Contains("--loadInputs must include a mapping", service, StringComparison.Ordinal);
        Assert.Contains("runtime-output-captured-unverified", artifactWriter, StringComparison.Ordinal);
        Assert.Contains("not runtime proof, real-model proof, or package-consumer proof", artifactWriter, StringComparison.Ordinal);
    }

    [Fact]
    public void TrtexecParserReportsMissingExternalOnnxClearly()
    {
        FileNotFoundException exception = Assert.Throws<FileNotFoundException>(() =>
            TrtexecLikeParser.Parse(new[] { "--onnx", Path.Combine(Path.GetTempPath(), "missing-model.onnx") }));

        Assert.Contains("ONNX model file was not found", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void OnnxToEngineProgramDelegatesBuildWorkToReusableService()
    {
        string program = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "OnnxToEngine", "Program.cs"));
        string toolsProject = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "JYPPX.TensorRtSharp.Tools.csproj"));
        string service = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "OnnxEngineBuildService.cs"));
        string diagnostics = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "src", "JYPPX.TensorRtSharp.Tools", "OnnxEngineBuildDiagnostics.cs"));
        string project = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "OnnxToEngine", "OnnxToEngine.csproj"));

        Assert.Contains("new OnnxEngineBuildService().Execute", program, StringComparison.Ordinal);
        Assert.Contains("JYPPX.CudaSharp.csproj", toolsProject, StringComparison.Ordinal);
        Assert.Contains("JYPPX.TensorRtSharp.csproj", toolsProject, StringComparison.Ordinal);
        Assert.Contains("public sealed class OnnxEngineBuildService", service, StringComparison.Ordinal);
        Assert.Contains("BuildSerializedNetwork", service, StringComparison.Ordinal);
        Assert.Contains("OnnxEngineBuildDiagnostics.WriteReport", service, StringComparison.Ordinal);
        Assert.Contains("IsRuntimeExecutionProof", diagnostics, StringComparison.Ordinal);
        Assert.Contains("ProofClassification", diagnostics, StringComparison.Ordinal);
        Assert.Contains("NormalizedCommandSha256", diagnostics, StringComparison.Ordinal);
        Assert.Contains("OnnxToEngine ReportPath=", program, StringComparison.Ordinal);
        Assert.Contains("OnnxToEngine ProofClassification=", program, StringComparison.Ordinal);
        Assert.Contains("OnnxToEngine NormalizedCommandSha256=", program, StringComparison.Ordinal);
        Assert.Contains("OnnxToEngine WorkspaceBytes=", program, StringComparison.Ordinal);
        Assert.Contains("OnnxToEngine State=", program, StringComparison.Ordinal);
        Assert.Contains("precheck", diagnostics, StringComparison.Ordinal);
        Assert.Contains("IsRealModelRuntimeProof", diagnostics, StringComparison.Ordinal);
        Assert.Contains("IsPackageConsumerRuntimeProof", diagnostics, StringComparison.Ordinal);
        Assert.Contains("ModelEvidence", diagnostics, StringComparison.Ordinal);
        Assert.Contains("EvidenceSidecarPath", diagnostics, StringComparison.Ordinal);
        Assert.Contains("EvidenceSidecarDiagnostics", diagnostics, StringComparison.Ordinal);
        Assert.Contains("--save-engine <path>", program, StringComparison.Ordinal);
        Assert.Contains("--load-engine <path>", program, StringComparison.Ordinal);
        Assert.Contains("--timingCache <path>", program, StringComparison.Ordinal);
        Assert.Contains("--verbose", program, StringComparison.Ordinal);
        Assert.Contains("Memory values accept MiB by default", program, StringComparison.Ordinal);
        Assert.Contains("--iterations <n> --warmUp <ms> --duration <sec> --streams <n> --useCudaGraph", program, StringComparison.Ordinal);
        Assert.Contains("--exportReport|--report <path.json|path.md>", program, StringComparison.Ordinal);
        Assert.Contains("src\\JYPPX.TensorRtSharp.Tools\\JYPPX.TensorRtSharp.Tools.csproj", project.Replace("/", "\\"), StringComparison.Ordinal);
    }

    [Fact]
    public void TensorRtExecAndYoloExternalAssetArticlesAreIndexed()
    {
        string index = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string toc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string roadmap = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "technical-article-roadmap.md"));
        string yoloExample = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "assets", "yolovision-yolox-s-example.json"));
        string candidatePlan = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "interface-coverage", "deferred-readonly-api-manual-review-candidates.md"));
        string trtexecCoverage = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "user-acceptance", "trtexec-option-coverage.md"));

        Assert.Contains("yolovision-real-asset-walkthrough.md", index, StringComparison.Ordinal);
        Assert.Contains("yolovision-multi-output-metadata-guide.md", index, StringComparison.Ordinal);
        Assert.Contains("onnx-to-engine-trtexec-conversion-guide.md", index, StringComparison.Ordinal);
        Assert.Contains("tensorrtexec-gui-user-guide.md", index, StringComparison.Ordinal);
        Assert.Contains("tensorrtexec-external-onnx-build-report.md", toc, StringComparison.Ordinal);
        Assert.Contains("yolovision-multi-output-metadata-guide.md", toc, StringComparison.Ordinal);
        Assert.Contains("onnx-to-engine-trtexec-conversion-guide.md", toc, StringComparison.Ordinal);
        Assert.Contains("tensorrtexec-gui-user-guide.md", toc, StringComparison.Ordinal);
        Assert.Contains("src/JYPPX.TensorRtSharp.Tools", roadmap, StringComparison.Ordinal);
        Assert.Contains("YoloVision 多输出 Metadata 指南", roadmap, StringComparison.Ordinal);
        Assert.Contains("ONNX 到 TensorRT Engine 转换指南", roadmap, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec GUI 使用教程", roadmap, StringComparison.Ordinal);
        Assert.Contains("YoloRuntimeOutputTensor", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "yolovision-multi-output-metadata-guide.md")), StringComparison.Ordinal);
        Assert.Contains("BuildEvidenceOnly", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "onnx-to-engine-trtexec-conversion-guide.md")), StringComparison.Ordinal);
        Assert.Contains("NormalizedCommandLine", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "tensorrtexec-gui-user-guide.md")), StringComparison.Ordinal);
        Assert.Contains("\"isSmokePassed\": false", yoloExample, StringComparison.Ordinal);
        Assert.Contains("\"proofClassification\": \"build-only\"", yoloExample, StringComparison.Ordinal);
        Assert.Contains("\"stdoutSummary\"", yoloExample, StringComparison.Ordinal);
        Assert.Contains("YOLOX", yoloExample, StringComparison.Ordinal);
        Assert.Contains("Manual Review Candidates", candidatePlan, StringComparison.Ordinal);
        Assert.Contains("Do not treat this table as completion evidence", candidatePlan, StringComparison.Ordinal);
        Assert.Contains("trtexec-like Option Coverage", trtexecCoverage, StringComparison.Ordinal);
        Assert.Contains("release readiness 在缺少真实外部 runtime proof 前仍应失败", trtexecCoverage, StringComparison.Ordinal);
        Assert.Contains("ArtifactProofBoundary", trtexecCoverage, StringComparison.Ordinal);
        Assert.Contains("RuntimeProofClass", trtexecCoverage, StringComparison.Ordinal);
        Assert.Contains("HasTensorOutputProof=false", trtexecCoverage, StringComparison.Ordinal);
        Assert.Contains("HasRawBindingProof=false", trtexecCoverage, StringComparison.Ordinal);
        Assert.Contains("synthetic-input-runtime is not real-model-runtime", trtexecCoverage, StringComparison.Ordinal);
        Assert.Contains("PreflightMetadata", trtexecCoverage, StringComparison.Ordinal);
        Assert.Contains("ArtifactProofBoundary", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "OnnxToEngine", "README.md")), StringComparison.Ordinal);
        Assert.Contains("HasTensorOutputProof", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "OnnxToEngine", "README.md")), StringComparison.Ordinal);
        Assert.Contains("HasRawBindingProof=false", File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "OnnxToEngine", "README.md")), StringComparison.Ordinal);
        Assert.Contains("--save-engine", trtexecCoverage, StringComparison.Ordinal);
        Assert.Contains("--timingCache", trtexecCoverage, StringComparison.Ordinal);
        Assert.Contains("--useCudaGraph", trtexecCoverage, StringComparison.Ordinal);
    }

    private static OnnxEngineBuildResult CreateResult(
        bool skipped,
        string state,
        bool inferenceRan,
        bool outputMatch)
    {
        return new OnnxEngineBuildResult(
            success: !skipped,
            skipped: skipped,
            state: state,
            tensorRtLine: JYPPX.Shared.Interop.TensorRtApiLine.TensorRt10,
            modelSource: "embedded-dynamic-identity",
            enginePath: "model.plan",
            parsed: !skipped,
            engineSaved: !skipped,
            engineFileRoundTrip: false,
            inferenceRan: inferenceRan,
            outputMatch: outputMatch,
            profileIndex: 0,
            elapsedMilliseconds: null,
            skipReason: skipped ? "runtime missing" : string.Empty,
            normalizedCommandLine: "--tensor-rt-line 10",
            diagnostics: Array.Empty<string>(),
            logLines: new[] { $"State={state}" });
    }
}

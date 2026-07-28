using System.Security.Cryptography;
using System.Text.Json;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Tools;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TensorRtExecMultiInputReferenceContractTests
{
    [Fact]
    public void ParserNormalizesStructuredReferenceValidationOptions()
    {
        TrtexecLikeOptions parsed = TrtexecLikeParser.Parse(new[]
        {
            "--loadInputs", "left:left.bin,right:right.bin",
            "--referenceOutputs", "sum:sum.reference.json,difference:difference.reference.json",
            "--referenceAbsTolerance", "0.0001",
            "--referenceRelTolerance", "0.001",
            "--referenceNaNPolicy", "equal",
            "--referenceInfinityPolicy", "reject",
            "--exportOutput", Path.Combine(Path.GetTempPath(), "multi-reference-output.json")
        });

        TrtexecLikeRuntimeOptions runtime = parsed.RuntimeOptions;
        Assert.True(runtime.RequestsReferenceValidation);
        Assert.True(runtime.RequestsOutputCapture);
        Assert.Equal(0.0001f, runtime.ReferenceAbsoluteTolerance);
        Assert.Equal(0.001f, runtime.ReferenceRelativeTolerance);
        Assert.Equal(TrtexecLikeReferenceNaNPolicy.Equal, runtime.ReferenceNaNPolicy);
        Assert.Equal(TrtexecLikeReferenceInfinityPolicy.Reject, runtime.ReferenceInfinityPolicy);
        Assert.Contains("--referenceOutputs", parsed.ToArgumentLine(), StringComparison.Ordinal);
        Assert.Contains("--referenceNaNPolicy equal", parsed.ToArgumentLine(), StringComparison.Ordinal);

        Assert.Throws<ArgumentException>(() => TrtexecLikeParser.Parse(new[] { "--referenceAbsTolerance", "-1" }));
        Assert.Throws<ArgumentException>(() => TrtexecLikeParser.Parse(new[] { "--referenceNaNPolicy", "ignore" }));
        Assert.Throws<ArgumentException>(() => TrtexecLikeParser.Parse(new[] { "--referenceInfinityPolicy", "equal" }));
    }

    [Fact]
    public void ReferenceTensorDtoCopiesCollectionsAndRoundTripsJson()
    {
        int[] shape = { 1, 4 };
        float[] values = { float.NaN, float.PositiveInfinity, -2.0f, 3.0f };
        OnnxEngineReferenceTensorData reference = new OnnxEngineReferenceTensorData(
            1,
            "sum",
            shape,
            values,
            "synthetic-generated");
        shape[1] = 99;
        values[2] = 99.0f;

        JsonSerializerOptions jsonOptions = new JsonSerializerOptions
        {
            NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals
        };
        string json = JsonSerializer.Serialize(reference, jsonOptions);
        OnnxEngineReferenceTensorData copy = JsonSerializer.Deserialize<OnnxEngineReferenceTensorData>(json, jsonOptions)!;

        Assert.Equal(new[] { 1, 4 }, reference.Shape);
        Assert.Equal(-2.0f, reference.Values[2]);
        Assert.Equal("sum", copy.TensorName);
        Assert.True(float.IsNaN(copy.Values[0]));
        Assert.True(float.IsPositiveInfinity(copy.Values[1]));
        Assert.Equal("synthetic-generated", copy.SourceClassification);
    }

    [Fact]
    public void ReferenceValueComparerAppliesToleranceAndSpecialValuePoliciesExplicitly()
    {
        Assert.True(Matches(10.001f, 10.0f, 0.01f, 0.0f, TrtexecLikeReferenceNaNPolicy.Reject, TrtexecLikeReferenceInfinityPolicy.Exact));
        Assert.True(Matches(100.1f, 100.0f, 0.0f, 0.002f, TrtexecLikeReferenceNaNPolicy.Reject, TrtexecLikeReferenceInfinityPolicy.Exact));
        Assert.False(Matches(10.1f, 10.0f, 0.01f, 0.001f, TrtexecLikeReferenceNaNPolicy.Reject, TrtexecLikeReferenceInfinityPolicy.Exact));
        Assert.False(Matches(float.NaN, float.NaN, 0.0f, 0.0f, TrtexecLikeReferenceNaNPolicy.Reject, TrtexecLikeReferenceInfinityPolicy.Exact));
        Assert.True(Matches(float.NaN, float.NaN, 0.0f, 0.0f, TrtexecLikeReferenceNaNPolicy.Equal, TrtexecLikeReferenceInfinityPolicy.Exact));
        Assert.True(Matches(float.PositiveInfinity, float.PositiveInfinity, 0.0f, 0.0f, TrtexecLikeReferenceNaNPolicy.Reject, TrtexecLikeReferenceInfinityPolicy.Exact));
        Assert.False(Matches(float.PositiveInfinity, float.NegativeInfinity, 0.0f, 0.0f, TrtexecLikeReferenceNaNPolicy.Reject, TrtexecLikeReferenceInfinityPolicy.Exact));
        Assert.False(Matches(float.PositiveInfinity, float.PositiveInfinity, 0.0f, 0.0f, TrtexecLikeReferenceNaNPolicy.Reject, TrtexecLikeReferenceInfinityPolicy.Reject));
        Assert.Throws<ArgumentOutOfRangeException>(() => OnnxEngineReferenceValueComparer.Matches(
            1.0f,
            1.0f,
            -1.0f,
            0.0f,
            TrtexecLikeReferenceNaNPolicy.Reject,
            TrtexecLikeReferenceInfinityPolicy.Exact,
            out _,
            out _));
    }

    [Fact]
    public void ReferenceTensorMetadataComparerReportsNameShapeAndCountMismatches()
    {
        OnnxEngineReferenceTensorData reference = new OnnxEngineReferenceTensorData(
            1,
            "scores",
            new[] { 1, 3 },
            new[] { 0.1f, 0.2f, 0.3f },
            "owner-reviewed-real-model");

        Assert.True(OnnxEngineReferenceTensorComparer.MetadataMatches("scores", new[] { 1, 3 }, 3, reference, out string matched));
        Assert.Empty(matched);
        Assert.False(OnnxEngineReferenceTensorComparer.MetadataMatches("boxes", new[] { 1, 3 }, 3, reference, out string nameMismatch));
        Assert.Contains("tensorName", nameMismatch, StringComparison.Ordinal);
        Assert.False(OnnxEngineReferenceTensorComparer.MetadataMatches("scores", new[] { 3, 1 }, 3, reference, out string shapeMismatch));
        Assert.Contains("shape", shapeMismatch, StringComparison.Ordinal);
        Assert.False(OnnxEngineReferenceTensorComparer.MetadataMatches("scores", new[] { 1, 3 }, 4, reference, out string countMismatch));
        Assert.Contains("count", countMismatch, StringComparison.Ordinal);
    }

    [Fact]
    public void RuntimeArtifactWritesOrderedInputsAndAllOutputReferenceComparisons()
    {
        string root = Path.Combine(Path.GetTempPath(), "jyppx-multi-reference-artifact-" + Guid.NewGuid().ToString("N"));
        try
        {
            string outputPath = Path.Combine(root, "output.json");
            string rawPath = Path.Combine(root, "bindings.raw");
            TrtexecLikeRuntimeOptions runtime = new TrtexecLikeRuntimeOptions(
                noDataTransfers: false,
                useSpinWait: false,
                threads: null,
                avgRuns: null,
                percentile: null,
                sleepTimeMilliseconds: null,
                idleTimeMilliseconds: null,
                infStreams: null,
                loadInputs: "left:left.bin,right:right.bin",
                dumpOutput: true,
                dumpRawBindingsToFile: rawPath,
                exportOutputPath: outputPath,
                exportTimesPath: string.Empty,
                exportProfilePath: string.Empty,
                saveProfilePath: string.Empty,
                referenceOutputs: "sum:sum.reference.json,difference:difference.reference.json",
                referenceAbsoluteTolerance: 1e-5f,
                referenceRelativeTolerance: 1e-4f,
                referenceNaNPolicy: TrtexecLikeReferenceNaNPolicy.Equal,
                referenceInfinityPolicy: TrtexecLikeReferenceInfinityPolicy.Exact);
            OnnxEngineBuildResult result = new OnnxEngineBuildResult(
                success: true,
                skipped: false,
                state: "external-onnx-reference-validated-runtime",
                tensorRtLine: TensorRtApiLine.TensorRt10,
                modelSource: "synthetic-multi-io.onnx",
                enginePath: "synthetic-multi-io.plan",
                parsed: true,
                engineSaved: true,
                engineFileRoundTrip: true,
                inferenceRan: true,
                outputMatch: true,
                profileIndex: 0,
                elapsedMilliseconds: 1.0f,
                skipReason: string.Empty,
                normalizedCommandLine: "--referenceOutputs sum:sum.reference.json,difference:difference.reference.json",
                diagnostics: Array.Empty<string>(),
                logLines: new[] { "ReferenceOutputValidation Requested=True Completed=True Passed=True" },
                runtimeOptions: runtime,
                outputValidated: true,
                identityOutputMatch: false);
            OnnxEngineReferenceValidationArtifact validation = new OnnxEngineReferenceValidationArtifact(
                requested: true,
                completed: true,
                passed: true,
                absoluteTolerance: 1e-5f,
                relativeTolerance: 1e-4f,
                nanPolicy: "equal",
                infinityPolicy: "exact",
                diagnostics: Array.Empty<string>(),
                tensorComparisons: new[]
                {
                    Comparison("sum", Path.Combine(root, "sum.reference.json")),
                    Comparison("difference", Path.Combine(root, "difference.reference.json"))
                });
            OnnxEngineRuntimeArtifactData data = OnnxEngineRuntimeArtifactData.CreateRuntimeEvidence(
                new[]
                {
                    new OnnxEngineRuntimeInputArtifact("left", new[] { 1, 4 }, new[] { 1f, 2f, 3f, 4f }, "load-input-file", Path.Combine(root, "left.bin")),
                    new OnnxEngineRuntimeInputArtifact("right", new[] { 1, 4 }, new[] { 4f, 3f, 2f, 1f }, "load-input-file", Path.Combine(root, "right.bin"))
                },
                new[]
                {
                    new OnnxEngineRuntimeOutputArtifact("sum", new[] { 1, 4 }, new[] { 5f, 5f, 5f, 5f }),
                    new OnnxEngineRuntimeOutputArtifact("difference", new[] { 1, 4 }, new[] { -3f, -1f, 1f, 3f })
                },
                validation,
                "profile=0 bound=4 synchronized=False ready=True",
                new[] { 1.0f });

            OnnxEngineRuntimeArtifactWriter.WriteArtifacts(result, data);

            using JsonDocument output = JsonDocument.Parse(File.ReadAllText(outputPath));
            JsonElement outputRoot = output.RootElement;
            Assert.True(outputRoot.GetProperty("OutputValidated").GetBoolean());
            Assert.Equal(2, outputRoot.GetProperty("InputTensorCount").GetInt32());
            Assert.Equal("left", outputRoot.GetProperty("InputTensors")[0].GetProperty("TensorName").GetString());
            Assert.Equal("right", outputRoot.GetProperty("InputTensors")[1].GetProperty("TensorName").GetString());
            Assert.Equal(8, outputRoot.GetProperty("InputElementCount").GetInt32());
            Assert.Equal(2, outputRoot.GetProperty("ReferenceValidation").GetProperty("TensorComparisons").GetArrayLength());
            Assert.Equal("runtime-reference-output-validated; every captured output matched a traceable structured reference under the recorded tolerances and special-value policies. Proof classification remains bounded by model and consumer evidence.", outputRoot.GetProperty("ArtifactProofBoundary").GetString());

            using JsonDocument report = JsonDocument.Parse(OnnxEngineBuildDiagnostics.ToJson(result));
            Assert.True(report.RootElement.GetProperty("OutputValidated").GetBoolean());
            Assert.False(report.RootElement.GetProperty("IdentityOutputMatch").GetBoolean());
            string[] applied = report.RootElement.GetProperty("OptionImplementationStatus").GetProperty("AppliedOptions")
                .EnumerateArray().Select(static item => item.GetString()!).ToArray();
            Assert.Contains("--referenceOutputs", applied);
            Assert.Contains("--referenceAbsTolerance", applied);

            using JsonDocument rawManifest = JsonDocument.Parse(File.ReadAllText(rawPath + ".manifest.json"));
            Assert.True(rawManifest.RootElement.GetProperty("OutputValidated").GetBoolean());
            Assert.Equal(2, rawManifest.RootElement.GetProperty("ReferenceValidation").GetProperty("TensorComparisons").GetArrayLength());
        }
        finally
        {
            if (Directory.Exists(root))
            {
                Directory.Delete(root, recursive: true);
            }
        }
    }

    [Fact]
    public void GenericRuntimeSourceDoesNotRetainSingleInputRestriction()
    {
        string service = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp.Tools",
            "Build",
            "OnnxEngineBuildService.cs"));

        Assert.DoesNotContain("supports exactly one input tensor", service, StringComparison.Ordinal);
        Assert.Contains("List<OnnxEngineRuntimeInput> runtimeInputs", service, StringComparison.Ordinal);
        Assert.Contains("foreach (OnnxEngineRuntimeInput input in inputs)", service, StringComparison.Ordinal);
        Assert.Contains("unknown input tensors", service, StringComparison.Ordinal);
        Assert.Contains("duplicate mapping", service, StringComparison.Ordinal);
        Assert.Contains("ReferenceOutputValidation Requested=True", service, StringComparison.Ordinal);
    }

    private static OnnxEngineReferenceTensorComparisonArtifact Comparison(string tensorName, string path)
    {
        return new OnnxEngineReferenceTensorComparisonArtifact(
            tensorName,
            path,
            Convert.ToHexString(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(tensorName))).ToLowerInvariant(),
            "synthetic-generated",
            new[] { 1, 4 },
            new[] { 1, 4 },
            4,
            4,
            4,
            0,
            -1,
            0.0f,
            0.0f,
            passed: true,
            diagnostic: "all reference values matched");
    }

    private static bool Matches(
        float actual,
        float expected,
        float absoluteTolerance,
        float relativeTolerance,
        TrtexecLikeReferenceNaNPolicy nanPolicy,
        TrtexecLikeReferenceInfinityPolicy infinityPolicy)
    {
        return OnnxEngineReferenceValueComparer.Matches(
            actual,
            expected,
            absoluteTolerance,
            relativeTolerance,
            nanPolicy,
            infinityPolicy,
            out _,
            out _);
    }
}

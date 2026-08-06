using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TensorRtExecMultiOutputArtifactContractTests
{
    [Fact]
    public void RuntimeOutputArtifactContractLocksMultiOutputShapeHashOffsetAndProofBoundaries()
    {
        string contractPath = Path.Combine(
            RepositoryPaths.Root,
            "applications",
            "TensorRtExec",
            "tensor-rt-exec-runtime-output-artifact-contract.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(contractPath));
        JsonElement root = document.RootElement;

        Assert.Equal("tensor-rt-exec-runtime-output-artifact-contract", root.GetProperty("contractId").GetString());
        Assert.Equal(2, root.GetProperty("formatVersion").GetInt32());
        Assert.Equal("float32", root.GetProperty("supportedDataType").GetString());
        Assert.Equal(3, root.GetProperty("runtimeActivation").GetProperty("outputCaptureOptions").GetArrayLength());
        Assert.Equal("deterministic-generated", root.GetProperty("runtimeActivation").GetProperty("inputWithoutLoadInputs").GetString());
        Assert.Equal(8, root.GetProperty("dumpOutput").GetProperty("maximumPreviewValuesPerTensor").GetInt32());
        Assert.Equal(".manifest.json", root.GetProperty("rawBindings").GetProperty("manifestSuffix").GetString());

        string[] requiredOutputFields = root.GetProperty("outputJson").GetProperty("requiredFields")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        string[] requiredSegmentFields = root.GetProperty("rawBindings").GetProperty("requiredSegmentFields")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        foreach (string field in new[] { "OutputCaptureAvailable", "OutputValidated", "InputTensorCount", "InputTensors", "ReferenceValidation", "OutputTensorCount", "OutputTensors", "HasTensorOutputProof", "HasRawBindingProof" })
        {
            Assert.Contains(field, requiredOutputFields);
        }

        foreach (string field in new[] { "TensorName", "Shape", "ElementCount", "ByteOffset", "ByteLength", "Sha256" })
        {
            Assert.Contains(field, requiredSegmentFields);
        }

        JsonElement invariants = root.GetProperty("proofInvariants");
        Assert.True(invariants.GetProperty("captureDoesNotImplyValidation").GetBoolean());
        Assert.False(invariants.GetProperty("unvalidatedOutputValidated").GetBoolean());
        Assert.False(invariants.GetProperty("unvalidatedHasTensorOutputProof").GetBoolean());
        Assert.False(invariants.GetProperty("unvalidatedHasRawBindingProof").GetBoolean());
        Assert.False(invariants.GetProperty("canPromoteRealModelRuntimeProof").GetBoolean());
        Assert.False(invariants.GetProperty("canPromotePackageConsumerRuntimeProof").GetBoolean());
        Assert.False(invariants.GetProperty("canPromoteReleaseProof").GetBoolean());
    }

    [Fact]
    public void RuntimeOutputArtifactImplementationAndMatricesStayAlignedWithTheContract()
    {
        string writer =
            Read("src", "JYPPX.TensorRtSharp.Tools", "Artifacts", "OnnxEngineRuntimeArtifactWriter.Output.cs") +
            Read("src", "JYPPX.TensorRtSharp.Tools", "Artifacts", "OnnxEngineRuntimeArtifactWriter.RawBindings.cs") +
            Read("src", "JYPPX.TensorRtSharp.Tools", "Artifacts", "OnnxEngineRuntimeArtifactWriter.ProofBoundary.cs");
        string service = Read(
            "src", "JYPPX.TensorRtSharp.Tools", "Build", "OnnxEngineBuildService.RuntimeExecution.cs");
        string runtimeOptions = Read("src", "JYPPX.TensorRtSharp.Tools", "Trtexec", "TrtexecLikeRuntimeOptions.cs");
        string capabilities = Read("src", "JYPPX.TensorRtSharp.Tools", "Trtexec", "TrtexecLikeOptionCapabilities.cs");
        string readme = Read("applications", "TensorRtExec", "README.md");
        string featureMatrix = Read("applications", "TensorRtExec", "tensor-rt-exec-feature-matrix.json");
        string parityMatrix = Read("applications", "TensorRtExec", "tensor-rt-exec-trtexec-parity-matrix.json");
        string fieldMap = Read("applications", "TensorRtExec", "tensor-rt-exec-gui-cli-field-map.json");
        string sampleMatrix = Read("applications", "OnnxToEngine", "trtexec-parity-matrix.json");

        Assert.Contains("CreateRuntimeEvidence", service, StringComparison.Ordinal);
        Assert.Contains("options.RuntimeOptions.RequestsOutputCapture", service, StringComparison.Ordinal);
        Assert.Contains("InputSource={inputSource}", service, StringComparison.Ordinal);
        Assert.Contains("foreach (OnnxEngineRuntimeOutputArtifact output in outputArtifacts)", service, StringComparison.Ordinal);
        Assert.Contains("bounded-output-capture-not-reference-validation", service, StringComparison.Ordinal);
        Assert.Contains("OutputTensors = data.OutputTensors.Select", writer, StringComparison.Ordinal);
        Assert.Contains("WriteJson(path + \".manifest.json\"", writer, StringComparison.Ordinal);
        Assert.Contains("RawBindingSha256", writer, StringComparison.Ordinal);
        Assert.Contains("CanCaptureRawBindings(result, data) && result.OutputMatch", writer, StringComparison.Ordinal);
        Assert.Contains("public bool RequestsOutputCapture", runtimeOptions, StringComparison.Ordinal);
        Assert.Contains("implemented-bounded-multi-output-capture", capabilities, StringComparison.Ordinal);
        Assert.Contains("applied-bounded-output-capture", capabilities, StringComparison.Ordinal);
        Assert.Contains("implemented-structured-reference-validation", capabilities, StringComparison.Ordinal);

        Assert.Contains("OutputCaptureAvailable=true", readme, StringComparison.Ordinal);
        Assert.Contains("OutputValidated=false", readme, StringComparison.Ordinal);
        Assert.Contains("<raw-path>.manifest.json", readme, StringComparison.Ordinal);
        Assert.Contains("implemented-bounded-multi-output-capture", featureMatrix, StringComparison.Ordinal);
        Assert.Contains("implemented-pointer-free-multi-input-binding-multi-output-artifacts-and-reference-validation", parityMatrix, StringComparison.Ordinal);
        Assert.Contains("implemented-bounded-multi-output-log", fieldMap, StringComparison.Ordinal);
        Assert.Contains("implemented-bounded-multi-output-raw-manifest", fieldMap, StringComparison.Ordinal);
        Assert.Contains("implemented-bounded-multi-output-json", fieldMap, StringComparison.Ordinal);
        Assert.Contains("applied-bounded-output-capture", sampleMatrix, StringComparison.Ordinal);

        using JsonDocument gaps = JsonDocument.Parse(Read(
            "applications",
            "TensorRtExec",
            "tensor-rt-exec-release-candidate-gap-list.json"));
        Assert.Equal(
            gaps.RootElement.GetProperty("items").GetArrayLength(),
            gaps.RootElement.GetProperty("summary").GetProperty("totalItems").GetInt32());
    }

    private static string Read(params string[] segments)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(segments).ToArray()));
    }
}

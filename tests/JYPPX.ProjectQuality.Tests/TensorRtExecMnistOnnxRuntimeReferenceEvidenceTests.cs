using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TensorRtExecMnistOnnxRuntimeReferenceEvidenceTests
{
    [Fact]
    public void CompactEvidenceRecordsIndependentCpuExecutionWithoutPromotingOwnerOrReleaseProof()
    {
        using JsonDocument document = ReadJson(
            "artifacts",
            "interface-coverage",
            "tensorrtexec-mnist-onnxruntime-reference-evidence.json");
        JsonElement root = document.RootElement;
        JsonElement runtime = root.GetProperty("runtime");
        JsonElement reference = root.GetProperty("reference");
        JsonElement comparison = root.GetProperty("tensorRtComparison");
        JsonElement owner = root.GetProperty("ownerReview");
        JsonElement boundary = root.GetProperty("proofBoundary");

        Assert.Equal("tensorrtexec-mnist-onnxruntime-reference-evidence.v1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("independent-framework-reference-candidate-runtime", root.GetProperty("evidenceClassification").GetString());
        Assert.Equal("ONNX Runtime", runtime.GetProperty("name").GetString());
        Assert.Equal("1.23.2", runtime.GetProperty("version").GetString());
        Assert.Equal("CPUExecutionProvider", runtime.GetProperty("requestedProvider").GetString());
        Assert.Equal(new[] { "CPUExecutionProvider" }, runtime.GetProperty("profileProviders").EnumerateArray().Select(static item => item.GetString()).ToArray());
        Assert.True(runtime.GetProperty("providerValidated").GetBoolean());
        Assert.Equal(4, runtime.GetProperty("packages").GetArrayLength());

        Assert.Equal("Plus214_Output_0", reference.GetProperty("tensorName").GetString());
        Assert.Equal(new[] { 1, 10 }, reference.GetProperty("shape").EnumerateArray().Select(static item => item.GetInt32()).ToArray());
        Assert.Equal(10, reference.GetProperty("elementCount").GetInt32());
        Assert.Equal("onnxruntime-cpu-1.23.2-derived-unreviewed", reference.GetProperty("sourceClassification").GetString());
        Assert.True(reference.GetProperty("deterministicOutput").GetBoolean());
        Assert.Equal(7, reference.GetProperty("predictedIndex").GetInt32());
        Assert.Equal("1babfa81c0d277a3d483fdc6288285b26008ba5b382868272f1e2187338bd571", reference.GetProperty("sha256").GetString());

        Assert.True(comparison.GetProperty("passed").GetBoolean());
        Assert.Equal(10, comparison.GetProperty("comparedElementCount").GetInt32());
        Assert.Equal(0, comparison.GetProperty("mismatchCount").GetInt32());
        Assert.Equal(-1, comparison.GetProperty("firstMismatchIndex").GetInt32());
        Assert.True(comparison.GetProperty("maximumAbsoluteError").GetSingle() <= comparison.GetProperty("absoluteTolerance").GetSingle());
        Assert.True(comparison.GetProperty("maximumRelativeError").GetSingle() <= comparison.GetProperty("relativeTolerance").GetSingle());

        Assert.Equal("not-provided", owner.GetProperty("status").GetString());
        Assert.False(owner.GetProperty("acceptedAsGoldenReference").GetBoolean());
        Assert.False(owner.GetProperty("acceptedForRepositoryRedistribution").GetBoolean());
        Assert.False(owner.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.True(boundary.GetProperty("independentFromTensorRtExecution").GetBoolean());
        Assert.True(boundary.GetProperty("independentFrameworkReferenceCandidate").GetBoolean());
        Assert.False(boundary.GetProperty("ownerReviewedGolden").GetBoolean());
        Assert.False(boundary.GetProperty("repositoryRedistributionApproved").GetBoolean());
        Assert.False(boundary.GetProperty("publicPackageProof").GetBoolean());
        Assert.False(boundary.GetProperty("postPublishProof").GetBoolean());
        Assert.False(boundary.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(boundary.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    [Fact]
    public void ReferenceSidecarAndStrictValidationArtifactsAreCheckedIn()
    {
        string referencePath = RepositoryPath(
            "artifacts",
            "real-case",
            "onnx-to-engine-mnist-trt10-runtime",
            "digit-7",
            "mnist-trt10-7.onnxruntime-cpu.reference.json");
        string sidecarPath = RepositoryPath(
            "artifacts",
            "real-case",
            "onnx-to-engine-mnist-trt10-runtime",
            "digit-7",
            "mnist-trt10-7.onnxruntime-cpu.reference.sidecar.json");
        using JsonDocument reference = JsonDocument.Parse(File.ReadAllText(referencePath));
        using JsonDocument sidecar = JsonDocument.Parse(File.ReadAllText(sidecarPath));
        using JsonDocument validation = ReadJson(
            "artifacts",
            "interface-coverage",
            "tensorrtexec-mnist-onnxruntime-reference-validation.json");

        Assert.Equal(1, reference.RootElement.GetProperty("schemaVersion").GetInt32());
        Assert.Equal(10, reference.RootElement.GetProperty("values").GetArrayLength());
        Assert.Equal("onnxruntime-cpu-1.23.2-derived-unreviewed", reference.RootElement.GetProperty("sourceClassification").GetString());
        Assert.Equal("1babfa81c0d277a3d483fdc6288285b26008ba5b382868272f1e2187338bd571", ComputeSha256(referencePath));

        Assert.Equal("tensorrtexec-mnist-onnxruntime-reference-sidecar.v1", sidecar.RootElement.GetProperty("schemaVersion").GetString());
        Assert.Equal("CPUExecutionProvider", sidecar.RootElement.GetProperty("onnxRuntime").GetProperty("requestedProvider").GetString());
        Assert.True(sidecar.RootElement.GetProperty("output").GetProperty("deterministic").GetBoolean());
        Assert.Equal("not-provided", sidecar.RootElement.GetProperty("ownerReview").GetProperty("status").GetString());
        Assert.False(sidecar.RootElement.GetProperty("ownerReview").GetProperty("acceptedAsGoldenReference").GetBoolean());

        Assert.Equal("tensorrtexec-mnist-onnxruntime-reference-validation.v1", validation.RootElement.GetProperty("schemaVersion").GetString());
        Assert.True(validation.RootElement.GetProperty("strict").GetBoolean());
        Assert.True(validation.RootElement.GetProperty("runtimeArtifactChecksRequired").GetBoolean());
        Assert.Equal(validation.RootElement.GetProperty("checkCount").GetInt32(), validation.RootElement.GetProperty("passedCount").GetInt32());
        Assert.Equal(0, validation.RootElement.GetProperty("failureCount").GetInt32());
        Assert.True(validation.RootElement.GetProperty("checkCount").GetInt32() >= 51);
    }

    [Fact]
    public void ProducerAndValidatorKeepOfflineRestoreCpuProfilingAndCleanCloneBoundariesExplicit()
    {
        string program = File.ReadAllText(RepositoryPath("samples", "Mnist.OnnxRuntimeReference", "Program.cs"));
        string project = File.ReadAllText(RepositoryPath("samples", "Mnist.OnnxRuntimeReference", "Mnist.OnnxRuntimeReference.csproj.template"));
        string readme = File.ReadAllText(RepositoryPath("samples", "Mnist.OnnxRuntimeReference", "README.md"));
        string runner = File.ReadAllText(RepositoryPath("eng", "Test-TensorRtExecMnistOnnxRuntimeReference.ps1"));
        string validator = File.ReadAllText(RepositoryPath("eng", "Test-TensorRtExecMnistOnnxRuntimeReferenceEvidence.ps1"));
        string solution = File.ReadAllText(RepositoryPath("TensorRtSharp.sln"));

        Assert.Contains("AppendExecutionProvider_CPU", program, StringComparison.Ordinal);
        Assert.Contains("EnableProfiling", program, StringComparison.Ordinal);
        Assert.Contains("CPUExecutionProvider", program, StringComparison.Ordinal);
        Assert.Contains("SequenceEqual(secondRaw)", program, StringComparison.Ordinal);
        Assert.Contains("Microsoft.ML.OnnxRuntime", project, StringComparison.Ordinal);
        Assert.DoesNotContain("Mnist.OnnxRuntimeReference", solution, StringComparison.Ordinal);

        Assert.Contains("Required cached NuGet package is missing; this runner will not download it", runner, StringComparison.Ordinal);
        Assert.Contains("<clear />", runner, StringComparison.Ordinal);
        Assert.Contains("sourcePackagesCopiedToTemporaryEdriveFeed", runner, StringComparison.Ordinal);
        Assert.Contains("workspaceRemovedAfterValidation", runner, StringComparison.Ordinal);
        Assert.Contains("RequireRuntimeArtifacts", validator, StringComparison.Ordinal);
        Assert.Contains("provider-profile-only-cpu", validator, StringComparison.Ordinal);
        Assert.Contains("owner-review-open", validator, StringComparison.Ordinal);
        Assert.Contains("no-release-promotion", validator, StringComparison.Ordinal);
        Assert.Contains("A clean clone retains the reference, sidecar, compact evidence, and validation summary", readme, StringComparison.Ordinal);
    }

    private static JsonDocument ReadJson(params string[] parts)
    {
        string path = RepositoryPath(parts);
        Assert.True(File.Exists(path), "Required JSON file is missing: " + path);
        return JsonDocument.Parse(File.ReadAllText(path));
    }

    private static string RepositoryPath(params string[] parts)
    {
        return Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray());
    }

    private static string ComputeSha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }
}

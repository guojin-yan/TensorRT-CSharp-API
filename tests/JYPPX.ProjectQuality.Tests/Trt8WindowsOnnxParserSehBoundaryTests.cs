using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class Trt8WindowsOnnxParserSehBoundaryTests
{
    [Fact]
    public void CompactEvidenceKeepsSyntheticRuntimeAndReleaseBoundariesExplicit()
    {
        using JsonDocument document = JsonDocument.Parse(ReadSource(
            "artifacts", "interface-coverage", "trt8-windows-onnx-parser-seh-boundary-evidence.json"));
        JsonElement root = document.RootElement;
        JsonElement runtime = root.GetProperty("runtimeProbe");
        JsonElement boundary = root.GetProperty("proofBoundary");

        Assert.Equal("external-onnx-reference-validated-runtime", root.GetProperty("state").GetString());
        Assert.Equal("dedicated-TensorRtExec-child-process", runtime.GetProperty("processIsolation").GetString());
        Assert.True(runtime.GetProperty("parsed").GetBoolean());
        Assert.True(runtime.GetProperty("engineFileRoundTrip").GetBoolean());
        Assert.True(runtime.GetProperty("inferenceRan").GetBoolean());
        Assert.True(runtime.GetProperty("outputValidated").GetBoolean());
        Assert.Equal("synthetic-input-runtime", runtime.GetProperty("proofClassification").GetString());
        Assert.Equal(0, runtime.GetProperty("referenceMismatchCount").GetInt32());
        Assert.True(boundary.GetProperty("isParserCreationSehBoundaryProof").GetBoolean());
        Assert.False(boundary.GetProperty("isModelAccuracyProof").GetBoolean());
        Assert.False(boundary.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("isReleaseProof").GetBoolean());
        Assert.False(boundary.GetProperty("canPublishPublicly").GetBoolean());
    }

    [Fact]
    public void StrictEvidenceValidationPassesEveryCheck()
    {
        using JsonDocument document = JsonDocument.Parse(ReadSource(
            "artifacts", "interface-coverage", "trt8-windows-onnx-parser-seh-boundary-validation.json"));
        JsonElement root = document.RootElement;

        Assert.Equal("passed", root.GetProperty("validationState").GetString());
        Assert.True(root.GetProperty("strict").GetBoolean());
        Assert.Equal(root.GetProperty("checkCount").GetInt32(), root.GetProperty("passedCount").GetInt32());
        Assert.True(root.GetProperty("checkCount").GetInt32() >= 25);
        Assert.Equal(0, root.GetProperty("failureCount").GetInt32());
    }

    private static string ReadSource(params string[] parts)
    {
        string current = AppContext.BaseDirectory;
        while (!string.IsNullOrWhiteSpace(current))
        {
            string candidate = Path.Combine(new[] { current }.Concat(parts).ToArray());
            if (File.Exists(candidate))
            {
                return File.ReadAllText(candidate);
            }

            DirectoryInfo? parent = Directory.GetParent(current);
            if (parent == null)
            {
                break;
            }

            current = parent.FullName;
        }

        throw new FileNotFoundException($"Could not locate source file: {Path.Combine(parts)}");
    }
}

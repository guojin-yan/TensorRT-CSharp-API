using System.Collections;
using System.Reflection;
using System.Text.Json;
using JYPPX.TensorRtSharp;
using Xunit;
using YoloVisionSample;

namespace JYPPX.ProjectQuality.Tests;

public sealed class SampleSupportMultiInputReferenceContractTests
{
    [Fact]
    public void NamedInputParserBuildsIndependentShapesProfilesSourcesAndReferencesWithoutGpu()
    {
        string root = CreateTempDirectory();
        try
        {
            string model = WriteBytes(root, "model.onnx", new byte[] { 1 });
            string left = WriteFloats(root, "left.bin", new[] { 1.0f, 2.0f, 3.0f, 4.0f });
            string right = WriteFloats(root, "right.bin", new[] { 4.0f, 3.0f, 2.0f, 1.0f });
            string sum = WriteBytes(root, "sum.json", new byte[] { (byte)'{', (byte)'}' });
            string difference = WriteBytes(root, "difference.json", new byte[] { (byte)'{', (byte)'}' });
            string shapes = "left:1x4,right:1x4";

            object options = ParseOptions(new[]
            {
                "--model", model,
                "--input-shapes", shapes,
                "--min-shapes", shapes,
                "--opt-shapes", shapes,
                "--max-shapes", shapes,
                "--load-inputs", $"left:{left},right:{right}",
                "--reference-outputs", $"sum:{sum},difference:{difference}",
                "--reference-abs-tolerance", "0.001",
                "--reference-rel-tolerance", "0.01",
                "--reference-nan-policy", "equal",
                "--reference-infinity-policy", "reject"
            });

            Assert.True((bool)Get(options, "UsesNamedInputContract"));
            object[] inputs = ((IEnumerable)Get(options, "Inputs")).Cast<object>().ToArray();
            Assert.Equal(2, inputs.Length);
            Assert.Equal(new[] { "left", "right" }, inputs.Select(input => (string)Get(input, "Name")));
            Assert.All(inputs, input => Assert.Equal(new[] { 1, 4 }, ((TensorRtDims)Get(input, "Shape")).Values));
            Assert.All(inputs, input => Assert.Equal("float-data-file", Get(input, "SourceClassification")));

            object references = Get(options, "ReferenceOutputs");
            Assert.True((bool)Get(references, "Requested"));
            Assert.Equal(0.001f, (float)Get(references, "AbsoluteTolerance"));
            Assert.Equal(0.01f, (float)Get(references, "RelativeTolerance"));
            Assert.Equal("Equal", Get(references, "NaNPolicy").ToString());
            Assert.Equal("Reject", Get(references, "InfinityPolicy").ToString());
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void NamedInputParserFailsClosedForMissingSourcesDuplicateSourcesAndIncompleteProfiles()
    {
        string root = CreateTempDirectory();
        try
        {
            string model = WriteBytes(root, "model.onnx", new byte[] { 1 });
            string left = WriteFloats(root, "left.bin", new[] { 1.0f });
            string right = WriteFloats(root, "right.bin", new[] { 1.0f });
            string shapes = "left:1x1,right:1x1";

            ArgumentException missing = Assert.Throws<ArgumentException>(() => ParseOptions(new[]
            {
                "--model", model,
                "--input-shapes", shapes,
                "--load-inputs", $"left:{left}"
            }));
            Assert.Contains("match --input-shapes exactly", missing.Message, StringComparison.Ordinal);

            ArgumentException duplicate = Assert.Throws<ArgumentException>(() => ParseOptions(new[]
            {
                "--model", model,
                "--input-shapes", shapes,
                "--load-inputs", $"left:{left},right:{right}",
                "--input-patterns", "right:ones"
            }));
            Assert.Contains("more than one value", duplicate.Message, StringComparison.Ordinal);

            ArgumentException incompleteProfile = Assert.Throws<ArgumentException>(() => ParseOptions(new[]
            {
                "--model", model,
                "--input-shapes", shapes,
                "--min-shapes", shapes,
                "--opt-shapes", shapes,
                "--load-inputs", $"left:{left},right:{right}"
            }));
            Assert.Contains("--max-shapes", incompleteProfile.Message, StringComparison.Ordinal);

            Assert.Throws<ArgumentException>(() => ParseOptions(new[]
            {
                "--model", model,
                "--input-shapes", shapes,
                "--input-shape", "1x1",
                "--load-inputs", $"left:{left},right:{right}"
            }));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void ReferenceValuePolicyHandlesToleranceNaNAndInfinityWithoutGpu()
    {
        Assert.True(Matches(10.001f, 10.0f, 0.01f, 0.0f, "reject", "exact", out float absoluteError, out _));
        Assert.InRange(absoluteError, 0.0009f, 0.0011f);
        Assert.True(Matches(100.1f, 100.0f, 0.0f, 0.002f, "reject", "exact", out _, out _));
        Assert.False(Matches(10.1f, 10.0f, 0.01f, 0.001f, "reject", "exact", out _, out _));
        Assert.False(Matches(float.NaN, float.NaN, 0.0f, 0.0f, "reject", "exact", out _, out _));
        Assert.True(Matches(float.NaN, float.NaN, 0.0f, 0.0f, "equal", "exact", out _, out _));
        Assert.True(Matches(float.PositiveInfinity, float.PositiveInfinity, 0.0f, 0.0f, "reject", "exact", out _, out _));
        Assert.False(Matches(float.PositiveInfinity, float.PositiveInfinity, 0.0f, 0.0f, "reject", "reject", out _, out _));
    }

    [Fact]
    public void RuntimeSourceSchemasAndSmokeExposeTheCompleteContract()
    {
        string support = Read("samples", "JYPPX.SampleSupport", "TensorRtOnnxSample.cs");
        string inputs = Read("samples", "JYPPX.SampleSupport", "TensorRtOnnxSample.Inputs.cs");
        string references = Read("samples", "JYPPX.SampleSupport", "TensorRtOnnxSample.References.cs");
        string smoke = Read("smoke", "OnnxToEngineSmokeRunner", "Program.cs");
        string yoloSchema = Read("samples", "YoloVision", "yolovision-output.schema.json");
        string classificationSchema = Read("samples", "Classification", "classification-output.schema.json");
        string referenceSchema = Read("samples", "JYPPX.SampleSupport", "onnx-sample-reference.schema.json");

        Assert.DoesNotContain("This sample supports one float input tensor", support, StringComparison.Ordinal);
        Assert.Contains("foreach (OnnxSampleNetworkInput input in networkInputs)", support, StringComparison.Ordinal);
        Assert.Contains("Named input contract must match model inputs exactly", support, StringComparison.Ordinal);
        Assert.Contains("must have exactly one source", inputs, StringComparison.Ordinal);
        Assert.Contains("--reference-outputs must match captured outputs exactly", references, StringComparison.Ordinal);
        Assert.Contains("FirstMismatchIndex", references, StringComparison.Ordinal);
        Assert.Contains("first actual=", references, StringComparison.Ordinal);
        Assert.Contains("expected=", references, StringComparison.Ordinal);
        Assert.Contains("RunSampleSupportMultiInput", smoke, StringComparison.Ordinal);
        Assert.Contains("SampleSupportMismatchRun", smoke, StringComparison.Ordinal);
        Assert.Contains("\"inputTensors\"", yoloSchema, StringComparison.Ordinal);
        Assert.Contains("\"referenceValidation\"", yoloSchema, StringComparison.Ordinal);
        Assert.Contains("\"runtimeReferenceValidation\"", classificationSchema, StringComparison.Ordinal);
        using JsonDocument schema = JsonDocument.Parse(referenceSchema);
        Assert.Equal(1, schema.RootElement.GetProperty("properties").GetProperty("schemaVersion").GetProperty("const").GetInt32());
        Assert.Contains("sourceClassification", schema.RootElement.GetProperty("required").EnumerateArray().Select(static item => item.GetString()));
    }

    private static object ParseOptions(string[] args)
    {
        Type type = GetSampleSupportType("JYPPX.SampleSupport.OnnxSampleOptions");
        MethodInfo method = type.GetMethod("FromArgs", BindingFlags.Public | BindingFlags.Static)
            ?? throw new InvalidOperationException("OnnxSampleOptions.FromArgs was not found.");
        try
        {
            return method.Invoke(null, new object[] { args, "1x1" })
                ?? throw new InvalidOperationException("OnnxSampleOptions.FromArgs returned null.");
        }
        catch (TargetInvocationException exception) when (exception.InnerException != null)
        {
            throw exception.InnerException;
        }
    }

    private static bool Matches(
        float actual,
        float expected,
        float absoluteTolerance,
        float relativeTolerance,
        string nanPolicy,
        string infinityPolicy,
        out float absoluteError,
        out float relativeError)
    {
        Type type = GetSampleSupportType("JYPPX.SampleSupport.TensorRtOnnxSample");
        MethodInfo method = type.GetMethod("ReferenceValuesMatchForTesting", BindingFlags.Public | BindingFlags.Static)
            ?? throw new InvalidOperationException("ReferenceValuesMatchForTesting was not found.");
        object[] arguments = { actual, expected, absoluteTolerance, relativeTolerance, nanPolicy, infinityPolicy, 0.0f, 0.0f };
        bool result = (bool)(method.Invoke(null, arguments) ?? false);
        absoluteError = (float)arguments[6];
        relativeError = (float)arguments[7];
        return result;
    }

    private static Type GetSampleSupportType(string name)
    {
        return typeof(YoloModelProfile).Assembly.GetType(name, throwOnError: true)!;
    }

    private static object Get(object instance, string propertyName)
    {
        return instance.GetType().GetProperty(propertyName, BindingFlags.Public | BindingFlags.Instance)?.GetValue(instance)
            ?? throw new InvalidOperationException($"Property '{propertyName}' was not found.");
    }

    private static string CreateTempDirectory()
    {
        string path = Path.Combine(Path.GetTempPath(), "jyppx-sample-support-contract-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(path);
        return path;
    }

    private static string WriteFloats(string root, string name, float[] values)
    {
        byte[] bytes = new byte[checked(values.Length * sizeof(float))];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return WriteBytes(root, name, bytes);
    }

    private static string WriteBytes(string root, string name, byte[] bytes)
    {
        string path = Path.Combine(root, name);
        File.WriteAllBytes(path, bytes);
        return path;
    }

    private static string Read(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray()));
    }
}

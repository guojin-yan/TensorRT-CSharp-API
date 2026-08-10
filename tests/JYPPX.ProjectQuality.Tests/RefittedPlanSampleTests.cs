using System.Xml.Linq;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class RefittedPlanSampleTests
{
    [Fact]
    public void SampleUsesPublicPackageAndPersistsTheRefittedPlan()
    {
        string root = Path.Combine(RepositoryPaths.Root, "samples", "Inference", "04.RefittedPlan");
        XDocument project = XDocument.Load(Path.Combine(root, "RefittedPlan.csproj"));
        string program = File.ReadAllText(Path.Combine(root, "Program.cs"));

        Assert.Equal("true", project.Descendants("JYPPXIncludeTensorRtSampleSupport").Single().Value);
        Assert.Equal("false", project.Descendants("IsPackable").Single().Value);
        Assert.Empty(project.Descendants("ProjectReference"));
        Assert.Contains("SampleCommandLine.HasSwitch(args, \"--help\")", program, StringComparison.Ordinal);
        Assert.True(program.IndexOf("--help", StringComparison.Ordinal) < program.IndexOf("TensorRtEnvironmentProbe.GetCurrent", StringComparison.Ordinal));
        Assert.Contains("config.SetFlag(TensorRtBuilderFlag.Refit)", program, StringComparison.Ordinal);
        Assert.Contains("parserRefitter.RefitFromFile(refitModelPath)", program, StringComparison.Ordinal);
        Assert.Contains("refitter.RefitCudaEngine()", program, StringComparison.Ordinal);
        Assert.Contains("engine.Serialize()", program, StringComparison.Ordinal);
        Assert.Contains("reloadRuntime.DeserializeFromFile(refittedPlanPath)", program, StringComparison.Ordinal);
        Assert.Contains("proofClassification = \"synthetic-input-runtime\"", program, StringComparison.Ordinal);
    }

    [Fact]
    public void SyntheticModelsHaveStableInitializerNamesAndNoPointerInterop()
    {
        string source = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "Inference",
            "04.RefittedPlan",
            "SyntheticRefitOnnxModel.cs"));

        Assert.Contains("scale-1x4-baseline.onnx", source, StringComparison.Ordinal);
        Assert.Contains("scale-1x4-refit.onnx", source, StringComparison.Ordinal);
        Assert.Contains("node.WriteString(4, \"Mul\")", source, StringComparison.Ordinal);
        Assert.Contains("tensor.WriteString(8, \"scale\")", source, StringComparison.Ordinal);
        Assert.Contains("new[] { 1.0f, 1.0f, 1.0f, 1.0f }", source, StringComparison.Ordinal);
        Assert.Contains("new[] { 2.0f, 2.0f, 2.0f, 2.0f }", source, StringComparison.Ordinal);
        Assert.DoesNotContain("IntPtr", source, StringComparison.Ordinal);
        Assert.DoesNotContain("nint", source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("README.md")]
    [InlineData("README.zh-CN.md")]
    public void ReadmeDocumentsHelpPersistenceAndEvidenceBoundary(string fileName)
    {
        string readme = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "Inference",
            "04.RefittedPlan",
            fileName));

        Assert.Contains("--help", readme, StringComparison.Ordinal);
        Assert.Contains("--synthetic", readme, StringComparison.Ordinal);
        Assert.Contains("--plan", readme, StringComparison.Ordinal);
        Assert.Contains("--output-json", readme, StringComparison.Ordinal);
        Assert.Contains("TensorRtOnnxParserRefitter", readme, StringComparison.Ordinal);
        Assert.Contains("synthetic-input-runtime", readme, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime", readme, StringComparison.Ordinal);
    }
}

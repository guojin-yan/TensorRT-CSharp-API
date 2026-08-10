using System.Xml.Linq;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class OnnxBuildAndRunSampleTests
{
    [Fact]
    public void SampleUsesPublicPackageAndHasOfflineHelp()
    {
        string root = Path.Combine(RepositoryPaths.Root, "samples", "Inference", "03.OnnxBuildAndRun");
        string projectPath = Path.Combine(root, "OnnxBuildAndRun.csproj");
        XDocument project = XDocument.Load(projectPath);
        string program = File.ReadAllText(Path.Combine(root, "Program.cs"));

        Assert.Equal("true", project.Descendants("JYPPXIncludeTensorRtOnnxSampleSupport").Single().Value);
        Assert.Equal("false", project.Descendants("IsPackable").Single().Value);
        Assert.Empty(project.Descendants("ProjectReference"));
        Assert.Contains("SampleCommandLine.HasSwitch(args, \"--help\")", program, StringComparison.Ordinal);
        Assert.True(program.IndexOf("--help", StringComparison.Ordinal) < program.IndexOf("TensorRtOnnxSample.RunSingleFloatInputOutput", StringComparison.Ordinal));
        Assert.Contains("--output-json", program, StringComparison.Ordinal);
        Assert.Contains("proofClassification = \"synthetic-input-runtime\"", program, StringComparison.Ordinal);
        Assert.Contains("enqueueCount = 1", program, StringComparison.Ordinal);
    }

    [Fact]
    public void SyntheticModelIsDeterministicAndPointerFree()
    {
        string source = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "Inference",
            "03.OnnxBuildAndRun",
            "SyntheticIdentityOnnxModel.cs"));

        Assert.Contains("identity-1x4.onnx", source, StringComparison.Ordinal);
        Assert.Contains("node.WriteString(4, \"Identity\")", source, StringComparison.Ordinal);
        Assert.Contains("tensorType.WriteVarint(1, 1)", source, StringComparison.Ordinal);
        Assert.DoesNotContain("IntPtr", source, StringComparison.Ordinal);
        Assert.DoesNotContain("nint", source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("README.md")]
    [InlineData("README.zh-CN.md")]
    public void ReadmeDocumentsHelpSyntheticSmokeAndEvidenceBoundary(string fileName)
    {
        string readme = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "Inference",
            "03.OnnxBuildAndRun",
            fileName));

        Assert.Contains("--help", readme, StringComparison.Ordinal);
        Assert.Contains("--synthetic", readme, StringComparison.Ordinal);
        Assert.Contains("--output-json", readme, StringComparison.Ordinal);
        Assert.Contains("synthetic-input-runtime", readme, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime", readme, StringComparison.Ordinal);
    }
}

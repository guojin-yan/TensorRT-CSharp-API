using System.Xml.Linq;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CallbackLifecycleSampleTests
{
    [Fact]
    public void SampleUsesPublicPackageAndKeepsHelpOffline()
    {
        string root = Path.Combine(RepositoryPaths.Root, "samples", "Diagnostics", "01.CallbackLifecycle");
        XDocument project = XDocument.Load(Path.Combine(root, "CallbackLifecycle.csproj"));
        string program = File.ReadAllText(Path.Combine(root, "Program.cs"));

        Assert.Equal("true", project.Descendants("JYPPXIncludeTensorRtSampleSupport").Single().Value);
        Assert.Equal("false", project.Descendants("IsPackable").Single().Value);
        Assert.Empty(project.Descendants("ProjectReference"));
        Assert.Contains("SampleCommandLine.HasSwitch(args, \"--help\")", program, StringComparison.Ordinal);
        Assert.True(program.IndexOf("--help", StringComparison.Ordinal) < program.IndexOf("TensorRtEnvironmentProbe.GetCurrent", StringComparison.Ordinal));
        Assert.Contains("proofClassification = \"synthetic-input-runtime\"", program, StringComparison.Ordinal);
        Assert.Contains("--output-json", program, StringComparison.Ordinal);
    }

    [Fact]
    public void CallbackOwnersAttachExecuteAndDetachInOrder()
    {
        string program = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "Diagnostics",
            "01.CallbackLifecycle",
            "Program.cs"));

        Assert.Contains("new TensorRtLogger(line, records.RecordLog", program, StringComparison.Ordinal);
        Assert.Contains("config.SetProgressMonitor(progressMonitor)", program, StringComparison.Ordinal);
        Assert.Contains("config.ClearProgressMonitor()", program, StringComparison.Ordinal);
        Assert.Contains("context.SetProfiler(profiler)", program, StringComparison.Ordinal);
        Assert.Contains("context.SetDebugListener(debugListener)", program, StringComparison.Ordinal);
        Assert.Contains("context.SetTensorDebugState(OutputName, true)", program, StringComparison.Ordinal);
        Assert.Contains("context.ClearDebugListener()", program, StringComparison.Ordinal);
        Assert.Contains("context.ClearProfiler()", program, StringComparison.Ordinal);
        Assert.Contains("borrowedPointerExposed = debugAttached.BorrowedPointerExposed", program, StringComparison.Ordinal);
        Assert.True(program.IndexOf("context.SetProfiler(profiler)", StringComparison.Ordinal) < program.IndexOf("context.EnqueueAsync(stream)", StringComparison.Ordinal));
        Assert.True(program.IndexOf("context.EnqueueAsync(stream)", StringComparison.Ordinal) < program.IndexOf("context.ClearProfiler()", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("README.md")]
    [InlineData("README.zh-CN.md")]
    public void ReadmeDocumentsAllCallbacksAndBorrowedPointerBoundary(string fileName)
    {
        string readme = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "Diagnostics",
            "01.CallbackLifecycle",
            fileName));

        Assert.Contains("--help", readme, StringComparison.Ordinal);
        Assert.Contains("--output-json", readme, StringComparison.Ordinal);
        Assert.Contains("Logger", readme, StringComparison.Ordinal);
        Assert.Contains("ProgressMonitor", readme, StringComparison.Ordinal);
        Assert.Contains("Profiler", readme, StringComparison.Ordinal);
        Assert.Contains("DebugListener", readme, StringComparison.Ordinal);
        Assert.Contains("borrowedPointerExposed=false", readme, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime", readme, StringComparison.Ordinal);
    }
}

using System.Text.Json;
using JYPPX.TensorRtSharp.Tools;
using Xunit;
using YoloVisionSample;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ToolCapabilityJsonSurfaceTests
{
    [Fact]
    public void TrtexecLikeHelpJsonDocumentsProofBoundaryAndFrozenReleaseState()
    {
        using JsonDocument document = JsonDocument.Parse(TrtexecLikeOptionCapabilities.FormatJson("OnnxToEngine"));
        JsonElement root = document.RootElement;

        Assert.Equal("trtexec-like-option-capabilities.v1", root.GetProperty("schema").GetString());
        Assert.Equal("OnnxToEngine", root.GetProperty("tool").GetString());
        Assert.Equal("source-quality-capability-surface", root.GetProperty("matrixState").GetString());
        Assert.True(root.GetProperty("releaseFrozen").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.Contains("not runtime proof", root.GetProperty("proofBoundary").GetString(), StringComparison.Ordinal);
        Assert.Equal(TrtexecLikeOptionCapabilities.Entries.Count, root.GetProperty("entryCount").GetInt32());
        Assert.True(root.GetProperty("entryCount").GetInt32() >= 30);

        JsonElement[] entries = root.GetProperty("entries").EnumerateArray().ToArray();
        Assert.DoesNotContain(entries, entry => entry.GetProperty("option").GetString() == "--help-json");
        Assert.Contains(entries, entry => entry.GetProperty("option").GetString() == "--int8/--calib" &&
                                          entry.GetProperty("status").GetString() == "blocked-calibrator-lifecycle");
        Assert.Contains(entries, entry => entry.GetProperty("option").GetString() == "--sleepTime" &&
                                          entry.GetProperty("status").GetString() == "parse-report-only");
        Assert.All(entries, entry =>
        {
            Assert.False(entry.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.False(entry.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
            Assert.True(entry.GetProperty("requiresOwnerEvidence").GetBoolean());
        });
    }

    [Fact]
    public void CliEntrypointsExposeMachineReadableCapabilitySwitches()
    {
        string tensorRtExecCommand = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "Console", "TensorRtExecCommand.cs"));
        string onnxToEngineProgram = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "OnnxToEngine", "Program.cs"));
        string onnxReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "OnnxToEngine", "README.md"));

        Assert.Contains("--help-json", tensorRtExecCommand, StringComparison.Ordinal);
        Assert.Contains("TrtexecLikeOptionCapabilities.FormatJson(\"TensorRtExec\")", tensorRtExecCommand, StringComparison.Ordinal);
        Assert.Contains("--capabilities-json", tensorRtExecCommand, StringComparison.Ordinal);
        Assert.Contains("--help-json", onnxToEngineProgram, StringComparison.Ordinal);
        Assert.Contains("TrtexecLikeOptionCapabilities.FormatJson(\"OnnxToEngine\")", onnxToEngineProgram, StringComparison.Ordinal);
        Assert.Contains("--capabilities-json", onnxToEngineProgram, StringComparison.Ordinal);
        Assert.Contains("--help-json", onnxReadme, StringComparison.Ordinal);
        Assert.Contains("machine-readable", onnxReadme, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void YoloVisionCapabilitySelfTestRunsOfflineAndKeepsRuntimeProofDisabled()
    {
        using StringWriter writer = new StringWriter();
        TextWriter original = Console.Out;
        Console.SetOut(writer);
        int exitCode;
        try
        {
            exitCode = YoloVisionCommand.Run(new[] { "--self-test-capabilities" });
        }
        finally
        {
            Console.SetOut(original);
        }

        string output = writer.ToString();
        Assert.Equal(0, exitCode);
        Assert.Contains("YoloVision CapabilitySelfTest Passed=True", output, StringComparison.Ordinal);
        Assert.Contains("Entries=60", output, StringComparison.Ordinal);
        Assert.Contains("Supported=55", output, StringComparison.Ordinal);
        Assert.Contains("Unsupported=5", output, StringComparison.Ordinal);
        Assert.Contains("IsRuntimeProof=False", output, StringComparison.Ordinal);
        Assert.Contains("IsPackageConsumerRuntimeProof=False", output, StringComparison.Ordinal);

        string yoloReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "README.md"));
        Assert.Contains("--self-test-capabilities", yoloReadme, StringComparison.Ordinal);
    }
}

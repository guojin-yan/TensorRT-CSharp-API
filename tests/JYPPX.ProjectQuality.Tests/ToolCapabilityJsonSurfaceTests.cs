using System.Text.Json;
using JYPPX.TensorRtSharp.Tools;
using Xunit;
using YoloVisionSample;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ToolCapabilityJsonSurfaceTests
{
    [Fact]
    public void TensorRtExecOptionLayeringGuideBindsCapabilityFieldMapGapListAndExecutionStages()
    {
        string article = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "docs",
            "articles",
            "zh-cn",
            "tensorrtexec-option-layering-deep-dive.md"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "TensorRtExec", "README.md"));
        string roadmap = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "technical-article-roadmap.md"));

        Assert.Contains("tensorrtexec-option-layering-deep-dive.md", readme, StringComparison.Ordinal);
        Assert.Contains("| 72 |", roadmap, StringComparison.Ordinal);
        Assert.Contains("完整教程已收口", roadmap, StringComparison.Ordinal);

        foreach (string marker in new[]
        {
            "## 先读四种事实来源",
            "33 entries",
            "29 implemented/bounded",
            "3 parse-or-diagnostic-only",
            "1 blocked",
            "90 fields",
            "20 items",
            "..\\downloads\\cases\\tensorrtexec-option-audit",
            "TrtexecLikeParser",
            "TensorRtExecOptions",
            "ToArgumentLine",
            "OnnxEngineBuildService",
            "OptionImplementationStatus",
            "ParsedOptions",
            "AppliedOptions",
            "ParseOnlyOptions",
            "--help-json",
            "--capabilities-json",
            "releaseFrozen=true",
            "canPromoteRuntimeProof=false",
            "## 第一层：Dry Run / Precheck",
            "## 第二层：Build-Only",
            "## 第三层：Readonly Engine Diagnostics",
            "## 第四层：Bounded Runtime",
            "--loadInputs images:",
            "runtime-output-captured-unverified",
            "tensor-rt-exec-report.schema.json",
            "Test-TensorRtExecReport.ps1",
            "Export-TensorRtExecGuiCliParityChecklist.ps1",
            "Test-TensorRtExecGuiCliParityChecklist.ps1 -Strict",
            "TRT8/TRT10/TRT11",
            "TrtexecAlignmentStatus=parse-only",
            "blocked-calibrator-lifecycle",
            "## 证据阶梯",
            "## 收尾检查清单"
        })
        {
            Assert.Contains(marker, article, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument capabilities = JsonDocument.Parse(TrtexecLikeOptionCapabilities.FormatJson("TensorRtExec"));
        Assert.Equal(33, capabilities.RootElement.GetProperty("entryCount").GetInt32());
        Assert.Equal(29, capabilities.RootElement.GetProperty("implementedCount").GetInt32());
        Assert.Equal(3, capabilities.RootElement.GetProperty("parseOrDiagnosticOnlyCount").GetInt32());
        Assert.Equal(1, capabilities.RootElement.GetProperty("blockedCount").GetInt32());

        using JsonDocument fieldMap = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "applications",
            "TensorRtExec",
            "tensor-rt-exec-gui-cli-field-map.json")));
        Assert.Equal(90, fieldMap.RootElement.GetProperty("fields").GetArrayLength());
        Assert.Contains("not runtime proof", fieldMap.RootElement.GetProperty("proofBoundary").GetString(), StringComparison.Ordinal);

        using JsonDocument gapList = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "applications",
            "TensorRtExec",
            "tensor-rt-exec-release-candidate-gap-list.json")));
        JsonElement summary = gapList.RootElement.GetProperty("summary");
        Assert.Equal(20, summary.GetProperty("totalItems").GetInt32());
        Assert.Equal(0, summary.GetProperty("runtimeProofItems").GetInt32());
        Assert.Equal(0, summary.GetProperty("packageConsumerRuntimeProofItems").GetInt32());
    }

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
                                          entry.GetProperty("status").GetString() == "implemented-bounded-runtime" &&
                                          entry.GetProperty("implementationClass").GetString() == "runtime-applied-when-benchmark-executes");
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
        string onnxToEngineProgram = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "OnnxToEngine", "Program.cs"));
        string onnxReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "OnnxToEngine", "README.md"));

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

        string yoloReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "YoloVision", "README.md"));
        Assert.Contains("--self-test-capabilities", yoloReadme, StringComparison.Ordinal);
    }
}

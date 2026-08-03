using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class TensorRtExecGuiCliParityChecklistTests
{
    [Fact]
    public void TensorRtExecGuiCliParityChecklistCoversCoreOptionsWithoutPromotingProof()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-TensorRtExecGuiCliParityChecklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-TensorRtExecGuiCliParityChecklist.ps1"), "-Strict");

        using JsonDocument checklistDocument = ReadFinalReleaseJson("tensor-rt-exec-gui-cli-parity-checklist.json");
        JsonElement checklist = checklistDocument.RootElement;

        Assert.Equal("tensor-rt-exec-gui-cli-parity-checklist", checklist.GetProperty("recordKind").GetString());
        Assert.Equal("release-candidate-gui-cli-parity-non-proof", checklist.GetProperty("checklistState").GetString());
        Assert.True(checklist.GetProperty("fieldMapPresent").GetBoolean());
        Assert.True(checklist.GetProperty("itemCount").GetInt32() >= 15);
        Assert.True(checklist.GetProperty("cliSupportedCount").GetInt32() >= 10);
        Assert.True(checklist.GetProperty("winFormsSupportedCount").GetInt32() >= 10);
        Assert.Equal(checklist.GetProperty("itemCount").GetInt32(), checklist.GetProperty("commandPreviewSupportedCount").GetInt32());
        Assert.Equal(0, checklist.GetProperty("runtimeProofItems").GetInt32());
        Assert.Equal(0, checklist.GetProperty("packageConsumerRuntimeProofItems").GetInt32());
        Assert.False(checklist.GetProperty("performsPublish").GetBoolean());
        Assert.False(checklist.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(checklist.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(checklist.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.Contains("not runtime proof", checklist.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package-consumer-runtime proof", checklist.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = checklist.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("applications/TensorRtExec/tensor-rt-exec-feature-matrix.json", sourceArtifacts);
        Assert.Contains("applications/TensorRtExec/tensor-rt-exec-gui-cli-field-map.json", sourceArtifacts);
        Assert.Contains("applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json", sourceArtifacts);
        Assert.Contains("applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json", sourceArtifacts);
        Assert.Contains("applications/TensorRtExec/Core/TensorRtExecOptions.cs", sourceArtifacts);
        Assert.Contains("applications/TensorRtExec/Console/TensorRtExecCommand.cs", sourceArtifacts);
        Assert.Contains("applications/TensorRtExec/WinForms/MainForm.cs", sourceArtifacts);

        JsonElement[] items = checklist.GetProperty("items").EnumerateArray().ToArray();
        foreach (string optionId in new[]
        {
            "onnx",
            "save-engine",
            "load-engine",
            "shape-profiles",
            "precision",
            "int8-calibration",
            "workspace-memory-pool",
            "timing-cache",
            "plugins",
            "profiling",
            "layer-info",
            "report-export",
            "runtime-benchmark",
            "wait-idle-controls",
            "binding-output",
            "safety-cache-policy",
            "device-dla"
        })
        {
            Assert.Contains(items, item => item.GetProperty("optionId").GetString() == optionId);
        }

        Assert.Contains(items, item =>
            item.GetProperty("optionId").GetString() == "timing-cache" &&
            item.GetProperty("status").GetString() == "implemented-build-cache-lifecycle");
        Assert.Contains(items, item =>
            item.GetProperty("optionId").GetString() == "device-dla" &&
            item.GetProperty("status").GetString() == "implemented-build-readback-with-version-guards" &&
            item.GetProperty("officialTrtexecOption").GetString()!.Contains("--tacticSources", StringComparison.Ordinal) &&
            item.GetProperty("officialTrtexecOption").GetString()!.Contains("--stronglyTyped", StringComparison.Ordinal));
        Assert.Contains(items, item =>
            item.GetProperty("optionId").GetString() == "safety-cache-policy" &&
            item.GetProperty("status").GetString()!.Contains("parse", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(items, item =>
            item.GetProperty("optionId").GetString() == "wait-idle-controls" &&
            item.GetProperty("status").GetString() == "implemented-bounded-runtime" &&
            item.GetProperty("officialTrtexecOption").GetString()!.Contains("--sleepTime", StringComparison.Ordinal) &&
            item.GetProperty("officialTrtexecOption").GetString()!.Contains("--idleTime", StringComparison.Ordinal) &&
            item.GetProperty("proofBoundary").GetString()!.Contains("bridge-owned native host-function delay", StringComparison.Ordinal));
        Assert.All(items, item =>
        {
            Assert.False(item.GetProperty("isRuntimeProof").GetBoolean());
            Assert.False(item.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
            Assert.True(item.GetProperty("commandPreviewSupported").GetBoolean());
        });

        using JsonDocument validationDocument = ReadFinalReleaseJson("tensor-rt-exec-gui-cli-parity-checklist-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("tensor-rt-exec-gui-cli-parity-checklist-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("tensor-rt-exec-gui-cli-parity-checklist-ready", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
    {
        using Process process = new();
        process.StartInfo.FileName = "pwsh";
        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(scriptPath);
        foreach (string argument in arguments)
        {
            process.StartInfo.ArgumentList.Add(argument);
        }

        process.StartInfo.WorkingDirectory = RepositoryPaths.Root;
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.RedirectStandardError = true;
        process.StartInfo.UseShellExecute = false;

        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptPath}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        return stdout;
    }
}

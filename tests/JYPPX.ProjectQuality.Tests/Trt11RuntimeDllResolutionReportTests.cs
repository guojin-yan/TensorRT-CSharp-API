using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class Trt11RuntimeDllResolutionReportTests
{
    [Fact]
    public void Trt11RuntimeDllResolutionReportClassifiesLocalDllOrderWithoutPromotingProof()
    {
        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-Trt11RuntimeDllResolutionReport.ps1"));
        Assert.Contains("TRT11 runtime DLL resolution report written", output, StringComparison.Ordinal);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "trt11-runtime-dll-resolution-report.json")));
        JsonElement root = document.RootElement;

        Assert.Equal("trt11-runtime-dll-resolution-report", root.GetProperty("recordKind").GetString());
        Assert.StartsWith("diagnostic-", root.GetProperty("reportState").GetString(), StringComparison.Ordinal);
        Assert.Equal("win-x64-trt11.0-cuda13.2-cudnn9.22", root.GetProperty("sourceRuntimeKey").GetString());
        Assert.Equal("compatible-host-bridge-package-runtime-failed", root.GetProperty("proofClassification").GetString());
        Assert.Equal("failed", root.GetProperty("smokeStatus").GetString());
        Assert.Equal(1, root.GetProperty("exitCode").GetInt32());
        Assert.Equal("createInferRuntime-null", root.GetProperty("failureSignature").GetString());
        Assert.Equal("trt11-create-runtime-null-cuda-runtime-error", root.GetProperty("rootCauseCategory").GetString());
        Assert.True(root.GetProperty("tensorRtAvailable").GetBoolean());
        Assert.True(root.GetProperty("cudaAvailable").GetBoolean());
        Assert.True(root.GetProperty("nativeBridgePresent").GetBoolean());
        Assert.True(root.GetProperty("nativeAssetCount").GetInt32() >= 1);
        Assert.True(root.GetProperty("cudnnAssetCount").GetInt32() >= 1);
        Assert.True(root.GetProperty("searchDirectoryCount").GetInt32() >= 1);
        Assert.True(root.GetProperty("requiredDllGroupCount").GetInt32() >= 8);
        Assert.True(root.TryGetProperty("runtimeCreateDiagnostic", out _));
        Assert.True(root.TryGetProperty("nativeCreateRuntimeDiagnosticAvailable", out _));
        Assert.True(root.TryGetProperty("nativeCreateRuntimeAttempted", out _));
        Assert.True(root.TryGetProperty("nativeCreateRuntimeReturnedNull", out _));
        Assert.True(root.TryGetProperty("nativeCreateRuntimeLastStatus", out _));
        Assert.Equal("after-createInferRuntime-null-guard-ok", root.GetProperty("nativeCreateRuntimePhase").GetString());
        Assert.True(int.Parse(root.GetProperty("nativeCreateRuntimeLoggerMessageCount").GetString()!) >= 1);
        Assert.Matches("Cuda Runtime|catchCudaError", root.GetProperty("nativeCreateRuntimeLastLoggerMessage").GetString()!);
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());

        JsonElement[] resolutionRows = root.GetProperty("dllResolution").EnumerateArray().ToArray();
        Assert.Contains(resolutionRows, item =>
            item.GetProperty("key").GetString() == "bridge" &&
            item.GetProperty("found").GetBoolean() &&
            item.GetProperty("presentAtCapture").GetBoolean());
        Assert.Contains(resolutionRows, item => item.GetProperty("key").GetString() == "tensorrt-runtime");
        Assert.Contains(resolutionRows, item =>
            item.GetProperty("key").GetString() == "cuda-runtime" &&
            item.GetProperty("resolvedPath").GetString()!.EndsWith("CUDA\\v13.2\\bin\\x64\\cudart64_13.dll", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(resolutionRows, item => item.GetProperty("key").GetString() == "cudnn-runtime");

        string[] searchDirectories = root.GetProperty("searchDirectories").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains(searchDirectories, item => item.EndsWith("CUDA\\v13.2\\bin\\x64", StringComparison.OrdinalIgnoreCase));

        string[] sourceArtifacts = root.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/package-consumer/bridge-runtime/win-x64-trt11.0-cuda13.2-cudnn9.22/bridge-package-runtime-consumer-proof.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/trt11-runtime-smoke-root-cause-report.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/trt10-vs-trt11-bridge-runtime-diagnostic-diff.json", sourceArtifacts);
        Assert.Contains("eng/Test-BridgePackageRuntimeConsumer.ps1", sourceArtifacts);
        Assert.Contains("eng/Resolve-RuntimeRoots.ps1", sourceArtifacts);

        string markdown = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "trt11-runtime-dll-resolution-report.md"));
        Assert.Contains("TRT11 Runtime DLL Resolution Report", markdown, StringComparison.Ordinal);
        Assert.Contains("DLL Resolution", markdown, StringComparison.Ordinal);
        Assert.Contains("createInferRuntime-null", markdown, StringComparison.Ordinal);
        Assert.Contains("native create-runtime diagnostic available", markdown, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("diagnostic blocker evidence only", markdown, StringComparison.OrdinalIgnoreCase);
    }

    private static string RunPowerShell(string scriptPath)
    {
        using Process process = new();
        process.StartInfo.FileName = "pwsh";
        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(scriptPath);
        process.StartInfo.WorkingDirectory = RepositoryPaths.Root;
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.RedirectStandardError = true;
        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptPath}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        return stdout + stderr;
    }
}

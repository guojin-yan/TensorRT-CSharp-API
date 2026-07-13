using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class Trt10VsTrt11BridgeRuntimeDiagnosticDiffTests
{
    [Fact]
    public void DiffCapturesTrt10PassAndTrt11CreateRuntimeNullWithoutPromotingProof()
    {
        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-Trt10VsTrt11BridgeRuntimeDiagnosticDiff.ps1"));
        Assert.Contains("TRT10 vs TRT11 bridge runtime diagnostic diff written", output, StringComparison.Ordinal);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "trt10-vs-trt11-bridge-runtime-diagnostic-diff.json")));
        JsonElement root = document.RootElement;

        Assert.Equal("trt10-vs-trt11-bridge-runtime-diagnostic-diff", root.GetProperty("recordKind").GetString());
        Assert.Equal("diagnostic-diff-ready-non-proof", root.GetProperty("reportState").GetString());
        Assert.True(root.GetProperty("trt10SmokePassed").GetBoolean());
        Assert.True(root.GetProperty("trt11SmokeFailed").GetBoolean());
        Assert.True(root.GetProperty("runtimeEnvironmentBothAvailable").GetBoolean());
        Assert.Equal("createInferRuntime-null", root.GetProperty("trt11FailureSignature").GetString());
        Assert.Equal("trt11-create-runtime-null-cuda-runtime-error", root.GetProperty("trt11RootCauseCategory").GetString());
        Assert.True(root.TryGetProperty("trt11RuntimeCreateDiagnostic", out _));
        Assert.Equal("after-createInferRuntime-null-guard-ok", root.GetProperty("trt11NativeCreateRuntimePhase").GetString());
        Assert.True(int.Parse(root.GetProperty("trt11NativeCreateRuntimeLoggerMessageCount").GetString()!) >= 1);
        Assert.Matches("Cuda Runtime|catchCudaError", root.GetProperty("trt11NativeCreateRuntimeLastLoggerMessage").GetString()!);
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());

        JsonElement trt10 = root.GetProperty("trt10");
        JsonElement trt11 = root.GetProperty("trt11");
        Assert.Equal("win-x64-trt10.11-cuda12.9-cudnn9.22", trt10.GetProperty("sourceRuntimeKey").GetString());
        Assert.Equal("passed", trt10.GetProperty("smokeStatus").GetString());
        Assert.Equal("win-x64-trt11.0-cuda13.2-cudnn9.22", trt11.GetProperty("sourceRuntimeKey").GetString());
        Assert.Equal("failed", trt11.GetProperty("smokeStatus").GetString());
        Assert.NotEqual(
            trt10.GetProperty("nativeAssets").GetProperty("bridgeSha256").GetString(),
            trt11.GetProperty("nativeAssets").GetProperty("bridgeSha256").GetString());

        string[] diffFields = root.GetProperty("diffItems").EnumerateArray().Select(static item => item.GetProperty("field").GetString()!).ToArray();
        Assert.Contains("runtimeEnvironmentLine", diffFields);
        Assert.Contains("searchDirectoryCount", diffFields);
        Assert.Contains("bridgeSha256", diffFields);
        Assert.Contains("stderrSha256", diffFields);
        Assert.Contains("runtimeCreateDiagnostic.available", diffFields);
        Assert.Contains("runtimeCreateDiagnostic.attempted", diffFields);
        Assert.Contains("runtimeCreateDiagnostic.returnedNull", diffFields);
        Assert.Contains("runtimeCreateDiagnostic.lastStatus", diffFields);

        string[] sourceArtifacts = root.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/package-consumer/bridge-runtime/win-x64-trt10.11-cuda12.9-cudnn9.22/bridge-package-runtime-consumer-proof.json", sourceArtifacts);
        Assert.Contains("artifacts/package-consumer/bridge-runtime/win-x64-trt11.0-cuda13.2-cudnn9.22/bridge-package-runtime-consumer-proof.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/trt11-runtime-smoke-root-cause-report.json", sourceArtifacts);

        string markdown = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "trt10-vs-trt11-bridge-runtime-diagnostic-diff.md"));
        Assert.Contains("TRT10 vs TRT11 Bridge Runtime Diagnostic Diff", markdown, StringComparison.Ordinal);
        Assert.Contains("createInferRuntime-null", markdown, StringComparison.Ordinal);
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
        process.StartInfo.UseShellExecute = false;

        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptPath}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        return stdout;
    }
}

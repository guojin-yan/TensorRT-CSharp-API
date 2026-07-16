using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class Trt11CudaRuntimePreflightOwnerActionTests
{
    [Fact]
    public void ConsumerAndReleaseReportsCarryCudaPreflightWithoutPromotingProof()
    {
        RunPowerShell("Export-Trt11RuntimeSmokeRootCauseReport.ps1");
        RunPowerShell("Export-Trt10VsTrt11BridgeRuntimeDiagnosticDiff.ps1");
        RunPowerShell("Export-Trt11RuntimeDllResolutionReport.ps1");
        RunPowerShell("Export-FinalProofReadinessBlockerDashboard.ps1");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");

        using JsonDocument consumer = ReadJson("artifacts", "package-consumer", "bridge-runtime", "win-x64-trt11.0-cuda13.2-cudnn9.22", "bridge-package-runtime-consumer-proof.json");
        JsonElement consumerRoot = consumer.RootElement;
        JsonElement consumerPreflight = consumerRoot.GetProperty("cudaPreflight");
        Assert.True(consumerPreflight.GetProperty("attempted").GetBoolean());
        Assert.Equal("Failed", consumerPreflight.GetProperty("getDeviceCountStatus").GetString());
        Assert.Equal("CudaPreflightFailed", consumerPreflight.GetProperty("initStatus").GetString());
        Assert.Equal("CudaException", consumerPreflight.GetProperty("lastErrorName").GetString());
        Assert.Contains("CUDA error 35", consumerPreflight.GetProperty("lastErrorMessage").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.False(consumerPreflight.GetProperty("canAttemptTensorRtRuntimeCreate").GetBoolean());
        Assert.False(consumerRoot.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(consumerRoot.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());

        using JsonDocument rootCause = ReadJson("artifacts", "final-release", "trt11-runtime-smoke-root-cause-report.json");
        JsonElement rootCauseRoot = rootCause.RootElement;
        Assert.Equal("trt11-create-runtime-null-cuda-runtime-error", rootCauseRoot.GetProperty("rootCauseCategory").GetString());
        Assert.Equal("cuda-driver-insufficient-for-runtime", rootCauseRoot.GetProperty("rootCauseSubcategory").GetString());
        Assert.True(rootCauseRoot.GetProperty("cudaPreflightAttempted").GetBoolean());
        Assert.Equal("Failed", rootCauseRoot.GetProperty("cudaPreflightGetDeviceCountStatus").GetString());
        Assert.Equal("CudaPreflightFailed", rootCauseRoot.GetProperty("cudaPreflightInitStatus").GetString());
        Assert.Equal("CudaException", rootCauseRoot.GetProperty("cudaPreflightLastErrorName").GetString());
        Assert.Contains("CUDA error 35", rootCauseRoot.GetProperty("cudaPreflightLastErrorMessage").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.True(rootCauseRoot.GetProperty("cudaPreflightDriverInsufficient").GetBoolean());
        Assert.False(rootCauseRoot.GetProperty("cudaPreflightCanAttemptTensorRtRuntimeCreate").GetBoolean());
        Assert.False(rootCauseRoot.GetProperty("canPromoteRuntimeProof").GetBoolean());

        using JsonDocument dllResolution = ReadJson("artifacts", "final-release", "trt11-runtime-dll-resolution-report.json");
        JsonElement dllRoot = dllResolution.RootElement;
        Assert.Equal("cuda-driver-insufficient-for-runtime", dllRoot.GetProperty("rootCauseSubcategory").GetString());
        Assert.True(dllRoot.GetProperty("cudaPreflightAttempted").GetBoolean());
        Assert.Equal("CudaPreflightFailed", dllRoot.GetProperty("cudaPreflightInitStatus").GetString());
        Assert.False(dllRoot.GetProperty("canPromoteRuntimeProof").GetBoolean());

        using JsonDocument diff = ReadJson("artifacts", "final-release", "trt10-vs-trt11-bridge-runtime-diagnostic-diff.json");
        JsonElement diffRoot = diff.RootElement;
        Assert.Equal("cuda-driver-insufficient-for-runtime", diffRoot.GetProperty("trt11RootCauseSubcategory").GetString());
        Assert.True(diffRoot.GetProperty("trt11CudaPreflightAttempted").GetBoolean());
        Assert.Equal("CudaPreflightFailed", diffRoot.GetProperty("trt11CudaPreflightInitStatus").GetString());
        string[] diffFields = diffRoot.GetProperty("diffItems").EnumerateArray().Select(static item => item.GetProperty("field").GetString()!).ToArray();
        Assert.Contains("cudaPreflight.available", diffFields);
        Assert.Contains("cudaPreflight.initStatus", diffFields);
        Assert.False(diffRoot.GetProperty("canPromoteRuntimeProof").GetBoolean());

        using JsonDocument dashboard = ReadJson("artifacts", "final-release", "final-proof-readiness-blocker-dashboard.json");
        JsonElement sourceStates = dashboard.RootElement.GetProperty("sourceStates");
        Assert.Equal("cuda-driver-insufficient-for-runtime", sourceStates.GetProperty("trt11RootCauseSubcategory").GetString());
        Assert.True(sourceStates.GetProperty("trt11RootCauseCudaPreflightAttempted").GetBoolean());
        Assert.Equal("CudaPreflightFailed", sourceStates.GetProperty("trt11RootCauseCudaPreflightInitStatus").GetString());
        Assert.False(sourceStates.GetProperty("trt11RootCauseCudaPreflightCanAttemptTensorRtRuntimeCreate").GetBoolean());
        Assert.False(sourceStates.GetProperty("trt11RootCauseCanPromoteRuntimeProof").GetBoolean());

        using JsonDocument releaseBundle = ReadJson("artifacts", "final-release", "release-evidence-bundle.json");
        JsonElement bundleRoot = releaseBundle.RootElement;
        Assert.Equal("cuda-driver-insufficient-for-runtime", bundleRoot.GetProperty("trt11RootCauseSubcategory").GetString());
        Assert.True(bundleRoot.GetProperty("trt11BridgeRuntimeConsumerCudaPreflightAttempted").GetBoolean());
        Assert.True(bundleRoot.GetProperty("trt11RootCauseCudaPreflightAttempted").GetBoolean());
        Assert.Equal("CudaPreflightFailed", bundleRoot.GetProperty("trt11RootCauseCudaPreflightInitStatus").GetString());
        Assert.Equal("CudaException", bundleRoot.GetProperty("trt11RootCauseCudaPreflightLastErrorName").GetString());
        Assert.Contains("CUDA error 35", bundleRoot.GetProperty("trt11RootCauseCudaPreflightLastErrorMessage").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.False(bundleRoot.GetProperty("trt11BridgeRuntimeConsumerCanPromoteRuntimeProof").GetBoolean());
        Assert.False(bundleRoot.GetProperty("trt11RootCauseCanPromoteRuntimeProof").GetBoolean());
        Assert.False(bundleRoot.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(bundleRoot.GetProperty("canPublishPublicly").GetBoolean());
    }

    [Fact]
    public void RuntimeConsumerScriptDefinesCudaPreflightMarkersAndProofBoundary()
    {
        string script = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-BridgePackageRuntimeConsumer.ps1"));

        Assert.Contains("WriteCudaPreflight", script, StringComparison.Ordinal);
        Assert.Contains("CudaPreflightAvailable=", script, StringComparison.Ordinal);
        Assert.Contains("CudaPreflightAttempted=", script, StringComparison.Ordinal);
        Assert.Contains("CudaPreflightDriverVersion=", script, StringComparison.Ordinal);
        Assert.Contains("CudaPreflightRuntimeVersion=", script, StringComparison.Ordinal);
        Assert.Contains("CudaPreflightDeviceCount=", script, StringComparison.Ordinal);
        Assert.Contains("CudaPreflightGetDeviceCountStatus=", script, StringComparison.Ordinal);
        Assert.Contains("CudaPreflightInitStatus=", script, StringComparison.Ordinal);
        Assert.Contains("CudaPreflightLastErrorName=", script, StringComparison.Ordinal);
        Assert.Contains("CudaPreflightLastErrorMessage=", script, StringComparison.Ordinal);
        Assert.Contains("CudaPreflightCanAttemptTensorRtRuntimeCreate=", script, StringComparison.Ordinal);
        Assert.Contains("Join-Path $Roots.CudaRoot \"bin\\x64\"", script, StringComparison.Ordinal);
        Assert.Contains("cudaPreflight = [ordered]@", script, StringComparison.Ordinal);
        Assert.Contains("cannot promote runtime proof by itself", script, StringComparison.Ordinal);
    }

    private static JsonDocument ReadJson(params string[] pathParts)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray())));
    }

    private static string RunPowerShell(string scriptName)
    {
        using Process process = new();
        process.StartInfo.FileName = "pwsh";
        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(Path.Combine(RepositoryPaths.Root, "eng", scriptName));
        process.StartInfo.WorkingDirectory = RepositoryPaths.Root;
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.RedirectStandardError = true;
        process.StartInfo.UseShellExecute = false;

        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptName}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        return stdout;
    }
}

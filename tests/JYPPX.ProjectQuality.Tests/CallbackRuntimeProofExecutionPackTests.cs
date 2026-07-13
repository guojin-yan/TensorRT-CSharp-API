using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class CallbackRuntimeProofExecutionPackTests
{
    [Fact]
    public void CallbackRuntimeProofExecutionPackExportsBlockedOwnerActionPlan()
    {
        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-CallbackRuntimeProofExecutionPack.ps1"));
        string validationOutput = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-CallbackRuntimeProofExecutionPack.ps1"));

        Assert.Contains("Callback runtime proof execution pack written", output, StringComparison.Ordinal);
        Assert.Contains("ValidationState=blocked-owner-action-required", validationOutput, StringComparison.Ordinal);

        using JsonDocument packDocument = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "callback-runtime-proof-execution-pack.json")));
        JsonElement pack = packDocument.RootElement;

        Assert.Equal("callback-runtime-proof-execution-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-action-required", pack.GetProperty("packState").GetString());
        Assert.False(pack.GetProperty("performsPublish").GetBoolean());
        Assert.False(pack.GetProperty("performsRuntimeExecution").GetBoolean());
        Assert.False(pack.GetProperty("canPromoteRealCallbackRuntime").GetBoolean());
        Assert.False(pack.GetProperty("isRealCallbackRuntimeProof").GetBoolean());
        Assert.True(pack.GetProperty("missingOwnerInputCount").GetInt32() >= 10);
        Assert.Contains("InvocationCount greater than zero", pack.GetProperty("missingOwnerInputs").EnumerateArray().Select(static item => item.GetString()!));
        Assert.Contains("TensorRtCallbackAllocatorReadinessSnapshot", pack.GetProperty("proofBoundary").GetString()!, StringComparison.Ordinal);
        Assert.Contains("RuntimeEvidenceKind=managed-readiness", pack.GetProperty("proofBoundary").GetString()!, StringComparison.Ordinal);

        string[] nonSubstitutes = pack.GetProperty("nonSubstituteProofKinds").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string item in new[] { "managed-readiness", "TensorRtCallbackAllocatorReadinessSnapshot", "precheck-only", "dry-run-only", "schema-only" })
        {
            Assert.Contains(item, nonSubstitutes);
        }

        using JsonDocument validationDocument = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "callback-runtime-proof-execution-pack-validation.json")));
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("callback-runtime-proof-execution-pack-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-action-required", validation.GetProperty("validationState").GetString());
        Assert.False(validation.GetProperty("canPromoteRealCallbackRuntime").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains("CallbackAllocatorReadinessSnapshot", validation.GetProperty("boundary").GetString()!, StringComparison.Ordinal);
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

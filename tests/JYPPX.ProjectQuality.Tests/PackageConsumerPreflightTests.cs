using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class PackageConsumerPreflightTests
{
    [Fact]
    public void PackageConsumerPreflightChecksMetadataAndKeepsOwnerGateBlocked()
    {
        RunPowerShell("Export-FinalOwnerNextDecisionGate.ps1");
        RunPowerShell("Test-FinalOwnerNextDecisionGate.ps1", "-Strict");
        RunPowerShell("Export-PackageConsumerPreflight.ps1");
        RunPowerShell("Test-PackageConsumerPreflight.ps1", "-Strict");

        using JsonDocument preflightDocument = ReadFinalReleaseJson("package-consumer-preflight.json");
        JsonElement preflight = preflightDocument.RootElement;
        Assert.Equal("package-consumer-preflight", preflight.GetProperty("recordKind").GetString());
        Assert.Equal("package-consumer-preflight-ready-non-proof", preflight.GetProperty("preflightState").GetString());
        Assert.Equal(0, preflight.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(preflight.GetProperty("metadataCheckCount").GetInt32() >= 8);
        Assert.True(preflight.GetProperty("runtimeProjectCount").GetInt32() >= 6);
        Assert.True(preflight.GetProperty("runtimeSplitProjectCount").GetInt32() >= 6);
        Assert.Equal("blocked-final-owner-next-decision-required", preflight.GetProperty("finalOwnerGateState").GetString());
        Assert.Equal("keep-blocked-wait-for-owner", preflight.GetProperty("recommendedOwnerDefault").GetString());
        AssertNonProofFlags(preflight);

        string raw = preflight.GetRawText();
        foreach (string expected in new[]
        {
            "JYPPX.TensorRT.CSharp.API",
            "PackageReadmeFile",
            "RepositoryUrl",
            "JYPPXRuntimeAssetsDir",
            "Test-PackageConsumer.ps1",
            "ReferenceOutputAssembly=false",
            "local feed",
            "ProjectReference",
            "direct .nupkg",
            "sidecar-only report",
            "TensorRtExec report",
            "not package publication",
            "never pushes packages"
        })
        {
            Assert.Contains(expected, raw, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("package-consumer-preflight-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("package-consumer-preflight-validation-ready-non-proof", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        AssertNonProofFlags(validation);
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    private static void AssertNonProofFlags(JsonElement element)
    {
        Assert.False(element.GetProperty("performsPublish").GetBoolean());
        Assert.False(element.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(element.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(element.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(element.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(element.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(element.GetProperty("isReleaseCloseProof").GetBoolean());
        Assert.False(element.GetProperty("canPromotePackageConsumerRuntimeProof").GetBoolean());
    }

    private static void RunPowerShell(string scriptName, params string[] arguments)
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", scriptName);
        using Process process = new Process();
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
        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(
            process.ExitCode == 0,
            $"PowerShell script failed: {scriptName}{Environment.NewLine}STDOUT:{Environment.NewLine}{output}{Environment.NewLine}STDERR:{Environment.NewLine}{error}");
    }
}

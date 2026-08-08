using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalOwnerExecutionRealInputStrictPreflightTests
{
    [Fact]
    public void StrictPreflightReportsActionRequiredWithoutFailingQualityGate()
    {
        RunPowerShell("Export-FinalOwnerExecutionInputSkeleton.ps1");
        RunPowerShell("Export-FinalOwnerExecutionRealInputTemplate.ps1");
        RunPowerShell("Import-FinalOwnerExecutionRealInput.ps1");
        RunPowerShell("Test-FinalOwnerExecutionRealInputStrictPreflight.ps1", "-Strict");

        using JsonDocument preflightDocument = ReadFinalReleaseJson("final-owner-execution-real-input-strict-preflight.json");
        JsonElement preflight = preflightDocument.RootElement;

        Assert.Equal("final-owner-execution-real-input-strict-preflight", preflight.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-real-input-required", preflight.GetProperty("preflightState").GetString());
        Assert.True(preflight.GetProperty("ownerActionRequired").GetBoolean());
        Assert.Equal(49, preflight.GetProperty("checkedFieldCount").GetInt32());
        Assert.True(preflight.GetProperty("findingCount").GetInt32() >= 50);
        Assert.Equal(0, preflight.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(preflight.GetProperty("failedActionRequiredCount").GetInt32() >= 50);
        Assert.Equal(2, preflight.GetProperty("dualPackageRouteProofRouteCount").GetInt32());
        Assert.Equal(8, preflight.GetProperty("dualPackageRouteProofFieldCount").GetInt32());
        Assert.Equal(0, preflight.GetProperty("dualPackageRouteProofReadyFieldCount").GetInt32());
        Assert.Equal(8, preflight.GetProperty("dualPackageRouteProofPlaceholderFieldCount").GetInt32());
        Assert.False(preflight.GetProperty("dualPackageRouteProofReadyForCloseValidation").GetBoolean());
        Assert.False(preflight.GetProperty("readyForCloseValidation").GetBoolean());
        AssertNonProof(preflight);

        string[] routeIds = preflight.GetProperty("dualPackageRouteIds").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("nuget-small-bridge-core", routeIds);
        Assert.Contains("github-packages-bridge", routeIds);

        string[] routeFieldPaths = preflight.GetProperty("dualPackageRouteProofFieldPaths").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("dualPackageRoutes.nugetSmallBridgeCore.ownerAuthorizationUrl", routeFieldPaths);
        Assert.Contains("dualPackageRoutes.githubPackagesFullRuntime.runtimeDllResolutionReportPath", routeFieldPaths);

        string preflightText = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "final-owner-execution-real-input-strict-preflight.json"));
        foreach (string expected in new[]
        {
            "dualPackageRoutes.nugetSmallBridgeCore.ownerAuthorizationUrl",
            "dualPackageRoutes.githubPackagesFullRuntime.runtimeDllResolutionReportPath",
            "nuget-small-bridge-core",
            "github-packages-bridge",
            "placeholder",
            "SHA256 invalid",
            "path missing",
            "local feed",
            "ProjectReference",
            "direct .nupkg",
            "pre-publish smoke reused as post-publish proof",
            "not runtime proof",
            "not post-publish proof",
            "not publish approval",
            "not release close approval",
            "not package push"
        })
        {
            Assert.Contains(expected, preflightText, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void FailOnNotReadyFailsWhenOwnerInputsAreStillPlaceholders()
    {
        RunPowerShell("Export-FinalOwnerExecutionInputSkeleton.ps1");
        RunPowerShell("Export-FinalOwnerExecutionRealInputTemplate.ps1");
        RunPowerShell("Import-FinalOwnerExecutionRealInput.ps1");

        string output = RunPowerShellExpectFailure("Test-FinalOwnerExecutionRealInputStrictPreflight.ps1", "-FailOnNotReady");
        Assert.Contains("not ready", output, StringComparison.OrdinalIgnoreCase);
    }

    private static void AssertNonProof(JsonElement root)
    {
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("performsRuntimeExecution").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(root.GetProperty("isReleaseCloseProof").GetBoolean());
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    private static string RunPowerShell(string scriptName, params string[] arguments)
    {
        using Process process = CreatePowerShellProcess(scriptName, arguments);
        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptName}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        return stdout;
    }

    private static string RunPowerShellExpectFailure(string scriptName, params string[] arguments)
    {
        using Process process = CreatePowerShellProcess(scriptName, arguments);
        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.NotEqual(0, process.ExitCode);
        return stdout + Environment.NewLine + stderr;
    }

    private static Process CreatePowerShellProcess(string scriptName, params string[] arguments)
    {
        Process process = new()
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = PowerShellHost.ResolveExecutable(),
                WorkingDirectory = RepositoryPaths.Root,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            },
        };

        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(Path.Combine(RepositoryPaths.Root, "eng", scriptName));
        foreach (string argument in arguments)
        {
            process.StartInfo.ArgumentList.Add(argument);
        }

        return process;
    }
}

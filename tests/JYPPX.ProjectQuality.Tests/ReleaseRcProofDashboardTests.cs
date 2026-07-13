using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseRcProofDashboardTests
{
    private static readonly string[] RequiredIds =
    {
        "owner-authorization",
        "package-consumer-runtime",
        "callback-runtime-proof",
        "linux-runner-proof",
        "real-model-runtime",
        "post-publish-verification",
        "stale-release-claims",
        "release-close-preflight",
    };

    private static readonly string[] RequiredNonSubstitutes =
    {
        "managed-readiness",
        "CallbackAllocatorReadinessSnapshot",
        "precheck-only",
        "dry-run-only",
        "schema-only",
    };

    [Fact]
    public void ReleaseRcProofDashboardExportsBlockedOwnerExecutableBoard()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseRcProofDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseRcProofDashboard.ps1"));

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "release-rc-proof-dashboard.json")));
        JsonElement root = document.RootElement;

        Assert.Equal("release-rc-proof-dashboard", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-proof-required", root.GetProperty("dashboardState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal(RequiredIds.Length, root.GetProperty("proofBlockerCount").GetInt32());
        Assert.Equal(0, root.GetProperty("readyProofBlockerCount").GetInt32());
        Assert.Contains(root.GetProperty("sourceArtifacts").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/release-owner-proof-input-record-template.json");
        Assert.Contains(root.GetProperty("sourceArtifacts").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/release-owner-proof-input-record-validation.json");

        string[] ids = root.GetProperty("proofBlockers").EnumerateArray().Select(static item => item.GetProperty("id").GetString()!).ToArray();
        Assert.Equal(RequiredIds.Order(StringComparer.Ordinal).ToArray(), ids.Order(StringComparer.Ordinal).ToArray());

        string[] nonSubstitutes = root.GetProperty("nonSubstituteProofKinds").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string item in RequiredNonSubstitutes)
        {
            Assert.Contains(item, nonSubstitutes);
        }

        foreach (JsonElement blocker in root.GetProperty("proofBlockers").EnumerateArray())
        {
            Assert.False(blocker.GetProperty("ready").GetBoolean());
            Assert.False(blocker.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(blocker.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(blocker.GetProperty("performsPublish").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(blocker.GetProperty("requiredValidator").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(blocker.GetProperty("nextCommand").GetString()));
            Assert.True(blocker.GetProperty("requiredOwnerInputs").GetArrayLength() >= 1);
            Assert.True(blocker.GetProperty("sourceArtifacts").GetArrayLength() >= 1);
            Assert.True(blocker.GetProperty("cannotUse").GetArrayLength() >= 1);
            Assert.True(blocker.GetProperty("nonSubstituteProofKinds").GetArrayLength() >= RequiredNonSubstitutes.Length);
        }

        using JsonDocument validation = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "release-rc-proof-dashboard-validation.json")));
        Assert.Equal("blocked-valid-owner-handoff", validation.RootElement.GetProperty("validationState").GetString());
        Assert.True(validation.RootElement.GetProperty("ownerProofInputArtifactsPresent").GetBoolean());
        Assert.False(validation.RootElement.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.RootElement.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    [Fact]
    public void ReleaseRcOwnerHandoffExportsBlockedExecutableOwnerActions()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseRcProofDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseRcProofDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseRcOwnerHandoff.ps1"));

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "release-rc-owner-handoff.json")));
        JsonElement root = document.RootElement;

        Assert.Equal("release-rc-owner-handoff", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-proof-required", root.GetProperty("handoffState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal(RequiredIds.Length, root.GetProperty("actionCount").GetInt32());
        Assert.Equal(0, root.GetProperty("readyActionCount").GetInt32());
        Assert.Contains(root.GetProperty("sourceArtifacts").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/release-owner-proof-input-record-template.json");
        Assert.Contains(root.GetProperty("sourceArtifacts").EnumerateArray(), static item => item.GetString() == "artifacts/final-release/release-owner-proof-input-record-validation.json");

        string[] nonSubstitutes = root.GetProperty("nonSubstituteProofKinds").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string item in RequiredNonSubstitutes)
        {
            Assert.Contains(item, nonSubstitutes);
        }

        foreach (JsonElement action in root.GetProperty("ownerActions").EnumerateArray())
        {
            Assert.False(action.GetProperty("ready").GetBoolean());
            Assert.False(action.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(action.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(action.GetProperty("performsPublish").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(action.GetProperty("ownerAction").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(action.GetProperty("validator").GetString()));
            Assert.True(action.GetProperty("requiredOwnerInputs").GetArrayLength() >= 1);
            Assert.True(action.GetProperty("sourceArtifacts").GetArrayLength() >= 1);
        }
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

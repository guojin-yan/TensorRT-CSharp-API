using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerPublicPublishAuthorizationGateTests
{
    [Fact]
    public void AuthorizationGateStaysFailClosedAndTrackedByEvidenceBundle()
    {
        RunPowerShell("Export-PublicPublishFinalOwnerExecutionPack.ps1");
        RunPowerShell("Test-PublicPublishFinalOwnerExecutionPack.ps1", "-Strict");
        RunPowerShell("Export-OwnerPublicPublishAuthorizationGate.ps1");
        RunPowerShell("Test-OwnerPublicPublishAuthorizationGate.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument gateDocument = ReadFinalReleaseJson("owner-public-publish-authorization-gate.json");
        JsonElement gate = gateDocument.RootElement;
        Assert.Equal("owner-public-publish-authorization-gate", gate.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-public-publish-authorization-required", gate.GetProperty("gateState").GetString());
        Assert.False(gate.GetProperty("ownerAuthorizedPublicPublish").GetBoolean());
        Assert.False(gate.GetProperty("authorizationPhraseMatches").GetBoolean());
        Assert.Equal(string.Empty, gate.GetProperty("materializedExecutableCommand").GetString());
        Assert.False(gate.GetProperty("performsPublish").GetBoolean());
        Assert.False(gate.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(gate.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains("never executes dotnet nuget push", gate.GetProperty("safetyBoundary").GetString(), StringComparison.OrdinalIgnoreCase);

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-public-publish-authorization-gate-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("owner-public-publish-authorization-gate-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-public-publish-authorization-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("blockedActionRequiredCount").GetInt32() > 0);
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("blocked-owner-public-publish-authorization-required", evidence.GetProperty("ownerPublicPublishAuthorizationGateValidationState").GetString());
        Assert.False(evidence.GetProperty("ownerPublicPublishAuthorizationGateCanPublishPublicly").GetBoolean());
        Assert.False(evidence.GetProperty("ownerPublicPublishAuthorizationGateCanCloseReleaseIssue").GetBoolean());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(item => item.GetProperty("id").GetString() == "owner-public-publish-authorization-gate");
        Assert.False(evidenceItem.GetProperty("passed").GetBoolean());
        string boundary = evidenceItem.GetProperty("boundary").GetString()!;
        Assert.Contains("fail-closed", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not workflow dispatch", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package push", boundary, StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts")
            .EnumerateArray()
            .Select(item => item.GetString()!)
            .ToArray();
        Assert.Contains("artifacts/final-release/owner-public-publish-authorization-gate.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-public-publish-authorization-gate-validation.json", sourceArtifacts);
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static string RunPowerShell(string scriptName, params string[] arguments)
    {
        using Process process = new();
        process.StartInfo.FileName = "pwsh";
        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(Path.Combine(RepositoryPaths.Root, "eng", scriptName));
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

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptName}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        return stdout;
    }
}

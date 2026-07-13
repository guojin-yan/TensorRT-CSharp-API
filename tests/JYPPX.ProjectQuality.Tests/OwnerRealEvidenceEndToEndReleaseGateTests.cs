using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerRealEvidenceEndToEndReleaseGateTests
{
    [Fact]
    public void EndToEndGateBlocksUntilEveryRealOwnerEvidenceLaneIsAccepted()
    {
        RunPowerShell("Import-OwnerRealProofStagingWorkspace.ps1");
        RunPowerShell("Test-OwnerRealProofStagingWorkspace.ps1", "-Strict");
        RunPowerShell("Import-OwnerPublicPublishExecutionResultCandidate.ps1");
        RunPowerShell("Test-OwnerPublicPublishExecutionResultCandidate.ps1", "-Strict");
        RunPowerShell("Import-PostPublishCleanConsumerRealProofFromOwnerResult.ps1");
        RunPowerShell("Test-PostPublishCleanConsumerRealProofFromOwnerResult.ps1", "-Strict");
        RunPowerShell("Test-FinalPublicPublishAcceptanceGate.ps1", "-Strict");
        RunPowerShell("Test-FinalOwnerRealProofConvergenceGate.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");
        string output = RunPowerShell("Test-OwnerRealEvidenceEndToEndReleaseGate.ps1", "-Strict");

        Assert.Contains("OwnerRealEvidenceEndToEndReleaseGateState=blocked-owner-real-evidence-end-to-end-release-required", output, StringComparison.Ordinal);

        using JsonDocument gateDocument = ReadFinalReleaseJson("owner-real-evidence-end-to-end-release-gate.json");
        JsonElement gate = gateDocument.RootElement;

        Assert.Equal("owner-real-evidence-end-to-end-release-gate", gate.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-evidence-end-to-end-release-required", gate.GetProperty("gateState").GetString());
        Assert.True(gate.GetProperty("blockedGateCheckCount").GetInt32() >= 5);
        Assert.False(gate.GetProperty("readyForReleaseClose").GetBoolean());
        Assert.True(gate.GetProperty("failedBlockerCountIsNotProof").GetBoolean());
        Assert.False(gate.GetProperty("performsPublish").GetBoolean());
        Assert.False(gate.GetProperty("approvesPublicRelease").GetBoolean());
        Assert.False(gate.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(gate.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(gate.GetProperty("isPostPublishProof").GetBoolean());
        Assert.Contains("never executes dotnet nuget push", gate.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("failedBlockerCount=0", gate.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        Assert.Contains(gate.GetProperty("checks").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "owner-real-proof-staging-workspace-ready" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(gate.GetProperty("checks").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "owner-public-publish-execution-result-accepted" &&
            !item.GetProperty("passed").GetBoolean());
        Assert.Contains(gate.GetProperty("checks").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "post-publish-clean-consumer-real-proof-accepted" &&
            !item.GetProperty("passed").GetBoolean());
    }

    [Fact]
    public void ReleaseEvidenceAndClassificationAuditCarryEndToEndGateAsNonProof()
    {
        RunPowerShell("Test-OwnerRealEvidenceEndToEndReleaseGate.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument bundleDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement bundle = bundleDocument.RootElement;
        Assert.Contains(bundle.GetProperty("evidenceItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "owner-real-evidence-end-to-end-release-gate" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("boundary").GetString()!.Contains("not package push", StringComparison.OrdinalIgnoreCase));

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Contains(audit.GetProperty("auditedItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "owner-real-evidence-end-to-end-release-gate" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("hasNonProofBoundary").GetBoolean());
        Assert.Contains(audit.GetProperty("requiredNonSubstituteMarkers").EnumerateArray(), static marker =>
            marker.GetString() == "owner real evidence end-to-end release gate");
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    private static string RunPowerShell(string scriptName, params string[] arguments)
    {
        using Process process = new()
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "pwsh",
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

        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptName}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        return stdout;
    }
}

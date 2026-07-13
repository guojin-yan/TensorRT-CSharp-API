using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalOwnerRealProofConvergenceGateTests
{
    [Fact]
    public void ConvergenceGateDefaultsToBlockedAndDoesNotTreatZeroFailedBlockersAsProof()
    {
        RunPowerShell("Test-FinalOwnerRealProofConvergenceGate.ps1", "-Strict");

        using JsonDocument gateDocument = ReadFinalReleaseJson("final-owner-real-proof-convergence-gate.json");
        JsonElement gate = gateDocument.RootElement;

        Assert.Equal("final-owner-real-proof-convergence-gate", gate.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-real-proof-convergence-required", gate.GetProperty("gateState").GetString());
        Assert.False(gate.GetProperty("readyForFinalClose").GetBoolean());
        Assert.True(gate.GetProperty("ownerActionRequired").GetBoolean());
        Assert.True(gate.GetProperty("failedBlockerCountIsNotProof").GetBoolean());
        Assert.True(gate.GetProperty("blockedGateCheckCount").GetInt32() >= 5);
        Assert.False(gate.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(gate.GetProperty("isReleaseCloseProof").GetBoolean());
        Assert.False(gate.GetProperty("performsPublish").GetBoolean());

        string[] checks = gate.GetProperty("checks").EnumerateArray().Select(static check => check.GetProperty("id").GetString()!).ToArray();
        foreach (string expected in new[]
        {
            "external-clean-consumer-proof-candidate-ready",
            "post-publish-clean-consumer-proof-candidate-ready",
            "rollback-review-present",
            "final-close-decision-present",
            "release-evidence-classification-audit-clean",
            "final-owner-close-readiness-all-checks-pass"
        })
        {
            Assert.Contains(expected, checks);
        }
    }

    [Fact]
    public void ReleaseEvidenceIncludesFinalOwnerRealProofItemsAsFailedNonProof()
    {
        RunPowerShell("Export-FinalOwnerRealProofExecutionPackage.ps1");
        RunPowerShell("Test-FinalOwnerRealProofExecutionPackage.ps1", "-Strict");
        RunPowerShell("Export-FinalOwnerRealProofGapMatrix.ps1");
        RunPowerShell("Test-FinalOwnerRealProofGapMatrix.ps1", "-Strict");
        RunPowerShell("Test-FinalOwnerRealProofConvergenceGate.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument bundleDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement bundle = bundleDocument.RootElement;

        foreach (string id in new[]
        {
            "final-owner-real-proof-execution-package",
            "final-owner-real-proof-gap-matrix",
            "final-owner-real-proof-convergence-gate"
        })
        {
            JsonElement item = bundle.GetProperty("evidenceItems").EnumerateArray().Single(item => item.GetProperty("id").GetString() == id);
            Assert.False(item.GetProperty("passed").GetBoolean());
            string boundary = item.GetProperty("boundary").GetString()!;
            Assert.Contains("not runtime proof", boundary, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("not post-publish proof", boundary, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("not publish approval", boundary, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("not release close approval", boundary, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("not package push", boundary, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
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

using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerRealEvidenceFinalIntakeChecklistTests
{
    [Fact]
    public void FinalIntakeChecklistDocumentsAllOwnerEvidenceLanesWithoutBecomingProof()
    {
        RunPowerShell("Export-OwnerRealEvidenceFinalIntakeChecklist.ps1");
        string output = RunPowerShell("Test-OwnerRealEvidenceFinalIntakeChecklist.ps1", "-Strict");

        Assert.Contains("OwnerRealEvidenceFinalIntakeChecklistValidationState=owner-real-evidence-final-intake-checklist-ready-non-proof", output, StringComparison.Ordinal);

        using JsonDocument checklistDocument = ReadFinalReleaseJson("owner-real-evidence-final-intake-checklist.json");
        JsonElement checklist = checklistDocument.RootElement;

        Assert.Equal("owner-real-evidence-final-intake-checklist", checklist.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-evidence-final-intake-required", checklist.GetProperty("checklistState").GetString());
        Assert.True(checklist.GetProperty("checklistItemCount").GetInt32() >= 15);
        Assert.True(checklist.GetProperty("failedBlockerCountIsNotProof").GetBoolean());
        Assert.False(checklist.GetProperty("performsPublish").GetBoolean());
        Assert.False(checklist.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(checklist.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(checklist.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(checklist.GetProperty("isPostPublishProof").GetBoolean());
        Assert.Contains("never executes dotnet nuget push", checklist.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("failedBlockerCount=0", checklist.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string[] groups = checklist.GetProperty("checklistItems")
            .EnumerateArray()
            .Select(static item => item.GetProperty("group").GetString()!)
            .Distinct(StringComparer.Ordinal)
            .ToArray();

        foreach (string group in new[]
        {
            "external-clean-consumer",
            "post-publish-clean-consumer",
            "public-package",
            "host-metadata",
            "package-metadata",
            "owner-governance",
            "owner-confirmations",
            "final-gates"
        })
        {
            Assert.Contains(group, groups);
        }

        string[] nonSubstitutes = checklist.GetProperty("nonSubstituteProofKinds")
            .EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        foreach (string marker in new[] { "local feed", "ProjectReference", "direct .nupkg", "pre-publish smoke", "template", "candidate", "dashboard", "runbook" })
        {
            Assert.Contains(marker, nonSubstitutes);
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("owner-real-evidence-final-intake-checklist-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("owner-real-evidence-final-intake-checklist-ready-non-proof", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() >= 15);
    }

    [Fact]
    public void ReleaseEvidenceAndClassificationAuditCarryFinalIntakeChecklistAsNonProof()
    {
        RunPowerShell("Export-OwnerRealEvidenceFinalIntakeChecklist.ps1");
        RunPowerShell("Test-OwnerRealEvidenceFinalIntakeChecklist.ps1", "-Strict");
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument bundleDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement bundle = bundleDocument.RootElement;
        Assert.Contains(bundle.GetProperty("evidenceItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "owner-real-evidence-final-intake-checklist" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("boundary").GetString()!.Contains("not package push", StringComparison.OrdinalIgnoreCase));

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Contains(audit.GetProperty("auditedItems").EnumerateArray(), static item =>
            item.GetProperty("id").GetString() == "owner-real-evidence-final-intake-checklist" &&
            !item.GetProperty("passed").GetBoolean() &&
            item.GetProperty("hasNonProofBoundary").GetBoolean());
        Assert.Contains(audit.GetProperty("requiredNonSubstituteMarkers").EnumerateArray(), static marker =>
            marker.GetString() == "owner real evidence final intake checklist");
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

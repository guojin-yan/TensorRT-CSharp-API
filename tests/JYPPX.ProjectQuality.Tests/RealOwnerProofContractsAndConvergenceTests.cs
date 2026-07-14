using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class RealOwnerProofContractsAndConvergenceTests
{
    [Fact]
    public void UnifiedContractsAndConvergenceCoverNineOwnerProofLanes()
    {
        RunPowerShell("Export-RealOwnerProofConvergenceDashboard.ps1");
        RunPowerShell("Test-RealOwnerProofConvergenceDashboard.ps1", "-Strict");

        using JsonDocument contractsDocument = ReadFinalReleaseJson("real-owner-proof-input-contracts.json");
        JsonElement contracts = contractsDocument.RootElement;
        Assert.Equal("real-owner-proof-input-contracts-ready-non-proof", contracts.GetProperty("contractState").GetString());
        Assert.Equal(9, contracts.GetProperty("laneCount").GetInt32());
        Assert.True(contracts.GetProperty("requiredFieldCount").GetInt32() >= 100);
        JsonElement closeContract = contracts.GetProperty("contracts").EnumerateArray().Single(item => item.GetProperty("id").GetString() == "final-close-decision");
        string[] closeFields = closeContract.GetProperty("requiredFields").EnumerateArray().Select(item => item.GetString()!).ToArray();
        Assert.Contains("releaseIssueUrl", closeFields);
        Assert.Contains("postPublishProofUrls", closeFields);
        Assert.Contains("confirmsNoIssueCloseByAutomation", closeFields);
        AssertFalseAggregateProofFlags(contracts);

        using JsonDocument preflightDocument = ReadFinalReleaseJson("real-owner-proof-admission-preflight.json");
        JsonElement preflight = preflightDocument.RootElement;
        Assert.Equal(9, preflight.GetProperty("laneCount").GetInt32());
        Assert.Equal(9, preflight.GetProperty("structuralReadyLaneCount").GetInt32());
        Assert.Equal(0, preflight.GetProperty("structuralBlockedLaneCount").GetInt32());
        Assert.Equal(9, preflight.GetProperty("lanes").GetArrayLength());
        Assert.Contains(preflight.GetProperty("rejectedNonProofSubstitutes").EnumerateArray(), item => item.GetString() == "sample-build-only");
        Assert.Contains(preflight.GetProperty("rejectedNonProofSubstitutes").EnumerateArray(), item => item.GetString() == "mock-output");
        AssertFalseAggregateProofFlags(preflight);

        using JsonDocument dashboardDocument = ReadFinalReleaseJson("real-owner-proof-convergence-dashboard.json");
        JsonElement dashboard = dashboardDocument.RootElement;
        Assert.Equal("real-owner-proof-convergence-dashboard", dashboard.GetProperty("recordKind").GetString());
        Assert.Equal(9, dashboard.GetProperty("laneCount").GetInt32());
        Assert.Equal(3, dashboard.GetProperty("gateCount").GetInt32());
        Assert.Equal(3, dashboard.GetProperty("categorySummaries").GetArrayLength());
        Assert.False(dashboard.GetProperty("allRealOwnerProofInputsAccepted").GetBoolean());
        Assert.True(dashboard.GetProperty("blockedLaneCount").GetInt32() > 0);
        Assert.Contains("not package push", dashboard.GetProperty("boundary").GetString()!, StringComparison.OrdinalIgnoreCase);
        AssertFalseAggregateProofFlags(dashboard);
    }

    [Fact]
    public void AcceptedCloseDecisionFixtureStillCannotCloseReleaseIssue()
    {
        string inputPath = Path.Combine(Path.GetTempPath(), $"tensor-rt-sharp-final-close-{Guid.NewGuid():N}.json");
        string hashA = new('a', 64);
        string hashB = new('b', 64);
        string hashC = new('c', 64);
        string hashD = new('d', 64);
        string hashE = new('e', 64);
        object ownerInput = new
        {
            recordKind = "synthetic-final-owner-close-decision-fixture",
            reviewer = "project-owner-fixture",
            reviewedAtUtc = "2026-07-14T08:00:00Z",
            ownerReviewed = true,
            decision = "approved-for-manual-close",
            closeReason = "Synthetic fixture verifies admission without issue-close authority.",
            acceptedRisk = "Synthetic fixture only; no public release claim.",
            rollbackPlan = "Use the reviewed rollback plan if Owner later executes a real release.",
            packageId = "JYPPX.TensorRT.CSharp.API",
            packageVersion = "4.0.0-fixture",
            releaseIssueId = "999999",
            releaseIssueUrl = "https://github.com/guojin-yan/TensorRT-CSharp-API/issues/999999",
            evidenceBundleSha256 = hashA,
            classificationAuditSha256 = hashB,
            publicPackageProofSha256 = hashC,
            externalCleanConsumerProofSha256 = hashD,
            postPublishProofSha256 = hashE,
            postPublishProofUrls = new[] { "https://github.com/guojin-yan/TensorRT-CSharp-API/actions/runs/999999" },
            externalCleanConsumerProofReady = true,
            postPublishProofReady = true,
            rollbackReviewReady = true,
            classificationAuditPassed = true,
            manualCloseOnlyConfirmation = true,
            confirmsNoIssueCloseByAutomation = true,
        };

        File.WriteAllText(inputPath, JsonSerializer.Serialize(ownerInput));
        try
        {
            RunPowerShell("Import-FinalOwnerCloseDecision.ps1", "-OwnerInputPath", inputPath);
            RunPowerShell("Test-FinalOwnerCloseDecision.ps1", "-Strict");

            using JsonDocument document = ReadFinalReleaseJson("final-owner-close-decision-import.json");
            JsonElement record = document.RootElement;
            Assert.Equal("final-owner-close-decision-ready", record.GetProperty("importState").GetString());
            Assert.True(record.GetProperty("finalCloseDecisionReady").GetBoolean());
            Assert.True(record.GetProperty("closeDecisionAdmissionReady").GetBoolean());
            Assert.True(record.GetProperty("dependenciesConfirmed").GetBoolean());
            Assert.True(record.GetProperty("manualCloseOnly").GetBoolean());
            Assert.True(record.GetProperty("issueCloseExecutionForbidden").GetBoolean());
            Assert.False(record.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(record.GetProperty("isReleaseCloseProof").GetBoolean());
            Assert.False(record.GetProperty("passed").GetBoolean());
            Assert.Contains("does not close the release issue", record.GetProperty("boundary").GetString()!, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            File.Delete(inputPath);
            RunPowerShell("Import-FinalOwnerCloseDecision.ps1");
            RunPowerShell("Test-FinalOwnerCloseDecision.ps1", "-Strict");
        }
    }

    [Fact]
    public void ReleaseEvidenceBundleAndClassificationAuditTrackUnifiedOwnerProofAdmission()
    {
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument bundleDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement bundle = bundleDocument.RootElement;
        Assert.Equal("real-owner-proof-input-contracts-validation-ready-non-proof", bundle.GetProperty("realOwnerProofInputContractsValidationState").GetString());
        Assert.Equal(9, bundle.GetProperty("realOwnerProofInputContractsLaneCount").GetInt32());
        Assert.True(bundle.GetProperty("realOwnerProofInputContractsRequiredFieldCount").GetInt32() >= 100);
        Assert.Equal("real-owner-proof-admission-preflight-validation-ready-non-proof", bundle.GetProperty("realOwnerProofAdmissionPreflightValidationState").GetString());
        Assert.Equal(9, bundle.GetProperty("realOwnerProofAdmissionPreflightStructuralReadyLaneCount").GetInt32());
        Assert.Equal("real-owner-proof-convergence-dashboard-validation-ready-non-proof", bundle.GetProperty("realOwnerProofConvergenceDashboardValidationState").GetString());
        Assert.Equal(9, bundle.GetProperty("realOwnerProofConvergenceDashboardLaneCount").GetInt32());

        string[] ids =
        {
            "real-owner-proof-input-contracts",
            "real-owner-proof-admission-preflight",
            "real-owner-proof-convergence-dashboard",
        };
        string[] sourceArtifacts = bundle.GetProperty("sourceArtifacts").EnumerateArray().Select(item => item.GetString()!).ToArray();
        foreach (string id in ids)
        {
            Assert.Contains($"artifacts/final-release/{id}.json", sourceArtifacts);
            Assert.Contains($"artifacts/final-release/{id}-validation.json", sourceArtifacts);
            Assert.Contains(
                bundle.GetProperty("evidenceItems").EnumerateArray(),
                item => item.GetProperty("id").GetString() == id
                    && !item.GetProperty("passed").GetBoolean()
                    && item.GetProperty("boundary").GetString()!.Contains("not package push", StringComparison.OrdinalIgnoreCase));
        }

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        foreach (string marker in new[]
        {
            "real owner proof input contracts",
            "real owner proof admission preflight",
            "real owner proof convergence dashboard",
        })
        {
            Assert.Contains(audit.GetProperty("requiredNonSubstituteMarkers").EnumerateArray(), item => item.GetString() == marker);
        }

        foreach (string id in ids)
        {
            Assert.Contains(
                audit.GetProperty("auditedItems").EnumerateArray(),
                item => item.GetProperty("id").GetString() == id
                    && !item.GetProperty("passed").GetBoolean()
                    && item.GetProperty("hasNonProofBoundary").GetBoolean());
        }
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", fileName)));
    }

    private static void AssertFalseAggregateProofFlags(JsonElement root)
    {
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("usesPublishToken").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(root.GetProperty("isReleaseCloseProof").GetBoolean());
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

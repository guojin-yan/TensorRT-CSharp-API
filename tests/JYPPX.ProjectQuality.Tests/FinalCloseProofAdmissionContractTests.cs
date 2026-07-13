using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalCloseProofAdmissionContractTests
{
    [Fact]
    public void FinalCloseProofAdmissionContractRequiresStrictAcceptedRealEvidence()
    {
        RunPowerShell("Export-DualPackagePublishPreflightMatrix.ps1");
        RunPowerShell("Test-DualPackagePublishPreflightMatrix.ps1", "-Strict");
        RunPowerShell("Export-FinalCloseGateConvergence.ps1");
        RunPowerShell("Test-FinalCloseGateConvergence.ps1", "-Strict");

        using JsonDocument convergenceDocument = ReadFinalReleaseJson("final-close-gate-convergence.json");
        JsonElement convergence = convergenceDocument.RootElement;
        Assert.Equal("final-close-gate-convergence", convergence.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-close-gate-owner-proof-required", convergence.GetProperty("convergenceState").GetString());
        Assert.Equal(RequiredAdmissionLaneIds.Length, convergence.GetProperty("acceptedProofAdmissionContractCount").GetInt32());
        Assert.False(convergence.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(convergence.GetProperty("canCloseReleaseIssue").GetBoolean());

        string[] admissionLaneIds = ReadStringArray(convergence, "acceptedProofAdmissionContractLaneIds");
        foreach (string laneId in RequiredAdmissionLaneIds)
        {
            Assert.Contains(laneId, admissionLaneIds);
        }

        string[] requiredFields = ReadStringArray(convergence, "finalCloseProofAdmissionRequiredFields");
        foreach (string field in RequiredAdmissionFields)
        {
            Assert.Contains(field, requiredFields);
        }

        string[] rejectedStates = ReadStringArray(convergence, "rejectedNonProofStates");
        foreach (string state in RequiredRejectedStates)
        {
            Assert.Contains(state, rejectedStates);
        }

        JsonElement[] contracts = convergence.GetProperty("acceptedProofAdmissionContract").EnumerateArray().ToArray();
        Assert.Equal(RequiredAdmissionLaneIds.Length, contracts.Length);
        foreach (JsonElement contract in contracts)
        {
            string laneId = contract.GetProperty("laneId").GetString()!;
            Assert.Contains(laneId, RequiredAdmissionLaneIds);
            Assert.StartsWith("accepted-", contract.GetProperty("requiredAcceptedState").GetString(), StringComparison.Ordinal);
            Assert.False(string.IsNullOrWhiteSpace(contract.GetProperty("strictValidator").GetString()));
            Assert.True(contract.GetProperty("acceptedOnlyAfterStrictValidator").GetBoolean());
            Assert.False(contract.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.True(ReadStringArray(contract, "requiredEvidenceFields").Length >= 5);

            string[] contractRejectedStates = ReadStringArray(contract, "rejectsNonProofStates");
            foreach (string state in RequiredRejectedStates)
            {
                Assert.Contains(state, contractRejectedStates);
            }
        }

        using JsonDocument validationDocument = ReadFinalReleaseJson("final-close-gate-convergence-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("blocked-final-close-gate-owner-proof-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(RequiredAdmissionLaneIds.Length, validation.GetProperty("acceptedProofAdmissionContractCount").GetInt32());
        Assert.Equal(RequiredAdmissionFields.Length, validation.GetProperty("finalCloseProofAdmissionRequiredFieldCount").GetInt32());
        Assert.Equal(RequiredRejectedStates.Length, validation.GetProperty("rejectedNonProofStateCount").GetInt32());
        AssertValidationItemPassed(validation, "accepted-proof-admission-contract");
        AssertValidationItemPassed(validation, "required-proof-field-contract");
        AssertValidationItemPassed(validation, "rejected-non-proof-states");

        string markdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "final-close-gate-convergence.md"));
        Assert.Contains("## Proof Admission Contract", markdown, StringComparison.Ordinal);
        foreach (string laneId in RequiredAdmissionLaneIds)
        {
            Assert.Contains(laneId, markdown, StringComparison.Ordinal);
        }

        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal(RequiredAdmissionLaneIds.Length, evidence.GetProperty("finalCloseGateConvergenceAcceptedProofAdmissionContractCount").GetInt32());
        Assert.Equal(RequiredAdmissionFields.Length, evidence.GetProperty("finalCloseGateConvergenceRequiredProofFieldCount").GetInt32());
        Assert.Equal(RequiredRejectedStates.Length, evidence.GetProperty("finalCloseGateConvergenceRejectedNonProofStateCount").GetInt32());

        JsonElement evidenceItem = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(static item => item.GetProperty("id").GetString() == "final-close-gate-convergence");
        string evidenceState = evidenceItem.GetProperty("state").GetString()!;
        Assert.Contains("acceptedProofAdmissionContracts=5", evidenceState, StringComparison.Ordinal);
        Assert.Contains("requiredProofFields=20", evidenceState, StringComparison.Ordinal);
        Assert.Contains("rejectedNonProofStates=10", evidenceState, StringComparison.Ordinal);
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static string[] ReadStringArray(JsonElement element, string propertyName)
    {
        return element.GetProperty(propertyName).EnumerateArray().Select(static item => item.GetString()!).ToArray();
    }

    private static void AssertValidationItemPassed(JsonElement validation, string id)
    {
        Assert.Contains(validation.GetProperty("validationItems").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == id &&
            item.GetProperty("passed").GetBoolean());
    }

    private static void RunPowerShell(string scriptName, params string[] arguments)
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", scriptName);
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(scriptPath, arguments);
    }

    private static readonly string[] RequiredAdmissionLaneIds =
    [
        "github-actions-run-proof",
        "owner-public-publish-result",
        "public-package-download-proof",
        "post-publish-clean-consumer-proof",
        "release-issue-close-record-strict-validation",
    ];

    private static readonly string[] RequiredAdmissionFields =
    [
        "publicPackageSourceUrl",
        "publicPackageDownloadUrl",
        "managedNupkgSha256",
        "runtimeNupkgSha256",
        "externalCleanConsumerProjectIdentity",
        "smokeCommandRuntimePackageKey",
        "hostCudaVersion",
        "hostTensorRtVersion",
        "hostCudnnVersion",
        "stdoutSha256",
        "stderrSha256",
        "mergedTranscriptSha256",
        "githubRunId",
        "githubHeadSha",
        "githubLogSha256",
        "githubArtifactSha256",
        "ownerReviewer",
        "ownerAuthorizationLink",
        "rollbackReview",
        "finalCloseDecision",
    ];

    private static readonly string[] RequiredRejectedStates =
    [
        "template-only",
        "candidate-only",
        "draft-rich-but-not-proof",
        "draft-blocked-by-cuda-driver",
        "not-requested",
        "validation-ready-without-proof-candidate",
        "dashboard-only",
        "runbook-only",
        "local-feed-only",
        "project-reference-only",
    ];
}

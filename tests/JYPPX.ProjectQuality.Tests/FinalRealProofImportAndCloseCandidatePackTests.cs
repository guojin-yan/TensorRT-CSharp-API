using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalRealProofImportAndCloseCandidatePackTests
{
    [Fact]
    public void FinalRealProofSweepAndCloseCandidateRemainBlockedWithoutOwnerInputs()
    {
        RunPipeline();

        using JsonDocument sweepDocument = ReadFinalReleaseJson("final-real-proof-input-availability-sweep.json");
        JsonElement sweep = sweepDocument.RootElement;
        Assert.Equal("final-real-proof-input-availability-sweep", sweep.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-real-proof-inputs-required", sweep.GetProperty("sweepState").GetString());
        Assert.Equal(12, sweep.GetProperty("phaseCount").GetInt32());
        Assert.Equal(0, sweep.GetProperty("availableRealInputPhaseCount").GetInt32());
        Assert.Equal(0, sweep.GetProperty("acceptedRealInputPhaseCount").GetInt32());
        Assert.Equal(12, sweep.GetProperty("missingRealInputPhaseCount").GetInt32());
        Assert.Equal(0, sweep.GetProperty("proofReadyPhaseCount").GetInt32());
        Assert.False(sweep.GetProperty("closeCandidateReady").GetBoolean());
        AssertFalseProofPublishCloseFlags(sweep);

        string[] forbiddenKinds = sweep.GetProperty("forbiddenSubstituteKinds").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        foreach (string required in RequiredForbiddenSubstitutes)
        {
            Assert.Contains(required, forbiddenKinds);
        }

        foreach (JsonElement phase in sweep.GetProperty("phases").EnumerateArray())
        {
            Assert.Equal("missing-real-owner-input", phase.GetProperty("phaseState").GetString());
            Assert.True(phase.GetProperty("blocked").GetBoolean());
            Assert.True(phase.GetProperty("missingRealInput").GetBoolean());
            Assert.False(phase.GetProperty("proofReady").GetBoolean());
            Assert.All(
                phase.GetProperty("acceptedOwnerInputPaths").EnumerateArray(),
                path =>
                {
                    string value = path.GetString()!;
                    Assert.EndsWith(".json", value, StringComparison.OrdinalIgnoreCase);
                    Assert.False(value.EndsWith(".template.json", StringComparison.OrdinalIgnoreCase));
                    Assert.False(value.EndsWith(".example.json", StringComparison.OrdinalIgnoreCase));
                    Assert.False(value.EndsWith(".draft.json", StringComparison.OrdinalIgnoreCase));
                    Assert.False(value.EndsWith(".misuse.json", StringComparison.OrdinalIgnoreCase));
                    Assert.False(value.EndsWith(".ready.json", StringComparison.OrdinalIgnoreCase));
                    Assert.False(value.EndsWith("-validation.json", StringComparison.OrdinalIgnoreCase));
                });
        }

        using JsonDocument sweepValidationDocument = ReadFinalReleaseJson("final-real-proof-input-availability-sweep-validation.json");
        JsonElement sweepValidation = sweepValidationDocument.RootElement;
        Assert.Equal("final-real-proof-input-availability-sweep-ready-non-proof", sweepValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, sweepValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(sweepValidation.GetProperty("closeCandidateReady").GetBoolean());
        AssertFalseProofPublishCloseFlags(sweepValidation);

        using JsonDocument candidateDocument = ReadFinalReleaseJson("final-real-proof-import-and-close-candidate-pack.json");
        JsonElement candidate = candidateDocument.RootElement;
        Assert.Equal("final-real-proof-import-and-close-candidate-pack", candidate.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-real-proof-import-and-close-candidate-owner-evidence-required", candidate.GetProperty("candidateState").GetString());
        Assert.Equal(12, candidate.GetProperty("phaseCount").GetInt32());
        Assert.Equal(0, candidate.GetProperty("acceptedProofPhaseCount").GetInt32());
        Assert.Equal(12, candidate.GetProperty("missingProofPhaseCount").GetInt32());
        Assert.Equal(0, candidate.GetProperty("invalidProofPhaseCount").GetInt32());
        Assert.False(candidate.GetProperty("closeCandidateReady").GetBoolean());
        Assert.True(candidate.GetProperty("closeDecisionBlockedByFinalBridge").GetBoolean());
        AssertFalseProofPublishCloseFlags(candidate);

        JsonElement closeDecisionPhase = candidate.GetProperty("phases").EnumerateArray()
            .Single(static phase => phase.GetProperty("id").GetString() == "release-issue-close-owner-decision");
        string[] closeDecisionBlockers = closeDecisionPhase.GetProperty("blockedReasons").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("approved-close-decision-rejected-while-final-bridge-blocked", closeDecisionBlockers);

        using JsonDocument candidateValidationDocument = ReadFinalReleaseJson("final-real-proof-import-and-close-candidate-pack-validation.json");
        JsonElement candidateValidation = candidateValidationDocument.RootElement;
        Assert.Equal("final-real-proof-import-and-close-candidate-pack-ready-non-proof", candidateValidation.GetProperty("validationState").GetString());
        Assert.Equal(0, candidateValidation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(candidateValidation.GetProperty("closeCandidateReady").GetBoolean());
        Assert.True(candidateValidation.GetProperty("closeDecisionBlockedByFinalBridge").GetBoolean());
        AssertFalseProofPublishCloseFlags(candidateValidation);
    }

    [Fact]
    public void FinalRealProofArtifactsAreInEvidenceBundleAndClassificationAudit()
    {
        RunPipeline();
        RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
        RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");

        using JsonDocument evidenceDocument = ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;
        Assert.Equal("final-real-proof-input-availability-sweep-ready-non-proof", evidence.GetProperty("finalRealProofInputAvailabilitySweepValidationState").GetString());
        Assert.Equal(12, evidence.GetProperty("finalRealProofInputAvailabilitySweepMissingRealInputPhaseCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("finalRealProofInputAvailabilitySweepProofReadyPhaseCount").GetInt32());
        Assert.False(evidence.GetProperty("finalRealProofInputAvailabilitySweepCanCloseReleaseIssue").GetBoolean());
        Assert.Equal("final-real-proof-import-and-close-candidate-pack-ready-non-proof", evidence.GetProperty("finalRealProofImportAndCloseCandidatePackValidationState").GetString());
        Assert.Equal(0, evidence.GetProperty("finalRealProofImportAndCloseCandidatePackAcceptedProofPhaseCount").GetInt32());
        Assert.Equal(12, evidence.GetProperty("finalRealProofImportAndCloseCandidatePackMissingProofPhaseCount").GetInt32());
        Assert.False(evidence.GetProperty("finalRealProofImportAndCloseCandidatePackCloseCandidateReady").GetBoolean());
        Assert.False(evidence.GetProperty("finalRealProofImportAndCloseCandidatePackCanCloseReleaseIssue").GetBoolean());

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("artifacts/final-release/final-real-proof-input-availability-sweep.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-real-proof-input-availability-sweep-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-real-proof-import-and-close-candidate-pack.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/final-real-proof-import-and-close-candidate-pack-validation.json", sourceArtifacts);

        string[] nonSubstituteMarkers = evidence.GetProperty("nonSubstituteProofKinds").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("final real proof input availability sweep", nonSubstituteMarkers);
        Assert.Contains("final real proof import and close candidate pack", nonSubstituteMarkers);

        foreach (string id in new[] { "final-real-proof-input-availability-sweep", "final-real-proof-import-and-close-candidate-pack" })
        {
            JsonElement item = evidence.GetProperty("evidenceItems").EnumerateArray()
                .Single(evidenceItem => evidenceItem.GetProperty("id").GetString() == id);
            Assert.False(item.GetProperty("passed").GetBoolean());
            Assert.Contains("not runtime proof", item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Contains("not release close approval", item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument auditDocument = ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());
        foreach (string id in new[] { "final-real-proof-input-availability-sweep", "final-real-proof-import-and-close-candidate-pack" })
        {
            Assert.Contains(
                audit.GetProperty("auditedItems").EnumerateArray(),
                item => item.GetProperty("id").GetString() == id
                    && item.GetProperty("passed").GetBoolean() == false
                    && item.GetProperty("hasNonProofBoundary").GetBoolean());
        }
    }

    [Fact]
    public void FinalRealProofSweepRejectsMisuseAndFixtureFilesAsProof()
    {
        RunPipeline();

        using JsonDocument sweepDocument = ReadFinalReleaseJson("final-real-proof-input-availability-sweep.json");
        JsonElement sweep = sweepDocument.RootElement;
        Assert.True(sweep.GetProperty("forbiddenSubstituteCount").GetInt32() > 0);

        string allForbiddenFiles = string.Join(
            '\n',
            sweep.GetProperty("forbiddenSubstituteFiles").EnumerateArray()
                .Select(static item => item.GetProperty("path").GetString()));
        Assert.Contains("public-package-download-proof-input.misuse.json", allForbiddenFiles, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("release-issue-close-owner-decision-input.misuse.json", allForbiddenFiles, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("template", allForbiddenFiles, StringComparison.OrdinalIgnoreCase);

        string[] forbiddenKinds = sweep.GetProperty("forbiddenSubstituteKinds").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        Assert.Contains("local feed", forbiddenKinds);
        Assert.Contains("ProjectReference", forbiddenKinds);
        Assert.Contains("direct .nupkg", forbiddenKinds);
        Assert.Contains("queued workflow", forbiddenKinds);
        Assert.Contains("missing runner", forbiddenKinds);
        Assert.Contains("dry-run", forbiddenKinds);

        foreach (JsonElement phase in sweep.GetProperty("phases").EnumerateArray())
        {
            Assert.Equal(0, phase.GetProperty("acceptedRealInputFileCount").GetInt32());
            Assert.False(phase.GetProperty("proofReady").GetBoolean());
        }
    }

    private static readonly string[] RequiredForbiddenSubstitutes =
    {
        "local feed",
        "ProjectReference",
        "direct .nupkg",
        "queued workflow",
        "missing runner",
        "dry-run",
        "template",
        "dashboard",
        "audit",
        "bundle",
    };

    private static void RunPipeline()
    {
        RunPowerShell("Export-OwnerPublicPublishExecutionFinalIntakePack.ps1");
        RunPowerShell("Test-OwnerPublicPublishExecutionFinalIntakePack.ps1", "-Strict");
        RunPowerShell("Export-FinalRealProofInputAvailabilitySweep.ps1");
        RunPowerShell("Test-FinalRealProofInputAvailabilitySweep.ps1", "-Strict");
        RunPowerShell("Export-FinalRealProofImportAndCloseCandidatePack.ps1");
        RunPowerShell("Test-FinalRealProofImportAndCloseCandidatePack.ps1", "-Strict");
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static void AssertFalseProofPublishCloseFlags(JsonElement root)
    {
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
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

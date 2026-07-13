using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class FinalPublishProofGateAndOwnerExecutionPackTests
{
    [Fact]
    public void YoloVisionOwnerProofExecutionPackExportsRequiredFieldDeltasAndBoundaries()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionRealAssetOwnerProofExecutionPack.ps1"));

        using JsonDocument packDocument = ReadJson("artifacts", "user-acceptance", "yolovision-real-asset-owner-proof-execution-pack.json");
        JsonElement pack = packDocument.RootElement;

        Assert.Equal("yolovision-real-asset-owner-proof-execution-pack", pack.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-action-required", pack.GetProperty("packState").GetString());
        Assert.Equal(6, pack.GetProperty("caseCount").GetInt32());
        Assert.True(pack.GetProperty("ownerRequiredFieldCount").GetInt32() > 100);
        Assert.False(pack.GetProperty("performsPublish").GetBoolean());
        Assert.False(pack.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.False(pack.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.Contains("TensorRtExec report is not proof", pack.GetProperty("proofBoundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("sample-run-evidence candidate is not package-consumer-runtime proof", pack.GetProperty("proofBoundary").GetString(), StringComparison.OrdinalIgnoreCase);

        JsonElement[] cases = pack.GetProperty("cases").EnumerateArray().ToArray();
        Assert.Equal(6, cases.Length);
        Assert.Contains(cases, item => item.GetProperty("task").GetString() == "sem");
        foreach (JsonElement item in cases)
        {
            Assert.True(item.GetProperty("requiredFieldCount").GetInt32() > 0);
            string commands = string.Join("\n", item.GetProperty("recommendedCommandSequence").EnumerateArray().Select(static command => command.GetProperty("command").GetString()));
            Assert.Contains("TensorRtExec", commands, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("YoloVision", commands, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog", commands, StringComparison.Ordinal);
        }

        string markdown = ReadText("artifacts", "user-acceptance", "yolovision-real-asset-owner-proof-execution-pack.md");
        Assert.Contains("Case Required Field Delta", markdown, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec report is not proof", markdown, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("sample-run-evidence candidate is not package-consumer-runtime proof", markdown, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ReleaseCloseProofLaneWorklistSeparatesProofLanesWithoutPromoting()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseProofLaneWorklist.ps1"));

        using JsonDocument worklistDocument = ReadJson("artifacts", "final-release", "release-close-proof-lane-worklist.json");
        JsonElement worklist = worklistDocument.RootElement;

        Assert.Equal("release-close-proof-lane-worklist", worklist.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-owner-public-postpublish-proof-required", worklist.GetProperty("worklistState").GetString());
        Assert.Equal(4, worklist.GetProperty("laneCount").GetInt32());
        Assert.False(worklist.GetProperty("performsPublish").GetBoolean());
        Assert.False(worklist.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(worklist.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(worklist.GetProperty("canPromoteRuntimeProof").GetBoolean());

        JsonElement[] lanes = worklist.GetProperty("lanes").EnumerateArray().ToArray();
        string[] laneIds = lanes.Select(static lane => lane.GetProperty("id").GetString()!).ToArray();
        Assert.Contains("real-model-runtime", laneIds);
        Assert.Contains("package-consumer-runtime", laneIds);
        Assert.Contains("post-publish-verification", laneIds);
        Assert.Contains("public-owner-confirmation", laneIds);

        JsonElement packageLane = lanes.Single(static lane => lane.GetProperty("id").GetString() == "package-consumer-runtime");
        string packageLaneText = packageLane.GetRawText();
        Assert.Contains("publicPackageSource", packageLaneText, StringComparison.Ordinal);
        Assert.Contains("managedPackageVersion", packageLaneText, StringComparison.Ordinal);
        Assert.Contains("runtimePackageVersion", packageLaneText, StringComparison.Ordinal);
        Assert.Contains("hostOs", packageLaneText, StringComparison.Ordinal);
        Assert.Contains("gpuName", packageLaneText, StringComparison.Ordinal);
        Assert.Contains("exitCode=0", packageLaneText, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("local feed", packageLaneText, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("ProjectReference", packageLaneText, StringComparison.Ordinal);
        Assert.Contains("direct .nupkg", packageLaneText, StringComparison.OrdinalIgnoreCase);

        Assert.Contains("sample-run-evidence", worklist.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("post-publish verification", worklist.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string markdown = ReadText("artifacts", "final-release", "release-close-proof-lane-worklist.md");
        Assert.Contains("Release Close Proof Lane Worklist", markdown, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime", markdown, StringComparison.Ordinal);
        Assert.Contains("post-publish-verification", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void TensorRtExecAdvancedProofReadinessChecklistDocumentsOwnerFields()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-TensorRtExecAdvancedProofReadinessChecklist.ps1"));

        using JsonDocument checklistDocument = ReadJson("artifacts", "final-release", "tensorrtexec-advanced-proof-readiness-checklist.json");
        JsonElement checklist = checklistDocument.RootElement;

        Assert.Equal("tensorrtexec-advanced-proof-readiness-checklist", checklist.GetProperty("recordKind").GetString());
        Assert.False(checklist.GetProperty("performsPublish").GetBoolean());
        Assert.False(checklist.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.False(checklist.GetProperty("canCloseReleaseIssue").GetBoolean());

        string timingCacheJson = ReadText("artifacts", "final-release", "tensorrtexec-timing-cache-owner-proof-checklist.template.json");
        string int8Json = ReadText("artifacts", "final-release", "tensorrtexec-int8-calibration-owner-proof-checklist.template.json");
        string combined = timingCacheJson + int8Json + ReadText("artifacts", "final-release", "tensorrtexec-advanced-proof-readiness-checklist.md");

        foreach (string marker in new[]
        {
            "--timingCacheFile",
            "--exportTimingCache",
            "cache content hash",
            "native import/export smoke",
            "tensorRtVersion",
            "cudaVersion",
            "gpuName",
            "--int8",
            "--calib",
            "calibrationDatasetProvenance",
            "calibratorOwnership",
            "callbackOwnership",
            "modelSpecificInt8AccuracyEvidence",
            "not package-consumer-runtime proof"
        })
        {
            Assert.Contains(marker, combined, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void FinalPublishProofGateBlocksUntilRealOwnerPublicAndPostPublishProof()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerExternalProofExecutionBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Import-OwnerExternalProofExecutionResult.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerExternalProofExecutionResultImport.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealExternalProofRecordImportValidator.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-RealExternalProofRecordImportValidator.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealProofRecordCandidateFromOwnerResultImport.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-RealProofRecordCandidateFromOwnerResultImport.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseProofLaneWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalPublishProofGate.ps1"), "-Strict");

        using JsonDocument gateDocument = ReadJson("artifacts", "final-release", "final-publish-proof-gate-report.json");
        JsonElement gate = gateDocument.RootElement;

        Assert.Equal("final-publish-proof-gate-report", gate.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-publish-real-proof-required", gate.GetProperty("validationState").GetString());
        Assert.False(gate.GetProperty("performsPublish").GetBoolean());
        Assert.False(gate.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(gate.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(gate.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.Equal(0, gate.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(gate.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
        Assert.Equal("blocked-owner-external-proof-execution-result-required", gate.GetProperty("ownerExternalProofResultImportState").GetString());
        Assert.Equal(6, gate.GetProperty("ownerExternalProofResultLaneCount").GetInt32());
        Assert.Equal(6, gate.GetProperty("ownerExternalProofResultBlockedLaneCount").GetInt32());
        Assert.Equal(0, gate.GetProperty("ownerExternalProofResultPromotableLaneCount").GetInt32());
        Assert.True(gate.GetProperty("ownerExternalProofResultFileMissingCount").GetInt32() >= 30);
        Assert.True(gate.GetProperty("ownerExternalProofResultInvalidSha256Count").GetInt32() >= 30);
        Assert.Equal(0, gate.GetProperty("ownerExternalProofResultHashMismatchCount").GetInt32());
        Assert.Equal(0, gate.GetProperty("ownerExternalProofResultOutsideAllowedEvidenceRootCount").GetInt32());
        Assert.True(gate.GetProperty("ownerExternalProofResultForbiddenSubstituteFindingCount").GetInt32() >= 0);
        Assert.Equal("blocked-owner-result-import-candidate-required", gate.GetProperty("ownerResultCandidateBridgeState").GetString());
        Assert.Equal(0, gate.GetProperty("ownerResultCandidateBridgeCandidateCount").GetInt32());
        Assert.Equal(0, gate.GetProperty("ownerResultCandidateBridgeStrictValidatorReadyCandidateCount").GetInt32());
        Assert.Equal(0, gate.GetProperty("ownerResultCandidateBridgePackageConsumerCandidateCount").GetInt32());
        Assert.Equal(0, gate.GetProperty("ownerResultCandidateBridgePostPublishCandidateCount").GetInt32());

        JsonElement[] items = gate.GetProperty("validationItems").EnumerateArray().ToArray();
        Assert.Contains(items, item => item.GetProperty("id").GetString() == "no-automatic-nuget-push" && item.GetProperty("passed").GetBoolean());
        Assert.Contains(items, item => item.GetProperty("id").GetString() == "no-yolodet-live" && item.GetProperty("passed").GetBoolean());
        Assert.Contains(items, item => item.GetProperty("id").GetString() == "no-sample-run-substitute-package-consumer" && item.GetProperty("passed").GetBoolean());
        Assert.Contains(items, item => item.GetProperty("id").GetString() == "owner-external-proof-result-import-structurally-safe" && item.GetProperty("passed").GetBoolean());
        Assert.Contains(items, item => item.GetProperty("id").GetString() == "owner-external-proof-result-import-owner-proof-required" && !item.GetProperty("passed").GetBoolean());
        Assert.Contains(items, item => item.GetProperty("id").GetString() == "owner-result-candidate-bridge-structurally-safe" && item.GetProperty("passed").GetBoolean());
        Assert.Contains(items, item => item.GetProperty("id").GetString() == "owner-result-candidate-bridge-real-proof-required" && !item.GetProperty("passed").GetBoolean());
        Assert.Contains(items, item => item.GetProperty("id").GetString() == "real-model-runtime-owner-proof-required" && !item.GetProperty("passed").GetBoolean());
        Assert.Contains(items, item => item.GetProperty("id").GetString() == "post-publish-verification-owner-proof-required" && !item.GetProperty("passed").GetBoolean());
        Assert.Contains(items, item => item.GetProperty("id").GetString() == "final-owner-real-input-template-pack-safe" && item.GetProperty("passed").GetBoolean());
        Assert.Contains(items, item => item.GetProperty("id").GetString() == "final-owner-real-input-template-pack-owner-input-required" && !item.GetProperty("passed").GetBoolean());
        Assert.True(gate.GetProperty("finalOwnerRealInputTemplatePackLaneCount").GetInt32() >= 5);
        Assert.True(gate.GetProperty("finalOwnerRealInputTemplatePackFailedActionRequiredCount").GetInt32() >= 5);

        Assert.Empty(gate.GetProperty("liveYoloDetMatches").EnumerateArray());
        Assert.Empty(gate.GetProperty("liveTensorRtLayerTensorInfoMatches").EnumerateArray());
        Assert.Contains("does not run dotnet nuget push", gate.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("direct .nupkg", gate.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string markdown = ReadText("artifacts", "final-release", "final-publish-proof-gate-report.md");
        Assert.Contains("Final Publish Proof Gate Report", markdown, StringComparison.Ordinal);
        Assert.Contains("blocked-final-publish-real-proof-required", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void FinalOwnerProofActionWorklistMapsAllPublishGateActionsWithoutPromoting()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerExternalProofExecutionBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Import-OwnerExternalProofExecutionResult.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerExternalProofExecutionResultImport.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealExternalProofRecordImportValidator.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-RealExternalProofRecordImportValidator.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-RealProofRecordCandidateFromOwnerResultImport.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-RealProofRecordCandidateFromOwnerResultImport.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseCloseProofLaneWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseProofDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseProofDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PublicDocsAndPackageMetadataGate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalReleaseCloseBlockerDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalPublishProofGate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerProofActionWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerProofActionWorklist.ps1"), "-Strict");

        using JsonDocument worklistDocument = ReadJson("artifacts", "final-release", "final-owner-proof-action-worklist.json");
        JsonElement worklist = worklistDocument.RootElement;

        Assert.Equal("final-owner-proof-action-worklist", worklist.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-proof-action-required", worklist.GetProperty("worklistState").GetString());
        Assert.Equal("blocked-final-publish-real-proof-required", worklist.GetProperty("finalGateState").GetString());
        Assert.Equal(0, worklist.GetProperty("finalGateFailedBlockerCount").GetInt32());
        Assert.Equal(6, worklist.GetProperty("finalGateActionRequiredCount").GetInt32());
        Assert.Equal(4, worklist.GetProperty("releaseDashboardOwnerProofActionRequiredLaneCount").GetInt32());
        Assert.Equal(0, worklist.GetProperty("publicDocsBlockedMatchCount").GetInt32());
        Assert.Equal(8, worklist.GetProperty("actionCount").GetInt32());
        Assert.Equal(8, worklist.GetProperty("blockedActionCount").GetInt32());
        Assert.Equal(0, worklist.GetProperty("missingActionRequiredIdCount").GetInt32());
        Assert.False(worklist.GetProperty("performsPublish").GetBoolean());
        Assert.False(worklist.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(worklist.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(worklist.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(worklist.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(worklist.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(worklist.GetProperty("isReleaseCloseProof").GetBoolean());

        JsonElement[] actions = worklist.GetProperty("actions").EnumerateArray().ToArray();
        string[] actionIds = actions.Select(static action => action.GetProperty("id").GetString()!).ToArray();
        string[] nextOwnerOrder = worklist.GetProperty("nextOwnerOrder").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Equal(actionIds.Length, nextOwnerOrder.Length);
        Assert.Empty(actionIds.Except(nextOwnerOrder));
        Assert.Empty(nextOwnerOrder.Except(actionIds));

        Assert.Contains("00-clean-external-package-consumer-owner-runbook", actionIds);
        Assert.Contains("00-post-publish-owner-verification-runbook", actionIds);
        Assert.Contains("01-real-model-runtime-owner-evidence", actionIds);
        Assert.Contains("02-package-consumer-runtime-clean-external-proof", actionIds);
        Assert.Contains("03-post-publish-verification-public-channel", actionIds);
        Assert.Contains("04-final-owner-real-input-template-pack", actionIds);
        Assert.Contains("05-owner-external-result-import-real-files", actionIds);
        Assert.Contains("06-owner-result-candidate-bridge-strict-promotion", actionIds);

        string[] actionRequiredIds = actions.Select(static action => action.GetProperty("actionRequiredId").GetString()!).ToArray();
        Assert.Contains("real-model-runtime-owner-proof-required", actionRequiredIds);
        Assert.Contains("package-consumer-runtime-owner-proof-required", actionRequiredIds);
        Assert.Contains("post-publish-verification-owner-proof-required", actionRequiredIds);
        Assert.Contains("final-owner-real-input-template-pack-owner-input-required", actionRequiredIds);
        Assert.Contains("owner-external-proof-result-import-owner-proof-required", actionRequiredIds);
        Assert.Contains("owner-result-candidate-bridge-real-proof-required", actionRequiredIds);

        Assert.All(actions, action =>
        {
            Assert.True(action.GetProperty("blocked").GetBoolean());
            Assert.True(action.GetProperty("requiredInputCount").GetInt32() >= 5);
            Assert.NotEmpty(action.GetProperty("ownerCommands").EnumerateArray());
            Assert.NotEmpty(action.GetProperty("validatorCommands").EnumerateArray());
            Assert.False(action.GetProperty("performsPublish").GetBoolean());
            Assert.False(action.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.False(action.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(action.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(action.GetProperty("isRuntimeExecutionProof").GetBoolean());
            Assert.False(action.GetProperty("isPostPublishProof").GetBoolean());
            Assert.False(action.GetProperty("isReleaseCloseProof").GetBoolean());
        });

        string raw = worklist.GetRawText();
        foreach (string marker in new[]
        {
            "local feed",
            "ProjectReference",
            "direct .nupkg",
            "build-only",
            "dry-run",
            "candidate",
            "dashboard",
            "blocked-by-driver"
        })
        {
            Assert.Contains(marker, raw, StringComparison.OrdinalIgnoreCase);
        }

        string boundary = worklist.GetProperty("boundary").GetString()!;
        Assert.Contains("does not publish", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("cannot close", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("non-proof", boundary, StringComparison.OrdinalIgnoreCase);

        using JsonDocument validationDocument = ReadJson("artifacts", "final-release", "final-owner-proof-action-worklist-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-owner-proof-action-worklist-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-proof-action-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());

        string markdown = ReadText("artifacts", "final-release", "final-owner-proof-action-worklist.md");
        Assert.Contains("Final Owner Proof Action Worklist", markdown, StringComparison.Ordinal);
        Assert.Contains("package-consumer-runtime-clean-external-proof", markdown, StringComparison.Ordinal);
        Assert.Contains("post-publish-verification-public-channel", markdown, StringComparison.Ordinal);
        Assert.Contains("final-owner-real-input-template-pack", markdown, StringComparison.Ordinal);
        Assert.Contains("owner-result-candidate-bridge-strict-promotion", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void FinalOwnerExecutionPackageMapsAllOwnerActionsToCommandsInputsResultsAndValidators()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerProofActionWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerProofActionWorklist.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerExecutionPackage.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionPackage.ps1"), "-Strict");

        using JsonDocument packageDocument = ReadJson("artifacts", "final-release", "final-owner-execution-package.json");
        JsonElement package = packageDocument.RootElement;

        Assert.Equal("final-owner-execution-package", package.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-execution-required", package.GetProperty("packageState").GetString());
        Assert.Equal("blocked-final-owner-proof-action-required", package.GetProperty("sourceWorklistState").GetString());
        Assert.Equal("blocked-final-owner-proof-action-required", package.GetProperty("sourceWorklistValidationState").GetString());
        Assert.Equal("blocked-owner-compatible-host-runtime-smoke-field-alignment", package.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentState").GetString());
        Assert.Equal("blocked-owner-compatible-host-runtime-smoke-field-alignment-valid", package.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState").GetString());
        Assert.Equal("Smoke=not-requested", package.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus").GetString());
        Assert.True(package.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount").GetInt32() >= 30);
        Assert.Equal(0, package.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount").GetInt32());
        Assert.Equal(0, package.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount").GetInt32());
        Assert.Equal(8, package.GetProperty("actionCount").GetInt32());
        Assert.Equal(8, package.GetProperty("executionStepCount").GetInt32());
        Assert.Equal(8, package.GetProperty("blockedExecutionStepCount").GetInt32());
        Assert.False(package.GetProperty("performsPublish").GetBoolean());
        Assert.False(package.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(package.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(package.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(package.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(package.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(package.GetProperty("isReleaseCloseProof").GetBoolean());
        Assert.False(package.GetProperty("isPackagePush").GetBoolean());
        Assert.True(package.GetProperty("requiresOwnerManualPublish").GetBoolean());
        Assert.True(package.GetProperty("requiresRealExternalProof").GetBoolean());
        Assert.True(package.GetProperty("requiresPostPublishProof").GetBoolean());
        Assert.True(package.GetProperty("requiresReleaseCloseOwnerDecision").GetBoolean());
        string[] packageSources = package.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment.json", packageSources);
        Assert.Contains("artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment-validation.json", packageSources);

        JsonElement[] steps = package.GetProperty("executionSteps").EnumerateArray().ToArray();
        string[] stepIds = steps.Select(static step => step.GetProperty("id").GetString()!).ToArray();
        Assert.Contains("01-real-model-runtime-owner-evidence", stepIds);
        Assert.Contains("02-package-consumer-runtime-clean-external-proof", stepIds);
        Assert.Contains("03-post-publish-verification-public-channel", stepIds);
        Assert.Contains("04-final-owner-real-input-template-pack", stepIds);
        Assert.Contains("05-owner-external-result-import-real-files", stepIds);
        Assert.Contains("06-owner-result-candidate-bridge-strict-promotion", stepIds);

        string[] actionRequiredIds = package.GetProperty("actionRequiredIds").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("real-model-runtime-owner-proof-required", actionRequiredIds);
        Assert.Contains("package-consumer-runtime-owner-proof-required", actionRequiredIds);
        Assert.Contains("post-publish-verification-owner-proof-required", actionRequiredIds);
        Assert.Contains("final-owner-real-input-template-pack-owner-input-required", actionRequiredIds);
        Assert.Contains("owner-external-proof-result-import-owner-proof-required", actionRequiredIds);
        Assert.Contains("owner-result-candidate-bridge-real-proof-required", actionRequiredIds);

        Assert.All(steps, step =>
        {
            Assert.True(step.GetProperty("blocked").GetBoolean());
            Assert.True(step.GetProperty("requiredInputCount").GetInt32() >= 5);
            Assert.True(step.GetProperty("ownerCommandCount").GetInt32() >= 2);
            Assert.True(step.GetProperty("validatorCommandCount").GetInt32() >= 1);
            Assert.True(step.GetProperty("expectedResultArtifactCount").GetInt32() >= 2);
            Assert.True(step.GetProperty("forbiddenSubstituteCount").GetInt32() >= 5);
            Assert.False(step.GetProperty("performsPublish").GetBoolean());
            Assert.False(step.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.False(step.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(step.GetProperty("canCloseReleaseIssue").GetBoolean());

            JsonElement contract = step.GetProperty("inputContract");
            Assert.True(contract.GetProperty("ownerMustProvideRealFiles").GetBoolean());
            Assert.True(contract.GetProperty("requiresExistingLog").GetBoolean());
            Assert.True(contract.GetProperty("requiresSha256").GetBoolean());
            Assert.False(contract.GetProperty("acceptsTemplateOnly").GetBoolean());
            Assert.False(contract.GetProperty("acceptsCandidateAsProof").GetBoolean());
            Assert.False(contract.GetProperty("acceptsDashboardAsProof").GetBoolean());

            string boundary = step.GetProperty("boundary").GetString()!;
            Assert.Contains("not runtime proof", boundary, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("not post-publish proof", boundary, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("not package push", boundary, StringComparison.OrdinalIgnoreCase);
        });

        string raw = package.GetRawText();
        foreach (string marker in new[]
        {
            "Test-SampleRunEvidenceRecord.ps1",
            "Test-PackageConsumerRuntimeProofRecord.ps1",
            "Test-PostPublishVerificationRecord.ps1",
            "Test-OwnerExternalProofExecutionResultImport.ps1",
            "Test-RealProofRecordCandidateFromOwnerResultImport.ps1",
            "sample-run-evidence-record-validation.json",
            "package-consumer-runtime-proof-record-validation.json",
            "post-publish-verification-validation.json",
            "final-owner-real-input-template-pack-validation.json",
            "owner-external-proof-execution-result-import-validation.json",
            "real-proof-record-candidate-from-owner-result-import-validation.json",
            "local feed",
            "ProjectReference",
            "direct .nupkg",
            "build-only",
            "dry-run",
            "candidate",
            "dashboard",
            "blocked-by-cuda-driver"
        })
        {
            Assert.Contains(marker, raw, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument validationDocument = ReadJson("artifacts", "final-release", "final-owner-execution-package-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-owner-execution-package-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-execution-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(8, validation.GetProperty("executionStepCount").GetInt32());
        Assert.Equal(8, validation.GetProperty("blockedExecutionStepCount").GetInt32());
        Assert.Equal("blocked-owner-compatible-host-runtime-smoke-field-alignment", validation.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentState").GetString());
        Assert.Equal("blocked-owner-compatible-host-runtime-smoke-field-alignment-valid", validation.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState").GetString());
        Assert.Equal(0, validation.GetProperty("packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());

        string markdown = ReadText("artifacts", "final-release", "final-owner-execution-package.md");
        Assert.Contains("Final Owner Execution Package", markdown, StringComparison.Ordinal);
        Assert.Contains("final-owner-real-input-template-pack", markdown, StringComparison.Ordinal);
        Assert.Contains("owner-result-candidate-bridge-strict-promotion", markdown, StringComparison.Ordinal);
        Assert.Contains("not package push", markdown, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void FinalOwnerExecutionRepairChecklistMirrorsEightOwnerStepsWithoutPromotingProof()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerProofActionWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerProofActionWorklist.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerExecutionPackage.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionPackage.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerExecutionRepairChecklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionRepairChecklist.ps1"), "-Strict");

        using JsonDocument checklistDocument = ReadJson("artifacts", "final-release", "final-owner-execution-repair-checklist.json");
        JsonElement checklist = checklistDocument.RootElement;

        Assert.Equal("final-owner-execution-repair-checklist", checklist.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-execution-repair-real-owner-evidence-required", checklist.GetProperty("checklistState").GetString());
        Assert.Equal(8, checklist.GetProperty("executionStepCount").GetInt32());
        Assert.Equal(8, checklist.GetProperty("repairItemCount").GetInt32());
        Assert.Equal(8, checklist.GetProperty("blockedRepairItemCount").GetInt32());
        Assert.Equal(0, checklist.GetProperty("readyRepairItemCount").GetInt32());
        Assert.True(checklist.GetProperty("requiredFileFieldCount").GetInt32() > 0);
        Assert.True(checklist.GetProperty("requiredSha256FieldCount").GetInt32() > 0);
        Assert.True(checklist.GetProperty("requiredIdentityFieldCount").GetInt32() > 0);
        Assert.True(checklist.GetProperty("requiredNonSubstituteConfirmationCount").GetInt32() > 0);
        Assert.False(checklist.GetProperty("performsPublish").GetBoolean());
        Assert.False(checklist.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(checklist.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(checklist.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(checklist.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(checklist.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(checklist.GetProperty("isReleaseCloseProof").GetBoolean());

        JsonElement[] repairItems = checklist.GetProperty("repairItems").EnumerateArray().ToArray();
        Assert.Equal(8, repairItems.Length);
        Assert.All(repairItems, item =>
        {
            Assert.Equal("blocked-real-owner-evidence-required", item.GetProperty("repairState").GetString());
            Assert.NotEmpty(item.GetProperty("ownerCommands").EnumerateArray());
            Assert.NotEmpty(item.GetProperty("validatorCommands").EnumerateArray());
            Assert.NotEmpty(item.GetProperty("expectedResultArtifacts").EnumerateArray());
            Assert.True(item.GetProperty("requiredFileFields").EnumerateArray().Count() >= 3);
            Assert.True(item.GetProperty("requiredSha256Fields").EnumerateArray().Count() >= 3);
            Assert.True(item.GetProperty("requiredIdentityFields").EnumerateArray().Count() >= 3);
            Assert.False(item.GetProperty("performsPublish").GetBoolean());
            Assert.False(item.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(item.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.Contains("Repair guidance only", item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Contains("not package push", item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        });

        string raw = checklist.GetRawText();
        foreach (string marker in new[]
        {
            "stdoutPath",
            "stderrPath",
            "mergedTranscriptPath",
            "SHA256",
            "exitCode",
            "hostIdentity",
            "packageIdentity",
            "ownerReviewer",
            "local feed",
            "ProjectReference",
            "direct .nupkg",
            "template",
            "draft",
            "dry-run",
            "dashboard",
            "candidate",
            "build-only",
            "dependency-probe-only",
            "blocked-by-cuda-driver"
        })
        {
            Assert.Contains(marker, raw, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument validationDocument = ReadJson("artifacts", "final-release", "final-owner-execution-repair-checklist-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-owner-execution-repair-checklist-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-owner-execution-repair-real-owner-evidence-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() > 0);

        string repairMarkdown = ReadText("artifacts", "final-release", "final-owner-execution-repair-checklist.md");
        Assert.Contains("Final Owner Execution Repair Checklist", repairMarkdown, StringComparison.Ordinal);
        Assert.Contains("direct .nupkg", repairMarkdown, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("blocked-by-cuda-driver", repairMarkdown, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void FinalOwnerExternalResultInputContractRequiresRealEvidenceAndRejectsSubstitutes()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerExecutionRepairChecklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionRepairChecklist.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerExecutionRepairInputSkeleton.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionRepairInputSkeleton.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerExecutionOwnerInputDraft.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionOwnerInputDraft.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerExecutionExternalResultInputContract.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionExternalResultInputContract.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionExternalResultInputPreflight.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseEvidenceClassificationAudit.ps1"), "-Strict");

        using JsonDocument contractDocument = ReadJson("artifacts", "final-release", "final-owner-execution-external-result-input-contract.json");
        JsonElement contract = contractDocument.RootElement;
        Assert.Equal("final-owner-execution-external-result-input-contract", contract.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-external-result-required", contract.GetProperty("contractState").GetString());
        Assert.Equal(8, contract.GetProperty("contractItemCount").GetInt32());
        Assert.Equal(8, contract.GetProperty("blockedContractItemCount").GetInt32());
        Assert.Equal(0, contract.GetProperty("readyForImportCount").GetInt32());
        Assert.True(contract.GetProperty("placeholderFieldCount").GetInt32() >= 180);
        Assert.False(contract.GetProperty("performsPublish").GetBoolean());
        Assert.False(contract.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(contract.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(contract.GetProperty("canPromoteRuntimeProof").GetBoolean());

        JsonElement[] contractItems = contract.GetProperty("contractItems").EnumerateArray().ToArray();
        Assert.Equal(8, contractItems.Length);
        Assert.All(contractItems, item =>
        {
            Assert.Equal("blocked-owner-real-external-result-required", item.GetProperty("ownerInputState").GetString());
            Assert.StartsWith("<owner-fill-real-", item.GetProperty("realExecutionRoot").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.StartsWith("<owner-fill-real-", item.GetProperty("stdoutPath").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.StartsWith("<owner-fill-real-", item.GetProperty("validatorOutputSha256").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.True(item.TryGetProperty("hostIdentity", out JsonElement hostIdentity));
            Assert.True(hostIdentity.TryGetProperty("cudaVersion", out _));
            Assert.True(hostIdentity.TryGetProperty("tensorrtVersion", out _));
            Assert.True(hostIdentity.TryGetProperty("driverVersion", out _));
            Assert.True(item.TryGetProperty("packageIdentity", out JsonElement packageIdentity));
            Assert.True(packageIdentity.TryGetProperty("packageSource", out _));
            Assert.True(packageIdentity.TryGetProperty("nupkgSha256", out _));
            Assert.True(packageIdentity.TryGetProperty("publishedPackageUrl", out _));
            Assert.False(item.GetProperty("readyForImport").GetBoolean());
            Assert.False(item.GetProperty("performsPublish").GetBoolean());
            Assert.False(item.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(item.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.Contains("not runtime proof", item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Contains("not package push", item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        });

        string raw = contract.GetRawText();
        foreach (string marker in new[]
        {
            "realExecutionRoot",
            "stdoutPath",
            "stderrPath",
            "mergedTranscriptPath",
            "validatorOutputPath",
            "stdoutSha256",
            "stderrSha256",
            "mergedTranscriptSha256",
            "validatorOutputSha256",
            "exitCode",
            "executedCommand",
            "executedAtUtc",
            "hostIdentity",
            "packageIdentity",
            "ownerReviewer",
            "ownerReviewTimestampUtc",
            "local feed",
            "ProjectReference",
            "direct .nupkg",
            "template",
            "draft",
            "dry-run",
            "dashboard",
            "candidate",
            "build-only",
            "dependency-probe-only",
            "blocked-by-cuda-driver"
        })
        {
            Assert.Contains(marker, raw, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument validationDocument = ReadJson("artifacts", "final-release", "final-owner-execution-external-result-input-contract-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-owner-execution-external-result-input-contract-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-external-result-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(8, validation.GetProperty("failedActionRequiredCount").GetInt32());

        using JsonDocument preflightDocument = ReadJson("artifacts", "final-release", "final-owner-execution-external-result-input-preflight.json");
        JsonElement preflight = preflightDocument.RootElement;
        Assert.Equal("final-owner-execution-external-result-input-preflight", preflight.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-external-result-required", preflight.GetProperty("preflightState").GetString());
        Assert.Equal(0, preflight.GetProperty("readyCandidateLaneCount").GetInt32());
        Assert.Equal(8, preflight.GetProperty("blockedLaneCount").GetInt32());
        Assert.True(preflight.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        Assert.False(preflight.GetProperty("performsPublish").GetBoolean());
        Assert.False(preflight.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(preflight.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument bundleDocument = ReadJson("artifacts", "final-release", "release-evidence-bundle.json");
        JsonElement contractEvidence = bundleDocument.RootElement.GetProperty("evidenceItems").EnumerateArray().Single(static item => item.GetProperty("id").GetString() == "final-owner-execution-external-result-input-contract");
        Assert.False(contractEvidence.GetProperty("passed").GetBoolean());
        Assert.Contains("Owner input contract only", contractEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package push", contractEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        JsonElement preflightEvidence = bundleDocument.RootElement.GetProperty("evidenceItems").EnumerateArray().Single(static item => item.GetProperty("id").GetString() == "final-owner-execution-external-result-input-preflight");
        Assert.False(preflightEvidence.GetProperty("passed").GetBoolean());
        Assert.Contains("candidate readiness screening only", preflightEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not runtime proof", preflightEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void FinalOwnerExternalResultCandidateRemainsNonProofUntilPostPublishCleanConsumerProof()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerExecutionRepairChecklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionRepairChecklist.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerExecutionRepairInputSkeleton.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionRepairInputSkeleton.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerExecutionOwnerInputDraft.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionOwnerInputDraft.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerExecutionExternalResultInputContract.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionExternalResultInputContract.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionExternalResultInputPreflight.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Import-FinalOwnerExecutionExternalResultCandidate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionExternalResultCandidate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseEvidenceClassificationAudit.ps1"), "-Strict");

        using JsonDocument candidateDocument = ReadJson("artifacts", "final-release", "final-owner-execution-external-result-candidate.json");
        JsonElement candidate = candidateDocument.RootElement;
        Assert.Equal("final-owner-execution-external-result-candidate", candidate.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-no-owner-external-result-candidate-ready", candidate.GetProperty("candidateState").GetString());
        Assert.Equal(8, candidate.GetProperty("contractItemCount").GetInt32());
        Assert.Equal(0, candidate.GetProperty("candidateItemCount").GetInt32());
        Assert.Equal(8, candidate.GetProperty("blockedCandidateItemCount").GetInt32());
        Assert.False(candidate.GetProperty("performsPublish").GetBoolean());
        Assert.False(candidate.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(candidate.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(candidate.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(candidate.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(candidate.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(candidate.GetProperty("isReleaseCloseProof").GetBoolean());

        string raw = candidate.GetRawText();
        foreach (string marker in new[]
        {
            "post-publish",
            "clean consumer",
            "not package push",
            "cannot substitute runtime proof",
            "post-publish proof",
            "release close proof"
        })
        {
            Assert.Contains(marker, raw, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument validationDocument = ReadJson("artifacts", "final-release", "final-owner-execution-external-result-candidate-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-owner-execution-external-result-candidate-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-no-owner-external-result-candidate-ready", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(8, validation.GetProperty("failedActionRequiredCount").GetInt32());

        using JsonDocument bundleDocument = ReadJson("artifacts", "final-release", "release-evidence-bundle.json");
        JsonElement candidateEvidence = bundleDocument.RootElement.GetProperty("evidenceItems").EnumerateArray().Single(static item => item.GetProperty("id").GetString() == "final-owner-execution-external-result-candidate");
        Assert.False(candidateEvidence.GetProperty("passed").GetBoolean());
        Assert.Contains("candidate evidence only", candidateEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not post-publish proof", candidateEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package push", candidateEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        JsonElement sourceArtifacts = bundleDocument.RootElement.GetProperty("sourceArtifacts");
        Assert.Contains(sourceArtifacts.EnumerateArray(), item => item.GetString() == "artifacts/final-release/final-owner-execution-external-result-candidate-validation.json");

        string candidateMarkdown = ReadText("artifacts", "final-release", "final-owner-execution-external-result-candidate.md");
        Assert.Contains("Final Owner Execution External Result Candidate", candidateMarkdown, StringComparison.Ordinal);
        Assert.Contains("not runtime proof", candidateMarkdown, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void FinalPostPublishCleanConsumerProofRecordContractRequiresPublicPackageAndCleanConsumerEvidence()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerExecutionRepairChecklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionRepairChecklist.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerExecutionRepairInputSkeleton.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionRepairInputSkeleton.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerExecutionOwnerInputDraft.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionOwnerInputDraft.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerExecutionExternalResultInputContract.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionExternalResultInputContract.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionExternalResultInputPreflight.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Import-FinalOwnerExecutionExternalResultCandidate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionExternalResultCandidate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalPostPublishCleanConsumerProofRecordContract.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalPostPublishCleanConsumerProofRecordContract.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalPostPublishCleanConsumerProofPreflight.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Import-FinalPostPublishCleanConsumerProofCandidate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalPostPublishCleanConsumerProofCandidate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseEvidenceClassificationAudit.ps1"), "-Strict");

        using JsonDocument contractDocument = ReadJson("artifacts", "final-release", "final-post-publish-clean-consumer-proof-record-contract.json");
        JsonElement contract = contractDocument.RootElement;
        Assert.Equal("final-post-publish-clean-consumer-proof-record-contract", contract.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-post-publish-clean-consumer-proof-required", contract.GetProperty("contractState").GetString());
        Assert.Equal(3, contract.GetProperty("contractItemCount").GetInt32());
        Assert.Equal(0, contract.GetProperty("readyForPreflightCount").GetInt32());
        Assert.True(contract.GetProperty("requiredEvidenceFieldCount").GetInt32() >= 80);
        Assert.False(contract.GetProperty("performsPublish").GetBoolean());
        Assert.False(contract.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(contract.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(contract.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(contract.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(contract.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(contract.GetProperty("isReleaseCloseProof").GetBoolean());

        JsonElement[] contractItems = contract.GetProperty("contractItems").EnumerateArray().ToArray();
        Assert.Equal(3, contractItems.Length);
        Assert.All(contractItems, item =>
        {
            Assert.Equal("blocked-post-publish-clean-consumer-proof-required", item.GetProperty("proofState").GetString());
            Assert.True(item.TryGetProperty("packageIdentity", out JsonElement packageIdentity));
            Assert.True(packageIdentity.TryGetProperty("packageId", out _));
            Assert.True(packageIdentity.TryGetProperty("packageVersion", out _));
            Assert.True(packageIdentity.TryGetProperty("packageSource", out _));
            Assert.True(packageIdentity.TryGetProperty("publicPackageUrl", out _));
            Assert.True(packageIdentity.TryGetProperty("nupkgSha256", out _));
            Assert.StartsWith("<owner-fill-public-", packageIdentity.GetProperty("publicPackageUrl").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.StartsWith("<owner-fill-clean-consumer-", item.GetProperty("cleanConsumerProjectRoot").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.StartsWith("<owner-fill-clean-consumer-", item.GetProperty("cleanConsumerRunLogSha256").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.False(item.GetProperty("readyForPreflight").GetBoolean());
            Assert.False(item.GetProperty("performsPublish").GetBoolean());
            Assert.False(item.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(item.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.False(item.GetProperty("isRuntimeExecutionProof").GetBoolean());
            Assert.False(item.GetProperty("isPostPublishProof").GetBoolean());
            Assert.False(item.GetProperty("isReleaseCloseProof").GetBoolean());
            Assert.Contains("not package push", item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        });

        string raw = contract.GetRawText();
        foreach (string marker in new[]
        {
            "publicPackageUrl",
            "packageSource",
            "nupkgSha256",
            "cleanConsumerProjectRoot",
            "cleanConsumerRestoreLogPath",
            "cleanConsumerBuildLogPath",
            "cleanConsumerRunLogPath",
            "cleanConsumerMergedTranscriptPath",
            "cleanConsumerValidatorOutputPath",
            "hostIdentity",
            "noProjectReferenceConfirmation",
            "noLocalFeedConfirmation",
            "noDirectNupkgConfirmation",
            "noSourceCheckoutReferenceConfirmation",
            "local feed",
            "ProjectReference",
            "direct .nupkg",
            "source checkout reference",
            "template",
            "draft",
            "dry-run",
            "dashboard",
            "candidate",
            "build-only",
            "dependency-probe-only",
            "blocked-by-cuda-driver"
        })
        {
            Assert.Contains(marker, raw, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument validationDocument = ReadJson("artifacts", "final-release", "final-post-publish-clean-consumer-proof-record-contract-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-post-publish-clean-consumer-proof-record-contract-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-post-publish-clean-consumer-proof-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(3, validation.GetProperty("failedActionRequiredCount").GetInt32());

        using JsonDocument preflightDocument = ReadJson("artifacts", "final-release", "final-post-publish-clean-consumer-proof-preflight.json");
        JsonElement preflight = preflightDocument.RootElement;
        Assert.Equal("final-post-publish-clean-consumer-proof-preflight", preflight.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-post-publish-clean-consumer-proof-required", preflight.GetProperty("preflightState").GetString());
        Assert.Equal(0, preflight.GetProperty("readyProofCandidateCount").GetInt32());
        Assert.Equal(3, preflight.GetProperty("blockedProofCandidateCount").GetInt32());
        Assert.Equal(0, preflight.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(preflight.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        Assert.False(preflight.GetProperty("performsPublish").GetBoolean());
        Assert.False(preflight.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(preflight.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument bundleDocument = ReadJson("artifacts", "final-release", "release-evidence-bundle.json");
        JsonElement contractEvidence = bundleDocument.RootElement.GetProperty("evidenceItems").EnumerateArray().Single(static item => item.GetProperty("id").GetString() == "final-post-publish-clean-consumer-proof-record-contract");
        Assert.False(contractEvidence.GetProperty("passed").GetBoolean());
        Assert.Contains("contract only", contractEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not post-publish proof", contractEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package push", contractEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        JsonElement preflightEvidence = bundleDocument.RootElement.GetProperty("evidenceItems").EnumerateArray().Single(static item => item.GetProperty("id").GetString() == "final-post-publish-clean-consumer-proof-preflight");
        Assert.False(preflightEvidence.GetProperty("passed").GetBoolean());
        Assert.Contains("screening only", preflightEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package push", preflightEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void FinalPostPublishCleanConsumerProofCandidateCannotCloseReleaseWithoutOwnerApproval()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalPostPublishCleanConsumerProofRecordContract.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalPostPublishCleanConsumerProofRecordContract.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalPostPublishCleanConsumerProofPreflight.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Import-FinalPostPublishCleanConsumerProofCandidate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalPostPublishCleanConsumerProofCandidate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseEvidenceClassificationAudit.ps1"), "-Strict");

        using JsonDocument candidateDocument = ReadJson("artifacts", "final-release", "final-post-publish-clean-consumer-proof-candidate.json");
        JsonElement candidate = candidateDocument.RootElement;
        Assert.Equal("final-post-publish-clean-consumer-proof-candidate", candidate.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-no-post-publish-clean-consumer-proof-candidate-ready", candidate.GetProperty("candidateState").GetString());
        Assert.Equal(3, candidate.GetProperty("contractItemCount").GetInt32());
        Assert.Equal(0, candidate.GetProperty("proofCandidateItemCount").GetInt32());
        Assert.Equal(3, candidate.GetProperty("blockedProofCandidateItemCount").GetInt32());
        Assert.False(candidate.GetProperty("performsPublish").GetBoolean());
        Assert.False(candidate.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(candidate.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(candidate.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(candidate.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(candidate.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(candidate.GetProperty("isReleaseCloseProof").GetBoolean());

        string raw = candidate.GetRawText();
        foreach (string marker in new[]
        {
            "Owner final release close decision",
            "rollback",
            "release notes",
            "final public package URL",
            "not package push",
            "cannot close the release"
        })
        {
            Assert.Contains(marker, raw, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument validationDocument = ReadJson("artifacts", "final-release", "final-post-publish-clean-consumer-proof-candidate-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-post-publish-clean-consumer-proof-candidate-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-no-post-publish-clean-consumer-proof-candidate-ready", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(3, validation.GetProperty("failedActionRequiredCount").GetInt32());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument bundleDocument = ReadJson("artifacts", "final-release", "release-evidence-bundle.json");
        JsonElement candidateEvidence = bundleDocument.RootElement.GetProperty("evidenceItems").EnumerateArray().Single(static item => item.GetProperty("id").GetString() == "final-post-publish-clean-consumer-proof-candidate");
        Assert.False(candidateEvidence.GetProperty("passed").GetBoolean());
        Assert.Contains("candidate only", candidateEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not release close approval", candidateEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package push", candidateEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        JsonElement sourceArtifacts = bundleDocument.RootElement.GetProperty("sourceArtifacts");
        Assert.Contains(sourceArtifacts.EnumerateArray(), item => item.GetString() == "artifacts/final-release/final-post-publish-clean-consumer-proof-candidate-validation.json");

        string candidateMarkdown = ReadText("artifacts", "final-release", "final-post-publish-clean-consumer-proof-candidate.md");
        Assert.Contains("Final Post-Publish Clean Consumer Proof Candidate", candidateMarkdown, StringComparison.Ordinal);
        Assert.Contains("cannot close the release", candidateMarkdown, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void FinalReleaseCloseOwnerApprovalContractRequiresRealOwnerDecisionsAndHashes()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalPostPublishCleanConsumerProofRecordContract.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalPostPublishCleanConsumerProofRecordContract.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalPostPublishCleanConsumerProofPreflight.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Import-FinalPostPublishCleanConsumerProofCandidate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalPostPublishCleanConsumerProofCandidate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalReleaseCloseOwnerApprovalContract.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalReleaseCloseOwnerApprovalContract.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalReleaseCloseOwnerApprovalPreflight.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Import-FinalReleaseCloseOwnerApprovalCandidate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalReleaseCloseOwnerApprovalCandidate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseEvidenceClassificationAudit.ps1"), "-Strict");

        using JsonDocument contractDocument = ReadJson("artifacts", "final-release", "final-release-close-owner-approval-contract.json");
        JsonElement contract = contractDocument.RootElement;
        Assert.Equal("final-release-close-owner-approval-contract", contract.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-release-close-owner-approval-required", contract.GetProperty("contractState").GetString());
        Assert.Equal(4, contract.GetProperty("approvalLaneCount").GetInt32());
        Assert.Equal(4, contract.GetProperty("blockedApprovalLaneCount").GetInt32());
        Assert.Equal(0, contract.GetProperty("readyForPreflightCount").GetInt32());
        Assert.True(contract.GetProperty("requiredOwnerInputFieldCount").GetInt32() >= 80);
        Assert.False(contract.GetProperty("performsPublish").GetBoolean());
        Assert.False(contract.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(contract.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(contract.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(contract.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(contract.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(contract.GetProperty("isReleaseCloseProof").GetBoolean());

        JsonElement[] lanes = contract.GetProperty("approvalLanes").EnumerateArray().ToArray();
        Assert.Equal(4, lanes.Length);
        Assert.Contains(lanes, lane => lane.GetProperty("id").GetString() == "owner-final-release-close-decision");
        Assert.Contains(lanes, lane => lane.GetProperty("id").GetString() == "owner-rollback-decision");
        Assert.Contains(lanes, lane => lane.GetProperty("id").GetString() == "owner-release-notes-approval");
        Assert.Contains(lanes, lane => lane.GetProperty("id").GetString() == "owner-final-public-package-url-and-hash-approval");
        Assert.All(lanes, lane =>
        {
            Assert.Equal("blocked-final-release-close-owner-approval-required", lane.GetProperty("approvalState").GetString());
            Assert.StartsWith("<owner-fill-final-release-close-", lane.GetProperty("ownerReviewer").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.StartsWith("<owner-fill-", lane.GetProperty("releaseIssueCloseDecision").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.StartsWith("<owner-fill-", lane.GetProperty("rollbackDecision").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.StartsWith("<owner-fill-", lane.GetProperty("releaseNotesPath").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.StartsWith("<owner-fill-", lane.GetProperty("releaseNotesSha256").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.StartsWith("<owner-fill-final-public-package-", lane.GetProperty("finalPublicPackageUrl").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.StartsWith("<owner-fill-final-public-package-", lane.GetProperty("finalPublicPackageSha256").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.True(lane.TryGetProperty("finalPackageIdentity", out JsonElement packageIdentity));
            Assert.True(packageIdentity.TryGetProperty("packageId", out _));
            Assert.True(packageIdentity.TryGetProperty("packageVersion", out _));
            Assert.True(packageIdentity.TryGetProperty("packageSource", out _));
            Assert.True(packageIdentity.TryGetProperty("publicPackageUrl", out _));
            Assert.True(packageIdentity.TryGetProperty("nupkgSha256", out _));
            Assert.True(lane.GetProperty("nonSubstituteConfirmations").EnumerateObject().Count() >= 8);
            Assert.False(lane.GetProperty("readyForPreflight").GetBoolean());
            Assert.False(lane.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(lane.GetProperty("isReleaseCloseProof").GetBoolean());
            Assert.Contains("not package push", lane.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Contains("cannot close the release", lane.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        });

        string raw = contract.GetRawText();
        foreach (string marker in new[]
        {
            "ownerReviewer",
            "ownerReviewTimestampUtc",
            "ownerApprovalDecision",
            "releaseIssueCloseDecision",
            "rollbackDecision",
            "releaseNotesPath",
            "releaseNotesSha256",
            "finalPublicPackageUrl",
            "finalPublicPackageSha256",
            "finalPackageIdentity",
            "postPublishCleanConsumerProofCandidateId",
            "classificationAuditPath",
            "releaseEvidenceBundlePath",
            "local feed",
            "ProjectReference",
            "direct .nupkg",
            "dry-run",
            "dashboard",
            "candidate",
            "blocked-by-driver"
        })
        {
            Assert.Contains(marker, raw, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument validationDocument = ReadJson("artifacts", "final-release", "final-release-close-owner-approval-contract-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-release-close-owner-approval-contract-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-release-close-owner-approval-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(4, validation.GetProperty("failedActionRequiredCount").GetInt32());

        using JsonDocument preflightDocument = ReadJson("artifacts", "final-release", "final-release-close-owner-approval-preflight.json");
        JsonElement preflight = preflightDocument.RootElement;
        Assert.Equal("final-release-close-owner-approval-preflight", preflight.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-release-close-owner-approval-required", preflight.GetProperty("preflightState").GetString());
        Assert.Equal(0, preflight.GetProperty("readyCloseCandidateCount").GetInt32());
        Assert.Equal(4, preflight.GetProperty("blockedCloseCandidateCount").GetInt32());
        Assert.Equal(0, preflight.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(preflight.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        Assert.False(preflight.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(preflight.GetProperty("isReleaseCloseProof").GetBoolean());

        using JsonDocument bundleDocument = ReadJson("artifacts", "final-release", "release-evidence-bundle.json");
        JsonElement contractEvidence = bundleDocument.RootElement.GetProperty("evidenceItems").EnumerateArray().Single(static item => item.GetProperty("id").GetString() == "final-release-close-owner-approval-contract");
        Assert.False(contractEvidence.GetProperty("passed").GetBoolean());
        Assert.Contains("contract only", contractEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package push", contractEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("cannot close the release", contractEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        JsonElement preflightEvidence = bundleDocument.RootElement.GetProperty("evidenceItems").EnumerateArray().Single(static item => item.GetProperty("id").GetString() == "final-release-close-owner-approval-preflight");
        Assert.False(preflightEvidence.GetProperty("passed").GetBoolean());
        Assert.Contains("screening only", preflightEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package push", preflightEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void FinalReleaseCloseOwnerApprovalCandidateCannotCloseReleaseWithoutRealApproval()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalReleaseCloseOwnerApprovalContract.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalReleaseCloseOwnerApprovalContract.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalReleaseCloseOwnerApprovalPreflight.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Import-FinalReleaseCloseOwnerApprovalCandidate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalReleaseCloseOwnerApprovalCandidate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseEvidenceClassificationAudit.ps1"), "-Strict");

        using JsonDocument candidateDocument = ReadJson("artifacts", "final-release", "final-release-close-owner-approval-candidate.json");
        JsonElement candidate = candidateDocument.RootElement;
        Assert.Equal("final-release-close-owner-approval-candidate", candidate.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-no-final-release-close-owner-approval-candidate-ready", candidate.GetProperty("candidateState").GetString());
        Assert.Equal(4, candidate.GetProperty("approvalLaneCount").GetInt32());
        Assert.Equal(0, candidate.GetProperty("closeCandidateItemCount").GetInt32());
        Assert.Equal(4, candidate.GetProperty("blockedCloseCandidateItemCount").GetInt32());
        Assert.False(candidate.GetProperty("performsPublish").GetBoolean());
        Assert.False(candidate.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(candidate.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(candidate.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(candidate.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(candidate.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(candidate.GetProperty("isReleaseCloseProof").GetBoolean());

        string raw = candidate.GetRawText();
        foreach (string marker in new[]
        {
            "No real Owner final release close approval",
            "No package push",
            "No release issue is closed",
            "not release close proof",
            "cannot close the release"
        })
        {
            Assert.Contains(marker, raw, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument validationDocument = ReadJson("artifacts", "final-release", "final-release-close-owner-approval-candidate-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-release-close-owner-approval-candidate-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-no-final-release-close-owner-approval-candidate-ready", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(4, validation.GetProperty("failedActionRequiredCount").GetInt32());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validation.GetProperty("isReleaseCloseProof").GetBoolean());

        using JsonDocument bundleDocument = ReadJson("artifacts", "final-release", "release-evidence-bundle.json");
        JsonElement candidateEvidence = bundleDocument.RootElement.GetProperty("evidenceItems").EnumerateArray().Single(static item => item.GetProperty("id").GetString() == "final-release-close-owner-approval-candidate");
        Assert.False(candidateEvidence.GetProperty("passed").GetBoolean());
        Assert.Contains("candidate only", candidateEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not release close proof", candidateEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package push", candidateEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        JsonElement sourceArtifacts = bundleDocument.RootElement.GetProperty("sourceArtifacts");
        Assert.Contains(sourceArtifacts.EnumerateArray(), item => item.GetString() == "artifacts/final-release/final-release-close-owner-approval-candidate-validation.json");

        string candidateMarkdown = ReadText("artifacts", "final-release", "final-release-close-owner-approval-candidate.md");
        Assert.Contains("Final Release Close Owner Approval Candidate", candidateMarkdown, StringComparison.Ordinal);
        Assert.Contains("cannot close the release", candidateMarkdown, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void FinalOwnerExecutionOwnerInputDraftKeepsOwnerFillPlaceholdersNonProof()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerExecutionRepairChecklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionRepairChecklist.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerExecutionRepairInputSkeleton.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionRepairInputSkeleton.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerExecutionOwnerInputDraft.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionOwnerInputDraft.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseEvidenceClassificationAudit.ps1"), "-Strict");

        using JsonDocument draftDocument = ReadJson("artifacts", "final-release", "final-owner-execution-owner-input-draft.json");
        JsonElement draft = draftDocument.RootElement;

        Assert.Equal("final-owner-execution-owner-input-draft", draft.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-fill-required", draft.GetProperty("draftState").GetString());
        Assert.Equal(8, draft.GetProperty("draftItemCount").GetInt32());
        Assert.Equal(8, draft.GetProperty("blockedDraftItemCount").GetInt32());
        Assert.Equal(0, draft.GetProperty("readyDraftItemCount").GetInt32());
        Assert.Equal(0, draft.GetProperty("readyForImportCount").GetInt32());
        Assert.True(draft.GetProperty("missingFileInputCount").GetInt32() >= 75);
        Assert.True(draft.GetProperty("missingSha256InputCount").GetInt32() >= 67);
        Assert.True(draft.GetProperty("missingIdentityInputCount").GetInt32() >= 80);
        Assert.True(draft.GetProperty("missingConfirmationCount").GetInt32() >= 96);
        Assert.True(draft.GetProperty("totalMissingInputCount").GetInt32() >= 318);
        Assert.False(draft.GetProperty("performsPublish").GetBoolean());
        Assert.False(draft.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(draft.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(draft.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(draft.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(draft.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(draft.GetProperty("isReleaseCloseProof").GetBoolean());

        JsonElement[] draftItems = draft.GetProperty("draftItems").EnumerateArray().ToArray();
        Assert.Equal(8, draftItems.Length);
        Assert.All(draftItems, item =>
        {
            Assert.Equal("owner-fill-required", item.GetProperty("ownerInputState").GetString());
            Assert.True(item.TryGetProperty("ownerProvidedFiles", out _));
            Assert.True(item.TryGetProperty("ownerProvidedSha256", out _));
            Assert.True(item.TryGetProperty("ownerProvidedIdentity", out _));
            Assert.True(item.GetProperty("ownerProvidedNonSubstituteConfirmations").EnumerateArray().Count() >= 8);
            Assert.False(item.GetProperty("readyForImport").GetBoolean());
            Assert.NotEmpty(item.GetProperty("expectedValidatorCommands").EnumerateArray());
            Assert.NotEmpty(item.GetProperty("expectedResultArtifacts").EnumerateArray());
            Assert.False(item.GetProperty("performsPublish").GetBoolean());
            Assert.False(item.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(item.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.False(item.GetProperty("isRuntimeExecutionProof").GetBoolean());
            Assert.False(item.GetProperty("isPostPublishProof").GetBoolean());
            Assert.False(item.GetProperty("isReleaseCloseProof").GetBoolean());
            Assert.Contains("Owner input draft only", item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Contains("not package push", item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        });

        string raw = draft.GetRawText();
        foreach (string marker in new[]
        {
            "ownerProvidedFiles",
            "ownerProvidedSha256",
            "ownerProvidedIdentity",
            "ownerProvidedNonSubstituteConfirmations",
            "missingFileInputCount",
            "missingSha256InputCount",
            "missingIdentityInputCount",
            "missingConfirmationCount",
            "readyForImport",
            "local feed",
            "ProjectReference",
            "direct .nupkg",
            "template",
            "draft",
            "dry-run",
            "dashboard",
            "candidate",
            "build-only",
            "dependency-probe-only",
            "blocked-by-cuda-driver"
        })
        {
            Assert.Contains(marker, raw, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument validationDocument = ReadJson("artifacts", "final-release", "final-owner-execution-owner-input-draft-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-owner-execution-owner-input-draft-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-fill-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(8, validation.GetProperty("failedActionRequiredCount").GetInt32());

        using JsonDocument bundleDocument = ReadJson("artifacts", "final-release", "release-evidence-bundle.json");
        JsonElement draftEvidence = bundleDocument.RootElement.GetProperty("evidenceItems").EnumerateArray().Single(static item => item.GetProperty("id").GetString() == "final-owner-execution-owner-input-draft");
        Assert.False(draftEvidence.GetProperty("passed").GetBoolean());
        Assert.Contains("owner input draft only", draftEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not runtime proof", draftEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package push", draftEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string draftMarkdown = ReadText("artifacts", "final-release", "final-owner-execution-owner-input-draft.md");
        Assert.Contains("Final Owner Execution Owner Input Draft", draftMarkdown, StringComparison.Ordinal);
        Assert.Contains("Owner input draft only", draftMarkdown, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void FinalOwnerExecutionRepairInputSkeletonExposesOwnerFillContractWithoutPromotingProof()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerExecutionRepairChecklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionRepairChecklist.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalOwnerExecutionRepairInputSkeleton.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalOwnerExecutionRepairInputSkeleton.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseEvidenceClassificationAudit.ps1"), "-Strict");

        using JsonDocument skeletonDocument = ReadJson("artifacts", "final-release", "final-owner-execution-repair-input-skeleton.json");
        JsonElement skeleton = skeletonDocument.RootElement;

        Assert.Equal("final-owner-execution-repair-input-skeleton", skeleton.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-evidence-input-required", skeleton.GetProperty("skeletonState").GetString());
        Assert.Equal(8, skeleton.GetProperty("skeletonItemCount").GetInt32());
        Assert.Equal(8, skeleton.GetProperty("blockedSkeletonItemCount").GetInt32());
        Assert.Equal(0, skeleton.GetProperty("readySkeletonItemCount").GetInt32());
        Assert.True(skeleton.GetProperty("fileInputCount").GetInt32() >= 75);
        Assert.True(skeleton.GetProperty("sha256InputCount").GetInt32() >= 67);
        Assert.True(skeleton.GetProperty("identityInputCount").GetInt32() >= 80);
        Assert.True(skeleton.GetProperty("nonSubstituteConfirmationCount").GetInt32() >= 96);
        Assert.False(skeleton.GetProperty("performsPublish").GetBoolean());
        Assert.False(skeleton.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(skeleton.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(skeleton.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(skeleton.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(skeleton.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(skeleton.GetProperty("isReleaseCloseProof").GetBoolean());

        JsonElement[] skeletonItems = skeleton.GetProperty("skeletonItems").EnumerateArray().ToArray();
        Assert.Equal(8, skeletonItems.Length);
        Assert.All(skeletonItems, item =>
        {
            Assert.Equal("owner-real-evidence-required", item.GetProperty("inputState").GetString());
            Assert.True(item.GetProperty("ownerMustFill").EnumerateArray().Count() >= 5);
            Assert.True(item.GetProperty("fileInputs").EnumerateArray().Count() >= 3);
            Assert.True(item.GetProperty("sha256Inputs").EnumerateArray().Count() >= 3);
            Assert.True(item.GetProperty("identityInputs").EnumerateArray().Count() >= 3);
            Assert.True(item.GetProperty("nonSubstituteConfirmations").EnumerateArray().Count() >= 8);
            Assert.NotEmpty(item.GetProperty("expectedValidatorCommands").EnumerateArray());
            Assert.NotEmpty(item.GetProperty("expectedResultArtifacts").EnumerateArray());
            Assert.True(item.GetProperty("evidenceRootPolicy").GetProperty("mustExist").GetBoolean());
            Assert.True(item.GetProperty("evidenceRootPolicy").GetProperty("mustStayUnderAllowedEvidenceRoot").GetBoolean());
            Assert.Equal("eng/Import-OwnerExternalProofExecutionResult.ps1", item.GetProperty("importPreflight").GetProperty("importer").GetString());
            Assert.True(item.GetProperty("importPreflight").GetProperty("requiresSha256Match").GetBoolean());
            Assert.False(item.GetProperty("performsPublish").GetBoolean());
            Assert.False(item.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(item.GetProperty("canPromoteRuntimeProof").GetBoolean());
            Assert.Contains("input skeleton only", item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Contains("not package push", item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        });

        string raw = skeleton.GetRawText();
        foreach (string marker in new[]
        {
            "stdoutPath",
            "stderrPath",
            "mergedTranscriptPath",
            "validatorOutputPath",
            "stdoutSha256",
            "stderrSha256",
            "mergedTranscriptSha256",
            "validatorOutputSha256",
            "exitCode",
            "hostIdentity",
            "packageIdentity",
            "ownerReviewer",
            "Import-OwnerExternalProofExecutionResult.ps1",
            "local feed",
            "ProjectReference",
            "direct .nupkg",
            "template",
            "draft",
            "dry-run",
            "dashboard",
            "candidate",
            "build-only",
            "dependency-probe-only",
            "blocked-by-cuda-driver"
        })
        {
            Assert.Contains(marker, raw, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument validationDocument = ReadJson("artifacts", "final-release", "final-owner-execution-repair-input-skeleton-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-owner-execution-repair-input-skeleton-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-real-evidence-input-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(8, validation.GetProperty("skeletonItemCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(8, validation.GetProperty("failedActionRequiredCount").GetInt32());

        using JsonDocument bundleDocument = ReadJson("artifacts", "final-release", "release-evidence-bundle.json");
        JsonElement skeletonEvidence = bundleDocument.RootElement.GetProperty("evidenceItems").EnumerateArray().Single(static item => item.GetProperty("id").GetString() == "final-owner-execution-repair-input-skeleton");
        Assert.False(skeletonEvidence.GetProperty("passed").GetBoolean());
        Assert.Contains("input skeleton only", skeletonEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not runtime proof", skeletonEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package push", skeletonEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        string skeletonMarkdown = ReadText("artifacts", "final-release", "final-owner-execution-repair-input-skeleton.md");
        Assert.Contains("Final Owner Execution Repair Input Skeleton", skeletonMarkdown, StringComparison.Ordinal);
        Assert.Contains("input skeleton only", skeletonMarkdown, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void FinalPublicPublishPreExecutionFreezeSummarizesAllRemainingOwnerActions()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalPostPublishCleanConsumerProofRecordContract.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalPostPublishCleanConsumerProofRecordContract.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalPostPublishCleanConsumerProofPreflight.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Import-FinalPostPublishCleanConsumerProofCandidate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalPostPublishCleanConsumerProofCandidate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalReleaseCloseOwnerApprovalContract.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalReleaseCloseOwnerApprovalContract.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalReleaseCloseOwnerApprovalPreflight.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Import-FinalReleaseCloseOwnerApprovalCandidate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalReleaseCloseOwnerApprovalCandidate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalPublicPublishPreExecutionFreeze.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalPublicPublishPreExecutionFreeze.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalPublicPublishOwnerActionWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalPublicPublishOwnerActionWorklist.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseEvidenceClassificationAudit.ps1"), "-Strict");

        using JsonDocument freezeDocument = ReadJson("artifacts", "final-release", "final-public-publish-pre-execution-freeze.json");
        JsonElement freeze = freezeDocument.RootElement;

        Assert.Equal("final-public-publish-pre-execution-freeze", freeze.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-public-publish-owner-action-required", freeze.GetProperty("freezeState").GetString());
        Assert.Equal(4, freeze.GetProperty("blockedCategoryCount").GetInt32());
        Assert.True(freeze.GetProperty("ownerActionCount").GetInt32() >= 10);
        Assert.True(freeze.GetProperty("requiredOwnerInputCount").GetInt32() > 0);
        Assert.True(freeze.GetProperty("missingPublicPackageProofCount").GetInt32() > 0);
        Assert.True(freeze.GetProperty("missingOwnerApprovalCount").GetInt32() > 0);
        Assert.True(freeze.GetProperty("missingReleaseNotesApprovalCount").GetInt32() > 0);
        Assert.True(freeze.GetProperty("missingFinalPackageUrlApprovalCount").GetInt32() > 0);
        Assert.True(freeze.GetProperty("mustRemainNonProofItemCount").GetInt32() >= 3);
        Assert.Equal(0, freeze.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(freeze.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        Assert.False(freeze.GetProperty("performsPublish").GetBoolean());
        Assert.False(freeze.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(freeze.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(freeze.GetProperty("isReleaseReady").GetBoolean());

        string raw = freeze.GetRawText();
        foreach (string marker in new[]
        {
            "public package identity",
            "外部 consumer",
            "runtime smoke",
            "release notes",
            "rollback",
            "release issue",
            "final public package URL",
            "classification audit",
            "failedBlockerCount=0 is not release ready"
        })
        {
            Assert.Contains(marker, raw, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument validationDocument = ReadJson("artifacts", "final-release", "final-public-publish-pre-execution-freeze-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-public-publish-pre-execution-freeze-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-public-publish-owner-action-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument worklistDocument = ReadJson("artifacts", "final-release", "final-public-publish-owner-action-worklist.json");
        JsonElement worklist = worklistDocument.RootElement;
        Assert.Equal("blocked-public-publish-owner-action-required", worklist.GetProperty("worklistState").GetString());
        Assert.True(worklist.GetProperty("ownerActionCount").GetInt32() >= 10);
        Assert.Equal(0, worklist.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(worklist.GetProperty("performsPublish").GetBoolean());
        Assert.False(worklist.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(worklist.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.All(worklist.GetProperty("actionItems").EnumerateArray(), action =>
        {
            Assert.False(action.GetProperty("performsPublish").GetBoolean());
            Assert.False(action.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(action.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.NotEmpty(action.GetProperty("ownerMustSupply").GetString()!);
            Assert.NotEmpty(action.GetProperty("validatorCommand").GetString()!);
            Assert.Contains("not proof", action.GetProperty("whyNotProof").GetString(), StringComparison.OrdinalIgnoreCase);
        });

        using JsonDocument bundleDocument = ReadJson("artifacts", "final-release", "release-evidence-bundle.json");
        JsonElement freezeEvidence = bundleDocument.RootElement.GetProperty("evidenceItems").EnumerateArray().Single(static item => item.GetProperty("id").GetString() == "final-public-publish-pre-execution-freeze");
        Assert.False(freezeEvidence.GetProperty("passed").GetBoolean());
        Assert.Contains("blocked dashboard", freezeEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not release ready", freezeEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        JsonElement worklistEvidence = bundleDocument.RootElement.GetProperty("evidenceItems").EnumerateArray().Single(static item => item.GetProperty("id").GetString() == "final-public-publish-owner-action-worklist");
        Assert.False(worklistEvidence.GetProperty("passed").GetBoolean());
        Assert.Contains("owner handoff", worklistEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        using JsonDocument auditDocument = ReadJson("artifacts", "final-release", "release-evidence-classification-audit.json");
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", auditDocument.RootElement.GetProperty("auditState").GetString());
        Assert.Equal(0, auditDocument.RootElement.GetProperty("findingCount").GetInt32());
    }

    [Fact]
    public void FinalPublicPublishCommandDryContractCannotPublishOrCloseRelease()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalPublicPublishPreExecutionFreeze.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalPublicPublishPreExecutionFreeze.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalPublicPublishOwnerActionWorklist.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalPublicPublishOwnerActionWorklist.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-FinalPublicPublishCommandDryContract.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalPublicPublishCommandDryContract.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseEvidenceClassificationAudit.ps1"), "-Strict");

        using JsonDocument contractDocument = ReadJson("artifacts", "final-release", "final-public-publish-command-dry-contract.json");
        JsonElement contract = contractDocument.RootElement;

        Assert.Equal("final-public-publish-command-dry-contract", contract.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-public-publish-owner-manual-execution-required", contract.GetProperty("contractState").GetString());
        Assert.True(contract.GetProperty("isDryContract").GetBoolean());
        Assert.False(contract.GetProperty("performsPublish").GetBoolean());
        Assert.False(contract.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(contract.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(contract.GetProperty("isReleaseReady").GetBoolean());
        Assert.False(contract.GetProperty("storesToken").GetBoolean());
        Assert.False(contract.GetProperty("executesDotnetNugetPush").GetBoolean());
        Assert.False(contract.GetProperty("executesGitHubReleaseUpload").GetBoolean());
        Assert.False(contract.GetProperty("closesReleaseIssue").GetBoolean());
        Assert.True(contract.GetProperty("commandGroupCount").GetInt32() >= 4);
        Assert.Equal(0, contract.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(contract.GetProperty("failedActionRequiredCount").GetInt32() > 0);

        string raw = contract.GetRawText();
        foreach (string marker in new[]
        {
            "dotnet nuget push",
            "gh release upload",
            "<OWNER_SUPPLIED_TOKEN>",
            "post-publish clean consumer",
            "final close validation",
            "performsPublish=false",
            "does not execute dotnet nuget push",
            "does not store tokens",
            "does not fake package URL/SHA256",
            "does not close release issue"
        })
        {
            Assert.Contains(marker, raw, StringComparison.OrdinalIgnoreCase);
        }

        Assert.All(contract.GetProperty("commandGroups").EnumerateArray(), group =>
        {
            Assert.True(group.GetProperty("manualOnly").GetBoolean());
            Assert.False(group.GetProperty("performsPublish").GetBoolean());
            Assert.False(group.GetProperty("canExecuteInAutomation").GetBoolean());
            Assert.NotEmpty(group.GetProperty("requiredBeforeExecution").EnumerateArray());
            Assert.Contains("only", group.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        });

        using JsonDocument validationDocument = ReadJson("artifacts", "final-release", "final-public-publish-command-dry-contract-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("final-public-publish-command-dry-contract-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-public-publish-owner-manual-execution-required", validation.GetProperty("validationState").GetString());
        Assert.True(validation.GetProperty("isDryContract").GetBoolean());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(validation.GetProperty("isReleaseReady").GetBoolean());
        Assert.False(validation.GetProperty("executesDotnetNugetPush").GetBoolean());
        Assert.False(validation.GetProperty("storesToken").GetBoolean());

        using JsonDocument bundleDocument = ReadJson("artifacts", "final-release", "release-evidence-bundle.json");
        JsonElement dryEvidence = bundleDocument.RootElement.GetProperty("evidenceItems").EnumerateArray().Single(static item => item.GetProperty("id").GetString() == "final-public-publish-command-dry-contract");
        Assert.False(dryEvidence.GetProperty("passed").GetBoolean());
        Assert.Contains("manual placeholder commands only", dryEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("isDryContract=true", dryEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not proof", dryEvidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);

        JsonElement sourceArtifacts = bundleDocument.RootElement.GetProperty("sourceArtifacts");
        Assert.Contains(sourceArtifacts.EnumerateArray(), item => item.GetString() == "artifacts/final-release/final-public-publish-command-dry-contract-validation.json");
    }

    [Fact]
    public void OwnerPublicPublishExecutionResultInputContractRequiresRealOwnerEvidenceAndHashes()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerPublicPublishExecutionResultInputContract.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerPublicPublishExecutionResultInputContract.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerPublicPublishExecutionResultInputTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerPublicPublishExecutionResultInputTemplate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerPublicPublishExecutionResultPreflight.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseEvidenceClassificationAudit.ps1"), "-Strict");

        using JsonDocument contractDocument = ReadJson("artifacts", "final-release", "owner-public-publish-execution-result-input-contract.json");
        JsonElement contract = contractDocument.RootElement;
        Assert.Equal("owner-public-publish-execution-result-input-contract", contract.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-public-publish-execution-result-input-required", contract.GetProperty("contractState").GetString());
        Assert.True(contract.GetProperty("requiredFieldCount").GetInt32() >= 100);
        Assert.False(contract.GetProperty("performsPublish").GetBoolean());
        Assert.False(contract.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(contract.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(contract.GetProperty("isReleaseReady").GetBoolean());

        string raw = contract.GetRawText();
        foreach (string marker in new[]
        {
            "publicPackageUrl",
            "publicPackageSha256",
            "githubReleaseAssetUrl",
            "githubReleaseAssetSha256",
            "nugetPushTranscriptPath",
            "cleanConsumerRestoreStdoutPath",
            "cleanConsumerBuildStdoutPath",
            "cleanConsumerRuntimeSmokeStdoutPath",
            "strictValidatorOutputPath",
            "hostCudaVersion",
            "hostTensorRtVersion",
            "hostCudnnVersion",
            "packageManagedPackageId",
            "packageNativeBridgePackageId",
            "packageRuntimePackageId",
            "releaseNotesPath",
            "rollbackDecision",
            "releaseIssueCloseDecision",
            "ownerReviewer",
            "ownerSignature",
            "noLocalFeedConfirmation",
            "noProjectReferenceConfirmation",
            "noDirectNupkgConfirmation",
            "noBuildOnlyConfirmation",
            "noDependencyProbeOnlyConfirmation",
            "noDryRunOnlyConfirmation",
            "noCandidateDashboardRunbookSubstitutionConfirmation"
        })
        {
            Assert.Contains(marker, raw, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument validationDocument = ReadJson("artifacts", "final-release", "owner-public-publish-execution-result-input-contract-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("owner-public-publish-execution-result-input-contract-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-public-publish-execution-result-input-required", validation.GetProperty("validationState").GetString());
        Assert.True(validation.GetProperty("requiredFieldCount").GetInt32() >= 100);
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() > 0);

        using JsonDocument preflightDocument = ReadJson("artifacts", "final-release", "owner-public-publish-execution-result-preflight.json");
        JsonElement preflight = preflightDocument.RootElement;
        Assert.Equal("owner-public-publish-execution-result-preflight", preflight.GetProperty("recordKind").GetString());
        Assert.Equal(0, preflight.GetProperty("readyCandidateCount").GetInt32());
        Assert.True(preflight.GetProperty("blockedRequiredFieldCount").GetInt32() >= 100);
        Assert.False(preflight.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(preflight.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument bundleDocument = ReadJson("artifacts", "final-release", "release-evidence-bundle.json");
        JsonElement[] evidenceItems = bundleDocument.RootElement.GetProperty("evidenceItems").EnumerateArray().ToArray();
        foreach (string id in new[]
        {
            "owner-public-publish-execution-result-input-contract",
            "owner-public-publish-execution-result-input-template",
            "owner-public-publish-execution-result-preflight"
        })
        {
            JsonElement evidence = evidenceItems.Single(item => item.GetProperty("id").GetString() == id);
            Assert.False(evidence.GetProperty("passed").GetBoolean());
            Assert.Contains("not proof", evidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument auditDocument = ReadJson("artifacts", "final-release", "release-evidence-classification-audit.json");
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", auditDocument.RootElement.GetProperty("auditState").GetString());
        Assert.Equal(0, auditDocument.RootElement.GetProperty("findingCount").GetInt32());
    }

    [Fact]
    public void OwnerPublicPublishExecutionResultCandidateCannotPromoteWithoutRealExternalProof()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerPublicPublishExecutionResultInputContract.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerPublicPublishExecutionResultInputTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerPublicPublishExecutionResultPreflight.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Import-OwnerPublicPublishExecutionResultCandidate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerPublicPublishExecutionResultCandidate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseEvidenceClassificationAudit.ps1"), "-Strict");

        using JsonDocument candidateDocument = ReadJson("artifacts", "final-release", "owner-public-publish-execution-result-candidate.json");
        JsonElement candidate = candidateDocument.RootElement;
        Assert.Equal("owner-public-publish-execution-result-candidate", candidate.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-public-publish-execution-result-input-required", candidate.GetProperty("candidateState").GetString());
        Assert.Equal(0, candidate.GetProperty("candidateItemCount").GetInt32());
        Assert.Equal(0, candidate.GetProperty("readyCandidateCount").GetInt32());
        Assert.False(candidate.GetProperty("performsPublish").GetBoolean());
        Assert.False(candidate.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(candidate.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(candidate.GetProperty("isReleaseReady").GetBoolean());
        Assert.False(candidate.GetProperty("isPostPublishProof").GetBoolean());

        using JsonDocument validationDocument = ReadJson("artifacts", "final-release", "owner-public-publish-execution-result-candidate-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("owner-public-publish-execution-result-candidate-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal(0, validation.GetProperty("candidateItemCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument bundleDocument = ReadJson("artifacts", "final-release", "release-evidence-bundle.json");
        JsonElement evidence = bundleDocument.RootElement.GetProperty("evidenceItems").EnumerateArray().Single(item => item.GetProperty("id").GetString() == "owner-public-publish-execution-result-candidate");
        Assert.False(evidence.GetProperty("passed").GetBoolean());
        Assert.Contains("candidate only", evidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package push", evidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void PostPublishAndFinalCloseRealInputImportsRemainBlockedWithoutOwnerEvidence()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerPublicPublishExecutionResultInputContract.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-OwnerPublicPublishExecutionResultInputTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerPublicPublishExecutionResultPreflight.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Import-OwnerPublicPublishExecutionResultCandidate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-OwnerPublicPublishExecutionResultCandidate.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Import-PostPublishCleanConsumerRealProofFromOwnerResult.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-PostPublishCleanConsumerRealProofFromOwnerResult.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Import-FinalReleaseCloseApprovalRealInputFromOwnerResult.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-FinalReleaseCloseApprovalRealInputFromOwnerResult.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseEvidenceBundle.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-ReleaseEvidenceClassificationAudit.ps1"), "-Strict");

        using JsonDocument postPublishDocument = ReadJson("artifacts", "final-release", "post-publish-clean-consumer-real-proof-from-owner-result-validation.json");
        JsonElement postPublish = postPublishDocument.RootElement;
        Assert.Equal("post-publish-clean-consumer-real-proof-from-owner-result-validation", postPublish.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-post-publish-clean-consumer-real-owner-proof-required", postPublish.GetProperty("validationState").GetString());
        Assert.Equal(0, postPublish.GetProperty("readyProofCount").GetInt32());
        Assert.Equal(0, postPublish.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(postPublish.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        Assert.False(postPublish.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(postPublish.GetProperty("isPostPublishProof").GetBoolean());

        using JsonDocument closeDocument = ReadJson("artifacts", "final-release", "final-release-close-approval-real-input-from-owner-result-validation.json");
        JsonElement close = closeDocument.RootElement;
        Assert.Equal("final-release-close-approval-real-input-from-owner-result-validation", close.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-final-release-close-owner-approval-real-input-required", close.GetProperty("validationState").GetString());
        Assert.Equal(0, close.GetProperty("readyCloseApprovalCount").GetInt32());
        Assert.Equal(0, close.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(close.GetProperty("failedActionRequiredCount").GetInt32() > 0);
        Assert.False(close.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(close.GetProperty("isReleaseReady").GetBoolean());
        Assert.False(close.GetProperty("isReleaseCloseProof").GetBoolean());
        Assert.False(close.GetProperty("closesReleaseIssue").GetBoolean());

        using JsonDocument bundleDocument = ReadJson("artifacts", "final-release", "release-evidence-bundle.json");
        JsonElement[] evidenceItems = bundleDocument.RootElement.GetProperty("evidenceItems").EnumerateArray().ToArray();
        foreach (string id in new[]
        {
            "post-publish-clean-consumer-real-proof-from-owner-result",
            "final-release-close-approval-real-input-from-owner-result"
        })
        {
            JsonElement evidence = evidenceItems.Single(item => item.GetProperty("id").GetString() == id);
            Assert.False(evidence.GetProperty("passed").GetBoolean());
            Assert.Contains("not proof", evidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
            Assert.Contains("cannot close", evidence.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument auditDocument = ReadJson("artifacts", "final-release", "release-evidence-classification-audit.json");
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", auditDocument.RootElement.GetProperty("auditState").GetString());
        Assert.Equal(0, auditDocument.RootElement.GetProperty("findingCount").GetInt32());
    }

    private static JsonDocument ReadJson(params string[] pathParts)
    {
        return JsonDocument.Parse(ReadText(pathParts));
    }

    private static string ReadText(params string[] pathParts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
    {
        using Process process = new();
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
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptPath}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        return stdout;
    }
}

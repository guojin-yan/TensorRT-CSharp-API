using System.Text.Json;
using System.Text.Json.Nodes;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerExternalProofExecutionBundleTests
{
    [Fact]
    public void OwnerExternalProofExecutionBundleExportsBlockedExecutionPlan()
    {
        RealExternalProofExecutionAndCloseOwnerInputPipeline.Run();

        using JsonDocument bundleDocument = RealExternalProofExecutionAndCloseOwnerInputPipeline.ReadFinalReleaseJson("owner-external-proof-execution-bundle.json");
        JsonElement bundle = bundleDocument.RootElement;

        Assert.Equal("owner-external-proof-execution-bundle", bundle.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-external-proof-execution-required", bundle.GetProperty("bundleState").GetString());
        Assert.Equal(6, bundle.GetProperty("executionBundleItemCount").GetInt32());
        Assert.Equal(6, bundle.GetProperty("blockedExecutionBundleItemCount").GetInt32());
        Assert.Equal(0, bundle.GetProperty("readyExecutionBundleItemCount").GetInt32());
        Assert.True(bundle.GetProperty("remainingCloseGapCount").GetInt32() >= 200);
        RealExternalProofExecutionAndCloseOwnerInputPipeline.AssertFalseProofPublishCloseFlags(bundle);

        JsonElement firstItem = bundle.GetProperty("executionBundleItems").EnumerateArray().First();
        Assert.Equal("blocked-owner-external-proof-execution-required", firstItem.GetProperty("bundleItemState").GetString());
        Assert.True(firstItem.GetProperty("remainingCloseGapCount").GetInt32() >= 30);
        Assert.False(firstItem.GetProperty("canPromoteRuntimeProof").GetBoolean());

        using JsonDocument validationDocument = RealExternalProofExecutionAndCloseOwnerInputPipeline.ReadFinalReleaseJson("owner-external-proof-execution-bundle-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("owner-external-proof-execution-bundle-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-external-proof-execution-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("failedActionRequiredCount").GetInt32());
        RealExternalProofExecutionAndCloseOwnerInputPipeline.AssertFalseProofPublishCloseFlags(validation);
    }

    [Fact]
    public void RealProofRecordCandidateFromOwnerResultImportKeepsFormalRunBlocked()
    {
        RealExternalProofExecutionAndCloseOwnerInputPipeline.Run();

        using JsonDocument candidateDocument = RealExternalProofExecutionAndCloseOwnerInputPipeline.ReadFinalReleaseJson("real-proof-record-candidate-from-owner-result-import.json");
        JsonElement candidateRecord = candidateDocument.RootElement;

        Assert.Equal("real-proof-record-candidate-from-owner-result-import", candidateRecord.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-result-import-candidate-required", candidateRecord.GetProperty("candidateState").GetString());
        Assert.Equal(6, candidateRecord.GetProperty("sourceCandidateContractCount").GetInt32());
        Assert.Equal(0, candidateRecord.GetProperty("sourceReadyCandidateContractCount").GetInt32());
        Assert.Equal(0, candidateRecord.GetProperty("candidateCount").GetInt32());
        Assert.Equal(0, candidateRecord.GetProperty("strictValidatorReadyCandidateCount").GetInt32());
        Assert.Equal(0, candidateRecord.GetProperty("packageConsumerRuntimeCandidateCount").GetInt32());
        Assert.Equal(0, candidateRecord.GetProperty("postPublishVerificationCandidateCount").GetInt32());
        RealExternalProofExecutionAndCloseOwnerInputPipeline.AssertFalseProofPublishCloseFlags(candidateRecord);
        Assert.False(candidateRecord.GetProperty("isPostPublishProof").GetBoolean());

        using JsonDocument validationDocument = RealExternalProofExecutionAndCloseOwnerInputPipeline.ReadFinalReleaseJson("real-proof-record-candidate-from-owner-result-import-validation.json");
        JsonElement validation = validationDocument.RootElement;

        Assert.Equal("real-proof-record-candidate-from-owner-result-import-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-result-import-candidate-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("candidateCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("strictValidatorReadyCandidateCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        RealExternalProofExecutionAndCloseOwnerInputPipeline.AssertFalseProofPublishCloseFlags(validation);
        Assert.False(validation.GetProperty("isPostPublishProof").GetBoolean());
    }
}

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerExternalProofExecutionResultImportTests
{
    [Fact]
    public void OwnerExternalProofExecutionResultImportRequiresRealEvidence()
    {
        RealExternalProofExecutionAndCloseOwnerInputPipeline.Run();

        using JsonDocument importDocument = RealExternalProofExecutionAndCloseOwnerInputPipeline.ReadFinalReleaseJson("owner-external-proof-execution-result-import.json");
        JsonElement import = importDocument.RootElement;

        Assert.Equal("owner-external-proof-execution-result-import", import.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-external-proof-execution-result-required", import.GetProperty("importState").GetString());
        Assert.Equal(6, import.GetProperty("resultImportItemCount").GetInt32());
        Assert.Equal(6, import.GetProperty("blockedResultImportItemCount").GetInt32());
        Assert.Equal(0, import.GetProperty("readyResultImportItemCount").GetInt32());
        Assert.Equal(0, import.GetProperty("promotableResultImportItemCount").GetInt32());
        Assert.Equal(6, import.GetProperty("ownerExternalProofResultLaneCount").GetInt32());
        Assert.Equal(6, import.GetProperty("ownerExternalProofResultBlockedLaneCount").GetInt32());
        Assert.Equal(0, import.GetProperty("ownerExternalProofResultReadyLaneCount").GetInt32());
        Assert.Equal(0, import.GetProperty("ownerExternalProofResultPromotableLaneCount").GetInt32());
        Assert.True(import.GetProperty("missingRealEvidenceCount").GetInt32() >= 100);
        Assert.True(import.GetProperty("fileMissingCount").GetInt32() >= 30);
        Assert.True(import.GetProperty("invalidSha256Count").GetInt32() >= 30);
        Assert.Equal(0, import.GetProperty("hashMismatchCount").GetInt32());
        Assert.Equal(0, import.GetProperty("outsideAllowedEvidenceRootCount").GetInt32());
        Assert.True(import.GetProperty("forbiddenSubstituteFindingCount").GetInt32() >= 0);
        RealExternalProofExecutionAndCloseOwnerInputPipeline.AssertFalseProofPublishCloseFlags(import);

        JsonElement firstItem = import.GetProperty("resultImportItems").EnumerateArray().First();
        Assert.Equal("blocked-owner-external-proof-execution-result-required", firstItem.GetProperty("importItemState").GetString());
        Assert.False(firstItem.GetProperty("readyForRealProofRecordImport").GetBoolean());
        Assert.False(firstItem.GetProperty("canPromoteLaneResult").GetBoolean());
        Assert.True(firstItem.GetProperty("forbiddenSubstituteFindingCount").GetInt32() >= 0);
        Assert.False(firstItem.GetProperty("exitCodeZero").GetBoolean());
        Assert.False(firstItem.GetProperty("ownerReviewReady").GetBoolean());
        Assert.False(firstItem.GetProperty("nonSubstituteConfirmationsReady").GetBoolean());
        Assert.True(firstItem.GetProperty("fileMissingCount").GetInt32() >= 5);
        Assert.True(firstItem.GetProperty("invalidSha256Count").GetInt32() >= 5);
        Assert.Equal(0, firstItem.GetProperty("hashMismatchCount").GetInt32());
        Assert.Equal(0, firstItem.GetProperty("outsideAllowedEvidenceRootCount").GetInt32());

        JsonElement[] laneSummaries = import.GetProperty("laneSummaries").EnumerateArray().ToArray();
        Assert.Equal(6, laneSummaries.Length);
        Assert.Contains(laneSummaries, lane => lane.GetProperty("proofLane").GetString() == "package-consumer-runtime");
        Assert.Contains(laneSummaries, lane => lane.GetProperty("proofLane").GetString() == "post-publish-verification");

        using JsonDocument validationDocument = RealExternalProofExecutionAndCloseOwnerInputPipeline.ReadFinalReleaseJson("owner-external-proof-execution-result-import-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("owner-external-proof-execution-result-import-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-external-proof-execution-result-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("failedActionRequiredCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("ownerExternalProofResultLaneCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("ownerExternalProofResultPromotableLaneCount").GetInt32());
        Assert.True(validation.GetProperty("fileMissingCount").GetInt32() >= 30);
        Assert.True(validation.GetProperty("invalidSha256Count").GetInt32() >= 30);
        Assert.Equal(0, validation.GetProperty("hashMismatchCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("outsideAllowedEvidenceRootCount").GetInt32());
        Assert.True(validation.GetProperty("forbiddenSubstituteFindingCount").GetInt32() >= 0);
        RealExternalProofExecutionAndCloseOwnerInputPipeline.AssertFalseProofPublishCloseFlags(validation);
    }

    [Fact]
    public void OwnerExternalProofExecutionResultImportClassifiesHashMismatch()
    {
        OwnerResultImportTestContext context = OwnerResultImportTestContext.Create("hash-mismatch");
        string inputPath = context.CreateSingleLaneInput(
            evidenceRoot: context.AllowedEvidenceRoot,
            makeHashMismatch: true,
            useForbiddenSubstituteText: false);

        using JsonDocument importDocument = context.RunImport(inputPath);
        JsonElement import = importDocument.RootElement;
        JsonElement item = import.GetProperty("resultImportItems").EnumerateArray().First();

        Assert.Equal("blocked-owner-external-proof-execution-result-required", item.GetProperty("importItemState").GetString());
        Assert.False(item.GetProperty("readyForRealProofRecordImport").GetBoolean());
        Assert.Equal(0, item.GetProperty("fileMissingCount").GetInt32());
        Assert.Equal(0, item.GetProperty("invalidSha256Count").GetInt32());
        Assert.True(item.GetProperty("hashMismatchCount").GetInt32() >= 1);
        Assert.Equal(0, item.GetProperty("outsideAllowedEvidenceRootCount").GetInt32());
        Assert.Equal(0, item.GetProperty("forbiddenSubstituteFindingCount").GetInt32());
        Assert.Equal(0, import.GetProperty("readyResultImportItemCount").GetInt32());
        RealExternalProofExecutionAndCloseOwnerInputPipeline.AssertFalseProofPublishCloseFlags(import);

        using JsonDocument validatorDocument = context.RunRealExternalValidator();
        JsonElement contract = validatorDocument.RootElement.GetProperty("candidateContracts").EnumerateArray().First();
        Assert.True(contract.GetProperty("hashMismatchCount").GetInt32() >= 1);
        Assert.Contains(contract.GetProperty("blockedReasons").EnumerateArray(), reason => reason.GetString()!.Contains("SHA256 mismatches", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void OwnerExternalProofExecutionResultImportBlocksOutsideAllowedEvidenceRoot()
    {
        OwnerResultImportTestContext context = OwnerResultImportTestContext.Create("outside-allowed-root");
        string inputPath = context.CreateSingleLaneInput(
            evidenceRoot: context.DisallowedEvidenceRoot,
            makeHashMismatch: false,
            useForbiddenSubstituteText: false);

        using JsonDocument importDocument = context.RunImport(inputPath);
        JsonElement import = importDocument.RootElement;
        JsonElement item = import.GetProperty("resultImportItems").EnumerateArray().First();

        Assert.Equal("blocked-owner-external-proof-execution-result-required", item.GetProperty("importItemState").GetString());
        Assert.False(item.GetProperty("readyForRealProofRecordImport").GetBoolean());
        Assert.Equal(0, item.GetProperty("fileMissingCount").GetInt32());
        Assert.Equal(0, item.GetProperty("invalidSha256Count").GetInt32());
        Assert.Equal(0, item.GetProperty("hashMismatchCount").GetInt32());
        Assert.True(item.GetProperty("outsideAllowedEvidenceRootCount").GetInt32() >= 1);
        Assert.Equal(0, import.GetProperty("readyResultImportItemCount").GetInt32());
        RealExternalProofExecutionAndCloseOwnerInputPipeline.AssertFalseProofPublishCloseFlags(import);

        using JsonDocument validatorDocument = context.RunRealExternalValidator();
        JsonElement contract = validatorDocument.RootElement.GetProperty("candidateContracts").EnumerateArray().First();
        Assert.True(contract.GetProperty("outsideAllowedEvidenceRootCount").GetInt32() >= 1);
        Assert.Contains(contract.GetProperty("blockedReasons").EnumerateArray(), reason => reason.GetString()!.Contains("outside allowed roots", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void OwnerExternalProofExecutionResultImportBlocksForbiddenSubstitutes()
    {
        OwnerResultImportTestContext context = OwnerResultImportTestContext.Create("forbidden-substitute");
        string inputPath = context.CreateSingleLaneInput(
            evidenceRoot: context.AllowedEvidenceRoot,
            makeHashMismatch: false,
            useForbiddenSubstituteText: true);

        using JsonDocument importDocument = context.RunImport(inputPath);
        JsonElement import = importDocument.RootElement;
        JsonElement item = import.GetProperty("resultImportItems").EnumerateArray().First();

        Assert.Equal("blocked-owner-external-proof-execution-result-required", item.GetProperty("importItemState").GetString());
        Assert.False(item.GetProperty("readyForRealProofRecordImport").GetBoolean());
        Assert.Equal(0, item.GetProperty("fileMissingCount").GetInt32());
        Assert.Equal(0, item.GetProperty("invalidSha256Count").GetInt32());
        Assert.Equal(0, item.GetProperty("hashMismatchCount").GetInt32());
        Assert.Equal(0, item.GetProperty("outsideAllowedEvidenceRootCount").GetInt32());
        Assert.True(item.GetProperty("forbiddenSubstituteFindingCount").GetInt32() >= 1);
        Assert.False(item.GetProperty("nonSubstituteConfirmationsReady").GetBoolean());
        Assert.Equal(0, import.GetProperty("readyResultImportItemCount").GetInt32());
        RealExternalProofExecutionAndCloseOwnerInputPipeline.AssertFalseProofPublishCloseFlags(import);

        using JsonDocument validatorDocument = context.RunRealExternalValidator();
        JsonElement contract = validatorDocument.RootElement.GetProperty("candidateContracts").EnumerateArray().First();
        Assert.True(contract.GetProperty("forbiddenSubstituteFindingCount").GetInt32() >= 1);
        Assert.Contains(contract.GetProperty("blockedReasons").EnumerateArray(), reason => reason.GetString()!.Contains("forbidden substitute", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void OwnerExternalProofExecutionResultImportAllowsSingleLaneReadyWithoutPromotion()
    {
        OwnerResultImportTestContext context = OwnerResultImportTestContext.Create("single-lane-ready");
        string inputPath = context.CreateSingleLaneInput(
            evidenceRoot: context.AllowedEvidenceRoot,
            makeHashMismatch: false,
            useForbiddenSubstituteText: false);

        using JsonDocument importDocument = context.RunImport(inputPath);
        JsonElement import = importDocument.RootElement;
        JsonElement item = import.GetProperty("resultImportItems").EnumerateArray().First();

        Assert.Equal("blocked-owner-external-proof-execution-result-required", import.GetProperty("importState").GetString());
        Assert.Equal(1, import.GetProperty("readyResultImportItemCount").GetInt32());
        Assert.Equal(1, import.GetProperty("ownerExternalProofResultReadyLaneCount").GetInt32());
        Assert.Equal(0, import.GetProperty("promotableResultImportItemCount").GetInt32());
        Assert.Equal(0, import.GetProperty("ownerExternalProofResultPromotableLaneCount").GetInt32());
        Assert.Equal("owner-external-proof-execution-result-ready", item.GetProperty("importItemState").GetString());
        Assert.True(item.GetProperty("readyForRealProofRecordImport").GetBoolean());
        Assert.False(item.GetProperty("canPromoteLaneResult").GetBoolean());
        Assert.Equal(0, item.GetProperty("fileMissingCount").GetInt32());
        Assert.Equal(0, item.GetProperty("invalidSha256Count").GetInt32());
        Assert.Equal(0, item.GetProperty("hashMismatchCount").GetInt32());
        Assert.Equal(0, item.GetProperty("outsideAllowedEvidenceRootCount").GetInt32());
        Assert.Equal(0, item.GetProperty("forbiddenSubstituteFindingCount").GetInt32());
        RealExternalProofExecutionAndCloseOwnerInputPipeline.AssertFalseProofPublishCloseFlags(import);

        using JsonDocument validatorDocument = context.RunRealExternalValidator();
        JsonElement validator = validatorDocument.RootElement;
        JsonElement contract = validator.GetProperty("candidateContracts").EnumerateArray().First();
        Assert.Equal("blocked-real-external-proof-record-import-required", validator.GetProperty("validatorState").GetString());
        Assert.Equal(1, validator.GetProperty("readyCandidateContractCount").GetInt32());
        Assert.Equal("real-external-proof-record-import-ready", contract.GetProperty("contractState").GetString());
        Assert.True(contract.GetProperty("readyForPromotionGuard").GetBoolean());
        Assert.False(contract.GetProperty("canPromoteRuntimeProof").GetBoolean());
    }

    [Fact]
    public void ReadyOwnerResultContractProjectsOnlyCandidateNotProof()
    {
        OwnerResultImportTestContext context = OwnerResultImportTestContext.Create("ready-contract-candidate-only");
        string inputPath = context.CreateSingleLaneInput(
            evidenceRoot: context.AllowedEvidenceRoot,
            makeHashMismatch: false,
            useForbiddenSubstituteText: false);

        using JsonDocument importDocument = context.RunImport(inputPath);
        Assert.Equal(1, importDocument.RootElement.GetProperty("readyResultImportItemCount").GetInt32());

        context.RunRealExternalValidator().Dispose();
        using JsonDocument candidateDocument = context.RunCandidateBridge();
        JsonElement candidateRecord = candidateDocument.RootElement;

        Assert.Equal("real-proof-record-candidate-from-owner-result-import", candidateRecord.GetProperty("recordKind").GetString());
        Assert.Equal("owner-result-import-candidate-ready-for-strict-validator", candidateRecord.GetProperty("candidateState").GetString());
        Assert.Equal(1, candidateRecord.GetProperty("candidateCount").GetInt32());
        Assert.Equal(1, candidateRecord.GetProperty("strictValidatorReadyCandidateCount").GetInt32());
        Assert.Equal(0, candidateRecord.GetProperty("postPublishVerificationCandidateCount").GetInt32());
        RealExternalProofExecutionAndCloseOwnerInputPipeline.AssertFalseProofPublishCloseFlags(candidateRecord);
        Assert.False(candidateRecord.GetProperty("isPostPublishProof").GetBoolean());

        JsonElement candidate = candidateRecord.GetProperty("candidateItems").EnumerateArray().Single();
        Assert.Equal("ready-for-strict-validator-input", candidate.GetProperty("candidateState").GetString());
        Assert.Equal("package-consumer-runtime", candidate.GetProperty("proofLane").GetString());
        Assert.True(candidate.GetProperty("strictValidatorInputReady").GetBoolean());
        Assert.True(candidate.GetProperty("ownerReviewReady").GetBoolean());
        Assert.True(candidate.GetProperty("nonSubstituteConfirmationsReady").GetBoolean());
        Assert.True(candidate.GetProperty("fileEvidenceCheckCount").GetInt32() >= 5);
        Assert.Equal(0, candidate.GetProperty("fileMissingCount").GetInt32());
        Assert.Equal(0, candidate.GetProperty("invalidSha256Count").GetInt32());
        Assert.Equal(0, candidate.GetProperty("hashMismatchCount").GetInt32());
        Assert.Equal(0, candidate.GetProperty("outsideAllowedEvidenceRootCount").GetInt32());
        Assert.Equal(0, candidate.GetProperty("forbiddenSubstituteFindingCount").GetInt32());
        Assert.False(candidate.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(candidate.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(candidate.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(candidate.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument validationDocument = context.RunCandidateBridgeValidation();
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("real-proof-record-candidate-from-owner-result-import-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("owner-result-import-candidate-ready-for-strict-validator", validation.GetProperty("validationState").GetString());
        Assert.Equal(1, validation.GetProperty("candidateCount").GetInt32());
        Assert.Equal(1, validation.GetProperty("strictValidatorReadyCandidateCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(validation.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(validation.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    [Fact]
    public void NonReadyOwnerResultContractDoesNotCreateCandidate()
    {
        OwnerResultImportTestContext context = OwnerResultImportTestContext.Create("non-ready-no-candidate");
        string inputPath = context.CreateSingleLaneInput(
            evidenceRoot: context.AllowedEvidenceRoot,
            makeHashMismatch: true,
            useForbiddenSubstituteText: false);

        using JsonDocument importDocument = context.RunImport(inputPath);
        Assert.Equal(0, importDocument.RootElement.GetProperty("readyResultImportItemCount").GetInt32());

        context.RunRealExternalValidator().Dispose();
        using JsonDocument candidateDocument = context.RunCandidateBridge();
        JsonElement candidateRecord = candidateDocument.RootElement;

        Assert.Equal("blocked-owner-result-import-candidate-required", candidateRecord.GetProperty("candidateState").GetString());
        Assert.Equal(0, candidateRecord.GetProperty("candidateCount").GetInt32());
        Assert.Equal(0, candidateRecord.GetProperty("strictValidatorReadyCandidateCount").GetInt32());
        RealExternalProofExecutionAndCloseOwnerInputPipeline.AssertFalseProofPublishCloseFlags(candidateRecord);
        Assert.False(candidateRecord.GetProperty("isPostPublishProof").GetBoolean());

        using JsonDocument validationDocument = context.RunCandidateBridgeValidation();
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("blocked-owner-result-import-candidate-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("candidateCount").GetInt32());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.False(validation.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validation.GetProperty("isPostPublishProof").GetBoolean());
    }

    [Fact]
    public void PostPublishLaneRemainsSeparateFromPackageConsumerRuntime()
    {
        OwnerResultImportTestContext context = OwnerResultImportTestContext.Create("post-publish-lane-separate");
        string inputPath = context.CreateSingleLaneInput(
            evidenceRoot: context.AllowedEvidenceRoot,
            makeHashMismatch: false,
            useForbiddenSubstituteText: false,
            proofLaneOverride: "post-publish-verification");

        using JsonDocument importDocument = context.RunImport(inputPath);
        Assert.Equal(1, importDocument.RootElement.GetProperty("readyResultImportItemCount").GetInt32());

        context.RunRealExternalValidator().Dispose();
        using JsonDocument candidateDocument = context.RunCandidateBridge();
        JsonElement candidateRecord = candidateDocument.RootElement;

        Assert.Equal(1, candidateRecord.GetProperty("candidateCount").GetInt32());
        Assert.Equal(0, candidateRecord.GetProperty("packageConsumerRuntimeCandidateCount").GetInt32());
        Assert.Equal(1, candidateRecord.GetProperty("postPublishVerificationCandidateCount").GetInt32());
        Assert.False(candidateRecord.GetProperty("isPostPublishProof").GetBoolean());

        JsonElement candidate = candidateRecord.GetProperty("candidateItems").EnumerateArray().Single();
        Assert.Equal("post-publish-verification", candidate.GetProperty("proofLane").GetString());
        Assert.Contains("package-consumer-runtime cannot substitute", candidate.GetProperty("postPublishBoundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.False(candidate.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(candidate.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(candidate.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(candidate.GetProperty("canCloseReleaseIssue").GetBoolean());

        using JsonDocument validationDocument = context.RunCandidateBridgeValidation();
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("owner-result-import-candidate-ready-for-strict-validator", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("packageConsumerRuntimeCandidateCount").GetInt32());
        Assert.Equal(1, validation.GetProperty("postPublishVerificationCandidateCount").GetInt32());
        Assert.False(validation.GetProperty("isPostPublishProof").GetBoolean());
    }
}

[Collection("ReleaseCloseProofArtifacts")]
public sealed class RealExternalProofRecordImportValidatorTests
{
    [Fact]
    public void RealExternalProofRecordImportValidatorKeepsContractsBlocked()
    {
        RealExternalProofExecutionAndCloseOwnerInputPipeline.Run();

        using JsonDocument validatorDocument = RealExternalProofExecutionAndCloseOwnerInputPipeline.ReadFinalReleaseJson("real-external-proof-record-import-validator.json");
        JsonElement validator = validatorDocument.RootElement;

        Assert.Equal("real-external-proof-record-import-validator", validator.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-external-proof-record-import-required", validator.GetProperty("validatorState").GetString());
        Assert.Equal(6, validator.GetProperty("candidateContractCount").GetInt32());
        Assert.Equal(6, validator.GetProperty("blockedCandidateContractCount").GetInt32());
        Assert.Equal(0, validator.GetProperty("readyCandidateContractCount").GetInt32());
        Assert.True(validator.GetProperty("blockedReasonCount").GetInt32() >= 6);
        RealExternalProofExecutionAndCloseOwnerInputPipeline.AssertFalseProofPublishCloseFlags(validator);

        JsonElement firstContract = validator.GetProperty("candidateContracts").EnumerateArray().First();
        Assert.Equal("blocked-real-external-proof-record-import-required", firstContract.GetProperty("contractState").GetString());
        Assert.False(firstContract.GetProperty("readyForPromotionGuard").GetBoolean());

        using JsonDocument validationDocument = RealExternalProofExecutionAndCloseOwnerInputPipeline.ReadFinalReleaseJson("real-external-proof-record-import-validator-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("real-external-proof-record-import-validator-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-external-proof-record-import-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("failedActionRequiredCount").GetInt32());
        RealExternalProofExecutionAndCloseOwnerInputPipeline.AssertFalseProofPublishCloseFlags(validation);
    }
}

[Collection("ReleaseCloseProofArtifacts")]
public sealed class ReleaseCloseOwnerInputBridgeTests
{
    [Fact]
    public void ReleaseCloseOwnerInputBridgeAndEvidenceBundleKeepCloseBlocked()
    {
        RealExternalProofExecutionAndCloseOwnerInputPipeline.Run();

        using JsonDocument bridgeDocument = RealExternalProofExecutionAndCloseOwnerInputPipeline.ReadFinalReleaseJson("release-close-owner-input-bridge.json");
        JsonElement bridge = bridgeDocument.RootElement;

        Assert.Equal("release-close-owner-input-bridge", bridge.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-release-close-owner-input-required", bridge.GetProperty("bridgeState").GetString());
        Assert.Equal(7, bridge.GetProperty("bridgeGateCount").GetInt32());
        Assert.Equal(6, bridge.GetProperty("blockedBridgeGateCount").GetInt32());
        Assert.Equal(1, bridge.GetProperty("readyBridgeGateCount").GetInt32());
        RealExternalProofExecutionAndCloseOwnerInputPipeline.AssertFalseProofPublishCloseFlags(bridge);

        using JsonDocument validationDocument = RealExternalProofExecutionAndCloseOwnerInputPipeline.ReadFinalReleaseJson("release-close-owner-input-bridge-validation.json");
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("release-close-owner-input-bridge-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-release-close-owner-input-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal(6, validation.GetProperty("failedActionRequiredCount").GetInt32());
        RealExternalProofExecutionAndCloseOwnerInputPipeline.AssertFalseProofPublishCloseFlags(validation);

        using JsonDocument evidenceDocument = RealExternalProofExecutionAndCloseOwnerInputPipeline.ReadFinalReleaseJson("release-evidence-bundle.json");
        JsonElement evidence = evidenceDocument.RootElement;

        Assert.Equal("blocked-owner-external-proof-execution-required", evidence.GetProperty("ownerExternalProofExecutionBundleState").GetString());
        Assert.Equal("blocked-owner-external-proof-execution-result-required", evidence.GetProperty("ownerExternalProofExecutionResultImportState").GetString());
        Assert.Equal("blocked-real-external-proof-record-import-required", evidence.GetProperty("realExternalProofRecordImportValidatorState").GetString());
        Assert.Equal("blocked-owner-result-import-candidate-required", evidence.GetProperty("realProofRecordCandidateFromOwnerResultImportState").GetString());
        Assert.Equal("blocked-owner-result-import-candidate-required", evidence.GetProperty("realProofRecordCandidateFromOwnerResultImportValidationState").GetString());
        Assert.Equal(0, evidence.GetProperty("realProofRecordCandidateFromOwnerResultImportCandidateCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("realProofRecordCandidateFromOwnerResultImportStrictValidatorReadyCandidateCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("realProofRecordCandidateFromOwnerResultImportPackageConsumerCandidateCount").GetInt32());
        Assert.Equal(0, evidence.GetProperty("realProofRecordCandidateFromOwnerResultImportPostPublishCandidateCount").GetInt32());
        Assert.Equal("blocked-release-close-owner-input-required", evidence.GetProperty("releaseCloseOwnerInputBridgeState").GetString());
        Assert.False(evidence.GetProperty("ownerExternalProofExecutionBundleCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("ownerExternalProofExecutionResultImportCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("realExternalProofRecordImportValidatorCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("realProofRecordCandidateFromOwnerResultImportCanPromoteRuntimeProof").GetBoolean());
        Assert.False(evidence.GetProperty("realProofRecordCandidateFromOwnerResultImportCanCloseReleaseIssue").GetBoolean());
        Assert.False(evidence.GetProperty("realProofRecordCandidateFromOwnerResultImportIsRuntimeExecutionProof").GetBoolean());
        Assert.False(evidence.GetProperty("realProofRecordCandidateFromOwnerResultImportIsPostPublishProof").GetBoolean());
        Assert.False(evidence.GetProperty("releaseCloseOwnerInputBridgeCanCloseReleaseIssue").GetBoolean());

        AssertBlockedEvidenceItem(evidence, "owner-external-proof-execution-bundle", "not proof");
        AssertBlockedEvidenceItem(evidence, "owner-external-proof-execution-result-import", "not runtime proof");
        AssertBlockedEvidenceItem(evidence, "real-external-proof-record-import-validator", "not proof");
        AssertBlockedEvidenceItem(evidence, "real-proof-record-candidate-from-owner-result-import", "strict-validator input candidates only");
        AssertBlockedEvidenceItem(evidence, "release-close-owner-input-bridge", "not release close approval");

        string[] sourceArtifacts = evidence.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/owner-external-proof-execution-bundle.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-external-proof-execution-bundle-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-external-proof-execution-result-import.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/owner-external-proof-execution-result-import-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-external-proof-record-import-validator.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-external-proof-record-import-validator-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-proof-record-candidate-from-owner-result-import.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-proof-record-candidate-from-owner-result-import-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-close-owner-input-bridge.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-close-owner-input-bridge-validation.json", sourceArtifacts);

        using JsonDocument auditDocument = RealExternalProofExecutionAndCloseOwnerInputPipeline.ReadFinalReleaseJson("release-evidence-classification-audit.json");
        JsonElement audit = auditDocument.RootElement;
        Assert.Equal("classification-audit-passed-non-proof-boundaries-intact", audit.GetProperty("auditState").GetString());
        Assert.Equal(0, audit.GetProperty("findingCount").GetInt32());

        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.md"));
        string readmeZh = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "README.zh-CN.md"));
        string releaseEvidenceDoc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "release-evidence-bundle.md"));

        Assert.Contains("articles/zh-cn/owner-external-proof-execution-bundle.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-close-owner-input-bridge.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("owner-external-proof-execution-result-import", readme, StringComparison.Ordinal);
        Assert.Contains("real-external-proof-record-import-validator", readmeZh, StringComparison.Ordinal);
        Assert.Contains("release-close-owner-input-bridge", releaseEvidenceDoc, StringComparison.Ordinal);
    }

    private static void AssertBlockedEvidenceItem(JsonElement evidence, string id, string boundaryText)
    {
        JsonElement item = evidence.GetProperty("evidenceItems")
            .EnumerateArray()
            .Single(item => item.GetProperty("id").GetString() == id);

        Assert.False(item.GetProperty("passed").GetBoolean());
        Assert.Contains(boundaryText, item.GetProperty("boundary").GetString(), StringComparison.OrdinalIgnoreCase);
    }
}

internal static class RealExternalProofExecutionAndCloseOwnerInputPipeline
{
    private static readonly object Sync = new();
    private static bool s_hasRun;

    internal static void Run()
    {
        lock (Sync)
        {
            if (s_hasRun)
            {
                return;
            }

            ReleaseCloseStrictDryRunSummaryTests.RunReleaseCloseStrictDryRunPipeline();
            RunPowerShell("Export-OwnerExternalProofExecutionBundle.ps1");
            RunPowerShell("Test-OwnerExternalProofExecutionBundle.ps1", "-Strict");
            RunPowerShell("Import-OwnerExternalProofExecutionResult.ps1");
            RunPowerShell("Test-OwnerExternalProofExecutionResultImport.ps1", "-Strict");
            RunPowerShell("Export-RealExternalProofRecordImportValidator.ps1");
            RunPowerShell("Test-RealExternalProofRecordImportValidator.ps1", "-Strict");
            RunPowerShell("Export-RealProofRecordCandidateFromOwnerResultImport.ps1");
            RunPowerShell("Test-RealProofRecordCandidateFromOwnerResultImport.ps1", "-Strict");
            RunPowerShell("Export-ReleaseCloseOwnerInputBridge.ps1");
            RunPowerShell("Test-ReleaseCloseOwnerInputBridge.ps1", "-Strict");
            RunPowerShell("Export-ReleaseEvidenceBundle.ps1");
            RunPowerShell("Test-ReleaseEvidenceClassificationAudit.ps1", "-Strict");
            s_hasRun = true;
        }
    }

    internal static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    internal static void AssertFalseProofPublishCloseFlags(JsonElement element)
    {
        Assert.False(element.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(element.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(element.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(element.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(element.GetProperty("isReleaseCloseProof").GetBoolean());
    }

    private static void RunPowerShell(string scriptName, params string[] arguments)
    {
        string scriptPath = Path.Combine(RepositoryPaths.Root, "eng", scriptName);
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(scriptPath, arguments);
    }
}

internal sealed class OwnerResultImportTestContext
{
    private OwnerResultImportTestContext(string caseName)
    {
        CaseName = caseName;
        CaseRoot = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "test-owner-result-inputs", caseName);
        OutputRoot = Path.Combine(CaseRoot, "output");
        AllowedEvidenceRoot = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "test-owner-result-evidence", caseName);
        DisallowedEvidenceRoot = Path.Combine(RepositoryPaths.Root, "artifacts", "owner-result-disallowed-evidence", caseName);

        Directory.CreateDirectory(CaseRoot);
        Directory.CreateDirectory(OutputRoot);
        Directory.CreateDirectory(AllowedEvidenceRoot);
        Directory.CreateDirectory(DisallowedEvidenceRoot);
    }

    internal string CaseName { get; }

    internal string CaseRoot { get; }

    internal string OutputRoot { get; }

    internal string AllowedEvidenceRoot { get; }

    internal string DisallowedEvidenceRoot { get; }

    internal static OwnerResultImportTestContext Create(string caseName)
    {
        RealExternalProofExecutionAndCloseOwnerInputPipeline.Run();
        return new OwnerResultImportTestContext(caseName);
    }

    internal string CreateSingleLaneInput(
        string evidenceRoot,
        bool makeHashMismatch,
        bool useForbiddenSubstituteText,
        string? proofLaneOverride = null)
    {
        string templatePath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "owner-runtime-proof-result-input.template.json");
        using JsonDocument templateDocument = JsonDocument.Parse(File.ReadAllText(templatePath));
        JsonElement templateItem = proofLaneOverride is null
            ? templateDocument.RootElement.GetProperty("resultInputs").EnumerateArray().First()
            : templateDocument.RootElement.GetProperty("resultInputs").EnumerateArray()
                .Single(item => item.GetProperty("proofLane").GetString() == proofLaneOverride);

        JsonObject input = new()
        {
            ["recordKind"] = "owner-external-proof-execution-result-test-input",
            ["caseName"] = CaseName,
            ["resultInputs"] = new JsonArray(CreateResultInput(templateItem, evidenceRoot, makeHashMismatch, useForbiddenSubstituteText))
        };

        string inputPath = Path.Combine(CaseRoot, $"{CaseName}.input.json");
        File.WriteAllText(inputPath, input.ToJsonString());
        return inputPath;
    }

    internal JsonDocument RunImport(string inputPath)
    {
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Import-OwnerExternalProofExecutionResult.ps1"),
            "-InputPath",
            inputPath,
            "-OutputRoot",
            OutputRoot);

        return JsonDocument.Parse(File.ReadAllText(Path.Combine(OutputRoot, "owner-external-proof-execution-result-import.json")));
    }

    internal JsonDocument RunRealExternalValidator()
    {
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Export-RealExternalProofRecordImportValidator.ps1"),
            "-InputPath",
            Path.Combine(OutputRoot, "owner-external-proof-execution-result-import.json"),
            "-ResultImportValidationPath",
            Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "owner-external-proof-execution-result-import-validation.json"),
            "-ExecutionBundlePath",
            Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "owner-external-proof-execution-bundle.json"),
            "-OutputRoot",
            OutputRoot);

        return JsonDocument.Parse(File.ReadAllText(Path.Combine(OutputRoot, "real-external-proof-record-import-validator.json")));
    }

    internal JsonDocument RunCandidateBridge()
    {
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Export-RealProofRecordCandidateFromOwnerResultImport.ps1"),
            "-ImportValidatorPath",
            Path.Combine(OutputRoot, "real-external-proof-record-import-validator.json"),
            "-OwnerResultImportPath",
            Path.Combine(OutputRoot, "owner-external-proof-execution-result-import.json"),
            "-OutputRoot",
            OutputRoot);

        return JsonDocument.Parse(File.ReadAllText(Path.Combine(OutputRoot, "real-proof-record-candidate-from-owner-result-import.json")));
    }

    internal JsonDocument RunCandidateBridgeValidation()
    {
        OwnerRealProofFieldDeltaPackTests.RunPowerShell(
            Path.Combine(RepositoryPaths.Root, "eng", "Test-RealProofRecordCandidateFromOwnerResultImport.ps1"),
            "-InputPath",
            Path.Combine(OutputRoot, "real-proof-record-candidate-from-owner-result-import.json"),
            "-OutputRoot",
            OutputRoot,
            "-Strict");

        return JsonDocument.Parse(File.ReadAllText(Path.Combine(OutputRoot, "real-proof-record-candidate-from-owner-result-import-validation.json")));
    }

    private JsonObject CreateResultInput(JsonElement templateItem, string evidenceRoot, bool makeHashMismatch, bool useForbiddenSubstituteText)
    {
        Directory.CreateDirectory(evidenceRoot);

        string nupkgPath = WriteEvidenceFile(evidenceRoot, "package.nupkg", "test package bytes");
        string stdoutPath = WriteEvidenceFile(evidenceRoot, "stdout.log", "stdout from owner test");
        string stderrPath = WriteEvidenceFile(evidenceRoot, "stderr.log", "stderr from owner test");
        string transcriptPath = WriteEvidenceFile(evidenceRoot, "merged-transcript.log", "merged transcript from owner test");
        string validatorPath = WriteEvidenceFile(evidenceRoot, "validator-output.json", "{\"validationState\":\"test-only\"}");

        string nupkgSha = Sha256(nupkgPath);
        if (makeHashMismatch)
        {
            nupkgSha = new string('0', 64);
        }

        string packageSource = useForbiddenSubstituteText ? "local feed ProjectReference direct .nupkg" : "public package source";
        string executedCommand = useForbiddenSubstituteText
            ? "dotnet test --build-only skipped"
            : "dotnet test clean external consumer smoke";

        return new JsonObject
        {
            ["resultInputId"] = templateItem.GetProperty("resultInputId").GetString(),
            ["executionInputId"] = templateItem.GetProperty("executionInputId").GetString(),
            ["candidateId"] = templateItem.GetProperty("candidateId").GetString(),
            ["proofLane"] = templateItem.GetProperty("proofLane").GetString(),
            ["runtimePackageKey"] = templateItem.GetProperty("runtimePackageKey").GetString(),
            ["hostMetadata"] = new JsonObject
            {
                ["os"] = "Windows test host",
                ["arch"] = "x64",
                ["gpu"] = "test GPU",
                ["driverVersion"] = "test-driver",
                ["cudaVersion"] = "test-cuda",
                ["tensorRtVersion"] = "test-tensorrt",
                ["dotnetVersion"] = Environment.Version.ToString()
            },
            ["packageIdentity"] = new JsonObject
            {
                ["packageId"] = "JYPPX.TensorRtSharp.Test",
                ["packageVersion"] = "0.0.0-test",
                ["nupkgPath"] = ToRepositoryRelativePath(nupkgPath),
                ["nupkgSha256"] = nupkgSha,
                ["packageSource"] = packageSource
            },
            ["executedCommandLine"] = executedCommand,
            ["workingDirectory"] = ToRepositoryRelativePath(evidenceRoot),
            ["stdoutPath"] = ToRepositoryRelativePath(stdoutPath),
            ["stderrPath"] = ToRepositoryRelativePath(stderrPath),
            ["stdoutSha256"] = Sha256(stdoutPath),
            ["stderrSha256"] = Sha256(stderrPath),
            ["mergedTranscriptPath"] = ToRepositoryRelativePath(transcriptPath),
            ["mergedTranscriptSha256"] = Sha256(transcriptPath),
            ["validatorOutputPath"] = ToRepositoryRelativePath(validatorPath),
            ["validatorOutputSha256"] = Sha256(validatorPath),
            ["exitCode"] = "0",
            ["passed"] = true,
            ["startedAtUtc"] = "2026-07-08T00:00:00Z",
            ["endedAtUtc"] = "2026-07-08T00:01:00Z",
            ["ownerReviewer"] = "test-owner",
            ["ownerReviewTimestampUtc"] = "2026-07-08T00:02:00Z",
            ["nonSubstituteConfirmations"] = new JsonArray
            {
                "public package source used where required",
                "no local-feed substitute used",
                "no project-reference substitute used",
                "no direct-package-file substitute used",
                "runtime command executed",
                "full smoke executed",
                "not dependency probe only",
                "not sidecar only",
                "not build simulation only",
                "hashes match the referenced files"
            }
        };
    }

    private static string WriteEvidenceFile(string directory, string fileName, string contents)
    {
        string path = Path.Combine(directory, fileName);
        File.WriteAllText(path, contents);
        return path;
    }

    private static string Sha256(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        byte[] hash = System.Security.Cryptography.SHA256.HashData(bytes);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private static string ToRepositoryRelativePath(string path)
    {
        string relative = Path.GetRelativePath(RepositoryPaths.Root, path);
        return relative.Replace(Path.DirectorySeparatorChar, '/');
    }
}

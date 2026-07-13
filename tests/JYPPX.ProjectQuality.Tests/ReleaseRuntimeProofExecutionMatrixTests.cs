using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseRuntimeProofExecutionMatrixTests
{
    [Fact]
    public void ReleaseProofReadinessSnapshotKeepsFiveBlockersBlockedAndActionable()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseRuntimeProofExecutionMatrix.ps1"));

        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseProofReadinessSnapshot.ps1"));
        Assert.Contains("Release proof readiness snapshot written", output, StringComparison.Ordinal);
        Assert.Contains("CanPublishPublicly=False", output, StringComparison.Ordinal);
        Assert.Contains("CanCloseReleaseIssue=False", output, StringComparison.Ordinal);

        string jsonPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-proof-readiness-snapshot.json");
        string markdownPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-proof-readiness-snapshot.md");
        Assert.True(File.Exists(jsonPath), $"Expected {jsonPath} to exist.");
        Assert.True(File.Exists(markdownPath), $"Expected {markdownPath} to exist.");

        using JsonDocument snapshot = JsonDocument.Parse(File.ReadAllText(jsonPath));
        JsonElement root = snapshot.RootElement;

        Assert.Equal("release-proof-readiness-snapshot", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-proof-required", root.GetProperty("readinessState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isReleaseProofComplete").GetBoolean());
        Assert.Equal(5, root.GetProperty("oneScreenReleaseHoldChecklistCount").GetInt32());
        Assert.Equal(5, root.GetProperty("readinessItemCount").GetInt32());
        Assert.Equal(0, root.GetProperty("readyProofItemCount").GetInt32());
        Assert.Equal(5, root.GetProperty("blockedProofItemCount").GetInt32());
        Assert.Equal("blocked-real-proof-required", root.GetProperty("ownerReleaseExecutionPackageState").GetString());
        Assert.Equal("blocked-real-proof-required", root.GetProperty("releaseRuntimeProofExecutionMatrixState").GetString());

        string[] expectedIds =
        {
            "owner-authorization",
            "package-consumer-runtime",
            "linux-runner-proof",
            "real-model-runtime",
            "post-publish-verification",
        };

        JsonElement[] items = root.GetProperty("readinessItems").EnumerateArray().ToArray();
        Assert.Equal(expectedIds.Order(StringComparer.Ordinal).ToArray(), items.Select(static item => item.GetProperty("id").GetString()!).Order(StringComparer.Ordinal).ToArray());
        Assert.All(items, static item =>
        {
            Assert.False(item.GetProperty("ready").GetBoolean());
            Assert.Equal("blocked", item.GetProperty("state").GetString());
            Assert.False(item.GetProperty("performsPublish").GetBoolean());
            Assert.False(item.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("currentState").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("requiredValidator").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("requiredRealInputs").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("ownerAction").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("cannotUse").GetString()));
        });

        JsonElement packageConsumer = Assert.Single(items, static item => item.GetProperty("id").GetString() == "package-consumer-runtime");
        Assert.Contains("Test-ExternalRuntimeProofRecord.ps1", packageConsumer.GetProperty("requiredValidator").GetString(), StringComparison.Ordinal);
        Assert.Contains("blocked-by-cuda-driver", packageConsumer.GetProperty("cannotUse").GetString(), StringComparison.Ordinal);
        Assert.Contains("ProjectReference", packageConsumer.GetProperty("cannotUse").GetString(), StringComparison.Ordinal);

        JsonElement postPublish = Assert.Single(items, static item => item.GetProperty("id").GetString() == "post-publish-verification");
        Assert.Contains("Test-PostPublishVerificationRecord.ps1", postPublish.GetProperty("requiredValidator").GetString(), StringComparison.Ordinal);
        Assert.Contains("real public package URL", postPublish.GetProperty("requiredRealInputs").GetString(), StringComparison.Ordinal);

        string[] sourceArtifacts = root.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/owner-release-execution-package.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-runtime-proof-execution-matrix.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/external-runtime-proof-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/post-publish-verification-validation.json", sourceArtifacts);

        string[] nonSubstitutes = root.GetProperty("nonSubstituteProofKinds").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("local feed", nonSubstitutes);
        Assert.Contains("ProjectReference", nonSubstitutes);
        Assert.Contains("blocked-by-cuda-driver", nonSubstitutes);
        Assert.Contains("sidecar-only", nonSubstitutes);

        string markdown = File.ReadAllText(markdownPath);
        foreach (string id in expectedIds)
        {
            Assert.Contains(id, markdown, StringComparison.Ordinal);
        }

        Assert.Contains("release-proof-readiness-snapshot", markdown, StringComparison.Ordinal);
        Assert.Contains("blocked-real-proof-required", markdown, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly", markdown, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue", markdown, StringComparison.Ordinal);
        Assert.DoesNotContain("canPublishPublicly=true", markdown, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("canCloseReleaseIssue=true", markdown, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void RuntimeProofExecutionMatrixKeepsRealProofBlockersExplicitAndBlocked()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredBTierProofClosureDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredBTierAliasProofClosureRecord.ps1"));

        string output = RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseRuntimeProofExecutionMatrix.ps1"));
        Assert.Contains("Release runtime proof execution matrix written", output, StringComparison.Ordinal);

        string matrixPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-runtime-proof-execution-matrix.json");
        string markdownPath = Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-runtime-proof-execution-matrix.md");
        Assert.True(File.Exists(matrixPath), $"Expected {matrixPath} to exist.");
        Assert.True(File.Exists(markdownPath), $"Expected {markdownPath} to exist.");

        using JsonDocument matrix = JsonDocument.Parse(File.ReadAllText(matrixPath));
        JsonElement root = matrix.RootElement;

        Assert.Equal("release-runtime-proof-execution-matrix", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-real-proof-required", root.GetProperty("matrixState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isReleaseProofComplete").GetBoolean());
        Assert.Equal(8, root.GetProperty("proofItemCount").GetInt32());
        Assert.Equal(8, root.GetProperty("blockedProofItemCount").GetInt32());
        Assert.Equal("covered-by-eight-template-paths", root.GetProperty("ownerProofTemplateCoverageState").GetString());
        Assert.Equal("covered-by-eight-validator-commands", root.GetProperty("validatorCoverageState").GetString());
        Assert.Equal("covered-by-article-slug-map", root.GetProperty("ownerProofExecutionArticleCoverageState").GetString());
        Assert.Equal(8, root.GetProperty("expectedInputTemplateCount").GetInt32());
        Assert.Equal(8, root.GetProperty("expectedValidatedRecordCount").GetInt32());
        Assert.Equal(8, root.GetProperty("validatorCommandCount").GetInt32());
        Assert.Equal(8, root.GetProperty("articleMappedProofItemCount").GetInt32());
        Assert.Equal("blocked-owner-action-required", root.GetProperty("packageConsumerProofPackState").GetString());
        Assert.Equal("blocked-owner-action-required", root.GetProperty("linuxRunnerProofPackState").GetString());
        Assert.False(root.GetProperty("packageConsumerProofPackCanPromote").GetBoolean());
        Assert.False(root.GetProperty("linuxRunnerProofPackCanPromote").GetBoolean());
        Assert.Equal("blocked-owner-action-required", root.GetProperty("realCaseProofPackState").GetString());
        Assert.True(root.GetProperty("realCaseProofCaseCount").GetInt32() >= 8);
        Assert.Equal(root.GetProperty("realCaseProofCaseCount").GetInt32(), root.GetProperty("realCaseProofBlockedCaseCount").GetInt32());
        Assert.False(root.GetProperty("realCaseProofCanPromote").GetBoolean());
        Assert.Contains("does not perform publish actions", root.GetProperty("boundary").GetString()!, StringComparison.Ordinal);

        string[] matrixSourceArtifacts = root.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/final-release/external-runtime-proof-record-template.json", matrixSourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-execution-pack.json", matrixSourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-pack-validation.json", matrixSourceArtifacts);
        Assert.Contains("artifacts/final-release/linux-runner-evidence-record.template.json", matrixSourceArtifacts);
        Assert.Contains("artifacts/final-release/linux-runner-proof-execution-pack.json", matrixSourceArtifacts);
        Assert.Contains("artifacts/final-release/linux-runner-proof-pack-validation.json", matrixSourceArtifacts);
        Assert.Contains("artifacts/user-acceptance/sample-run-evidence-record-validation.json", matrixSourceArtifacts);
        Assert.Contains("artifacts/user-acceptance/onnx-engine-build-evidence-sidecar-audit.json", matrixSourceArtifacts);
        Assert.Contains("artifacts/final-release/real-model-and-package-proof-input-package.json", matrixSourceArtifacts);
        Assert.Contains("artifacts/final-release/real-case-proof-execution-pack.json", matrixSourceArtifacts);
        Assert.Contains("artifacts/final-release/real-case-evidence-record-template.json", matrixSourceArtifacts);
        Assert.Contains("artifacts/final-release/real-case-evidence-record-validation.json", matrixSourceArtifacts);
        Assert.Contains("artifacts/final-release/post-publish-verification-record-template.json", matrixSourceArtifacts);
        Assert.Contains("artifacts/final-release/post-publish-verification-validation.json", matrixSourceArtifacts);

        string[] expectedPostPublishRequiredEvidence =
        {
            "selectedChannel",
            "channelSourceUri",
            "publishedPackageUrl",
            "managedPackageUrl",
            "runtimePackageUrl",
            "managedNupkgSha256",
            "runtimeNupkgSha256",
            "cleanConsumerRootOutsideRepository",
            "consumerProjectPath",
            "noProjectReference",
            "noLocalPackageSource",
            "noLocalNupkgPackageReference",
            "restoreLogPath",
            "nativeAssetListingSha256",
            "dependencyProbeLogPath",
            "dependencyProbeLogSha256",
            "runtimeSmokeLogPath",
            "runtimeSmokeLogSha256",
            "runtimeSmokePassed",
            "runtimeSmokeExitCode",
            "stdoutSummary",
            "stderrSummary",
            "hostMetadata",
        };
        string[] postPublishRequiredEvidence = root.GetProperty("postPublishRequiredEvidence").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Equal(expectedPostPublishRequiredEvidence.Length, root.GetProperty("postPublishRequiredEvidenceCount").GetInt32());
        Assert.Equal(expectedPostPublishRequiredEvidence.Order(StringComparer.Ordinal).ToArray(), postPublishRequiredEvidence.Order(StringComparer.Ordinal).ToArray());

        string[] expectedIds =
        {
            "owner-authorization",
            "package-consumer-runtime",
            "package-consumer-proof-pack",
            "linux-runner-proof",
            "linux-runner-proof-pack",
            "real-model-runtime",
            "real-case-proof-pack",
            "post-publish-verification",
        };
        JsonElement[] proofItems = root.GetProperty("proofItems").EnumerateArray().ToArray();
        Assert.Equal(expectedIds.Order(StringComparer.Ordinal).ToArray(), proofItems.Select(static item => item.GetProperty("id").GetString()!).Order(StringComparer.Ordinal).ToArray());

        JsonElement postPublishItem = Assert.Single(proofItems, static item => item.GetProperty("id").GetString() == "post-publish-verification");
        string[] postPublishRequiredLogFields = postPublishItem.GetProperty("requiredLogFields").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        string[] postPublishRequiredSha256Fields = postPublishItem.GetProperty("requiredSha256Fields").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        string[] postPublishRequiredArtifacts = postPublishItem.GetProperty("requiredArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        foreach (string fieldName in expectedPostPublishRequiredEvidence.Except(postPublishRequiredSha256Fields, StringComparer.Ordinal))
        {
            Assert.Contains(fieldName, postPublishRequiredLogFields);
        }
        foreach (string fieldName in new[]
        {
            "managedNupkgSha256",
            "runtimeNupkgSha256",
            "nativeAssetListingSha256",
            "dependencyProbeLogSha256",
            "runtimeSmokeLogSha256",
        })
        {
            Assert.Contains(fieldName, postPublishRequiredSha256Fields);
        }
        Assert.Contains("published package URL", postPublishRequiredArtifacts);
        Assert.Contains("clean external consumer root outside repository", postPublishRequiredArtifacts);
        Assert.Contains("runtime smoke log/hash", postPublishRequiredArtifacts);
        Assert.Contains("host metadata", postPublishRequiredArtifacts);

        Assert.All(proofItems, static item =>
        {
            Assert.False(item.GetProperty("passed").GetBoolean());
            Assert.False(item.GetProperty("performsPublish").GetBoolean());
            Assert.False(item.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(item.GetProperty("isRuntimeExecutionProof").GetBoolean());
            Assert.True(item.GetProperty("requiredArtifacts").GetArrayLength() >= 2);
            Assert.True(item.GetProperty("nonSubstituteProofKinds").GetArrayLength() >= 4);
            Assert.Contains("B-tier alias proof closure record", item.GetProperty("nonSubstituteProofKinds").EnumerateArray().Select(static value => value.GetString()!));

            foreach (string propertyName in new[]
            {
                "validatorCommand",
                "expectedInputTemplate",
                "expectedValidatedRecord",
                "ownerAction",
                "compatibleHostRequirement",
                "localCollectability",
                "blocker",
                "articleSlug",
            })
            {
                Assert.True(item.TryGetProperty(propertyName, out JsonElement property), $"Expected proof item property '{propertyName}'.");
                Assert.False(string.IsNullOrWhiteSpace(property.GetString()));
            }

            Assert.True(item.GetProperty("articleTopicId").GetInt32() > 0);
            Assert.True(item.GetProperty("requiredLogFields").GetArrayLength() >= 4);
            Assert.True(item.GetProperty("requiredSha256Fields").GetArrayLength() >= 1);
        });

        string markdown = File.ReadAllText(markdownPath);
        foreach (string marker in expectedIds)
        {
            Assert.Contains(marker, markdown, StringComparison.Ordinal);
        }

        Assert.Contains("blocked-real-proof-required", markdown, StringComparison.Ordinal);
        Assert.Contains("real-case-proof-pack", markdown, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly", markdown, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue", markdown, StringComparison.Ordinal);
    }

    [Fact]
    public void ReleaseFreezeConsumesAliasClosureAndRuntimeMatrixWithoutUnlockingRelease()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredReadOnlyApiCandidatePlan.ps1"), "-IncludeMediumRisk", "-MaxItems", "60");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredBTierProofClosureDashboard.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-DeferredBTierAliasProofClosureRecord.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseRuntimeProofExecutionMatrix.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseProofReadinessSnapshot.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-ReleaseFreezeFinalVerification.ps1"));

        using JsonDocument aliasRecord = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "interface-coverage",
            "deferred-btier-alias-proof-closure-record.json")));
        using JsonDocument matrix = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "release-runtime-proof-execution-matrix.json")));
        using JsonDocument readinessSnapshot = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "release-proof-readiness-snapshot.json")));
        using JsonDocument freeze = JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            "release-freeze-final-verification.json")));

        JsonElement freezeRoot = freeze.RootElement;
        Assert.Equal("blocked-real-proof-required", freezeRoot.GetProperty("verificationState").GetString());
        Assert.False(freezeRoot.GetProperty("performsPublish").GetBoolean());
        Assert.False(freezeRoot.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(freezeRoot.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Equal("alias-proof-ready-engineering-record", freezeRoot.GetProperty("deferredBTierAliasProofClosureRecordState").GetString());
        Assert.Equal(aliasRecord.RootElement.GetProperty("closureCandidateCount").GetInt32(), freezeRoot.GetProperty("deferredBTierAliasProofClosureRecordCandidateCount").GetInt32());
        Assert.False(freezeRoot.GetProperty("deferredBTierAliasProofClosureRecordIsReleaseProof").GetBoolean());
        Assert.False(freezeRoot.GetProperty("deferredBTierAliasProofClosureRecordCanPublishPublicly").GetBoolean());
        Assert.False(freezeRoot.GetProperty("deferredBTierAliasProofClosureRecordCanCloseReleaseIssue").GetBoolean());
        Assert.False(freezeRoot.GetProperty("deferredBTierAliasProofClosureRecordCanDeleteDeferredRecords").GetBoolean());
        Assert.Equal("blocked-real-proof-required", freezeRoot.GetProperty("releaseRuntimeProofExecutionMatrixState").GetString());
        Assert.Equal(matrix.RootElement.GetProperty("proofItemCount").GetInt32(), freezeRoot.GetProperty("releaseRuntimeProofExecutionMatrixProofItemCount").GetInt32());
        Assert.Equal(matrix.RootElement.GetProperty("blockedProofItemCount").GetInt32(), freezeRoot.GetProperty("releaseRuntimeProofExecutionMatrixBlockedProofItemCount").GetInt32());
        Assert.False(freezeRoot.GetProperty("releaseRuntimeProofExecutionMatrixIsReleaseProofComplete").GetBoolean());
        Assert.False(freezeRoot.GetProperty("releaseRuntimeProofExecutionMatrixCanPublishPublicly").GetBoolean());
        Assert.False(freezeRoot.GetProperty("releaseRuntimeProofExecutionMatrixCanCloseReleaseIssue").GetBoolean());
        Assert.Equal("blocked-real-proof-required", freezeRoot.GetProperty("releaseProofReadinessSnapshotState").GetString());
        Assert.Equal(5, freezeRoot.GetProperty("releaseProofReadinessSnapshotItemCount").GetInt32());
        Assert.Equal(0, freezeRoot.GetProperty("releaseProofReadinessSnapshotReadyProofItemCount").GetInt32());
        Assert.Equal(5, freezeRoot.GetProperty("releaseProofReadinessSnapshotBlockedProofItemCount").GetInt32());
        Assert.Equal(
            readinessSnapshot.RootElement.GetProperty("blockedProofItemCount").GetInt32(),
            freezeRoot.GetProperty("releaseProofReadinessSnapshotBlockedProofItemCount").GetInt32());
        Assert.False(freezeRoot.GetProperty("releaseProofReadinessSnapshotIsReleaseProofComplete").GetBoolean());
        Assert.False(freezeRoot.GetProperty("releaseProofReadinessSnapshotCanPublishPublicly").GetBoolean());
        Assert.False(freezeRoot.GetProperty("releaseProofReadinessSnapshotCanCloseReleaseIssue").GetBoolean());
        Assert.Contains("status view only", freezeRoot.GetProperty("releaseProofReadinessSnapshotBoundary").GetString()!, StringComparison.Ordinal);
        Assert.Equal("covered-by-eight-template-paths", freezeRoot.GetProperty("ownerProofTemplateCoverageState").GetString());
        Assert.Equal(8, freezeRoot.GetProperty("ownerProofTemplateCoverageCount").GetInt32());
        Assert.Equal(8, freezeRoot.GetProperty("ownerProofValidatedRecordCoverageCount").GetInt32());
        Assert.Equal("covered-by-eight-validator-commands", freezeRoot.GetProperty("validatorCoverageState").GetString());
        Assert.Equal(8, freezeRoot.GetProperty("validatorCoverageCount").GetInt32());
        Assert.Equal("covered-by-article-slug-map", freezeRoot.GetProperty("ownerProofExecutionArticleCoverageState").GetString());
        Assert.Equal(8, freezeRoot.GetProperty("ownerProofExecutionArticleMappedCount").GetInt32());
        Assert.Equal(matrix.RootElement.GetProperty("postPublishRequiredEvidenceCount").GetInt32(), freezeRoot.GetProperty("releaseRuntimeProofExecutionMatrixPostPublishRequiredEvidenceCount").GetInt32());
        Assert.Equal(
            matrix.RootElement.GetProperty("postPublishRequiredEvidence").EnumerateArray().Select(static item => item.GetString()!).Order(StringComparer.Ordinal).ToArray(),
            freezeRoot.GetProperty("releaseRuntimeProofExecutionMatrixPostPublishRequiredEvidence").EnumerateArray().Select(static item => item.GetString()!).Order(StringComparer.Ordinal).ToArray());
        Assert.Equal(8, freezeRoot.GetProperty("releaseProofFinalAuditItemCount").GetInt32());
        Assert.Equal(8, freezeRoot.GetProperty("releaseProofFinalAuditBlockedItemCount").GetInt32());
        Assert.Equal("blocked-owner-action-required", freezeRoot.GetProperty("packageConsumerProofPackState").GetString());
        Assert.Equal("blocked-owner-action-required", freezeRoot.GetProperty("linuxRunnerProofPackState").GetString());
        Assert.False(freezeRoot.GetProperty("packageConsumerProofPackCanPromote").GetBoolean());
        Assert.False(freezeRoot.GetProperty("linuxRunnerProofPackCanPromote").GetBoolean());
        Assert.Equal("blocked-owner-action-required", freezeRoot.GetProperty("realCaseProofPackState").GetString());
        Assert.True(freezeRoot.GetProperty("realCaseProofCaseCount").GetInt32() >= 8);
        Assert.Equal(freezeRoot.GetProperty("realCaseProofCaseCount").GetInt32(), freezeRoot.GetProperty("realCaseProofBlockedCaseCount").GetInt32());
        Assert.False(freezeRoot.GetProperty("realCaseProofCanPromote").GetBoolean());
        JsonElement[] finalAuditItems = freezeRoot.GetProperty("releaseProofFinalAuditItems").EnumerateArray().ToArray();
        string[] expectedFinalAuditIds =
        {
            "owner-authorization",
            "package-consumer-runtime",
            "package-consumer-proof-pack",
            "linux-runner-proof",
            "linux-runner-proof-pack",
            "real-model-runtime",
            "real-case-proof-pack",
            "post-publish-verification",
        };
        Assert.Equal(expectedFinalAuditIds.Order(StringComparer.Ordinal).ToArray(), finalAuditItems.Select(static item => item.GetProperty("proofId").GetString()!).Order(StringComparer.Ordinal).ToArray());
        Assert.All(finalAuditItems, static item =>
        {
            Assert.False(item.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(item.GetProperty("canCloseReleaseIssue").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("currentState").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("templatePath").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("expectedValidatedRecord").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("validatorCommand").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(item.GetProperty("missingOwnerInput").GetString()));
            Assert.True(item.GetProperty("nonSubstituteProofKinds").GetArrayLength() >= 8);
        });
        Assert.Equal("template-only", freezeRoot.GetProperty("packageConsumerRuntimeProofTemplateState").GetString());
        Assert.Equal("external-runtime-proof-record-template", freezeRoot.GetProperty("packageConsumerRuntimeProofTemplateKind").GetString());
        string packageConsumerRuntimeValidatorState = freezeRoot.GetProperty("packageConsumerRuntimeValidatorState").GetString()!;
        Assert.Contains(packageConsumerRuntimeValidatorState, new[]
        {
            "template-only",
            "draft-blocked-by-cuda-driver",
            "draft-rich-but-not-proof",
            "blocked-by-cuda-driver",
            "dependency-probe-only",
            "incomplete-runtime-proof",
        });
        Assert.Contains(freezeRoot.GetProperty("packageConsumerRuntimeProofClassification").GetString(), new[]
        {
            "template-only",
            "package-consumer-runtime",
            "dependency-probe-only",
            "synthetic-input-runtime",
            "real-model-runtime",
            "build-only",
            "precheck",
        });
        Assert.False(freezeRoot.GetProperty("packageConsumerRuntimeProofCanPromote").GetBoolean());
        Assert.False(freezeRoot.GetProperty("packageConsumerRuntimeProofCanPublishPublicly").GetBoolean());
        Assert.False(freezeRoot.GetProperty("packageConsumerRuntimeProofCanCloseReleaseIssue").GetBoolean());
        Assert.Equal("template-only", freezeRoot.GetProperty("linuxRunnerProofTemplateState").GetString());
        Assert.Equal("linux-runner-evidence-record-template", freezeRoot.GetProperty("linuxRunnerProofTemplateKind").GetString());
        string linuxRunnerValidatorState = freezeRoot.GetProperty("linuxRunnerValidatorState").GetString()!;
        Assert.Contains(linuxRunnerValidatorState, new[]
        {
            "template-only",
            "incomplete-linux-runner-evidence",
            "invalid-record",
        });
        Assert.False(freezeRoot.GetProperty("linuxRunnerProofCanPromote").GetBoolean());
        Assert.False(freezeRoot.GetProperty("linuxRunnerProofIsReal").GetBoolean());
        Assert.False(freezeRoot.GetProperty("linuxRunnerProofCanPublishPublicly").GetBoolean());
        Assert.False(freezeRoot.GetProperty("linuxRunnerProofCanCloseReleaseIssue").GetBoolean());
        Assert.Equal("owner-action-required", freezeRoot.GetProperty("realModelRuntimeProofInputPackageState").GetString());
        Assert.Equal("real-model-and-package-proof-input-package", freezeRoot.GetProperty("realModelRuntimeProofInputPackageKind").GetString());
        Assert.Equal("owner-action-required", freezeRoot.GetProperty("sampleRunEvidenceValidatorState").GetString());
        Assert.Equal("template-only", freezeRoot.GetProperty("sampleRunEvidenceProofClassification").GetString());
        Assert.True(freezeRoot.GetProperty("sampleRunEvidenceTemplateOnly").GetBoolean());
        Assert.False(freezeRoot.GetProperty("sampleRunEvidenceCanPromoteRealModelRuntime").GetBoolean());
        Assert.False(freezeRoot.GetProperty("sampleRunEvidenceRealModelEvidenceReady").GetBoolean());
        Assert.Equal("owner-action-required", freezeRoot.GetProperty("onnxEngineBuildEvidenceSidecarAuditState").GetString());
        Assert.False(freezeRoot.GetProperty("realModelRuntimeProofCanPromote").GetBoolean());
        Assert.False(freezeRoot.GetProperty("realModelRuntimeProofCanPublishPublicly").GetBoolean());
        Assert.False(freezeRoot.GetProperty("realModelRuntimeProofCanCloseReleaseIssue").GetBoolean());
        Assert.Equal("post-publish-verification-record-template", freezeRoot.GetProperty("postPublishVerificationTemplateKind").GetString());
        Assert.True(freezeRoot.GetProperty("postPublishVerificationTemplateOnly").GetBoolean());
        Assert.Contains(freezeRoot.GetProperty("postPublishVerificationValidatorState").GetString(), new[]
        {
            "template-only",
            "incomplete-post-publish-verification",
        });
        Assert.Contains(freezeRoot.GetProperty("postPublishVerificationProofClassification").GetString(), new[]
        {
            "template-only",
            "owner-action-required",
        });
        Assert.False(freezeRoot.GetProperty("postPublishVerificationIsProof").GetBoolean());
        Assert.False(freezeRoot.GetProperty("postPublishVerificationCanCloseReleaseIssue").GetBoolean());
        Assert.False(freezeRoot.GetProperty("postPublishVerificationCanPublishPublicly").GetBoolean());
        Assert.Contains("cannot substitute owner authorization", freezeRoot.GetProperty("proofNonSubstituteBoundary").GetString()!, StringComparison.Ordinal);
        Assert.Equal(5, freezeRoot.GetProperty("releaseProofBlockers").GetArrayLength());

        string[] sourceArtifacts = freezeRoot.GetProperty("sourceArtifacts").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        Assert.Contains("artifacts/interface-coverage/deferred-btier-alias-proof-closure-record.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-runtime-proof-execution-matrix.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/release-proof-readiness-snapshot.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-execution-pack.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/package-consumer-runtime-proof-pack-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/user-acceptance/sample-run-evidence-record-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/linux-runner-proof-execution-pack.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/linux-runner-proof-pack-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/user-acceptance/onnx-engine-build-evidence-sidecar-audit.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-model-and-package-proof-input-package.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-case-proof-execution-pack.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-case-evidence-record-template.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/real-case-evidence-record-validation.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/post-publish-verification-record-template.json", sourceArtifacts);
        Assert.Contains("artifacts/final-release/post-publish-verification-validation.json", sourceArtifacts);

        string markdown = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "artifacts", "final-release", "release-freeze-final-verification.md"));
        Assert.Contains("B-tier Alias Proof Closure Record", markdown, StringComparison.Ordinal);
        Assert.Contains("Runtime Proof Execution Matrix", markdown, StringComparison.Ordinal);
        Assert.Contains("Release Proof Readiness Snapshot", markdown, StringComparison.Ordinal);
        Assert.Contains("Package Consumer Runtime Proof", markdown, StringComparison.Ordinal);
        Assert.Contains("Linux Runner Proof", markdown, StringComparison.Ordinal);
        Assert.Contains("Real Model Runtime Proof", markdown, StringComparison.Ordinal);
        Assert.Contains("Post Publish Verification Proof", markdown, StringComparison.Ordinal);
        Assert.Contains("Release Proof Final Audit", markdown, StringComparison.Ordinal);
        Assert.Contains("Missing owner input", markdown, StringComparison.Ordinal);
        Assert.Contains("owner proof template coverage state", markdown, StringComparison.Ordinal);
        Assert.Contains("Proof Non-Substitute Boundary", markdown, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly=false", markdown, StringComparison.Ordinal);
        Assert.Contains("canCloseReleaseIssue=false", markdown, StringComparison.Ordinal);
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
    {
        ProcessStartInfo startInfo = new()
        {
            FileName = "pwsh",
            WorkingDirectory = RepositoryPaths.Root,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };

        startInfo.ArgumentList.Add("-NoProfile");
        startInfo.ArgumentList.Add("-ExecutionPolicy");
        startInfo.ArgumentList.Add("Bypass");
        startInfo.ArgumentList.Add("-File");
        startInfo.ArgumentList.Add(scriptPath);

        foreach (string argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        using Process process = Process.Start(startInfo)!;
        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"PowerShell command failed with exit code {process.ExitCode}:{Environment.NewLine}{output}{error}");
        return output + error;
    }
}

using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseCandidatePackageConsumerClosureTests
{
    [Fact]
    public void ClosureMapKeepsAllReleaseProofTracksBlockedUntilRealOwnerEvidenceArrives()
    {
        using JsonDocument document = ReadFinalReleaseJson("release-candidate-package-consumer-closure-map.json");
        JsonElement root = document.RootElement;

        Assert.Equal("release-candidate-package-consumer-closure-map", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-action-required", root.GetProperty("mapState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("approvesPublicRelease").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isRealModelRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishVerificationProof").GetBoolean());

        string boundary = root.GetProperty("proofBoundary").GetString()!;
        Assert.Contains("does not run models", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("publish packages", boundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("runtime proof", boundary, StringComparison.OrdinalIgnoreCase);

        string[] trackIds = root.GetProperty("proofTracks").EnumerateArray()
            .Select(static track => track.GetProperty("id").GetString()!)
            .ToArray();

        foreach (string trackId in new[]
        {
            "real-model-runtime",
            "package-consumer-runtime",
            "post-publish-verification",
            "release-issue-close"
        })
        {
            Assert.Contains(trackId, trackIds);
        }

        foreach (JsonElement track in root.GetProperty("proofTracks").EnumerateArray())
        {
            Assert.False(track.GetProperty("canPromote").GetBoolean());
            Assert.True(track.GetProperty("requiredBeforePromotion").GetArrayLength() >= 4);
            Assert.True(track.GetProperty("cannotUse").GetArrayLength() >= 5);
            Assert.Contains("eng/Test-", track.GetProperty("validator").GetString()!, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ClosureMapAlignsPackageConsumerRequiredFieldsWithSchemaAndValidation()
    {
        using JsonDocument mapDocument = ReadFinalReleaseJson("release-candidate-package-consumer-closure-map.json");
        using JsonDocument schemaDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-owner-input.schema.json");
        using JsonDocument templateDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-owner-input.template.json");
        using JsonDocument validationDocument = ReadFinalReleaseJson("package-consumer-runtime-proof-record-validation.json");

        JsonElement map = mapDocument.RootElement;
        string[] mappedFields = map.GetProperty("fieldAlignment").GetProperty("packageConsumerRequiredFields").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        string[] schemaFields = schemaDocument.RootElement.GetProperty("fields").EnumerateArray()
            .Select(static item => item.GetProperty("name").GetString()!)
            .ToArray();
        JsonElement template = templateDocument.RootElement;
        string validationText = validationDocument.RootElement.GetRawText();

        foreach (string field in mappedFields)
        {
            Assert.Contains(field, schemaFields);
            Assert.True(template.TryGetProperty(field, out _), $"Owner input template is missing {field}.");
        }

        foreach (string validatorId in new[]
        {
            "no-project-reference",
            "no-local-feed",
            "no-direct-nupkg",
            "smoke-exit-code",
            "smoke-status",
            "native-assets-copied",
            "smoke-log-hash-match",
            "host-gpuName",
            "host-cudnnVersion",
            "command-restoreCommand",
            "command-buildCommand"
        })
        {
            Assert.Contains(validatorId, validationText, StringComparison.Ordinal);
        }

        Assert.Equal("template-only", validationDocument.RootElement.GetProperty("validationState").GetString());
        Assert.False(validationDocument.RootElement.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(validationDocument.RootElement.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validationDocument.RootElement.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    [Fact]
    public void ClosureMapAlignsRealCaseSemanticCoverageAndPostPublishCloseGates()
    {
        using JsonDocument mapDocument = ReadFinalReleaseJson("release-candidate-package-consumer-closure-map.json");
        using JsonDocument realCaseDocument = ReadFinalReleaseJson("real-case-proof-execution-pack.json");
        using JsonDocument postPublishDocument = ReadFinalReleaseJson("post-publish-verification-validation.json");
        using JsonDocument closeDocument = ReadFinalReleaseJson("release-issue-close-record-validation.json");

        JsonElement map = mapDocument.RootElement;
        string[] requiredTasks = map.GetProperty("fieldAlignment").GetProperty("realCaseRequiredTasks").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();
        string[] realCaseTasks = realCaseDocument.RootElement.GetProperty("requiredTaskCoverage").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        foreach (string task in new[] { "det", "cls", "seg", "obb", "pose", "sem" })
        {
            Assert.Contains(task, requiredTasks);
            Assert.Contains(task, realCaseTasks);
        }

        string realCaseText = realCaseDocument.RootElement.GetRawText();
        Assert.Contains("yolovision-semantic", realCaseText, StringComparison.Ordinal);
        Assert.Equal(9, realCaseDocument.RootElement.GetProperty("caseCount").GetInt32());
        Assert.Equal(9, realCaseDocument.RootElement.GetProperty("blockedCaseCount").GetInt32());
        Assert.Equal(6, realCaseDocument.RootElement.GetProperty("yoloVisionCaseCount").GetInt32());

        Assert.Contains(postPublishDocument.RootElement.GetProperty("validationState").GetString(), new[]
        {
            "template-only",
            "incomplete-post-publish-verification",
        });
        Assert.Contains(postPublishDocument.RootElement.GetProperty("postPublishProofClassification").GetString(), new[]
        {
            "template-only",
            "owner-action-required",
        });
        Assert.Equal(0, postPublishDocument.RootElement.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(postPublishDocument.RootElement.GetProperty("failedProofItemCount").GetInt32() > 0);
        Assert.False(postPublishDocument.RootElement.GetProperty("isPostPublishVerificationProof").GetBoolean());
        Assert.False(postPublishDocument.RootElement.GetProperty("canCloseReleaseIssue").GetBoolean());

        string postPublishText = postPublishDocument.RootElement.GetRawText();
        foreach (string marker in new[]
        {
            "post-publish-package-consumer-runtime",
            "no-project-reference",
            "no-local-package-source",
            "no-local-nupkg-reference",
            "runtime-key-smoke-command",
            "compatible-host-smoke"
        })
        {
            Assert.Contains(marker, postPublishText, StringComparison.Ordinal);
        }

        Assert.Equal("blocked-template-only", closeDocument.RootElement.GetProperty("validationState").GetString());
        Assert.False(closeDocument.RootElement.GetProperty("canPromoteReleaseIssueCloseRecord").GetBoolean());
        Assert.False(closeDocument.RootElement.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(closeDocument.RootElement.GetProperty("canCloseReleaseIssue").GetBoolean());

        string closeText = closeDocument.RootElement.GetRawText();
        foreach (string marker in new[]
        {
            "owner-final-close-decision",
            "post-publish-verification-proof",
            "release-close-preflight-passed",
            "evidence-bundle-sha256-matches",
            "rollback-plan-ready"
        })
        {
            Assert.Contains(marker, closeText, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ClosureMapIsDocumentedAndCrossLinked()
    {
        string artifactMarkdown = ReadFinalReleaseText("release-candidate-package-consumer-closure-map.md");
        string article = ReadText("docs", "articles", "zh-cn", "release-candidate-package-consumer-closure-map.md");
        string docsIndex = ReadText("docs", "index.md");
        string docsToc = ReadText("docs", "toc.yml");

        foreach (string marker in new[]
        {
            "Release Candidate Package Consumer Closure Map",
            "package-consumer-runtime",
            "post-publish-verification",
            "release-issue-close",
            "direct `.nupkg`",
            "ProjectReference"
        })
        {
            Assert.Contains(marker, artifactMarkdown, StringComparison.Ordinal);
        }

        foreach (string marker in new[]
        {
            "发布候选包消费闭环图",
            "real-model-runtime",
            "package-consumer-runtime",
            "post-publish verification",
            "release issue close",
            "无 ProjectReference",
            "无 local feed",
            "无 direct `.nupkg`"
        })
        {
            Assert.Contains(marker, article, StringComparison.Ordinal);
        }

        Assert.Contains("articles/zh-cn/release-candidate-package-consumer-closure-map.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("artifacts/final-release/release-candidate-package-consumer-closure-map.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-candidate-package-consumer-closure-map.md", docsToc, StringComparison.Ordinal);
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(ReadFinalReleaseText(fileName));
    }

    private static string ReadFinalReleaseText(string fileName)
    {
        return ReadText("artifacts", "final-release", fileName);
    }

    private static string ReadText(params string[] pathParts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(pathParts).ToArray()));
    }
}

using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ReleaseEvidenceClosureIndexTests
{
    [Fact]
    public void ReleaseEvidenceClosureIndexCentralizesSampleToolArticleAndProofBoundaries()
    {
        using JsonDocument document = ReadFinalReleaseJson("release-evidence-closure-index.json");
        JsonElement root = document.RootElement;

        Assert.Equal("release-evidence-closure-index", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-action-required", root.GetProperty("indexState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("approvesPublicRelease").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isRealModelRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());

        string proofBoundary = root.GetProperty("proofBoundary").GetString()!;
        Assert.Contains("not runtime proof", proofBoundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not package-consumer-runtime proof", proofBoundary, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not post-publish proof", proofBoundary, StringComparison.OrdinalIgnoreCase);

        string[] sourceArtifacts = root.GetProperty("sourceArtifacts").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        foreach (string required in new[]
        {
            "applications/YoloVision/yolo-model-matrix.json",
            "applications/YoloVision/yolovision-output.schema.json",
            "applications/OnnxToEngine/trtexec-parity-matrix.json",
            "applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json",
            "applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json",
            "applications/TensorRtExec/tensor-rt-exec-report.schema.json",
            "docs/articles/zh-cn/publishing/article-roadmap-30plus.json",
            "docs/articles/zh-cn/source-build-cmake-windows-guide.md",
            "artifacts/final-release/release-evidence-bundle.json",
            "artifacts/final-release/release-evidence-classification-audit.json",
            "artifacts/final-release/real-case-proof-execution-pack.json",
            "artifacts/final-release/clean-consumer-proof-owner-execution-pack.json",
            "artifacts/final-release/package-consumer-runtime-proof-record-validation.json"
        })
        {
            Assert.Contains(required, sourceArtifacts);
        }
    }

    [Fact]
    public void ReleaseEvidenceClosureIndexKeepsEveryLaneNonProofUntilOwnerEvidenceArrives()
    {
        using JsonDocument document = ReadFinalReleaseJson("release-evidence-closure-index.json");
        JsonElement[] lanes = document.RootElement.GetProperty("lanes").EnumerateArray().ToArray();

        Assert.True(lanes.Length >= 6);

        foreach (JsonElement lane in lanes)
        {
            Assert.False(lane.GetProperty("isRuntimeProof").GetBoolean());
            Assert.True(lane.GetProperty("requiredBeforePromotion").GetArrayLength() >= 3);
            Assert.True(lane.GetProperty("nonSubstitutes").GetArrayLength() >= 3);
        }

        string[] laneIds = lanes.Select(static item => item.GetProperty("id").GetString()!).ToArray();
        foreach (string laneId in new[]
        {
            "yolovision-real-model",
            "onnx-to-engine-build-report",
            "tensorrtexec-tool",
            "article-system",
            "package-consumer-runtime",
            "post-publish-verification"
        })
        {
            Assert.Contains(laneId, laneIds);
        }

        string raw = document.RootElement.GetRawText();
        foreach (string marker in new[]
        {
            "owner-action-required",
            "build-report-only",
            "documentation-ready-not-proof",
            "template-only",
            "blocked-real-publication-required"
        })
        {
            Assert.Contains(marker, raw, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void ReleaseEvidenceClosureIndexLocksForbiddenSubstitutesAndCloseGates()
    {
        using JsonDocument document = ReadFinalReleaseJson("release-evidence-closure-index.json");
        JsonElement root = document.RootElement;

        string[] forbiddenSubstitutes = root.GetProperty("forbiddenSubstitutes").EnumerateArray()
            .Select(static item => item.GetString()!)
            .ToArray();

        foreach (string forbidden in new[]
        {
            "TensorRtExec report",
            "YoloVision matrix",
            "OnnxToEngine report",
            "build-only",
            "dry-run",
            "preflight-only",
            "sidecar-only",
            "template-only",
            "Skipped=True",
            "blocked-by-cuda-driver",
            "dependency-probe-only",
            "local feed",
            "ProjectReference",
            "direct .nupkg",
            "article",
            "roadmap"
        })
        {
            Assert.Contains(forbidden, forbiddenSubstitutes);
        }

        JsonElement[] gates = root.GetProperty("closureGates").EnumerateArray().ToArray();
        Assert.All(gates, gate => Assert.True(gate.GetProperty("requiredForClose").GetBoolean()));

        string gateText = string.Join('\n', gates.Select(static gate => gate.GetRawText()));
        foreach (string validator in new[]
        {
            "eng/Test-RealCaseEvidenceRecord.ps1",
            "eng/Test-PackageConsumerRuntimeProofRecord.ps1",
            "eng/Test-PostPublishVerificationRecord.ps1",
            "eng/Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady"
        })
        {
            Assert.Contains(validator, gateText, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ReleaseEvidenceClosureIndexIsDocumentedInDocsAndMarkdownMirror()
    {
        string artifactMarkdown = ReadFinalReleaseText("release-evidence-closure-index.md");
        string article = ReadText("docs", "articles", "zh-cn", "release-evidence-closure-index.md");
        string docsIndex = ReadText("docs", "index.md");
        string docsToc = ReadText("docs", "toc.yml");

        foreach (string marker in new[]
        {
            "Release Evidence Closure Index",
            "blocked-owner-action-required",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "TensorRtExec report",
            "YoloVision matrix",
            "OnnxToEngine report",
            "ProjectReference",
            "direct `.nupkg`"
        })
        {
            Assert.Contains(marker, artifactMarkdown, StringComparison.Ordinal);
        }

        foreach (string marker in new[]
        {
            "发布证据闭环索引",
            "不是 runtime proof",
            "YoloVision",
            "TensorRtExec",
            "package-consumer-runtime-proof-record.json",
            "eng/Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady"
        })
        {
            Assert.Contains(marker, article, StringComparison.Ordinal);
        }

        Assert.Contains("articles/zh-cn/release-evidence-closure-index.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("artifacts/final-release/release-evidence-closure-index.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/release-evidence-closure-index.md", docsToc, StringComparison.Ordinal);
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

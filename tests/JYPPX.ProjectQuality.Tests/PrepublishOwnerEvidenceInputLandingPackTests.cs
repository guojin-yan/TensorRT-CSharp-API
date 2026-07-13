using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class PrepublishOwnerEvidenceInputLandingPackTests
{
    [Fact]
    public void LandingPackKeepsReleaseBlockedAndRejectsSubstitutes()
    {
        using JsonDocument document = ReadFinalReleaseJson("prepublish-owner-evidence-input-landing-pack.json");
        JsonElement root = document.RootElement;

        Assert.Equal("prepublish-owner-evidence-input-landing-pack", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-action-required", root.GetProperty("landingState").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("approvesPublicRelease").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishVerificationProof").GetBoolean());

        string boundary = root.GetProperty("proofBoundary").GetString()!;
        foreach (string marker in new[]
        {
            "does not run models",
            "publish packages",
            "generate real proof records",
            "templates",
            "matrices",
            "Skipped=True",
            "blocked-by-cuda-driver",
            "local feed",
            "ProjectReference",
            "direct .nupkg"
        })
        {
            Assert.Contains(marker, boundary, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void LandingPackDefinesOwnerInputsLogsHashesHostMetadataAndValidators()
    {
        using JsonDocument document = ReadFinalReleaseJson("prepublish-owner-evidence-input-landing-pack.json");
        JsonElement root = document.RootElement;

        string[] expectedOrder =
        {
            "real-model-runtime",
            "package-consumer-runtime",
            "post-publish-verification",
            "release-issue-close"
        };

        Assert.Equal(expectedOrder, root.GetProperty("ownerInputOrder").EnumerateArray().Select(static item => item.GetString()!).ToArray());

        JsonElement[] lanes = root.GetProperty("lanes").EnumerateArray().ToArray();
        Assert.Equal(4, lanes.Length);

        for (int index = 0; index < expectedOrder.Length; index++)
        {
            JsonElement lane = Assert.Single(lanes, item => item.GetProperty("id").GetString() == expectedOrder[index]);

            Assert.Equal(index + 1, lane.GetProperty("ownerInputOrder").GetInt32());
            Assert.False(lane.GetProperty("canPromote").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("ownerInputFilePath").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("expectedFinalRecordPath").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("recordLandingState").GetString()));
            Assert.True(lane.GetProperty("requiredSourceLogs").GetArrayLength() >= 4);
            Assert.True(lane.GetProperty("requiredSha256CommandExamples").GetArrayLength() >= 3);
            Assert.True(lane.GetProperty("requiredHostMetadataCaptureCommands").GetArrayLength() >= 3);
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("strictValidatorCommand").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("validatorOutputArchivePath").GetString()));
            Assert.True(lane.GetProperty("commonFailureReasons").GetArrayLength() >= 5);
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("repairAction").GetString()));
            Assert.True(lane.GetProperty("forbiddenSubstitutions").GetArrayLength() >= 7);
        }
    }

    [Fact]
    public void LandingPackCapturesRealModelAndPackageConsumerSpecificRequirements()
    {
        using JsonDocument document = ReadFinalReleaseJson("prepublish-owner-evidence-input-landing-pack.json");
        JsonElement[] lanes = document.RootElement.GetProperty("lanes").EnumerateArray().ToArray();

        JsonElement realModelLane = Assert.Single(lanes, item => item.GetProperty("id").GetString() == "real-model-runtime");
        Assert.Equal(
            new[] { "det", "cls", "seg", "obb", "pose", "sem" },
            realModelLane.GetProperty("requiredTaskCoverage").EnumerateArray().Select(static item => item.GetString()!).ToArray());
        string realModelText = realModelLane.GetRawText();
        foreach (string marker in new[] { "model source URL", "model license", "ONNX model", "serialized engine", "stdout and stderr logs" })
        {
            Assert.Contains(marker, realModelText, StringComparison.Ordinal);
        }

        JsonElement packageLane = Assert.Single(lanes, item => item.GetProperty("id").GetString() == "package-consumer-runtime");
        string packageText = packageLane.GetRawText();
        foreach (string marker in new[]
        {
            "clean consumer root must be outside the repository",
            "public package source must be traceable",
            "no ProjectReference",
            "no local feed",
            "no direct .nupkg",
            "runtime smoke log must exist",
            "runtime smoke log hash must match owner input"
        })
        {
            Assert.Contains(marker, packageText, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void LandingPackIsLinkedFromReleaseClosureSurfaces()
    {
        string landingPath = "artifacts/final-release/prepublish-owner-evidence-input-landing-pack.json";

        foreach (string fileName in new[]
        {
            "real-release-proof-backfill-final-close-gate.json",
            "owner-execution-result-import-prepublish-recheck.json",
            "release-candidate-owner-one-screen-execution-pack.json",
            "real-external-execution-backfill-final-freeze.json",
            "owner-real-proof-import-master-pack.json",
            "release-evidence-closure-index.json"
        })
        {
            using JsonDocument linkedDocument = ReadFinalReleaseJson(fileName);
            Assert.Contains(landingPath, linkedDocument.RootElement.GetRawText(), StringComparison.Ordinal);
            Assert.False(linkedDocument.RootElement.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(linkedDocument.RootElement.GetProperty("canCloseReleaseIssue").GetBoolean());
        }
    }

    [Fact]
    public void LandingPackIsDocumentedAndIndexed()
    {
        string artifact = ReadFinalReleaseText("prepublish-owner-evidence-input-landing-pack.md");
        string article = ReadText("docs", "articles", "zh-cn", "prepublish-owner-evidence-input-landing-pack.md");
        string docsIndex = ReadText("docs", "index.md");
        string docsToc = ReadText("docs", "toc.yml");

        foreach (string marker in new[]
        {
            "Prepublish Owner Evidence Input Landing Pack",
            "blocked-owner-action-required",
            "Can publish publicly: `false`",
            "Can close release issue: `false`",
            "Owner input templates must not be marked passed",
            "package-consumer-runtime-proof-owner-input.json"
        })
        {
            Assert.Contains(marker, artifact, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string marker in new[]
        {
            "公开发布前 Owner 实证输入落地包",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "det/cls/seg/obb/pose/sem",
            "无 ProjectReference",
            "无 local feed",
            "无 direct `.nupkg`"
        })
        {
            Assert.Contains(marker, article, StringComparison.Ordinal);
        }

        Assert.Contains("articles/zh-cn/prepublish-owner-evidence-input-landing-pack.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("artifacts/final-release/prepublish-owner-evidence-input-landing-pack.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/prepublish-owner-evidence-input-landing-pack.md", docsToc, StringComparison.Ordinal);
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

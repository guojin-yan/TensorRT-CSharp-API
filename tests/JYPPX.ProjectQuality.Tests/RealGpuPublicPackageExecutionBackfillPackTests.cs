using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class RealGpuPublicPackageExecutionBackfillPackTests
{
    [Fact]
    public void BackfillPackKeepsReleaseBlockedAndCommandPackNonProof()
    {
        using JsonDocument document = ReadFinalReleaseJson("real-gpu-public-package-execution-backfill-pack.json");
        JsonElement root = document.RootElement;

        Assert.Equal("real-gpu-public-package-execution-backfill-pack", root.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-owner-action-required", root.GetProperty("backfillState").GetString());
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
            "execution checklist only",
            "does not execute NuGet publish",
            "generate fake passed records",
            "provide real hashes",
            "command examples",
            "local feed",
            "ProjectReference",
            "direct .nupkg"
        })
        {
            Assert.Contains(marker, boundary, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void BackfillPackDefinesEvidenceDirectoriesAndLaneCommandPacks()
    {
        using JsonDocument document = ReadFinalReleaseJson("real-gpu-public-package-execution-backfill-pack.json");
        JsonElement root = document.RootElement;

        foreach (string directory in new[]
        {
            "artifacts/final-release/owner-evidence/real-model-runtime/{task}/",
            "artifacts/final-release/owner-evidence/package-consumer-runtime/",
            "artifacts/final-release/owner-evidence/post-publish-verification/",
            "artifacts/final-release/owner-evidence/release-issue-close/"
        })
        {
            Assert.Contains(directory, root.GetProperty("evidenceDirectoryLayout").GetRawText(), StringComparison.Ordinal);
        }

        JsonElement[] lanes = root.GetProperty("lanes").EnumerateArray().ToArray();
        Assert.Equal(4, lanes.Length);

        for (int index = 0; index < lanes.Length; index++)
        {
            JsonElement lane = lanes[index];
            Assert.Equal(index + 1, lane.GetProperty("executionOrder").GetInt32());
            Assert.False(lane.GetProperty("canPromote").GetBoolean());
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("evidenceDirectory").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("ownerInputTarget").GetString()));
            Assert.False(string.IsNullOrWhiteSpace(lane.GetProperty("validatorOutputArchivePath").GetString()));
            Assert.True(lane.GetProperty("environmentCaptureCommands").GetArrayLength() >= 3);
            Assert.True(lane.GetProperty("executionCommandTemplates").GetArrayLength() >= 3);
            Assert.True(lane.GetProperty("logRedirectionRules").GetArrayLength() >= 3);
            Assert.True(lane.GetProperty("hashCollectionCommands").GetArrayLength() >= 3);
            Assert.Contains("*>", lane.GetRawText(), StringComparison.Ordinal);
            Assert.Contains("-FailOn", lane.GetProperty("strictValidatorCommand").GetString(), StringComparison.Ordinal);
        }
    }

    [Fact]
    public void BackfillPackCapturesYoloVisionSixTasksAndPublicPackageChecks()
    {
        using JsonDocument document = ReadFinalReleaseJson("real-gpu-public-package-execution-backfill-pack.json");
        JsonElement[] lanes = document.RootElement.GetProperty("lanes").EnumerateArray().ToArray();

        JsonElement realModelLane = Assert.Single(lanes, item => item.GetProperty("id").GetString() == "real-model-runtime");
        Assert.Equal(
            new[] { "det", "cls", "seg", "obb", "pose", "sem" },
            realModelLane.GetProperty("requiredTaskCoverage").EnumerateArray().Select(static item => item.GetString()!).ToArray());
        string realModelText = realModelLane.GetRawText();
        foreach (string marker in new[] { "--task det", "--task cls", "--task seg", "--task obb", "--task pose", "--task sem", "modelSourceUrl", "license", "onnxSha256", "engineSha256" })
        {
            Assert.Contains(marker, realModelText, StringComparison.Ordinal);
        }

        JsonElement packageLane = Assert.Single(lanes, item => item.GetProperty("id").GetString() == "package-consumer-runtime");
        string packageText = packageLane.GetRawText();
        foreach (string marker in new[]
        {
            "clean consumer root outside repository",
            "dotnet restore --no-cache --force-evaluate",
            "check .csproj has no ProjectReference",
            "check NuGet config does not point to local feed",
            "check PackageReference does not use direct .nupkg",
            "smoke log hash must match"
        })
        {
            Assert.Contains(marker, packageText, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void BackfillPackIsLinkedFromReleaseClosureSurfaces()
    {
        string packPath = "artifacts/final-release/real-gpu-public-package-execution-backfill-pack.json";

        foreach (string fileName in new[]
        {
            "prepublish-owner-evidence-input-landing-pack.json",
            "real-release-proof-backfill-final-close-gate.json",
            "owner-execution-result-import-prepublish-recheck.json",
            "release-candidate-owner-one-screen-execution-pack.json",
            "real-external-execution-backfill-final-freeze.json",
            "owner-real-proof-import-master-pack.json",
            "release-evidence-closure-index.json"
        })
        {
            using JsonDocument linkedDocument = ReadFinalReleaseJson(fileName);
            Assert.Contains(packPath, linkedDocument.RootElement.GetRawText(), StringComparison.Ordinal);
            Assert.False(linkedDocument.RootElement.GetProperty("canPublishPublicly").GetBoolean());
            Assert.False(linkedDocument.RootElement.GetProperty("canCloseReleaseIssue").GetBoolean());
        }
    }

    [Fact]
    public void BackfillPackIsDocumentedAndIndexed()
    {
        string artifact = ReadFinalReleaseText("real-gpu-public-package-execution-backfill-pack.md");
        string article = ReadText("docs", "articles", "zh-cn", "real-gpu-public-package-execution-backfill-pack.md");
        string docsIndex = ReadText("docs", "index.md");
        string docsToc = ReadText("docs", "toc.yml");

        foreach (string marker in new[]
        {
            "Real GPU Public Package Execution Backfill Pack",
            "blocked-owner-action-required",
            "Can publish publicly: `false`",
            "Can close release issue: `false`",
            "The command pack itself is not proof"
        })
        {
            Assert.Contains(marker, artifact, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string marker in new[]
        {
            "真实 GPU 与公开包源执行回填包",
            "canPublishPublicly=false",
            "canCloseReleaseIssue=false",
            "det/cls/seg/obb/pose/sem",
            "dotnet restore --no-cache --force-evaluate",
            "无 ProjectReference",
            "无 local feed",
            "无 direct `.nupkg`"
        })
        {
            Assert.Contains(marker, article, StringComparison.Ordinal);
        }

        Assert.Contains("articles/zh-cn/real-gpu-public-package-execution-backfill-pack.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("artifacts/final-release/real-gpu-public-package-execution-backfill-pack.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/real-gpu-public-package-execution-backfill-pack.md", docsToc, StringComparison.Ordinal);
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

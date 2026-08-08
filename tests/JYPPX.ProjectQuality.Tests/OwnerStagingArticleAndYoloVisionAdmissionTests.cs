using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

[Collection("ReleaseCloseProofArtifacts")]
public sealed class OwnerStagingArticleAndYoloVisionAdmissionTests
{
    [Fact]
    public void ArticleAndYoloVisionStagingAdmissionDefaultsToBlockedNonProof()
    {
        ResetDefaultStagingArtifacts();

        using JsonDocument articleDocument = ReadFinalReleaseJson("article-publication-proof-from-staging-workspace.json");
        JsonElement article = articleDocument.RootElement;
        Assert.Equal("article-publication-proof-from-staging-workspace", article.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-article-publication-staging-owner-proof-required", article.GetProperty("importState").GetString());
        Assert.False(article.GetProperty("ownerEvidenceShapeValid").GetBoolean());
        Assert.Equal(0, article.GetProperty("articleProofRecordCount").GetInt32());
        Assert.True(article.GetProperty("failedActionRequiredCount").GetInt32() >= 3);
        AssertNonProof(article);

        using JsonDocument yoloDocument = ReadFinalReleaseJson("yolovision-real-model-proof-from-staging-workspace.json");
        JsonElement yolo = yoloDocument.RootElement;
        Assert.Equal("yolovision-real-model-proof-from-staging-workspace", yolo.GetProperty("recordKind").GetString());
        Assert.Equal("blocked-yolovision-real-model-staging-owner-proof-required", yolo.GetProperty("importState").GetString());
        Assert.False(yolo.GetProperty("ownerEvidenceShapeValid").GetBoolean());
        Assert.Equal(12, yolo.GetProperty("assetFileCount").GetInt32());
        Assert.Equal(0, yolo.GetProperty("existingAssetFileCount").GetInt32());
        Assert.True(yolo.GetProperty("failedActionRequiredCount").GetInt32() >= 12);
        AssertNonProof(yolo);
    }

    [Fact]
    public void StrictExternalStagingWorkspaceCanBeShapeValidWithoutClaimingProof()
    {
        string stagingRoot = CreateValidStagingWorkspace();

        try
        {
            RunPowerShell(
                "Import-OwnerRealProofStagingWorkspace.ps1",
                "-OwnerStagingRoot",
                stagingRoot,
                "-RequireExistingFiles",
                "-RequireHashMatch");
            RunPowerShell("Test-OwnerRealProofStagingWorkspace.ps1", "-Strict");
            RunPowerShell("Import-ArticlePublicationProofFromStagingWorkspace.ps1");
            RunPowerShell("Test-ArticlePublicationProofFromStagingWorkspace.ps1", "-Strict");
            RunPowerShell("Import-YoloVisionRealModelProofFromStagingWorkspace.ps1");
            RunPowerShell("Test-YoloVisionRealModelProofFromStagingWorkspace.ps1", "-Strict");

            using JsonDocument articleDocument = ReadFinalReleaseJson("article-publication-proof-from-staging-workspace.json");
            JsonElement article = articleDocument.RootElement;
            Assert.Equal("article-publication-staging-shape-valid-non-proof", article.GetProperty("importState").GetString());
            Assert.True(article.GetProperty("ownerEvidenceShapeValid").GetBoolean());
            Assert.Equal(1, article.GetProperty("articleProofRecordCount").GetInt32());
            Assert.Equal(1, article.GetProperty("articleProofReadyRecordCount").GetInt32());
            Assert.True(article.GetProperty("articleProofManifestHashMatches").GetBoolean());
            Assert.True(article.GetProperty("screenshotsArchiveHashReady").GetBoolean());
            Assert.Equal(0, article.GetProperty("failedActionRequiredCount").GetInt32());
            AssertNonProof(article);

            using JsonDocument yoloDocument = ReadFinalReleaseJson("yolovision-real-model-proof-from-staging-workspace.json");
            JsonElement yolo = yoloDocument.RootElement;
            Assert.Equal("yolovision-real-model-staging-shape-valid-non-proof", yolo.GetProperty("importState").GetString());
            Assert.True(yolo.GetProperty("ownerEvidenceShapeValid").GetBoolean());
            Assert.Equal(12, yolo.GetProperty("assetFileCount").GetInt32());
            Assert.Equal(12, yolo.GetProperty("existingAssetFileCount").GetInt32());
            Assert.Equal(12, yolo.GetProperty("sha256ValidFileCount").GetInt32());
            Assert.Equal(11, yolo.GetProperty("assetManifestHashMatchCount").GetInt32());
            Assert.True(yolo.GetProperty("assetManifestLinkageReady").GetBoolean());
            Assert.True(yolo.GetProperty("modelLicenseReady").GetBoolean());
            Assert.True(yolo.GetProperty("realModelExecutionConfirmationReady").GetBoolean());
            Assert.True(yolo.GetProperty("runtimeTranscriptLinkageReady").GetBoolean());
            Assert.Equal(0, yolo.GetProperty("failedActionRequiredCount").GetInt32());
            AssertNonProof(yolo);
        }
        finally
        {
            ResetDefaultStagingArtifacts();
            if (Directory.Exists(stagingRoot))
            {
                Directory.Delete(stagingRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void ArticleStagingRejectsDraftPublicationArtifacts()
    {
        string stagingRoot = CreateValidStagingWorkspace();
        WriteArticleProof(stagingRoot, isDraft: true);

        try
        {
            ImportStrictOwnerStaging(stagingRoot);
            RunPowerShell("Import-ArticlePublicationProofFromStagingWorkspace.ps1");
            RunPowerShell("Test-ArticlePublicationProofFromStagingWorkspace.ps1", "-Strict");

            using JsonDocument document = ReadFinalReleaseJson("article-publication-proof-from-staging-workspace.json");
            JsonElement root = document.RootElement;
            Assert.Equal("blocked-article-publication-staging-owner-proof-required", root.GetProperty("importState").GetString());
            Assert.False(root.GetProperty("ownerEvidenceShapeValid").GetBoolean());
            Assert.True(root.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
            Assert.Contains(root.GetProperty("findings").EnumerateArray(), static finding =>
                finding.GetProperty("category").GetString() == "draft-article");
            AssertNonProof(root);
        }
        finally
        {
            ResetDefaultStagingArtifacts();
            if (Directory.Exists(stagingRoot))
            {
                Directory.Delete(stagingRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void YoloVisionStagingRejectsMatrixOnlyRuntimeTranscript()
    {
        string stagingRoot = CreateValidStagingWorkspace();
        WriteYoloVisionProof(stagingRoot, transcriptContainsForbiddenSubstitute: true);

        try
        {
            ImportStrictOwnerStaging(stagingRoot);
            RunPowerShell("Import-YoloVisionRealModelProofFromStagingWorkspace.ps1");
            RunPowerShell("Test-YoloVisionRealModelProofFromStagingWorkspace.ps1", "-Strict");

            using JsonDocument document = ReadFinalReleaseJson("yolovision-real-model-proof-from-staging-workspace.json");
            JsonElement root = document.RootElement;
            Assert.Equal("blocked-yolovision-real-model-staging-owner-proof-required", root.GetProperty("importState").GetString());
            Assert.False(root.GetProperty("ownerEvidenceShapeValid").GetBoolean());
            Assert.True(root.GetProperty("failedActionRequiredCount").GetInt32() >= 1);
            Assert.Contains(root.GetProperty("findings").EnumerateArray(), static finding =>
                finding.GetProperty("category").GetString() == "forbidden-substitute");
            AssertNonProof(root);
        }
        finally
        {
            ResetDefaultStagingArtifacts();
            if (Directory.Exists(stagingRoot))
            {
                Directory.Delete(stagingRoot, recursive: true);
            }
        }
    }

    private static string CreateValidStagingWorkspace()
    {
        RunPowerShell("Export-OwnerRealProofStagingWorkspaceContract.ps1");

        string stagingRoot = Path.Combine(
            Path.GetTempPath(),
            "TensorRtSharp-owner-article-yolovision-staging-" + Guid.NewGuid().ToString("N"));

        using JsonDocument contractDocument = ReadFinalReleaseJson("owner-real-proof-staging-workspace-contract.json");
        foreach (JsonElement file in contractDocument.RootElement.GetProperty("requiredFiles").EnumerateArray())
        {
            string relativePath = file.GetProperty("relativePath").GetString()!;
            WriteText(stagingRoot, relativePath, $"owner-staging-placeholder:{relativePath}");
        }

        WriteArticleProof(stagingRoot);
        WriteYoloVisionProof(stagingRoot);
        return stagingRoot;
    }

    private static void ImportStrictOwnerStaging(string stagingRoot)
    {
        RunPowerShell(
            "Import-OwnerRealProofStagingWorkspace.ps1",
            "-OwnerStagingRoot",
            stagingRoot,
            "-RequireExistingFiles",
            "-RequireHashMatch");
        RunPowerShell("Test-OwnerRealProofStagingWorkspace.ps1", "-Strict");
    }

    private static void WriteArticleProof(string stagingRoot, bool isDraft = false)
    {
        string recordsRelativePath = "article-publication/article-proof-records.json";
        string manifestRelativePath = "article-publication/article-proof-manifest.json";
        string screenshotsRelativePath = "article-publication/screenshots.zip";

        WriteBytes(stagingRoot, screenshotsRelativePath, Encoding.UTF8.GetBytes("published article screenshots archive fixture"));

        var records = new
        {
            articleProofRecords = new[]
            {
                new
                {
                    articleId = "article-proof-001",
                    title = "TensorRtSharp 4.0 package consumer proof",
                    publicUrl = "https://example.com/tensorrtsharp/articles/package-consumer-proof",
                    publishedAtUtc = "2026-07-14T00:00:00Z",
                    platform = "owner-publication-channel",
                    contentSha256 = HashText("article-proof-content"),
                    screenshotPath = "screenshots/article-proof-001.png",
                    screenshotSha256 = HashFile(FullPath(stagingRoot, screenshotsRelativePath)),
                    isPublic = true,
                    isDraft,
                    ownerReviewed = true,
                },
            },
        };
        WriteJson(stagingRoot, recordsRelativePath, records);

        var manifest = new
        {
            articleProofCount = 1,
            articleProofRecordsSha256 = HashFile(FullPath(stagingRoot, recordsRelativePath)),
            screenshotsArchiveSha256 = HashFile(FullPath(stagingRoot, screenshotsRelativePath)),
            manifestSha256 = HashText("article-proof-manifest"),
            generatedAtUtc = "2026-07-14T00:00:00Z",
            publicProofManifestUrl = "https://example.com/tensorrtsharp/articles/proof-manifest.json",
        };
        WriteJson(stagingRoot, manifestRelativePath, manifest);
    }

    private static void WriteYoloVisionProof(
        string stagingRoot,
        bool transcriptContainsForbiddenSubstitute = false,
        bool confirmationRejectsSubstitutes = true)
    {
        const string taskName = "YoloVisionRealModelFixture";

        WriteJson(stagingRoot, "yolovision/task-metadata.json", new
        {
            taskName,
            modelFamily = "YOLO",
            taskType = "object-detection",
        });
        WriteBytes(stagingRoot, "yolovision/model.onnx", Encoding.UTF8.GetBytes("onnx-model-fixture"));
        WriteJson(stagingRoot, "yolovision/model-license.json", new
        {
            modelSourceUrl = "https://example.com/models/yolovision-real-model.onnx",
            licenseName = "MIT",
            redistributionAllowed = true,
            ownerLicenseReviewed = true,
        });
        WriteText(stagingRoot, "yolovision/labels.txt", "person\ncar\n");
        WriteBytes(stagingRoot, "yolovision/input.bin", new byte[] { 1, 2, 3, 4, 5, 6 });
        WriteJson(stagingRoot, "yolovision/output.json", new
        {
            detections = new[]
            {
                new { label = "person", confidence = 0.92, x = 1, y = 2, width = 3, height = 4 },
            },
        });
        WriteText(stagingRoot, "yolovision/stdout.log", "TensorRtSharp YoloVision inference completed.\n");
        WriteText(stagingRoot, "yolovision/stderr.log", "No stderr output.\n");
        WriteText(
            stagingRoot,
            "yolovision/runtime-transcript.log",
            transcriptContainsForbiddenSubstitute
                ? "matrix-only fixture must be rejected by staging admission.\n"
                : "TensorRtSharp YoloVision full inference command executed with model assets and captured outputs.\n");
        WriteJson(stagingRoot, "yolovision/host-metadata.json", new
        {
            osDescription = "Windows test fixture",
            cudaVersion = "13.0",
            tensorRtVersion = "11.0",
        });
        WriteJson(stagingRoot, "yolovision/real-model-execution-confirmation.json", new
        {
            taskName,
            realModelExecutionConfirmed = true,
            notReadinessOnly = confirmationRejectsSubstitutes,
            notTutorialOnly = confirmationRejectsSubstitutes,
            notMatrixOnly = confirmationRejectsSubstitutes,
        });

        WriteJson(stagingRoot, "yolovision/asset-manifest.json", new
        {
            taskMetadataPath = "yolovision/task-metadata.json",
            taskMetadataSha256 = HashFile(FullPath(stagingRoot, "yolovision/task-metadata.json")),
            modelPath = "yolovision/model.onnx",
            modelSha256 = HashFile(FullPath(stagingRoot, "yolovision/model.onnx")),
            modelLicensePath = "yolovision/model-license.json",
            modelLicenseSha256 = HashFile(FullPath(stagingRoot, "yolovision/model-license.json")),
            labelsPath = "yolovision/labels.txt",
            labelsSha256 = HashFile(FullPath(stagingRoot, "yolovision/labels.txt")),
            inputPath = "yolovision/input.bin",
            inputImageSha256 = HashFile(FullPath(stagingRoot, "yolovision/input.bin")),
            outputJsonPath = "yolovision/output.json",
            outputJsonSha256 = HashFile(FullPath(stagingRoot, "yolovision/output.json")),
            stdoutLogPath = "yolovision/stdout.log",
            stdoutLogSha256 = HashFile(FullPath(stagingRoot, "yolovision/stdout.log")),
            stderrLogPath = "yolovision/stderr.log",
            stderrLogSha256 = HashFile(FullPath(stagingRoot, "yolovision/stderr.log")),
            runtimeTranscriptPath = "yolovision/runtime-transcript.log",
            runtimeTranscriptSha256 = HashFile(FullPath(stagingRoot, "yolovision/runtime-transcript.log")),
            hostMetadataPath = "yolovision/host-metadata.json",
            hostMetadataSha256 = HashFile(FullPath(stagingRoot, "yolovision/host-metadata.json")),
            realModelExecutionConfirmationPath = "yolovision/real-model-execution-confirmation.json",
            realModelExecutionConfirmationSha256 = HashFile(FullPath(stagingRoot, "yolovision/real-model-execution-confirmation.json")),
        });
    }

    private static void ResetDefaultStagingArtifacts()
    {
        RunPowerShell("Import-OwnerRealProofStagingWorkspace.ps1");
        RunPowerShell("Test-OwnerRealProofStagingWorkspace.ps1", "-Strict");
        RunPowerShell("Import-ArticlePublicationProofFromStagingWorkspace.ps1");
        RunPowerShell("Test-ArticlePublicationProofFromStagingWorkspace.ps1", "-Strict");
        RunPowerShell("Import-YoloVisionRealModelProofFromStagingWorkspace.ps1");
        RunPowerShell("Test-YoloVisionRealModelProofFromStagingWorkspace.ps1", "-Strict");
    }

    private static void AssertNonProof(JsonElement root)
    {
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("usesPublishToken").GetBoolean());
        Assert.False(root.GetProperty("performsRuntimeExecution").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.False(root.GetProperty("isRuntimeExecutionProof").GetBoolean());
        Assert.False(root.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("isPostPublishProof").GetBoolean());
        Assert.False(root.GetProperty("isReleaseCloseProof").GetBoolean());
    }

    private static JsonDocument ReadFinalReleaseJson(string fileName)
    {
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "final-release",
            fileName)));
    }

    private static string FullPath(string root, string relativePath)
    {
        return Path.Combine(root, relativePath.Replace('/', Path.DirectorySeparatorChar));
    }

    private static void WriteJson(string root, string relativePath, object value)
    {
        WriteText(root, relativePath, JsonSerializer.Serialize(value, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static void WriteText(string root, string relativePath, string content)
    {
        string fullPath = FullPath(root, relativePath);
        Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
        File.WriteAllText(fullPath, content, Encoding.UTF8);
    }

    private static void WriteBytes(string root, string relativePath, byte[] content)
    {
        string fullPath = FullPath(root, relativePath);
        Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
        File.WriteAllBytes(fullPath, content);
    }

    private static string HashText(string text)
    {
        using SHA256 sha256 = SHA256.Create();
        return ToLowerHex(sha256.ComputeHash(Encoding.UTF8.GetBytes(text)));
    }

    private static string HashFile(string path)
    {
        using SHA256 sha256 = SHA256.Create();
        using FileStream stream = File.OpenRead(path);
        return ToLowerHex(sha256.ComputeHash(stream));
    }

    private static string ToLowerHex(byte[] hash)
    {
        return BitConverter.ToString(hash).Replace("-", string.Empty, StringComparison.Ordinal).ToLowerInvariant();
    }

    private static string RunPowerShell(string scriptName, params string[] arguments)
    {
        using Process process = new()
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = PowerShellHost.ResolveExecutable(),
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

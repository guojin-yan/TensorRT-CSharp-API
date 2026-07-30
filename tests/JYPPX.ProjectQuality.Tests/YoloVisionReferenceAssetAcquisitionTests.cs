using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class YoloVisionReferenceAssetAcquisitionTests
{
    [Fact]
    public void YoloV8SegmentationOfficialManifestPinsReleaseAssetLicenseAndOutputRoles()
    {
        string manifestPath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "yolovision-yolov8n-seg-official-assets.json");
        string scriptPath = Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Acquire-YoloV8SegOfficialAssets.ps1");
        Assert.True(File.Exists(manifestPath), manifestPath);
        Assert.True(File.Exists(scriptPath), scriptPath);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement root = document.RootElement;
        Assert.Equal(
            "yolovision-yolov8n-seg-official-asset-acquisition-manifest",
            root.GetProperty("recordKind").GetString());
        Assert.Equal(177482232, root.GetProperty("upstreamReleaseId").GetInt64());
        Assert.Equal("v8.3.0", root.GetProperty("upstreamReleaseTag").GetString());
        Assert.Equal(40, root.GetProperty("upstreamSourceCommit").GetString()!.Length);
        Assert.Equal("AGPL-3.0-only", root.GetProperty("license").GetProperty("spdxId").GetString());
        Assert.False(root.GetProperty("license").GetProperty("publicRedistributionOwnerApproval").GetBoolean());

        JsonElement[] assets = root.GetProperty("assets").EnumerateArray().ToArray();
        Assert.Equal(2, assets.Length);
        Assert.All(assets, static asset =>
        {
            Assert.True(asset.GetProperty("expectedLength").GetInt64() > 0);
            Assert.Equal(64, asset.GetProperty("expectedSha256").GetString()!.Length);
            Assert.False(string.IsNullOrWhiteSpace(asset.GetProperty("hashProvenance").GetString()));
        });
        JsonElement modelAsset = assets.Single(static asset => asset.GetProperty("id").GetString() == "yolov8n-seg-pt");
        Assert.Equal(195720083, modelAsset.GetProperty("githubReleaseAssetId").GetInt64());
        Assert.Contains("upstream-release-api-digest-was-empty", modelAsset.GetProperty("hashProvenance").GetString(), StringComparison.Ordinal);

        JsonElement[] outputs = root.GetProperty("modelContract").GetProperty("outputs").EnumerateArray().ToArray();
        Assert.Equal(new[] { "output0", "output1" }, outputs.Select(static output => output.GetProperty("name").GetString()).ToArray());
        Assert.Equal("detection-rows-with-mask-coefficients", outputs[0].GetProperty("role").GetString());
        Assert.Equal(new[] { 1, 116, 8400 }, outputs[0].GetProperty("shape").EnumerateArray().Select(static value => value.GetInt32()).ToArray());
        Assert.Equal(32, outputs[0].GetProperty("maskCoefficientCount").GetInt32());
        Assert.Equal("mask-prototypes", outputs[1].GetProperty("role").GetString());
        Assert.Equal(new[] { 1, 32, 160, 160 }, outputs[1].GetProperty("shape").EnumerateArray().Select(static value => value.GetInt32()).ToArray());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("performsPublish").GetBoolean());

        string script = File.ReadAllText(scriptPath);
        Assert.Contains("Test-DriveIsNotC", script, StringComparison.Ordinal);
        Assert.Contains("expectedLength", script, StringComparison.Ordinal);
        Assert.Contains("expectedSha256", script, StringComparison.Ordinal);
        Assert.Contains("upstream Release did not publish a digest", script, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("performsExport = $false", script, StringComparison.Ordinal);
        Assert.Contains("performsRuntime = $false", script, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", script, StringComparison.Ordinal);
        Assert.DoesNotContain("dotnet nuget push", script, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("gh release upload", script, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void YoloV8SegmentationRealRuntimeEvidenceKeepsPromotionAndReleaseBoundariesSeparate()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "yolovision-yolov8n-seg-real-model-runtime-evidence.json");
        Assert.True(File.Exists(evidencePath), evidencePath);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(evidencePath));
        JsonElement root = document.RootElement;
        Assert.Equal("sample-run-evidence-record", root.GetProperty("recordKind").GetString());
        Assert.Equal("real-model-runtime", root.GetProperty("proofClassification").GetString());
        Assert.False(root.GetProperty("templateOnly").GetBoolean());
        Assert.True(root.GetProperty("isSmokePassed").GetBoolean());
        Assert.True(root.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.Equal("real-model-runtime", root.GetProperty("validatorState").GetString());
        Assert.Equal(
            "08b5c61368d4ddec5e647522fc55a93c42a9e0c581770aae48b87bba65a9b21d",
            root.GetProperty("modelSha256").GetString());

        JsonElement contract = root.GetProperty("modelContract");
        Assert.Equal(
            new[] { 1, 3, 640, 640 },
            contract.GetProperty("input").GetProperty("shape").EnumerateArray().Select(static value => value.GetInt32()).ToArray());
        JsonElement[] outputs = contract.GetProperty("outputs").EnumerateArray().ToArray();
        Assert.Equal(new[] { "output0", "output1" }, outputs.Select(static output => output.GetProperty("name").GetString()).ToArray());
        Assert.Equal(new[] { 1, 116, 8400 }, outputs[0].GetProperty("shape").EnumerateArray().Select(static value => value.GetInt32()).ToArray());
        Assert.Equal(new[] { 1, 32, 160, 160 }, outputs[1].GetProperty("shape").EnumerateArray().Select(static value => value.GetInt32()).ToArray());

        JsonElement runtimeReference = root.GetProperty("runtimeReferenceValidation");
        Assert.True(runtimeReference.GetProperty("passed").GetBoolean());
        JsonElement[] referenceTensors = runtimeReference.GetProperty("tensors").EnumerateArray().ToArray();
        Assert.Equal(2, referenceTensors.Length);
        Assert.Equal(1_793_600, referenceTensors.Sum(static tensor => tensor.GetProperty("comparedValueCount").GetInt32()));
        Assert.All(referenceTensors, static tensor =>
        {
            Assert.Equal(0, tensor.GetProperty("mismatchCount").GetInt32());
            Assert.Equal(64, tensor.GetProperty("referenceSha256").GetString()!.Length);
        });

        JsonElement postprocess = root.GetProperty("segmentationPostprocessValidation");
        Assert.True(postprocess.GetProperty("passed").GetBoolean());
        JsonElement thresholds = postprocess.GetProperty("thresholds");
        JsonElement[] predictions = postprocess.GetProperty("predictions").EnumerateArray().ToArray();
        Assert.Equal(4, predictions.Length);
        Assert.Equal(new[] { "dog", "bicycle", "truck", "car" }, predictions.Select(static prediction => prediction.GetProperty("className").GetString()).ToArray());
        Assert.All(predictions, prediction =>
        {
            Assert.True(prediction.GetProperty("passed").GetBoolean());
            Assert.True(prediction.GetProperty("boxIoU").GetDouble() >= thresholds.GetProperty("minimumBoxIoU").GetDouble());
            Assert.True(prediction.GetProperty("maskIoU").GetDouble() >= thresholds.GetProperty("minimumMaskIoU").GetDouble());
        });

        JsonElement negative = root.GetProperty("controlledNegativeValidation");
        Assert.Equal(2, negative.GetProperty("exitCode").GetInt32());
        Assert.True(negative.GetProperty("completed").GetBoolean());
        Assert.False(negative.GetProperty("passed").GetBoolean());
        Assert.Equal(974_400, negative.GetProperty("comparedValueCount").GetInt32());
        Assert.Equal(1, negative.GetProperty("mismatchCount").GetInt32());
        Assert.Equal(0, negative.GetProperty("firstMismatchIndex").GetInt32());

        JsonElement artifactIntegrity = root.GetProperty("controlledArtifactIntegrityValidation");
        Assert.True(artifactIntegrity.GetProperty("failClosed").GetBoolean());
        Assert.False(artifactIntegrity.GetProperty("passed").GetBoolean());
        Assert.Contains("SHA256 does not match", artifactIntegrity.GetProperty("diagnostic").GetString(), StringComparison.Ordinal);

        JsonElement boundary = root.GetProperty("proofBoundary");
        Assert.True(boundary.GetProperty("sourceTreeRealModelRuntime").GetBoolean());
        foreach (string name in new[]
        {
            "publicRedistributionApproved",
            "packageConsumerRuntimeProof",
            "publicPackageProof",
            "postPublishProof",
            "ownerReleaseAcceptance",
            "releaseProof",
            "performsPublish",
            "uploadsAssets"
        })
        {
            Assert.False(boundary.GetProperty(name).GetBoolean(), name);
        }

        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "README.md"));
        string tutorial = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "yolovision-yolov8-seg-real-asset-tutorial.md"));
        string maskGuide = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "yolovision-segmentation-mask-postprocess-guide.md"));
        foreach (string text in new[] { readme, tutorial, maskGuide })
        {
            Assert.Contains("--segmentation-mask-output-directory", text, StringComparison.Ordinal);
            Assert.Contains("real-model-runtime", text, StringComparison.Ordinal);
            Assert.Contains("package", text, StringComparison.OrdinalIgnoreCase);
        }
        Assert.Contains("Mismatches=1", tutorial, StringComparison.Ordinal);
        Assert.Contains("mask IoU", tutorial, StringComparison.Ordinal);
        Assert.Contains("publicRedistributionApproved", File.ReadAllText(evidencePath), StringComparison.Ordinal);
    }

    [Fact]
    public void RepositoryManifestKeepsHashVerifiedTensorRtAssetsBehindLicenseReview()
    {
        string path = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "yolovision-reference-assets.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        JsonElement root = document.RootElement;

        Assert.Equal("yolovision-reference-asset-acquisition-manifest", root.GetProperty("recordKind").GetString());
        Assert.Equal("source-files-hash-verified-license-review-required", root.GetProperty("status").GetString());
        Assert.Equal("owner-review-required", root.GetProperty("licenseBoundary").GetProperty("state").GetString());
        Assert.False(root.GetProperty("licenseBoundary").GetProperty("allRedistributionApproved").GetBoolean());
        Assert.False(root.GetProperty("licenseBoundary").GetProperty("canPromoteRealModelRuntime").GetBoolean());

        JsonElement[] assets = root.GetProperty("assets").EnumerateArray().ToArray();
        Assert.Equal(3, assets.Length);
        Assert.Equal(new[] { "model", "labels", "input-image" }, assets.Select(static asset => asset.GetProperty("role").GetString()).ToArray());
        Assert.All(assets, static asset =>
        {
            Assert.Equal(64, asset.GetProperty("expectedSha256").GetString()!.Length);
            Assert.True(asset.GetProperty("expectedLength").GetInt64() > 0);
            Assert.Equal("owner-review-required", asset.GetProperty("license").GetProperty("status").GetString());
            Assert.False(asset.GetProperty("license").GetProperty("redistributionApproved").GetBoolean());
        });
    }

    [Fact]
    public void AcquisitionScriptVerifiesAndCopiesFilesWithoutPromotingUnreviewedLicenses()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "jyppx-yolovision-assets-" + Guid.NewGuid().ToString("N"));
        string sourceRoot = Path.Combine(tempRoot, "source");
        string outputRoot = Path.Combine(tempRoot, "output");
        Directory.CreateDirectory(Path.Combine(sourceRoot, "bin"));
        Directory.CreateDirectory(Path.Combine(sourceRoot, "samples"));

        try
        {
            byte[] model = Encoding.UTF8.GetBytes("model-fixture");
            byte[] labels = Encoding.UTF8.GetBytes("label-a\nlabel-b\n");
            File.WriteAllBytes(Path.Combine(sourceRoot, "bin", "model.onnx"), model);
            File.WriteAllBytes(Path.Combine(sourceRoot, "samples", "labels.txt"), labels);

            object manifest = new
            {
                schemaVersion = 1,
                recordKind = "yolovision-reference-asset-acquisition-manifest",
                assetSetId = "test-fixture",
                sourceRootHints = Array.Empty<string>(),
                assets = new object[]
                {
                    NewAsset("model", "model", @"bin\model.onnx", "model.onnx", model),
                    NewAsset("labels", "labels", @"samples\labels.txt", "labels.txt", labels),
                },
            };
            string manifestPath = Path.Combine(tempRoot, "manifest.json");
            File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifest));

            string output = RunPowerShell(
                Path.Combine(RepositoryPaths.Root, "eng", "Acquire-YoloVisionReferenceAssets.ps1"),
                "-ManifestPath",
                manifestPath,
                "-SourceRoot",
                sourceRoot,
                "-OutputRoot",
                outputRoot);

            Assert.Contains("AcquisitionState=verified-local-source-owner-review-required", output, StringComparison.Ordinal);
            string reportPath = Path.Combine(outputRoot, "acquisition-report.json");
            using JsonDocument reportDocument = JsonDocument.Parse(File.ReadAllText(reportPath));
            JsonElement report = reportDocument.RootElement;
            Assert.Equal("verified-local-source-owner-review-required", report.GetProperty("acquisitionState").GetString());
            Assert.True(report.GetProperty("allFilesReady").GetBoolean());
            Assert.False(report.GetProperty("allLicensesReady").GetBoolean());
            Assert.False(report.GetProperty("canPromoteRealModelRuntime").GetBoolean());
            Assert.False(report.GetProperty("canRedistributeInRepository").GetBoolean());
            Assert.False(report.GetProperty("performsPublish").GetBoolean());
            Assert.Equal(2, report.GetProperty("assets").GetArrayLength());
            Assert.All(report.GetProperty("assets").EnumerateArray(), static asset =>
            {
                Assert.True(asset.GetProperty("fileReady").GetBoolean());
                Assert.True(asset.GetProperty("copied").GetBoolean());
                Assert.False(asset.GetProperty("licenseReady").GetBoolean());
            });

            Assert.Equal(model, File.ReadAllBytes(Path.Combine(outputRoot, "model.onnx")));
            Assert.Equal(labels, File.ReadAllBytes(Path.Combine(outputRoot, "labels.txt")));

            string script = File.ReadAllText(Path.Combine(
                RepositoryPaths.Root,
                "eng",
                "Acquire-YoloVisionReferenceAssets.ps1"));
            Assert.Contains("AllowDownload", script, StringComparison.Ordinal);
            Assert.Contains("RequireLicenseReady", script, StringComparison.Ordinal);
            Assert.Contains("Invoke-WebRequest", script, StringComparison.Ordinal);
            Assert.Contains("canPromoteRealModelRuntime", script, StringComparison.Ordinal);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    private static object NewAsset(
        string id,
        string role,
        string relativeSourcePath,
        string cacheFileName,
        byte[] contents)
    {
        return new
        {
            id,
            role,
            relativeSourcePath,
            cacheFileName,
            expectedLength = contents.LongLength,
            expectedSha256 = Convert.ToHexString(SHA256.HashData(contents)).ToLowerInvariant(),
            sourceUrl = "test-fixture",
            downloadUrl = "",
            license = new
            {
                name = "test owner review required",
                url = "",
                status = "owner-review-required",
                redistributionApproved = false,
            },
        };
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
    {
        ProcessStartInfo startInfo = new()
        {
            FileName = "pwsh",
            WorkingDirectory = RepositoryPaths.Root,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
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

        using Process process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("Failed to start PowerShell.");
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        Assert.True(process.WaitForExit(180_000), $"PowerShell timed out.{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        Assert.Equal(0, process.ExitCode);
        Assert.True(string.IsNullOrWhiteSpace(stderr), stderr);
        return stdout;
    }
}

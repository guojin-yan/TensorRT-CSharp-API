using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class YoloVisionReferenceAssetAcquisitionTests
{
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

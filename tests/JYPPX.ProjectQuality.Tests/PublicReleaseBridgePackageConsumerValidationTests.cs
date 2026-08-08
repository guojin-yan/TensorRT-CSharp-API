using System.Diagnostics;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class PublicReleaseBridgePackageConsumerValidationTests
{
    private const string RuntimeKey = "win-x64-trt10.11-cuda12.9-cudnn9.22";
    private const string Repository = "guojin-yan/TensorRT-CSharp-API";
    private const string ManagedId = "JYPPX.TensorRT.CSharp.API";
    private const string BridgeId = "JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge";
    private const string Version = "4.0.test";
    private const string ManagedCommit = "1111111111111111111111111111111111111111";
    private const string DifferentBridgeCommit = "2222222222222222222222222222222222222222";

    [Fact]
    public void SourceContractsPinReferencedHashesAndKeepDiagnosticsNonPromotable()
    {
        string executor = ReadEng("Invoke-PublicReleaseBridgePackageConsumer.ps1");
        string validator = ReadEng("Test-PublicReleaseBridgePackageConsumer.ps1");
        string runtime = ReadEng("Test-BridgePackageRuntimeConsumer.ps1");

        Assert.Contains("runtimeReportSha256 = $runtimeProofSha256", executor, StringComparison.Ordinal);
        Assert.Contains("if ($packageSourceCommitAligned -and $AllowCrossCommitPair.IsPresent)", executor, StringComparison.Ordinal);
        Assert.Contains("if (-not $packageSourceCommitAligned -and -not $AllowCrossCommitPair.IsPresent)", executor, StringComparison.Ordinal);
        Assert.Contains("-AllowCrossCommitPair is diagnostic-only", executor, StringComparison.Ordinal);
        Assert.Contains("Test-PublicReleaseBridgePackageConsumer.ps1", executor, StringComparison.Ordinal);
        Assert.Contains("-RequireReferencedFiles", executor, StringComparison.Ordinal);
        Assert.Contains("-FailOnNotEvidence", executor, StringComparison.Ordinal);

        foreach (string marker in new[]
        {
            "inputSha256",
            "runtimeReportSha256Verified",
            "invocationLogHashesVerified",
            "sourceCommitsAligned",
            "verified-cross-commit-diagnostic-only",
            "canPromotePublicReleaseAssetConsumerEvidence",
            "isPackageConsumerRuntimeProof = $false",
            "isPostPublishProof = $false",
            "currentHeadBindingVerified = $false",
            "Cross-commit execution must remain explicitly diagnostic and non-promotable",
        })
        {
            Assert.Contains(marker, validator, StringComparison.Ordinal);
        }

        Assert.Contains("$Result | ConvertTo-Json -Depth 6", runtime, StringComparison.Ordinal);
        Assert.Contains("compatible-host-bridge-package-runtime-diagnostic", runtime, StringComparison.Ordinal);
        Assert.Contains("canPromoteCompatibleHostRuntimeProof = $runtimeSmokePassed -and -not $SkipInstalledVendorAssetHashing.IsPresent", runtime, StringComparison.Ordinal);
    }

    [Fact]
    public void SameCommitFixtureCanPromoteOnlyPublicReleaseAssetEvidence()
    {
        if (!PowerShellCoreIsAvailable())
        {
            return;
        }

        string root = CreateFixture(crossCommit: false);
        try
        {
            ProcessResult result = RunValidator(root, failOnNotEvidence: true);
            Assert.Equal(0, result.ExitCode);

            using JsonDocument validation = ReadValidation(root);
            JsonElement record = validation.RootElement;
            Assert.Equal("verified-public-release-assets-compatible-host-runtime", record.GetProperty("validationState").GetString());
            Assert.True(record.GetProperty("sourceCommitsAligned").GetBoolean());
            Assert.True(record.GetProperty("referencedFilesFullyVerified").GetBoolean());
            Assert.True(record.GetProperty("canPromotePublicReleaseAssetConsumerEvidence").GetBoolean());
            Assert.False(record.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
            Assert.False(record.GetProperty("isPostPublishProof").GetBoolean());
            Assert.False(record.GetProperty("currentHeadBindingVerified").GetBoolean());
            Assert.Equal(0, record.GetProperty("failedBlockerCount").GetInt32());
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void CrossCommitFixtureIsDiagnosticAndCannotPassEvidencePromotion()
    {
        if (!PowerShellCoreIsAvailable())
        {
            return;
        }

        string root = CreateFixture(crossCommit: true);
        try
        {
            ProcessResult strict = RunValidator(root, failOnNotEvidence: false);
            Assert.Equal(0, strict.ExitCode);
            using (JsonDocument validation = ReadValidation(root))
            {
                JsonElement record = validation.RootElement;
                Assert.Equal("verified-cross-commit-diagnostic-only", record.GetProperty("validationState").GetString());
                Assert.False(record.GetProperty("sourceCommitsAligned").GetBoolean());
                Assert.True(record.GetProperty("diagnosticOnly").GetBoolean());
                Assert.True(record.GetProperty("referencedFilesFullyVerified").GetBoolean());
                Assert.False(record.GetProperty("canPromotePublicReleaseAssetConsumerEvidence").GetBoolean());
                Assert.Equal(0, record.GetProperty("failedBlockerCount").GetInt32());
            }

            ProcessResult promotion = RunValidator(root, failOnNotEvidence: true);
            Assert.NotEqual(0, promotion.ExitCode);
            Assert.Contains("not promotable public asset evidence", promotion.CombinedOutput, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void RuntimeReportHashTamperingFailsStrictValidation()
    {
        if (!PowerShellCoreIsAvailable())
        {
            return;
        }

        string root = CreateFixture(crossCommit: false);
        try
        {
            File.AppendAllText(Path.Combine(root, "runtime.json"), Environment.NewLine, Encoding.UTF8);
            ProcessResult result = RunValidator(root, failOnNotEvidence: false);
            Assert.NotEqual(0, result.ExitCode);

            using JsonDocument validation = ReadValidation(root);
            JsonElement hashCheck = validation.RootElement.GetProperty("validationItems")
                .EnumerateArray()
                .Single(static item => item.GetProperty("id").GetString() == "runtime-report-hash-match");
            Assert.False(hashCheck.GetProperty("passed").GetBoolean());
            Assert.True(validation.RootElement.GetProperty("failedBlockerCount").GetInt32() > 0);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static string CreateFixture(bool crossCommit)
    {
        string root = Path.Combine(Path.GetTempPath(), "jyppx-public-release-validation-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);

        string bridgeCommit = crossCommit ? DifferentBridgeCommit : ManagedCommit;
        string managedPath = Path.Combine(root, $"{ManagedId}.{Version}.nupkg");
        string bridgePath = Path.Combine(root, $"{BridgeId}.{Version}.nupkg");
        CreatePackage(managedPath, ManagedId, ManagedCommit, nativeEntry: null);
        CreatePackage(bridgePath, BridgeId, bridgeCommit, "runtimes/win-x64/native/jyppxtrtbridge.dll");
        string managedHash = Sha256(managedPath);
        string bridgeHash = Sha256(bridgePath);

        string stdoutPath = Path.Combine(root, "runtime.stdout.log");
        string stderrPath = Path.Combine(root, "runtime.stderr.log");
        File.WriteAllText(stdoutPath, "EngineSerializedBytes=8652\nEnqueueCompleted=True\n", new UTF8Encoding(false));
        File.WriteAllText(stderrPath, string.Empty, new UTF8Encoding(false));

        object runtimeRecord = new
        {
            sourceRuntimeKey = RuntimeKey,
            proofClassification = crossCommit
                ? "compatible-host-bridge-package-runtime-diagnostic"
                : "compatible-host-bridge-package-runtime",
            smokeStatus = "passed",
            exitCode = 0,
            isRuntimeExecutionProof = true,
            isPackageConsumerRuntimeProof = false,
            canPromoteCompatibleHostRuntimeProof = !crossCommit,
            installedVendorAssetHashingSkipped = crossCommit,
            installedVendorAssetInventorySkipped = crossCommit,
            nativeAssetHashesComplete = !crossCommit,
            packages = new
            {
                managed = new { id = ManagedId, version = Version, sha256 = managedHash },
                bridge = new { id = BridgeId, version = Version, sha256 = bridgeHash },
            },
        };
        string runtimePath = Path.Combine(root, "runtime.json");
        WriteJson(runtimePath, runtimeRecord);

        string releaseTag = "v4.0.test";
        string repositoryUrl = $"https://github.com/{Repository}";
        object outerRecord = new
        {
            schemaVersion = 1,
            recordKind = "public-release-bridge-package-consumer",
            generatedAtUtc = DateTimeOffset.UtcNow.ToString("O"),
            repository = Repository,
            sourceRuntimeKey = RuntimeKey,
            publicationPolicy = "bridge-only",
            channel = "github-release-assets",
            proofClassification = crossCommit
                ? "cross-commit-public-assets-diagnostic-only"
                : "public-release-assets-compatible-host-runtime",
            publicReleaseAssetProvenanceVerified = true,
            remoteDigestVerified = true,
            packageIdentityVerified = true,
            packageSourceCommitAligned = !crossCommit,
            crossCommitDiagnosticOverride = crossCommit,
            externalVendorRuntimePackagePolicyPassed = true,
            restoreUsesDownloadedAssetStaging = true,
            stagingIsLocallyBuiltPackageFeed = false,
            directNupkgReferenceUsed = false,
            packageReferenceOnly = true,
            vendorRuntimeBundled = false,
            systemInstalledVendorDependenciesRequired = true,
            consumerRootOutsideRepository = true,
            runtimeSmokePassed = true,
            runtimeScriptExitCode = 0,
            isRuntimeExecutionProof = true,
            isPublicReleaseAssetConsumerEvidence = !crossCommit,
            isPackageConsumerRuntimeProof = false,
            isPostPublishProof = false,
            currentHeadBindingVerified = false,
            canPromoteCurrentHeadPackageConsumerProof = false,
            performsPublish = false,
            canPublishPublicly = false,
            canCloseReleaseIssue = false,
            releaseAssets = new
            {
                managed = CreateAsset(1, releaseTag, ManagedId, managedPath, managedHash, ManagedCommit, Array.Empty<string>(), repositoryUrl),
                bridge = CreateAsset(2, releaseTag, BridgeId, bridgePath, bridgeHash, bridgeCommit, new[] { "runtimes/win-x64/native/jyppxtrtbridge.dll" }, repositoryUrl),
            },
            consumer = new
            {
                outputRoot = root,
                runtimeReportPath = runtimePath,
                runtimeReportSha256 = Sha256(runtimePath),
                projectPath = Path.Combine(root, "Consumer.csproj"),
                projectSha256 = new string('3', 64),
                smokeExitCode = 0,
                smokeStatus = "passed",
                identityOutputMatch = true,
                enqueueCompleted = true,
                invocationStdoutPath = stdoutPath,
                invocationStdoutSha256 = Sha256(stdoutPath),
                invocationStderrPath = stderrPath,
                invocationStderrSha256 = Sha256(stderrPath),
            },
            boundary = crossCommit
                ? "This cross-commit record is diagnostic-only and cannot become post-publish proof."
                : "This same-commit evidence does not prove current HEAD publication or post-publish release closure.",
        };
        WriteJson(Path.Combine(root, "input.json"), outerRecord);
        return root;
    }

    private static object CreateAsset(
        long assetId,
        string releaseTag,
        string packageId,
        string path,
        string sha256,
        string commit,
        string[] nativeEntries,
        string repositoryUrl)
    {
        string assetName = $"{packageId}.{Version}.nupkg";
        return new
        {
            releaseTag,
            releaseUrl = $"{repositoryUrl}/releases/tag/{releaseTag}",
            assetId,
            assetName,
            assetUrl = $"{repositoryUrl}/releases/download/{releaseTag}/{assetName}",
            githubDigest = "sha256:" + sha256,
            downloadedPath = path,
            downloadedSha256 = sha256,
            downloadTransport = "public-https",
            lengthBytes = new FileInfo(path).Length,
            packageId,
            packageVersion = Version,
            repositoryUrl,
            repositoryCommit = commit,
            nativeEntries,
        };
    }

    private static void CreatePackage(string path, string packageId, string commit, string? nativeEntry)
    {
        using ZipArchive archive = ZipFile.Open(path, ZipArchiveMode.Create);
        ZipArchiveEntry nuspec = archive.CreateEntry(packageId + ".nuspec");
        using (StreamWriter writer = new(nuspec.Open(), new UTF8Encoding(false)))
        {
            writer.Write($"""
                <?xml version="1.0" encoding="utf-8"?>
                <package>
                  <metadata>
                    <id>{packageId}</id>
                    <version>{Version}</version>
                    <authors>JYPPX</authors>
                    <description>Validation fixture</description>
                    <repository type="git" url="https://github.com/{Repository}" commit="{commit}" />
                  </metadata>
                </package>
                """);
        }

        if (nativeEntry is not null)
        {
            ZipArchiveEntry native = archive.CreateEntry(nativeEntry);
            using Stream stream = native.Open();
            stream.Write(new byte[] { 0x4A, 0x59, 0x50, 0x50, 0x58 });
        }
    }

    private static ProcessResult RunValidator(string root, bool failOnNotEvidence)
    {
        using Process process = new();
        process.StartInfo.FileName = PowerShellHost.ResolveExecutable();
        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(Path.Combine(RepositoryPaths.Root, "eng", "Test-PublicReleaseBridgePackageConsumer.ps1"));
        process.StartInfo.ArgumentList.Add("-InputPath");
        process.StartInfo.ArgumentList.Add(Path.Combine(root, "input.json"));
        process.StartInfo.ArgumentList.Add("-OutputDirectory");
        process.StartInfo.ArgumentList.Add(root);
        process.StartInfo.ArgumentList.Add("-ExpectedSourceRuntimeKey");
        process.StartInfo.ArgumentList.Add(RuntimeKey);
        process.StartInfo.ArgumentList.Add("-RepositoryRoot");
        process.StartInfo.ArgumentList.Add(RepositoryPaths.Root);
        process.StartInfo.ArgumentList.Add("-RequireReferencedFiles");
        process.StartInfo.ArgumentList.Add("-Strict");
        if (failOnNotEvidence)
        {
            process.StartInfo.ArgumentList.Add("-FailOnNotEvidence");
        }

        process.StartInfo.WorkingDirectory = RepositoryPaths.Root;
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.RedirectStandardError = true;
        process.StartInfo.UseShellExecute = false;
        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();
        return new ProcessResult(process.ExitCode, stdout + Environment.NewLine + stderr);
    }

    private static bool PowerShellCoreIsAvailable()
    {
        try
        {
            using Process process = Process.Start(new ProcessStartInfo
            {
                FileName = PowerShellHost.ResolveExecutable(),
                Arguments = "-NoProfile -Command \"exit 0\"",
                CreateNoWindow = true,
                UseShellExecute = false,
            })!;
            process.WaitForExit();
            return process.ExitCode == 0;
        }
        catch
        {
            return false;
        }
    }

    private static JsonDocument ReadValidation(string root) =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(root, "public-release-bridge-package-consumer-validation.json")));

    private static string ReadEng(string fileName) =>
        File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", fileName));

    private static string Sha256(string path) =>
        Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();

    private static void WriteJson(string path, object value) =>
        File.WriteAllText(path, JsonSerializer.Serialize(value, new JsonSerializerOptions { WriteIndented = true }), new UTF8Encoding(false));

    private sealed record ProcessResult(int ExitCode, string CombinedOutput);
}

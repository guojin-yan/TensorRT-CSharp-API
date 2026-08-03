using System.Text.Json;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Xml.Linq;
using YoloVisionSample;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class YoloVisionLocalPackageConsumerTests
{
    [Fact]
    public void YoloVisionPackageExposesReusablePointerFreeCommand()
    {
        string project = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "YoloVision.csproj"));
        string program = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "Program.cs"));

        Assert.Contains("<IsPackable>true</IsPackable>", project, StringComparison.Ordinal);
        Assert.Contains("<PackageId>JYPPX.TensorRT.CSharp.API.YoloVision</PackageId>", project, StringComparison.Ordinal);
        Assert.Contains("<GenerateDocumentationFile>true</GenerateDocumentationFile>", project, StringComparison.Ordinal);
        Assert.Contains("JYPPX.TensorRtSharp.csproj", project, StringComparison.Ordinal);
        Assert.DoesNotContain("JYPPX.CudaSharp.csproj", project, StringComparison.Ordinal);
        Assert.Contains("public static class YoloVisionCommand", program, StringComparison.Ordinal);
        Assert.Contains("public static int Run(string[] args)", program, StringComparison.Ordinal);
        Assert.Contains("return YoloVisionCommand.Run(args);", program, StringComparison.Ordinal);
        foreach (string forbidden in new[] { "IntPtr", "nint", "UIntPtr", "SafeHandle" })
        {
            Assert.DoesNotContain(forbidden, program, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ConsumerTemplateUsesOnlyThreePackageReferences()
    {
        string templateRoot = Path.Combine(RepositoryPaths.Root, "samples", "YoloVision.PackageConsumer");
        string project = File.ReadAllText(Path.Combine(templateRoot, "YoloVision.PackageConsumer.csproj.template"));
        string program = File.ReadAllText(Path.Combine(templateRoot, "Program.cs"));
        string readme = File.ReadAllText(Path.Combine(templateRoot, "README.md"));

        Assert.Equal(3, project.Split("<PackageReference ", StringSplitOptions.None).Length - 1);
        Assert.DoesNotContain("ProjectReference", project, StringComparison.Ordinal);
        Assert.Contains("JYPPX.TensorRT.CSharp.API\"", project, StringComparison.Ordinal);
        Assert.Contains("JYPPX.TensorRT.CSharp.API.YoloVision", project, StringComparison.Ordinal);
        Assert.Contains("__BRIDGE_PACKAGE_ID__", project, StringComparison.Ordinal);
        Assert.DoesNotContain("trt10.11.cuda12.9", project, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("ProjectReference=False", program, StringComparison.Ordinal);
        Assert.Contains("BridgeTensorRt=", program, StringComparison.Ordinal);
        Assert.Contains("TensorRtEnvironmentProbe.GetCurrent()", program, StringComparison.Ordinal);
        Assert.Contains("YoloVisionCommand.Run(args)", program, StringComparison.Ordinal);
        Assert.Contains("TRT8, TRT10, or TRT11", readme, StringComparison.Ordinal);
        Assert.Contains("local-package-consumer-runtime", readme, StringComparison.Ordinal);
        Assert.Contains("not proof", readme, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ConsumerScriptKeepsWorkspaceAndRestoreCacheOffCDriveAndFailsClosed()
    {
        string script = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionLocalPackageConsumer.ps1"));

        Assert.Contains("[string]$RuntimePackageKey", script, StringComparison.Ordinal);
        Assert.Contains("[ValidateSet(\"8\", \"10\", \"11\")][string]$TensorRtLine", script, StringComparison.Ordinal);
        Assert.Contains("split-runtime-packages.manifest.json", script, StringComparison.Ordinal);
        Assert.Contains("sourceRuntimeKey", script, StringComparison.Ordinal);
        Assert.Contains("__BRIDGE_PACKAGE_ID__", script, StringComparison.Ordinal);
        Assert.Contains("Bridge TensorRT build version", script, StringComparison.Ordinal);
        Assert.Contains("bridgeBuildTensorRtLineMatches = $true", script, StringComparison.Ordinal);
        Assert.Contains("consumer-workspaces\\yolovision-yolox-local-package-trt$TensorRtLine", script, StringComparison.Ordinal);
        Assert.Contains("Assert-NonCDrivePath", script, StringComparison.Ordinal);
        Assert.Contains("<clear />", script, StringComparison.Ordinal);
        Assert.Contains("--packages", script, StringComparison.Ordinal);
        Assert.Contains("projectReferenceCount = 0", script, StringComparison.Ordinal);
        Assert.Contains("restoredProjectLibraryCount", script, StringComparison.Ordinal);
        Assert.Contains("isolated-local-feeds", script, StringComparison.Ordinal);
        Assert.Contains("one-selected-nupkg-per-feed", script, StringComparison.Ordinal);
        Assert.Contains("restoredPackageHashesMatchSelected", script, StringComparison.Ordinal);
        Assert.Contains("Restored package '$($package.Id)' SHA256 does not match", script, StringComparison.Ordinal);
        Assert.Contains("YoloVision Passed=True", script, StringComparison.Ordinal);
        Assert.Contains("passed-local-package-consumer-runtime", script, StringComparison.Ordinal);
        Assert.Contains("isPackageConsumerRuntimeProof = $false", script, StringComparison.Ordinal);
        Assert.Contains("packagesDownloadedFromPublicFeed = $false", script, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly = $false", script, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", script, StringComparison.Ordinal);
        Assert.DoesNotContain("api.nuget.org", script, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("dotnet nuget push", script, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void SegmentationConsumerUsesOnlyLocalPackagesAndRequiresIndependentMaskEvidence()
    {
        string runner = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionLocalPackageConsumer.ps1"));
        string entrypoint = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionSegmentationLocalPackageConsumer.ps1"));
        string exporter = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionSegmentationLocalPackageConsumerEvidence.ps1"));
        string mutation = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "New-YoloVisionReferenceMutation.py"));
        string independent = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Invoke-YoloVisionSegmentationReference.py"));

        Assert.Contains("yolov8-segmentation", entrypoint, StringComparison.Ordinal);
        Assert.Contains("Test-YoloVisionLocalPackageConsumer.ps1", entrypoint, StringComparison.Ordinal);
        foreach (string term in new[]
        {
            "--output-role-map", "output0:det,output1:mask-prototypes", "--mask-coefficient-count", "--mask-spatial-transform",
            "--reference-outputs", "--reference-abs-tolerance", "--reference-rel-tolerance",
            "segmentation-mask-artifacts.manifest.json", "Invoke-YoloVisionSegmentationReference.py",
            "local-package-consumer-runtime", "New-YoloVisionReferenceMutation.py",
            "Controlled raw-reference mutation", "Thresholded mask SHA256 does not match the manifest.",
            "EnvironmentVariablesToRemove", "JYPPX_NATIVE_BRIDGE_PATH", "directAssemblyReferenceCount = 0",
            "vendorRuntimePackageEntryCount", "bridgeNativePackageEntryCount", "Remove-DirectoryTree"
        })
        {
            Assert.Contains(term, runner, StringComparison.Ordinal);
        }

        Assert.DoesNotContain(".ArgumentList", runner, StringComparison.Ordinal);
        Assert.Contains("controlled-single-value-mutation", mutation, StringComparison.Ordinal);
        Assert.Contains("expectedRuntimeOutcome", mutation, StringComparison.Ordinal);
        Assert.Contains("local-package-consumer-runtime", independent, StringComparison.Ordinal);
        Assert.Contains("PublicPackageProof=False", exporter, StringComparison.Ordinal);
        Assert.Contains("OwnerReleaseAcceptance=False", exporter, StringComparison.Ordinal);
        foreach (string script in new[] { runner, entrypoint, exporter })
        {
            Assert.DoesNotContain("api.nuget.org", script, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("dotnet nuget push", script, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("gh release upload", script, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void SegmentationLocalPackageConsumerEvidenceClosesRawMaskAndNegativeChecksWithoutPromotion()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "yolovision-yolov8n-seg-local-package-consumer-runtime-evidence.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(evidencePath));
        JsonElement root = document.RootElement;

        Assert.Equal("yolovision-local-package-consumer-runtime-evidence", root.GetProperty("recordKind").GetString());
        Assert.Equal("local-package-consumer-runtime", root.GetProperty("proofClassification").GetString());
        Assert.Matches("^[0-9a-f]{64}$", root.GetProperty("fullLocalReportSha256").GetString()!);

        JsonElement consumer = root.GetProperty("packageConsumer");
        Assert.Equal("local-file-feed-only", consumer.GetProperty("packageSourceKind").GetString());
        Assert.Equal(0, consumer.GetProperty("remotePackageSourceCount").GetInt32());
        Assert.Equal(3, consumer.GetProperty("packageCount").GetInt32());
        Assert.Equal(0, consumer.GetProperty("projectReferenceCount").GetInt32());
        Assert.Equal(0, consumer.GetProperty("directAssemblyReferenceCount").GetInt32());
        Assert.Equal(0, consumer.GetProperty("restoredProjectLibraryCount").GetInt32());
        Assert.Equal("one-selected-nupkg-per-feed", consumer.GetProperty("packageSourceIsolation").GetString());
        Assert.True(consumer.GetProperty("restoredPackageHashesMatchSelected").GetBoolean());
        Assert.False(consumer.GetProperty("nativeBridgePathEnvironmentVariableSet").GetBoolean());
        Assert.True(consumer.GetProperty("tensorRtCudaAndCudnnAreExternalDependencies").GetBoolean());
        Assert.Equal(0, consumer.GetProperty("vendorRuntimePackageEntryCount").GetInt32());
        Assert.Equal(1, consumer.GetProperty("bridgeNativePackageEntryCount").GetInt32());
        Assert.Equal("E:", consumer.GetProperty("workspaceDrive").GetString());
        Assert.True(consumer.GetProperty("workspaceRemovedAfterValidation").GetBoolean());
        JsonElement[] packages = consumer.GetProperty("packages").EnumerateArray().ToArray();
        Assert.Equal(new[] { "managed-api", "yolovision", "bridge-only" }, packages.Select(static package => package.GetProperty("role").GetString()).ToArray());
        Assert.All(packages, static package =>
        {
            Assert.True(package.GetProperty("length").GetInt64() > 0);
            Assert.Matches("^[0-9a-f]{64}$", package.GetProperty("sha256").GetString()!);
        });
        Assert.EndsWith(".Bridge", packages[2].GetProperty("id").GetString(), StringComparison.Ordinal);

        JsonElement raw = root.GetProperty("rawTensorReferenceValidation");
        Assert.Equal(2, raw.GetProperty("tensorCount").GetInt32());
        Assert.Equal(1_793_600, raw.GetProperty("comparedElementCount").GetInt64());
        Assert.Equal(0, raw.GetProperty("mismatchCount").GetInt64());
        Assert.True(raw.GetProperty("passed").GetBoolean());
        Assert.Equal(new[] { "output0", "output1" }, raw.GetProperty("tensors").EnumerateArray().Select(static tensor => tensor.GetProperty("tensorName").GetString()).ToArray());

        JsonElement masks = root.GetProperty("maskArtifacts");
        Assert.Equal(4, masks.GetProperty("predictionCount").GetInt32());
        Assert.Equal(4, masks.GetProperty("sourceThresholdedMaskCount").GetInt32());
        Assert.True(masks.GetProperty("spatialTransformApplied").GetBoolean());
        JsonElement independentComparison = root.GetProperty("independentPostprocessValidation");
        Assert.Equal("local-package-consumer-runtime", independentComparison.GetProperty("evidenceClassification").GetString());
        Assert.True(independentComparison.GetProperty("passed").GetBoolean());
        Assert.All(independentComparison.GetProperty("comparisons").EnumerateArray(), static comparison =>
        {
            Assert.True(comparison.GetProperty("passed").GetBoolean());
            Assert.True(comparison.GetProperty("boxIoU").GetDouble() >= 0.995);
            Assert.True(comparison.GetProperty("maskIoU").GetDouble() >= 0.99);
        });

        JsonElement rawNegative = root.GetProperty("controlledNegativeValidation");
        Assert.Equal(1, rawNegative.GetProperty("exitCode").GetInt32());
        Assert.Equal(1, rawNegative.GetProperty("mismatchCount").GetInt64());
        Assert.Equal(0, rawNegative.GetProperty("firstMismatchIndex").GetInt64());
        Assert.True(rawNegative.GetProperty("failClosed").GetBoolean());
        JsonElement maskNegative = root.GetProperty("controlledArtifactIntegrityValidation");
        Assert.Equal(1, maskNegative.GetProperty("exitCode").GetInt32());
        Assert.True(maskNegative.GetProperty("failClosed").GetBoolean());

        JsonElement boundary = root.GetProperty("proofBoundary");
        Assert.True(boundary.GetProperty("localPackageConsumerRuntimeEvidence").GetBoolean());
        foreach (string name in new[]
        {
            "sourceTreeRuntimeProof", "publicPackageProof", "packagesDownloadedFromPublicFeed", "postPublishProof",
            "publicRedistributionOwnerApproval", "ownerReleaseAcceptance", "releaseProof", "performsPublish", "uploadsAssets"
        })
        {
            Assert.False(boundary.GetProperty(name).GetBoolean());
        }
    }

    [Fact]
    public void SemanticConsumerUsesOnlyLocalPackagesAndRequiresFullResolutionClassIndexEvidence()
    {
        string runner = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionLocalPackageConsumer.ps1"));
        string entrypoint = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionSemanticLocalPackageConsumer.ps1"));
        string validator = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionSemanticMapArtifact.ps1"));
        string exporter = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionSemanticLocalPackageConsumerEvidence.ps1"));

        Assert.Contains("torchvision-lraspp-semantic", entrypoint, StringComparison.Ordinal);
        Assert.Contains("Test-YoloVisionLocalPackageConsumer.ps1", entrypoint, StringComparison.Ordinal);
        foreach (string term in new[]
        {
            "--semantic-artifact-output-directory", "--mean", "0.485,0.456,0.406", "--std", "0.229,0.224,0.225",
            "--reference-outputs", "semantic:$ReferenceOutput0Path", "--reference-abs-tolerance", "0.0001",
            "semantic-map-artifacts.manifest.json", "semantic-class-index-onnxruntime.i32.bin",
            "Test-YoloVisionSemanticMapArtifact.ps1", "controlled-semantic-artifact-tamper",
            "artifact-sha256", "JYPPX_NATIVE_BRIDGE_PATH", "vendorRuntimePackageEntryCount",
            "assetsRemainOnEDrive = $true", "uploadsAssets = $false"
        })
        {
            Assert.Contains(term, runner, StringComparison.Ordinal);
        }

        foreach (string term in new[]
        {
            "yolovision-semantic-map-artifacts.v1", "int32-little-endian", "row-major-hw",
            "reference-class-index-sha256", "class-histogram", "isPackageConsumerProof = $false",
            "performsPublish = $false", "uploadsAssets = $false"
        })
        {
            Assert.Contains(term, validator, StringComparison.Ordinal);
        }

        Assert.Contains("RawValues=", exporter, StringComparison.Ordinal);
        Assert.Contains("Pixels=", exporter, StringComparison.Ordinal);
        Assert.Contains("PublicPackageProof=False", exporter, StringComparison.Ordinal);
        Assert.Contains("OwnerReleaseAcceptance=False", exporter, StringComparison.Ordinal);
        foreach (string script in new[] { runner, entrypoint, validator, exporter })
        {
            Assert.DoesNotContain("api.nuget.org", script, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("dotnet nuget push", script, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("gh release upload", script, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void SemanticLocalPackageConsumerEvidenceClosesRawClassIndexAndNegativeChecksWithoutPromotion()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "yolovision-lraspp-semantic-local-package-consumer-runtime-evidence.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(evidencePath));
        JsonElement root = document.RootElement;

        Assert.Equal("yolovision-local-package-consumer-runtime-evidence", root.GetProperty("recordKind").GetString());
        Assert.Equal("local-package-consumer-runtime", root.GetProperty("proofClassification").GetString());
        Assert.Equal("torchvision-lraspp-mobilenet-v3-large", root.GetProperty("family").GetString());
        Assert.Equal("sem", root.GetProperty("task").GetString());
        Assert.Matches("^\\d{4}-\\d{2}-\\d{2}T", root.GetProperty("generatedAtUtc").GetString()!);
        Assert.Matches("^[0-9a-f]{64}$", root.GetProperty("fullLocalReportSha256").GetString()!);

        JsonElement consumer = root.GetProperty("packageConsumer");
        Assert.Equal("local-file-feed-only", consumer.GetProperty("packageSourceKind").GetString());
        Assert.Equal(0, consumer.GetProperty("remotePackageSourceCount").GetInt32());
        Assert.Equal(3, consumer.GetProperty("packageCount").GetInt32());
        Assert.Equal(0, consumer.GetProperty("projectReferenceCount").GetInt32());
        Assert.Equal(0, consumer.GetProperty("directAssemblyReferenceCount").GetInt32());
        Assert.Equal(0, consumer.GetProperty("restoredProjectLibraryCount").GetInt32());
        Assert.Equal("one-selected-nupkg-per-feed", consumer.GetProperty("packageSourceIsolation").GetString());
        Assert.True(consumer.GetProperty("restoredPackageHashesMatchSelected").GetBoolean());
        Assert.False(consumer.GetProperty("nativeBridgePathEnvironmentVariableSet").GetBoolean());
        Assert.True(consumer.GetProperty("tensorRtCudaAndCudnnAreExternalDependencies").GetBoolean());
        Assert.Equal(0, consumer.GetProperty("vendorRuntimePackageEntryCount").GetInt32());
        Assert.Equal(1, consumer.GetProperty("bridgeNativePackageEntryCount").GetInt32());
        Assert.Equal("E:", consumer.GetProperty("workspaceDrive").GetString());
        Assert.True(consumer.GetProperty("workspaceRemovedAfterValidation").GetBoolean());
        Assert.Equal(new[] { "managed-api", "yolovision", "bridge-only" },
            consumer.GetProperty("packages").EnumerateArray().Select(static package => package.GetProperty("role").GetString()).ToArray());

        JsonElement assets = root.GetProperty("assets");
        Assert.True(assets.GetProperty("onnxStoredUnderWorkspaceModelsDirectory").GetBoolean());
        Assert.True(assets.GetProperty("heavyAssetsRemainOutsideGit").GetBoolean());
        Assert.False(assets.GetProperty("publicRedistributionOwnerApproval").GetBoolean());
        Assert.Equal("3cb94e561bdefe606ed7d1a2c4d0296409bec066f3a39a9fe9dabd72b23728f8", assets.GetProperty("onnxSha256").GetString());

        JsonElement raw = root.GetProperty("rawTensorReferenceValidation");
        Assert.Equal(1, raw.GetProperty("tensorCount").GetInt32());
        Assert.Equal(2_150_400, raw.GetProperty("comparedElementCount").GetInt64());
        Assert.Equal(0, raw.GetProperty("mismatchCount").GetInt64());
        Assert.True(raw.GetProperty("passed").GetBoolean());

        JsonElement classIndex = root.GetProperty("classIndexArtifactValidation");
        Assert.Equal(102_400, classIndex.GetProperty("pixelCount").GetInt32());
        Assert.Equal(0, classIndex.GetProperty("mismatchCount").GetInt32());
        Assert.Equal(-1, classIndex.GetProperty("firstMismatchIndex").GetInt32());
        Assert.Equal("fdd15b95222eadf137fc6880e56990aa507ee7d2429f471deaae9eab31268414", classIndex.GetProperty("classIndexSha256").GetString());
        Assert.True(classIndex.GetProperty("classIndexMatches").GetBoolean());
        Assert.True(classIndex.GetProperty("histogramMatches").GetBoolean());
        JsonElement[] nonEmptyRows = classIndex.GetProperty("histogram").EnumerateArray()
            .Where(static row => row.GetProperty("pixelCount").GetInt64() > 0)
            .ToArray();
        Assert.Equal(new[] { 0, 12 }, nonEmptyRows.Select(static row => row.GetProperty("classId").GetInt32()).ToArray());
        Assert.Equal(new long[] { 65_193, 37_207 }, nonEmptyRows.Select(static row => row.GetProperty("pixelCount").GetInt64()).ToArray());

        JsonElement rawNegative = root.GetProperty("controlledNegativeValidation");
        Assert.Equal(1, rawNegative.GetProperty("exitCode").GetInt32());
        Assert.Equal(1, rawNegative.GetProperty("mismatchCount").GetInt64());
        Assert.Equal(0, rawNegative.GetProperty("firstMismatchIndex").GetInt64());
        Assert.True(rawNegative.GetProperty("failClosed").GetBoolean());
        JsonElement artifactNegative = root.GetProperty("controlledArtifactIntegrityValidation");
        Assert.Equal(1, artifactNegative.GetProperty("exitCode").GetInt32());
        Assert.Contains("artifact-sha256", artifactNegative.GetProperty("findings").EnumerateArray().Select(static item => item.GetString()));
        Assert.True(artifactNegative.GetProperty("failClosed").GetBoolean());

        JsonElement boundary = root.GetProperty("proofBoundary");
        Assert.True(boundary.GetProperty("localPackageConsumerRuntimeEvidence").GetBoolean());
        foreach (string name in new[]
        {
            "sourceTreeRuntimeProof", "publicPackageProof", "packagesDownloadedFromPublicFeed", "postPublishProof",
            "publicRedistributionOwnerApproval", "ownerReleaseAcceptance", "releaseProof", "performsPublish", "uploadsAssets"
        })
        {
            Assert.False(boundary.GetProperty(name).GetBoolean());
        }
    }

    [Fact]
    public void ClassificationConsumerUsesOnlySelectedLocalPackagesAndRequiresFullProbabilityEvidence()
    {
        string runner = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionLocalPackageConsumer.ps1"));
        string entrypoint = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionClassificationLocalPackageConsumer.ps1"));
        string exporter = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionClassificationLocalPackageConsumerEvidence.ps1"));

        Assert.Contains("yolov8-classification", entrypoint, StringComparison.Ordinal);
        Assert.Contains("Test-YoloVisionLocalPackageConsumer.ps1", entrypoint, StringComparison.Ordinal);
        foreach (string term in new[]
        {
            "--classification-output", "--classification-score-mode", "probabilities",
            "--no-nms", "--nms-mode", "none", "output0:$ReferenceOutput0Path",
            "--reference-abs-tolerance", "0.001", "--reference-rel-tolerance",
            "input-ultralytics-1x3x224x224.fp32.bin", "minibus,police_van,trolleybus,golfcart,jinrikisha",
            "controlled-reference-negative", "one-selected-nupkg-per-feed", "restoredPackageHashesMatchSelected",
            "vendorRuntimePackageEntryCount", "assetsRemainOnEDrive = $true", "uploadsAssets = $false"
        })
        {
            Assert.Contains(term, runner, StringComparison.Ordinal);
        }

        Assert.Contains("RawValues=", exporter, StringComparison.Ordinal);
        Assert.Contains("Top5=", exporter, StringComparison.Ordinal);
        Assert.Contains("PublicPackageProof=False", exporter, StringComparison.Ordinal);
        Assert.Contains("OwnerReleaseAcceptance=False", exporter, StringComparison.Ordinal);
        foreach (string script in new[] { runner, entrypoint, exporter })
        {
            Assert.DoesNotContain("api.nuget.org", script, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("dotnet nuget push", script, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("gh release upload", script, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void ClassificationLocalPackageConsumerEvidenceClosesRawTop5AndNegativeChecksWithoutPromotion()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "yolovision-yolov8n-cls-local-package-consumer-runtime-evidence.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(evidencePath));
        JsonElement root = document.RootElement;

        Assert.Equal("yolovision-local-package-consumer-runtime-evidence", root.GetProperty("recordKind").GetString());
        Assert.Equal("local-package-consumer-runtime", root.GetProperty("proofClassification").GetString());
        Assert.Equal("yolov8", root.GetProperty("family").GetString());
        Assert.Equal("cls", root.GetProperty("task").GetString());
        Assert.Matches("^\\d{4}-\\d{2}-\\d{2}T", root.GetProperty("generatedAtUtc").GetString()!);
        Assert.Matches("^[0-9a-f]{64}$", root.GetProperty("fullLocalReportSha256").GetString()!);

        JsonElement consumer = root.GetProperty("packageConsumer");
        Assert.Equal("local-file-feed-only", consumer.GetProperty("packageSourceKind").GetString());
        Assert.Equal("one-selected-nupkg-per-feed", consumer.GetProperty("packageSourceIsolation").GetString());
        Assert.Equal(0, consumer.GetProperty("remotePackageSourceCount").GetInt32());
        Assert.Equal(3, consumer.GetProperty("packageCount").GetInt32());
        Assert.Equal(0, consumer.GetProperty("projectReferenceCount").GetInt32());
        Assert.Equal(0, consumer.GetProperty("directAssemblyReferenceCount").GetInt32());
        Assert.Equal(0, consumer.GetProperty("restoredProjectLibraryCount").GetInt32());
        Assert.True(consumer.GetProperty("restoredPackageHashesMatchSelected").GetBoolean());
        Assert.False(consumer.GetProperty("nativeBridgePathEnvironmentVariableSet").GetBoolean());
        Assert.True(consumer.GetProperty("tensorRtCudaAndCudnnAreExternalDependencies").GetBoolean());
        Assert.Equal(0, consumer.GetProperty("vendorRuntimePackageEntryCount").GetInt32());
        Assert.Equal(1, consumer.GetProperty("bridgeNativePackageEntryCount").GetInt32());
        Assert.Equal("E:", consumer.GetProperty("workspaceDrive").GetString());
        Assert.True(consumer.GetProperty("workspaceRemovedAfterValidation").GetBoolean());
        Assert.Equal(new[] { "managed-api", "yolovision", "bridge-only" },
            consumer.GetProperty("packages").EnumerateArray().Select(static package => package.GetProperty("role").GetString()).ToArray());

        JsonElement assets = root.GetProperty("assets");
        Assert.True(assets.GetProperty("onnxStoredUnderWorkspaceModelsDirectory").GetBoolean());
        Assert.True(assets.GetProperty("heavyAssetsRemainOutsideGit").GetBoolean());
        Assert.False(assets.GetProperty("publicRedistributionOwnerApproval").GetBoolean());
        Assert.Equal("630c022a99885d59f633ab5a614738f8a49be7f361e340fd3ff89b8c19b0768f", assets.GetProperty("onnxSha256").GetString());
        Assert.Equal("05e47521b07652eee70942902ab5bf070edc9da7067466a5433c7cdd29fb1a62", assets.GetProperty("authoritativeInputTensorSha256").GetString());

        JsonElement raw = root.GetProperty("rawTensorReferenceValidation");
        Assert.Equal(1, raw.GetProperty("tensorCount").GetInt32());
        Assert.Equal(1_000, raw.GetProperty("comparedElementCount").GetInt64());
        Assert.Equal(0, raw.GetProperty("mismatchCount").GetInt64());
        Assert.True(raw.GetProperty("passed").GetBoolean());

        JsonElement top5 = root.GetProperty("top5Validation");
        Assert.Equal(5, top5.GetProperty("predictionCount").GetInt32());
        Assert.True(top5.GetProperty("sameIndicesAndOrderAsIndependentReference").GetBoolean());
        Assert.True(top5.GetProperty("passed").GetBoolean());
        Assert.Equal(new[] { "minibus", "police_van", "trolleybus", "golfcart", "jinrikisha" },
            top5.GetProperty("predictions").EnumerateArray().Select(static item => item.GetProperty("className").GetString()).ToArray());

        JsonElement negative = root.GetProperty("controlledNegativeValidation");
        Assert.Equal(1, negative.GetProperty("exitCode").GetInt32());
        Assert.Equal(1, negative.GetProperty("mismatchCount").GetInt64());
        Assert.Equal(0, negative.GetProperty("firstMismatchIndex").GetInt64());
        Assert.True(negative.GetProperty("failClosed").GetBoolean());

        JsonElement boundary = root.GetProperty("proofBoundary");
        Assert.True(boundary.GetProperty("localPackageConsumerRuntimeEvidence").GetBoolean());
        foreach (string name in new[]
        {
            "sourceTreeRuntimeProof", "publicPackageProof", "packagesDownloadedFromPublicFeed", "postPublishProof",
            "publicRedistributionOwnerApproval", "ownerReleaseAcceptance", "releaseProof", "performsPublish", "uploadsAssets"
        })
        {
            Assert.False(boundary.GetProperty(name).GetBoolean());
        }
    }

    [Fact]
    public void PoseConsumerUsesOnlySelectedLocalPackagesAndRequiresIndependentKeypointEvidence()
    {
        string runner = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionLocalPackageConsumer.ps1"));
        string entrypoint = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionPoseLocalPackageConsumer.ps1"));
        string exporter = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionPoseLocalPackageConsumerEvidence.ps1"));

        Assert.Contains("yolov8-pose", entrypoint, StringComparison.Ordinal);
        Assert.Contains("Test-YoloVisionLocalPackageConsumer.ps1", entrypoint, StringComparison.Ordinal);
        foreach (string term in new[]
        {
            "--task", "pose", "--keypoint-count", "17", "--keypoint-stride", "3",
            "--aux-channel-start", "5", "--aux-layout", "channels-first",
            "output0:$ReferenceOutput0Path", "--reference-abs-tolerance", "1.25",
            "--reference-rel-tolerance", "0.05", "Invoke-YoloVisionPoseReference.py",
            "470400", "four person poses with 17 keypoints each", "controlled-reference-negative",
            "one-selected-nupkg-per-feed", "restoredPackageHashesMatchSelected",
            "vendorRuntimePackageEntryCount", "assetsRemainOnEDrive = $true", "uploadsAssets = $false"
        })
        {
            Assert.Contains(term, runner, StringComparison.Ordinal);
        }

        Assert.Contains("RawValues=", exporter, StringComparison.Ordinal);
        Assert.Contains("Poses=", exporter, StringComparison.Ordinal);
        Assert.Contains("PublicPackageProof=False", exporter, StringComparison.Ordinal);
        Assert.Contains("OwnerReleaseAcceptance=False", exporter, StringComparison.Ordinal);
        foreach (string script in new[] { runner, entrypoint, exporter })
        {
            Assert.DoesNotContain("api.nuget.org", script, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("dotnet nuget push", script, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("gh release upload", script, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void PoseLocalPackageConsumerEvidenceClosesRawKeypointAndNegativeChecksWithoutPromotion()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "yolovision-yolov8n-pose-local-package-consumer-runtime-evidence.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(evidencePath));
        JsonElement root = document.RootElement;

        Assert.Equal("yolovision-local-package-consumer-runtime-evidence", root.GetProperty("recordKind").GetString());
        Assert.Equal("local-package-consumer-runtime", root.GetProperty("proofClassification").GetString());
        Assert.Equal("yolov8", root.GetProperty("family").GetString());
        Assert.Equal("pose", root.GetProperty("task").GetString());
        Assert.Matches("^\\d{4}-\\d{2}-\\d{2}T", root.GetProperty("generatedAtUtc").GetString()!);
        Assert.Matches("^[0-9a-f]{64}$", root.GetProperty("fullLocalReportSha256").GetString()!);

        JsonElement consumer = root.GetProperty("packageConsumer");
        Assert.Equal("local-file-feed-only", consumer.GetProperty("packageSourceKind").GetString());
        Assert.Equal("one-selected-nupkg-per-feed", consumer.GetProperty("packageSourceIsolation").GetString());
        Assert.Equal(0, consumer.GetProperty("remotePackageSourceCount").GetInt32());
        Assert.Equal(3, consumer.GetProperty("packageCount").GetInt32());
        Assert.Equal(0, consumer.GetProperty("projectReferenceCount").GetInt32());
        Assert.Equal(0, consumer.GetProperty("directAssemblyReferenceCount").GetInt32());
        Assert.Equal(0, consumer.GetProperty("restoredProjectLibraryCount").GetInt32());
        Assert.True(consumer.GetProperty("restoredPackageHashesMatchSelected").GetBoolean());
        Assert.False(consumer.GetProperty("nativeBridgePathEnvironmentVariableSet").GetBoolean());
        Assert.True(consumer.GetProperty("tensorRtCudaAndCudnnAreExternalDependencies").GetBoolean());
        Assert.Equal(0, consumer.GetProperty("vendorRuntimePackageEntryCount").GetInt32());
        Assert.Equal(1, consumer.GetProperty("bridgeNativePackageEntryCount").GetInt32());
        Assert.Equal("E:", consumer.GetProperty("workspaceDrive").GetString());
        Assert.True(consumer.GetProperty("workspaceRemovedAfterValidation").GetBoolean());

        JsonElement assets = root.GetProperty("assets");
        Assert.True(assets.GetProperty("onnxStoredUnderWorkspaceModelsDirectory").GetBoolean());
        Assert.True(assets.GetProperty("heavyAssetsRemainOutsideGit").GetBoolean());
        Assert.False(assets.GetProperty("publicRedistributionOwnerApproval").GetBoolean());
        Assert.Equal("ed1e8d2d2aeb8a2c66e642a16295a72a2990393e3a3843325537da7e11c8a899", assets.GetProperty("onnxSha256").GetString());
        Assert.Equal("46a0278967f1230ef8db59b0b8311a3aba3dce45233f1ba68821418ae02a574d", assets.GetProperty("generatedInputTensorSha256").GetString());

        JsonElement preprocessing = root.GetProperty("preprocessingValidation");
        Assert.True(preprocessing.GetProperty("matchesAuthoritativeTensor").GetBoolean());
        Assert.Equal(1_228_800, preprocessing.GetProperty("elementCount").GetInt64());
        Assert.True(preprocessing.GetProperty("passed").GetBoolean());

        JsonElement raw = root.GetProperty("rawTensorReferenceValidation");
        Assert.Equal(1, raw.GetProperty("tensorCount").GetInt32());
        Assert.Equal(470_400, raw.GetProperty("comparedElementCount").GetInt64());
        Assert.Equal(0, raw.GetProperty("mismatchCount").GetInt64());
        Assert.True(raw.GetProperty("passed").GetBoolean());

        JsonElement independent = root.GetProperty("independentPostprocessValidation");
        Assert.Equal(4, independent.GetProperty("predictionCount").GetInt32());
        Assert.Equal(4, independent.GetProperty("comparisons").GetArrayLength());
        Assert.True(independent.GetProperty("minimumObservedBoxIoU").GetDouble() >= 0.98);
        Assert.True(independent.GetProperty("maximumObservedScoreError").GetDouble() <= 0.03);
        Assert.True(independent.GetProperty("maximumObservedKeypointCoordinateError").GetDouble() <= 5.0);
        Assert.True(independent.GetProperty("maximumObservedKeypointScoreError").GetDouble() <= 0.03);
        Assert.All(independent.GetProperty("comparisons").EnumerateArray(), static comparison =>
        {
            Assert.Equal("person", comparison.GetProperty("className").GetString());
            Assert.True(comparison.GetProperty("passed").GetBoolean());
        });
        Assert.True(independent.GetProperty("passed").GetBoolean());

        JsonElement negative = root.GetProperty("controlledNegativeValidation");
        Assert.Equal(1, negative.GetProperty("exitCode").GetInt32());
        Assert.Equal(1, negative.GetProperty("mismatchCount").GetInt64());
        Assert.Equal(0, negative.GetProperty("firstMismatchIndex").GetInt64());
        Assert.True(negative.GetProperty("failClosed").GetBoolean());

        JsonElement boundary = root.GetProperty("proofBoundary");
        Assert.True(boundary.GetProperty("localPackageConsumerRuntimeEvidence").GetBoolean());
        foreach (string name in new[]
        {
            "sourceTreeRuntimeProof", "publicPackageProof", "packagesDownloadedFromPublicFeed", "postPublishProof",
            "publicRedistributionOwnerApproval", "ownerReleaseAcceptance", "releaseProof", "performsPublish", "uploadsAssets"
        })
        {
            Assert.False(boundary.GetProperty(name).GetBoolean());
        }
    }

    [Fact]
    public void ObbConsumerUsesOnlySelectedLocalPackagesAndRequiresIndependentRotatedBoxEvidence()
    {
        string runner = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionLocalPackageConsumer.ps1"));
        string entrypoint = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionObbLocalPackageConsumer.ps1"));
        string exporter = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionObbLocalPackageConsumerEvidence.ps1"));
        string tutorial = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "yolovision-yolov8n-obb-local-package-consumer-tutorial.md"));

        Assert.Contains("yolov8-obb", entrypoint, StringComparison.Ordinal);
        Assert.Contains("Test-YoloVisionLocalPackageConsumer.ps1", entrypoint, StringComparison.Ordinal);
        foreach (string term in new[]
        {
            "--task", "obb", "--class-count", "15", "--aux-channel-start", "19",
            "--aux-layout", "channels-first", "--angle-radians", "--nms-mode", "class-aware",
            "output0:$ReferenceOutput0Path", "--reference-abs-tolerance", "4.25",
            "--reference-rel-tolerance", "0.05", "Invoke-YoloVisionObbReference.py",
            "430080", "exactly 40 ship oriented boxes", "controlled-reference-negative",
            "minimumObservedRotatedIoU", "maximumObservedAngleErrorRadians",
            "one-selected-nupkg-per-feed", "restoredPackageHashesMatchSelected",
            "vendorRuntimePackageEntryCount", "assetsRemainOnEDrive = $true", "uploadsAssets = $false"
        })
        {
            Assert.Contains(term, runner, StringComparison.Ordinal);
        }

        Assert.Contains("RawValues=", exporter, StringComparison.Ordinal);
        Assert.Contains("ObbPredictions=", exporter, StringComparison.Ordinal);
        Assert.Contains("PublicPackageProof=False", exporter, StringComparison.Ordinal);
        Assert.Contains("OwnerReleaseAcceptance=False", exporter, StringComparison.Ordinal);
        Assert.Contains("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-obb.pt", tutorial, StringComparison.Ordinal);
        Assert.Contains("yolo export", tutorial, StringComparison.Ordinal);
        Assert.Contains("models\\YoloVision\\OrientedBoundingBox", tutorial, StringComparison.Ordinal);
        Assert.Contains("Model Zoo", tutorial, StringComparison.Ordinal);
        foreach (string script in new[] { runner, entrypoint, exporter })
        {
            Assert.DoesNotContain("api.nuget.org", script, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("dotnet nuget push", script, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("gh release upload", script, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void ObbLocalPackageConsumerEvidenceClosesRawRotatedGeometryAndNegativeChecksWithoutPromotion()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "yolovision-yolov8n-obb-local-package-consumer-runtime-evidence.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(evidencePath));
        JsonElement root = document.RootElement;

        Assert.Equal("yolovision-local-package-consumer-runtime-evidence", root.GetProperty("recordKind").GetString());
        Assert.Equal("local-package-consumer-runtime", root.GetProperty("proofClassification").GetString());
        Assert.Equal("yolov8", root.GetProperty("family").GetString());
        Assert.Equal("obb", root.GetProperty("task").GetString());
        Assert.Matches("^\\d{4}-\\d{2}-\\d{2}T", root.GetProperty("generatedAtUtc").GetString()!);
        Assert.Matches("^[0-9a-f]{64}$", root.GetProperty("fullLocalReportSha256").GetString()!);

        JsonElement consumer = root.GetProperty("packageConsumer");
        Assert.Equal("local-file-feed-only", consumer.GetProperty("packageSourceKind").GetString());
        Assert.Equal("one-selected-nupkg-per-feed", consumer.GetProperty("packageSourceIsolation").GetString());
        Assert.Equal(0, consumer.GetProperty("remotePackageSourceCount").GetInt32());
        Assert.Equal(3, consumer.GetProperty("packageCount").GetInt32());
        Assert.Equal(0, consumer.GetProperty("projectReferenceCount").GetInt32());
        Assert.Equal(0, consumer.GetProperty("directAssemblyReferenceCount").GetInt32());
        Assert.Equal(0, consumer.GetProperty("restoredProjectLibraryCount").GetInt32());
        Assert.True(consumer.GetProperty("restoredPackageHashesMatchSelected").GetBoolean());
        Assert.False(consumer.GetProperty("nativeBridgePathEnvironmentVariableSet").GetBoolean());
        Assert.True(consumer.GetProperty("tensorRtCudaAndCudnnAreExternalDependencies").GetBoolean());
        Assert.Equal(0, consumer.GetProperty("vendorRuntimePackageEntryCount").GetInt32());
        Assert.Equal(1, consumer.GetProperty("bridgeNativePackageEntryCount").GetInt32());
        Assert.Equal("E:", consumer.GetProperty("workspaceDrive").GetString());
        Assert.True(consumer.GetProperty("workspaceRemovedAfterValidation").GetBoolean());

        JsonElement assets = root.GetProperty("assets");
        Assert.True(assets.GetProperty("onnxStoredUnderWorkspaceModelsDirectory").GetBoolean());
        Assert.True(assets.GetProperty("heavyAssetsRemainOutsideGit").GetBoolean());
        Assert.False(assets.GetProperty("publicRedistributionOwnerApproval").GetBoolean());
        Assert.Equal("5f2701ef5326fb5a691999438cfc55a69656323c21ffddebaff8968ab6de2e92", assets.GetProperty("onnxSha256").GetString());
        Assert.Equal("c56c027619088bce94f9160a3f602b4ad81fe323001867b1ee456100040fec6e", assets.GetProperty("generatedInputTensorSha256").GetString());

        JsonElement preprocessing = root.GetProperty("preprocessingValidation");
        Assert.True(preprocessing.GetProperty("matchesAuthoritativeTensor").GetBoolean());
        Assert.Equal(3_145_728, preprocessing.GetProperty("elementCount").GetInt64());
        Assert.True(preprocessing.GetProperty("passed").GetBoolean());

        JsonElement raw = root.GetProperty("rawTensorReferenceValidation");
        Assert.Equal(1, raw.GetProperty("tensorCount").GetInt32());
        Assert.Equal(430_080, raw.GetProperty("comparedElementCount").GetInt64());
        Assert.Equal(0, raw.GetProperty("mismatchCount").GetInt64());
        Assert.True(raw.GetProperty("passed").GetBoolean());

        JsonElement independent = root.GetProperty("independentPostprocessValidation");
        Assert.Equal(40, independent.GetProperty("predictionCount").GetInt32());
        Assert.Equal(40, independent.GetProperty("comparisons").GetArrayLength());
        Assert.True(independent.GetProperty("minimumObservedRotatedIoU").GetDouble() >= 0.98);
        Assert.True(independent.GetProperty("maximumObservedCoordinateError").GetDouble() <= 5.0);
        Assert.True(independent.GetProperty("maximumObservedAngleErrorRadians").GetDouble() <= 0.02);
        Assert.True(independent.GetProperty("maximumObservedScoreError").GetDouble() <= 0.03);
        Assert.All(independent.GetProperty("comparisons").EnumerateArray(), static comparison =>
        {
            Assert.Equal(1, comparison.GetProperty("classId").GetInt32());
            Assert.Equal("ship", comparison.GetProperty("className").GetString());
            Assert.True(comparison.GetProperty("passed").GetBoolean());
        });
        Assert.True(independent.GetProperty("passed").GetBoolean());

        JsonElement negative = root.GetProperty("controlledNegativeValidation");
        Assert.Equal(1, negative.GetProperty("exitCode").GetInt32());
        Assert.Equal(1, negative.GetProperty("mismatchCount").GetInt64());
        Assert.Equal(0, negative.GetProperty("firstMismatchIndex").GetInt64());
        Assert.True(negative.GetProperty("failClosed").GetBoolean());

        JsonElement boundary = root.GetProperty("proofBoundary");
        Assert.True(boundary.GetProperty("localPackageConsumerRuntimeEvidence").GetBoolean());
        foreach (string name in new[]
        {
            "sourceTreeRuntimeProof", "publicPackageProof", "packagesDownloadedFromPublicFeed", "postPublishProof",
            "publicRedistributionOwnerApproval", "ownerReleaseAcceptance", "releaseProof", "performsPublish", "uploadsAssets"
        })
        {
            Assert.False(boundary.GetProperty(name).GetBoolean());
        }
    }

    [Fact]
    public void MultiVersionMatrixRunsEverySupportedLineAndKeepsBlockedRowsNonProof()
    {
        string script = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionLocalPackageConsumerMatrix.ps1"));

        Assert.Contains("win-x64-trt8.6-cuda12.1-cudnn8.9", script, StringComparison.Ordinal);
        Assert.Contains("win-x64-trt10.11-cuda12.9-cudnn9.22", script, StringComparison.Ordinal);
        Assert.Contains("win-x64-trt11.0-cuda12.9-cudnn9.22", script, StringComparison.Ordinal);
        Assert.Contains("Resolve-RuntimeRoots.ps1", script, StringComparison.Ordinal);
        Assert.Contains("runtime-attempt-blocked", script, StringComparison.Ordinal);
        Assert.Contains("blockedRowsAreRuntimeExecutionProof = $false", script, StringComparison.Ordinal);
        Assert.Contains("isPackageConsumerRuntimeProof = $false", script, StringComparison.Ordinal);
        Assert.Contains("packagesDownloadedFromPublicFeed = $false", script, StringComparison.Ordinal);
        Assert.Contains("canPublishPublicly = $false", script, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", script, StringComparison.Ordinal);
        Assert.Contains("Remove-DirectoryWithRetry -Path $resolvedLineOutputRoot", script, StringComparison.Ordinal);
        Assert.DoesNotContain("dotnet nuget push", script, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ExportedYoloVisionSurfaceAndXmlCommandContractRemainPointerFree()
    {
        Assembly assembly = typeof(YoloVisionCommand).Assembly;
        Type[] exportedTypes = assembly.GetExportedTypes();
        Assert.NotEmpty(exportedTypes);

        List<string> forbidden = new();
        List<string> sampleInternalLeaks = new();
        foreach (Type type in exportedTypes)
        {
            InspectType(type, type.FullName ?? type.Name, forbidden, sampleInternalLeaks);
            foreach (MemberInfo member in type.GetMembers(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly))
            {
                foreach (Type signatureType in GetSignatureTypes(member))
                {
                    InspectType(signatureType, $"{type.FullName}.{member.Name}", forbidden, sampleInternalLeaks);
                }
            }
        }

        Assert.Empty(forbidden);
        Assert.Empty(sampleInternalLeaks);

        string xmlPath = Path.ChangeExtension(assembly.Location, ".xml");
        Assert.True(File.Exists(xmlPath), $"YoloVision XML documentation was not generated at {xmlPath}.");
        XDocument xml = XDocument.Load(xmlPath);
        string[] memberNames = xml.Descendants("member")
            .Select(static member => member.Attribute("name")?.Value ?? string.Empty)
            .ToArray();
        Assert.Contains("T:YoloVisionSample.YoloVisionCommand", memberNames);
        Assert.Contains("M:YoloVisionSample.YoloVisionCommand.Run(System.String[])", memberNames);
    }

    [Fact]
    public void CompactProofRecordsLocalRuntimeWithoutPublicProofPromotion()
    {
        string proofPath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "interface-coverage",
            "yolox-local-package-consumer-runtime-proof-closure.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(proofPath));
        JsonElement root = document.RootElement;

        Assert.Equal("local-package-consumer-runtime", root.GetProperty("evidenceClassification").GetString());
        Assert.Equal(3, root.GetProperty("packages").GetArrayLength());
        Assert.Equal(0, root.GetProperty("consumer").GetProperty("projectReferenceCount").GetInt32());
        Assert.Equal(0, root.GetProperty("consumer").GetProperty("restoredProjectLibraryCount").GetInt32());
        Assert.Equal("E:", root.GetProperty("consumer").GetProperty("workspaceDrive").GetString());
        Assert.True(root.GetProperty("consumer").GetProperty("workspaceRemovedAfterValidation").GetBoolean());
        Assert.Equal(5, root.GetProperty("runtime").GetProperty("predictionCount").GetInt32());
        Assert.Equal("bicycle", root.GetProperty("runtime").GetProperty("topPrediction").GetString());
        Assert.Equal("YoloVision Passed=True", root.GetProperty("runtime").GetProperty("passedMarker").GetString());
        JsonElement cDriveAudit = root.GetProperty("cDriveAudit");
        Assert.Equal(0, cDriveAudit.GetProperty("yoloXOrConsumerAssetMatchCount").GetInt32());
        Assert.False(cDriveAudit.GetProperty("packageCacheRootExists").GetBoolean());
        Assert.False(cDriveAudit.GetProperty("consumerWorkspaceUsedCDrive").GetBoolean());
        Assert.False(cDriveAudit.GetProperty("unrelatedUserOrSystemFilesRemoved").GetBoolean());
        JsonElement boundary = root.GetProperty("boundary");
        Assert.True(boundary.GetProperty("isLocalPackageConsumerRuntimeEvidence").GetBoolean());
        Assert.False(boundary.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("packagesDownloadedFromPublicFeed").GetBoolean());
        Assert.False(boundary.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(boundary.GetProperty("performsPublish").GetBoolean());
    }

    [Fact]
    public void MultiVersionCompactProofKeepsTrt8BlockedAndPromotesOnlyLocalTrt10AndTrt11Rows()
    {
        string proofPath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "interface-coverage",
            "yolox-multi-version-local-package-consumer-runtime-proof-closure.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(proofPath));
        JsonElement root = document.RootElement;

        Assert.Equal("passed-path-free-matrix-proof-closure", root.GetProperty("validationState").GetString());
        Assert.Equal(3, root.GetProperty("requestedRuntimeCount").GetInt32());
        Assert.Equal(2, root.GetProperty("passedRuntimeCount").GetInt32());
        Assert.Equal(1, root.GetProperty("blockedRuntimeCount").GetInt32());
        Assert.Equal(5, root.GetProperty("packages").GetArrayLength());

        JsonElement[] rows = root.GetProperty("rows").EnumerateArray().ToArray();
        JsonElement trt8 = rows.Single(static row => row.GetProperty("tensorRtLine").GetString() == "8");
        JsonElement trt10 = rows.Single(static row => row.GetProperty("tensorRtLine").GetString() == "10");
        JsonElement trt11 = rows.Single(static row => row.GetProperty("tensorRtLine").GetString() == "11");
        Assert.False(trt8.GetProperty("runtimePassed").GetBoolean());
        Assert.Equal("runtime-attempt-blocked", trt8.GetProperty("evidenceClassification").GetString());
        Assert.Equal(0, trt8.GetProperty("cudnnRuntimeDllCount").GetInt32());
        Assert.Contains("ONNX parser", trt8.GetProperty("diagnostic").GetString(), StringComparison.Ordinal);
        Assert.True(trt10.GetProperty("runtimePassed").GetBoolean());
        Assert.Equal(5, trt10.GetProperty("predictionCount").GetInt32());
        Assert.True(trt11.GetProperty("runtimePassed").GetBoolean());
        Assert.Equal(5, trt11.GetProperty("predictionCount").GetInt32());
        Assert.Equal("existing-assembled-runtime", trt11.GetProperty("tensorRtRuntimeRootSource").GetString());
        Assert.All(rows, static row => Assert.True(row.GetProperty("bridgeBuildTensorRtLineMatches").GetBoolean()));
        Assert.All(rows, static row => Assert.True(row.GetProperty("workspaceRemovedAfterValidation").GetBoolean()));

        JsonElement surface = root.GetProperty("packageSurfaceAudit");
        Assert.True(surface.GetProperty("valid").GetBoolean());
        Assert.Equal(0, surface.GetProperty("forbiddenPointerOrHandleFindingCount").GetInt32());
        Assert.Equal(0, surface.GetProperty("sampleInternalTypeLeakFindingCount").GetInt32());
        Assert.Equal(2, root.GetProperty("ownerHandoff").GetProperty("strictValidatorCommandCount").GetInt32());
        Assert.Equal(0, root.GetProperty("cDriveAudit").GetProperty("testDirectoryMatchCount").GetInt32());
        Assert.Equal(0, root.GetProperty("cDriveAudit").GetProperty("yoloXOrConsumerAssetMatchCount").GetInt32());

        JsonElement boundary = root.GetProperty("boundary");
        Assert.False(boundary.GetProperty("blockedRowsAreRuntimeExecutionProof").GetBoolean());
        Assert.False(boundary.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("packagesDownloadedFromPublicFeed").GetBoolean());
        Assert.False(boundary.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(boundary.GetProperty("performsPublish").GetBoolean());
    }

    [Fact]
    public void PublicPackageOwnerHandoffListsExactHashesAndExcludesBlockedTrt8FromRuntimeCommands()
    {
        string handoffPath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "interface-coverage",
            "yolovision-public-package-owner-handoff.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(handoffPath));
        JsonElement root = document.RootElement;

        Assert.Equal("owner-action-required-public-feed-not-executed", root.GetProperty("state").GetString());
        Assert.Equal(5, root.GetProperty("packages").GetArrayLength());
        Assert.Equal(2, root.GetProperty("publicConsumerCandidateRuntimeKeys").GetArrayLength());
        Assert.Equal(1, root.GetProperty("blockedRuntimeKeys").GetArrayLength());
        Assert.Equal("win-x64-trt8.6-cuda12.1-cudnn8.9", root.GetProperty("blockedRuntimeKeys")[0].GetString());
        Assert.Equal(2, root.GetProperty("cleanExternalCommands").GetArrayLength());
        Assert.Equal(2, root.GetProperty("strictValidatorCommands").GetArrayLength());
        Assert.All(root.GetProperty("cleanExternalCommands").EnumerateArray(), static command =>
            Assert.DoesNotContain("trt8.6", command.GetString(), StringComparison.Ordinal));

        JsonElement[] packages = root.GetProperty("packages").EnumerateArray().ToArray();
        Assert.All(packages, static package =>
        {
            Assert.Matches("^[0-9a-f]{64}$", package.GetProperty("localSha256").GetString()!);
            Assert.StartsWith("https://api.nuget.org/v3-flatcontainer/", package.GetProperty("expectedNuGetFlatContainerUrl").GetString(), StringComparison.Ordinal);
        });
        JsonElement trt8 = packages.Single(static package => package.GetProperty("id").GetString()!.Contains("trt8.6", StringComparison.Ordinal));
        Assert.False(trt8.GetProperty("eligibleForYoloVisionPublicRuntimeHandoff").GetBoolean());
        Assert.True(trt8.GetProperty("ownerMustRebuildAndRefreezeBeforeYoloVisionPublish").GetBoolean());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canCloseReleaseIssue").GetBoolean());
    }

    [Fact]
    public void DetectionConsumerUsesOnlySelectedLocalPackagesAndRequiresIndependentBoxEvidence()
    {
        string runner = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionLocalPackageConsumer.ps1"));
        string entrypoint = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionDetectionLocalPackageConsumer.ps1"));
        string exporter = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionDetectionLocalPackageConsumerEvidence.ps1"));
        string tutorial = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "yolovision-yolov8n-det-local-package-consumer-tutorial.md"));

        Assert.Contains("yolov8-detection", entrypoint, StringComparison.Ordinal);
        Assert.Contains("Test-YoloVisionLocalPackageConsumer.ps1", entrypoint, StringComparison.Ordinal);
        foreach (string term in new[]
        {
            "--task", "det", "--class-count", "80", "--layout", "channels-first",
            "--has-objectness", "false", "--nms-mode", "class-aware",
            "output0:$ReferenceOutput0Path", "--reference-abs-tolerance", "0.02",
            "--reference-rel-tolerance", "0.05", "Invoke-YoloVisionDetectionReference.py",
            "705600", "four persons and one bus", "controlled-reference-negative",
            "minimumObservedBoxIoU", "maximumObservedScoreError",
            "one-selected-nupkg-per-feed", "restoredPackageHashesMatchSelected",
            "vendorRuntimePackageEntryCount", "assetsRemainOnEDrive = $true", "uploadsAssets = $false"
        })
        {
            Assert.Contains(term, runner, StringComparison.Ordinal);
        }

        Assert.Contains("RawValues=", exporter, StringComparison.Ordinal);
        Assert.Contains("DetectionPredictions=", exporter, StringComparison.Ordinal);
        Assert.Contains("PublicPackageProof=False", exporter, StringComparison.Ordinal);
        Assert.Contains("OwnerReleaseAcceptance=False", exporter, StringComparison.Ordinal);
        Assert.Contains("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt", tutorial, StringComparison.Ordinal);
        Assert.Contains("yolo export", tutorial, StringComparison.Ordinal);
        Assert.Contains("models/YoloVision/Detection", tutorial, StringComparison.Ordinal);
        Assert.Contains("Model Zoo", tutorial, StringComparison.Ordinal);
        foreach (string script in new[] { runner, entrypoint, exporter })
        {
            Assert.DoesNotContain("api.nuget.org", script, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("dotnet nuget push", script, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("gh release upload", script, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void DetectionLocalPackageConsumerEvidenceClosesRawBoxesAndNegativeChecksWithoutPromotion()
    {
        string evidencePath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "yolovision-yolov8n-det-local-package-consumer-runtime-evidence.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(evidencePath));
        JsonElement root = document.RootElement;

        Assert.Equal("yolovision-local-package-consumer-runtime-evidence", root.GetProperty("recordKind").GetString());
        Assert.Equal("local-package-consumer-runtime", root.GetProperty("proofClassification").GetString());
        Assert.Equal("yolov8", root.GetProperty("family").GetString());
        Assert.Equal("det", root.GetProperty("task").GetString());
        Assert.Matches("^\\d{4}-\\d{2}-\\d{2}T", root.GetProperty("generatedAtUtc").GetString()!);
        Assert.Matches("^[0-9a-f]{64}$", root.GetProperty("fullLocalReportSha256").GetString()!);

        JsonElement consumer = root.GetProperty("packageConsumer");
        Assert.Equal("local-file-feed-only", consumer.GetProperty("packageSourceKind").GetString());
        Assert.Equal("one-selected-nupkg-per-feed", consumer.GetProperty("packageSourceIsolation").GetString());
        Assert.Equal(0, consumer.GetProperty("remotePackageSourceCount").GetInt32());
        Assert.Equal(3, consumer.GetProperty("packageCount").GetInt32());
        Assert.Equal(0, consumer.GetProperty("projectReferenceCount").GetInt32());
        Assert.Equal(0, consumer.GetProperty("directAssemblyReferenceCount").GetInt32());
        Assert.Equal(0, consumer.GetProperty("restoredProjectLibraryCount").GetInt32());
        Assert.True(consumer.GetProperty("restoredPackageHashesMatchSelected").GetBoolean());
        Assert.False(consumer.GetProperty("nativeBridgePathEnvironmentVariableSet").GetBoolean());
        Assert.True(consumer.GetProperty("tensorRtCudaAndCudnnAreExternalDependencies").GetBoolean());
        Assert.Equal(0, consumer.GetProperty("vendorRuntimePackageEntryCount").GetInt32());
        Assert.Equal(1, consumer.GetProperty("bridgeNativePackageEntryCount").GetInt32());
        Assert.Equal("E:", consumer.GetProperty("workspaceDrive").GetString());
        Assert.True(consumer.GetProperty("workspaceRemovedAfterValidation").GetBoolean());

        JsonElement assets = root.GetProperty("assets");
        Assert.True(assets.GetProperty("onnxStoredUnderWorkspaceModelsDirectory").GetBoolean());
        Assert.True(assets.GetProperty("heavyAssetsRemainOutsideGit").GetBoolean());
        Assert.False(assets.GetProperty("publicRedistributionOwnerApproval").GetBoolean());
        Assert.Equal("db28a49ffbb0425f39ae56252e7e0b43d06b357416c7da58872e285560b4221e", assets.GetProperty("onnxSha256").GetString());
        Assert.Equal("46a0278967f1230ef8db59b0b8311a3aba3dce45233f1ba68821418ae02a574d", assets.GetProperty("generatedInputTensorSha256").GetString());

        JsonElement preprocessing = root.GetProperty("preprocessingValidation");
        Assert.True(preprocessing.GetProperty("matchesAuthoritativeTensor").GetBoolean());
        Assert.Equal(1_228_800, preprocessing.GetProperty("elementCount").GetInt64());
        Assert.True(preprocessing.GetProperty("passed").GetBoolean());

        JsonElement raw = root.GetProperty("rawTensorReferenceValidation");
        Assert.Equal(1, raw.GetProperty("tensorCount").GetInt32());
        Assert.Equal(705_600, raw.GetProperty("comparedElementCount").GetInt64());
        Assert.Equal(0, raw.GetProperty("mismatchCount").GetInt64());
        Assert.True(raw.GetProperty("passed").GetBoolean());

        JsonElement independent = root.GetProperty("independentPostprocessValidation");
        Assert.Equal(5, independent.GetProperty("predictionCount").GetInt32());
        Assert.Equal(5, independent.GetProperty("comparisons").GetArrayLength());
        Assert.True(independent.GetProperty("minimumObservedBoxIoU").GetDouble() >= 0.995);
        Assert.True(independent.GetProperty("maximumObservedScoreError").GetDouble() <= 0.01);
        JsonElement[] comparisons = independent.GetProperty("comparisons").EnumerateArray().ToArray();
        Assert.Equal(4, comparisons.Count(static comparison => comparison.GetProperty("classId").GetInt32() == 0 && comparison.GetProperty("className").GetString() == "person"));
        Assert.Single(comparisons.Where(static comparison => comparison.GetProperty("classId").GetInt32() == 5 && comparison.GetProperty("className").GetString() == "bus"));
        Assert.All(comparisons, static comparison => Assert.True(comparison.GetProperty("passed").GetBoolean()));
        Assert.True(independent.GetProperty("passed").GetBoolean());

        JsonElement negative = root.GetProperty("controlledNegativeValidation");
        Assert.Equal(1, negative.GetProperty("exitCode").GetInt32());
        Assert.Equal(1, negative.GetProperty("mismatchCount").GetInt64());
        Assert.Equal(0, negative.GetProperty("firstMismatchIndex").GetInt64());
        Assert.True(negative.GetProperty("failClosed").GetBoolean());

        JsonElement boundary = root.GetProperty("proofBoundary");
        Assert.True(boundary.GetProperty("localPackageConsumerRuntimeEvidence").GetBoolean());
        foreach (string name in new[]
        {
            "sourceTreeRuntimeProof", "publicPackageProof", "packagesDownloadedFromPublicFeed", "postPublishProof",
            "publicRedistributionOwnerApproval", "ownerReleaseAcceptance", "releaseProof", "performsPublish", "uploadsAssets"
        })
        {
            Assert.False(boundary.GetProperty(name).GetBoolean());
        }
    }

    [Fact]
    public void PublicConsumerAndValidatorRequireNuGetSourceMetadataAndNeverPublish()
    {
        string publicConsumer = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionPublicPackageConsumer.ps1"));
        string validator = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionPublicPackageProof.ps1"));
        string surfaceAudit = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionPackageSurface.ps1"));

        Assert.Contains("<clear />", publicConsumer, StringComparison.Ordinal);
        Assert.Contains("https://api.nuget.org/v3/index.json", publicConsumer, StringComparison.Ordinal);
        Assert.Contains(".nupkg.metadata", publicConsumer, StringComparison.Ordinal);
        Assert.Contains("downloadedFromPublicFeed = $true", publicConsumer, StringComparison.Ordinal);
        Assert.Contains("isPackageConsumerRuntimeProof = $true", publicConsumer, StringComparison.Ordinal);
        Assert.Contains("Test-YoloVisionPublicPackageProof.ps1", publicConsumer, StringComparison.Ordinal);
        Assert.Contains("ExpectedHandoffPath", validator, StringComparison.Ordinal);
        Assert.Contains("localSha256", validator, StringComparison.Ordinal);
        Assert.Contains("package-metadata-source", validator, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", validator, StringComparison.Ordinal);
        Assert.Contains("GetExportedTypes", surfaceAudit, StringComparison.Ordinal);
        Assert.Contains("OnnxSampleOptions", surfaceAudit, StringComparison.Ordinal);
        Assert.Contains("SafeHandle", surfaceAudit, StringComparison.Ordinal);
        foreach (string script in new[] { publicConsumer, validator, surfaceAudit })
        {
            Assert.DoesNotContain("dotnet nuget push", script, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("gh release upload", script, StringComparison.OrdinalIgnoreCase);
        }
    }

    private static IEnumerable<Type> GetSignatureTypes(MemberInfo member)
    {
        switch (member)
        {
            case MethodInfo method:
                yield return method.ReturnType;
                foreach (ParameterInfo parameter in method.GetParameters())
                {
                    yield return parameter.ParameterType;
                }
                break;
            case ConstructorInfo constructor:
                foreach (ParameterInfo parameter in constructor.GetParameters())
                {
                    yield return parameter.ParameterType;
                }
                break;
            case PropertyInfo property:
                yield return property.PropertyType;
                foreach (ParameterInfo parameter in property.GetIndexParameters())
                {
                    yield return parameter.ParameterType;
                }
                break;
            case FieldInfo field:
                yield return field.FieldType;
                break;
            case EventInfo eventInfo when eventInfo.EventHandlerType != null:
                yield return eventInfo.EventHandlerType;
                break;
        }
    }

    private static void InspectType(Type type, string surface, ICollection<string> forbidden, ICollection<string> sampleInternalLeaks)
    {
        Type candidate = type;
        while (candidate.HasElementType && candidate.GetElementType() != null)
        {
            candidate = candidate.GetElementType()!;
        }

        if (candidate.IsPointer || candidate == typeof(IntPtr) || candidate == typeof(UIntPtr) || typeof(SafeHandle).IsAssignableFrom(candidate))
        {
            forbidden.Add($"{surface}: {type}");
        }
        if ((candidate.FullName ?? string.Empty).Contains("OnnxSampleOptions", StringComparison.Ordinal) ||
            string.Equals(candidate.Namespace, "JYPPX.SampleSupport", StringComparison.Ordinal))
        {
            sampleInternalLeaks.Add($"{surface}: {type}");
        }
        if (candidate.IsGenericType)
        {
            foreach (Type argument in candidate.GetGenericArguments())
            {
                InspectType(argument, surface, forbidden, sampleInternalLeaks);
            }
        }
    }
}

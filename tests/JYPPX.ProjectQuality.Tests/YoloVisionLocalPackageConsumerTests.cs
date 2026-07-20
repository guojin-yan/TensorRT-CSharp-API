using System.Text.Json;
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
        Assert.Contains("JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge", project, StringComparison.Ordinal);
        Assert.Contains("ProjectReference=False", program, StringComparison.Ordinal);
        Assert.Contains("YoloVisionCommand.Run(args)", program, StringComparison.Ordinal);
        Assert.Contains("local-package-consumer-runtime", readme, StringComparison.Ordinal);
        Assert.Contains("not proof", readme, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ConsumerScriptKeepsWorkspaceAndRestoreCacheOffCDriveAndFailsClosed()
    {
        string script = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionLocalPackageConsumer.ps1"));

        Assert.Contains("consumer-workspaces\\yolovision-yolox-local-package", script, StringComparison.Ordinal);
        Assert.Contains("Assert-NonCDrivePath", script, StringComparison.Ordinal);
        Assert.Contains("<clear />", script, StringComparison.Ordinal);
        Assert.Contains("--packages", script, StringComparison.Ordinal);
        Assert.Contains("projectReferenceCount = 0", script, StringComparison.Ordinal);
        Assert.Contains("restoredProjectLibraryCount", script, StringComparison.Ordinal);
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
}

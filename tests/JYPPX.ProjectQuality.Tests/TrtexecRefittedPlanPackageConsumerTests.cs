using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TrtexecRefittedPlanPackageConsumerTests
{
    [Fact]
    public void ConsumerUsesPublicOwnerSafeRuntimeWithoutSourceAssemblyLoading()
    {
        string program = ReadSource("tests", "fixtures", "package-consumers", "RefittedPlan.PackageConsumer", "Program.cs");
        string project = ReadSource(
            "tests", "fixtures", "package-consumers",
            "RefittedPlan.PackageConsumer",
            "RefittedPlan.PackageConsumer.csproj.template");

        int runCall = program.IndexOf("RunPersistedPlan(", StringComparison.Ordinal);
        int ownerExit = program.IndexOf("OwnerScopeExited=True", runCall, StringComparison.Ordinal);
        int runtime = program.IndexOf("using TensorRtRuntime runtime", StringComparison.Ordinal);
        int deserialize = program.IndexOf("runtime.DeserializeFromFile(planPath)", runtime, StringComparison.Ordinal);
        int engine = program.IndexOf("using TensorRtEngine engine", runtime, StringComparison.Ordinal);
        int context = program.IndexOf("using TensorRtExecutionContext context", engine, StringComparison.Ordinal);
        int bindings = program.IndexOf("using TensorRtInferenceBindings bindings", context, StringComparison.Ordinal);
        int enqueue = program.IndexOf("bindings.EnqueueAsync", bindings, StringComparison.Ordinal);
        int readback = program.IndexOf("bindings.ReadOutputSingles", enqueue, StringComparison.Ordinal);

        Assert.True(runtime >= 0 && engine > runtime && deserialize > engine && context > deserialize);
        Assert.True(bindings > context && enqueue > bindings && readback > enqueue);
        Assert.True(runCall >= 0 && ownerExit > runCall);
        Assert.Contains("PackageReferenceOnly=True", program, StringComparison.Ordinal);
        Assert.Contains("ManualManagedAssemblyLoad=False", program, StringComparison.Ordinal);
        Assert.Contains("OutputExactMatch=", program, StringComparison.Ordinal);
        Assert.Contains("ReadReference", program, StringComparison.Ordinal);
        Assert.Contains("ValidateReference", program, StringComparison.Ordinal);
        Assert.Contains("ReferenceValidationPassed=", program, StringComparison.Ordinal);
        Assert.DoesNotContain("Assembly.LoadFrom", program, StringComparison.Ordinal);
        Assert.DoesNotContain("Assembly.LoadFile", program, StringComparison.Ordinal);
        Assert.DoesNotContain("JYPPX_NATIVE_BRIDGE_PATH", program, StringComparison.Ordinal);
        Assert.Contains("<PackageReference", project, StringComparison.Ordinal);
        Assert.Contains("<RestorePackagesPath>", project, StringComparison.Ordinal);
        Assert.DoesNotContain("<ProjectReference", project, StringComparison.Ordinal);
    }

    [Fact]
    public void RunnerCreatesIsolatedLocalFeedConsumerAndCleansItsWorkspace()
    {
        string runner = ReadSource("eng", "Test-TrtexecRefittedPlanPackageConsumer.ps1");
        string validator = ReadSource("eng", "Test-TrtexecRefittedPlanPackageConsumerEvidence.ps1");

        Assert.Contains("<clear />", runner, StringComparison.Ordinal);
        Assert.Contains("jyppx-managed-local", runner, StringComparison.Ordinal);
        Assert.Contains("jyppx-bridge-local", runner, StringComparison.Ordinal);
        Assert.DoesNotContain("api.nuget.org", runner, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("$restorePackagesPath", runner, StringComparison.Ordinal);
        Assert.Contains("Copy-Item -LiteralPath $SourcePlanPath", runner, StringComparison.Ordinal);
        Assert.Contains("Copy-Item -LiteralPath $SourceInputPath", runner, StringComparison.Ordinal);
        Assert.Contains("Copy-Item -LiteralPath $SourceReferencePath", runner, StringComparison.Ordinal);
        Assert.Contains("ReferenceValidationPassed=", runner, StringComparison.Ordinal);
        Assert.Contains("$env:JYPPX_NATIVE_BRIDGE_PATH = $null", runner, StringComparison.Ordinal);
        Assert.Contains("$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = $null", runner, StringComparison.Ordinal);
        Assert.Contains("Remove-SafeConsumerDirectory", runner, StringComparison.Ordinal);
        Assert.Contains("workspaceRemovedAfterValidation", runner, StringComparison.Ordinal);
        Assert.Contains("compact-evidence-path-free", validator, StringComparison.Ordinal);
        Assert.Contains("plan-hash-cross-check", validator, StringComparison.Ordinal);
        Assert.Contains("output-hash-cross-check", validator, StringComparison.Ordinal);
        Assert.Contains("[IO.Directory]::EnumerateFileSystemEntries", runner, StringComparison.Ordinal);
        Assert.Contains("$PSVersionTable.PSVersion.Major", runner, StringComparison.Ordinal);
        Assert.Contains("[Security.Cryptography.SHA256]::Create()", runner, StringComparison.Ordinal);
        Assert.Contains("MakeRelativeUri", runner, StringComparison.Ordinal);
        Assert.Contains("[IO.Directory]::Delete($extendedPath, $true)", runner, StringComparison.Ordinal);
        Assert.DoesNotContain("ConvertFrom-Json -Depth", runner, StringComparison.Ordinal);
        Assert.DoesNotContain("[Security.Cryptography.SHA256]::HashData", runner, StringComparison.Ordinal);
        Assert.DoesNotContain("[IO.Path]::GetRelativePath", runner, StringComparison.Ordinal);
        Assert.DoesNotContain("ConvertFrom-Json -Depth", validator, StringComparison.Ordinal);
        Assert.Contains("[IO.File]::WriteAllText", validator, StringComparison.Ordinal);
    }

    [Fact]
    public void CompactEvidenceProvesLocalPackageConsumerEnqueueAndBaselineMatch()
    {
        using JsonDocument evidenceDocument = ReadJson(
            "artifacts",
            "interface-coverage",
            "trtexec-refitted-plan-package-consumer-evidence.json");
        using JsonDocument priorDocument = ReadJson(
            "artifacts",
            "interface-coverage",
            "trtexec-refitted-plan-persistence-evidence.json");
        JsonElement root = evidenceDocument.RootElement;
        JsonElement contract = root.GetProperty("packageContract");
        JsonElement consumer = root.GetProperty("consumer");
        JsonElement artifacts = root.GetProperty("artifacts");
        JsonElement runtime = root.GetProperty("runtime");
        JsonElement priorTrt10 = priorDocument.RootElement.GetProperty("tensorRt10");

        Assert.Equal("trtexec-refitted-plan-package-consumer-evidence.v1", root.GetProperty("schemaVersion").GetString());
        Assert.Equal("local-package-consumer-refitted-plan-runtime-passed", root.GetProperty("state").GetString());
        Assert.Equal(2, contract.GetProperty("declaredLocalSourceCount").GetInt32());
        Assert.False(contract.GetProperty("nugetOrgEnabled").GetBoolean());
        Assert.True(contract.GetProperty("usesPackageReferenceOnly").GetBoolean());
        Assert.False(contract.GetProperty("usesProjectReference").GetBoolean());
        Assert.False(contract.GetProperty("manualManagedAssemblyLoad").GetBoolean());
        Assert.True(consumer.GetProperty("managedAssemblyOutsideSourceTree").GetBoolean());
        Assert.True(consumer.GetProperty("workspaceRemovedAfterValidation").GetBoolean());
        Assert.Equal(
            priorTrt10.GetProperty("persistedPlanSha256").GetString(),
            artifacts.GetProperty("copiedPlanSha256").GetString());
        Assert.Equal(
            priorTrt10.GetProperty("baselineOutputSha256").GetString(),
            artifacts.GetProperty("outputSha256").GetString());
        Assert.Equal(
            "artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/mnist-trt10-7.reference.json",
            artifacts.GetProperty("sourceReference").GetString());
        Assert.True(artifacts.GetProperty("referenceCopyPathDistinct").GetBoolean());
        Assert.Equal(
            artifacts.GetProperty("sourceReferenceSha256").GetString(),
            artifacts.GetProperty("copiedReferenceSha256").GetString());
        Assert.Equal("Plus214_Output_0", artifacts.GetProperty("referenceTensorName").GetString());
        Assert.Equal(10, artifacts.GetProperty("referenceElementCount").GetInt32());
        Assert.Equal(
            "repository-mnist-runtime-output-derived-unreviewed",
            artifacts.GetProperty("referenceSourceClassification").GetString());
        Assert.True(artifacts.GetProperty("outputExactMatch").GetBoolean());
        Assert.True(runtime.GetProperty("bindingsReadyForEnqueue").GetBoolean());
        Assert.True(runtime.GetProperty("enqueueCompleted").GetBoolean());
        Assert.True(runtime.GetProperty("ownerScopeExited").GetBoolean());
        Assert.True(runtime.GetProperty("referenceValidationCompleted").GetBoolean());
        Assert.True(runtime.GetProperty("referenceValidationPassed").GetBoolean());
        Assert.Equal(10, runtime.GetProperty("referenceComparedElementCount").GetInt32());
        Assert.Equal(0, runtime.GetProperty("referenceMismatchCount").GetInt32());
        Assert.Equal(-1, runtime.GetProperty("referenceFirstMismatchIndex").GetInt32());
        Assert.Equal(7, runtime.GetProperty("predictedIndex").GetInt32());
        Assert.True(root.GetProperty("proofBoundary").GetProperty("isLocalPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("isPackageConsumerRuntimeProof").GetBoolean());

        string programPath = Path.Combine(
            RepositoryPaths.Root,
            "tests",
            "fixtures",
            "package-consumers",
            "RefittedPlan.PackageConsumer",
            "Program.cs");
        Assert.Equal(
            ComputeSha256(programPath),
            consumer.GetProperty("programSha256").GetString());
    }

    [Fact]
    public void StrictValidationArtifactIsCheckedInAndComplete()
    {
        using JsonDocument document = ReadJson(
            "artifacts",
            "interface-coverage",
            "trtexec-refitted-plan-package-consumer-validation.json");
        JsonElement root = document.RootElement;

        Assert.Equal("trtexec-refitted-plan-package-consumer-validation.v1", root.GetProperty("schemaVersion").GetString());
        Assert.True(root.GetProperty("strict").GetBoolean());
        Assert.True(root.GetProperty("checkCount").GetInt32() >= 40);
        Assert.Equal(root.GetProperty("checkCount").GetInt32(), root.GetProperty("passedCount").GetInt32());
        Assert.Equal(0, root.GetProperty("failureCount").GetInt32());
    }

    [Fact]
    public void TechnicalArticleDocumentsTheFullModelPackageAndRuntimeFlow()
    {
        string article = ReadSource(
            "docs",
            "articles",
            "zh-cn",
            "tensorrtexec-refitted-plan-local-package-consumer.md");

        foreach (string heading in new[]
        {
            "## 本文使用的项目与库",
            "## 模型获取与许可证",
            "## ONNX 转换与暂存",
            "## 生成可部署的 Refitted Plan",
            "## 创建公开包消费项目",
            "## 编写程序入口",
            "## 编译并运行",
            "## 已验证结果",
            "## 复查与边界"
        })
        {
            Assert.Contains(heading, article, StringComparison.Ordinal);
        }

        Assert.Contains(
            "https://github.com/onnx/models/tree/main/validated/vision/classification/mnist",
            article,
            StringComparison.Ordinal);
        Assert.Contains(
            "models/OnnxToEngine/MNIST/nvidia-tensorrt-10.11/mnist.onnx",
            article,
            StringComparison.Ordinal);
        Assert.Contains("上游文件已经是 ONNX", article, StringComparison.Ordinal);
        Assert.Contains("--saveRefittedEngine", article, StringComparison.Ordinal);
        Assert.Contains("JYPPX.TensorRtSharp", article, StringComparison.Ordinal);
        Assert.Contains("JYPPX.CudaSharp", article, StringComparison.Ordinal);
        Assert.Contains(
            "../../images/tensorrtexec-refitted-plan-package-consumer-runtime.png",
            article,
            StringComparison.Ordinal);
        Assert.Contains(
            "samples/assets/tensorrtexec-refitted-plan-package-consumer-article-runtime-evidence.json",
            article,
            StringComparison.Ordinal);
        Assert.Contains("53/53", article, StringComparison.Ordinal);
        Assert.Contains("powershell.exe -NoProfile -ExecutionPolicy Bypass", article, StringComparison.Ordinal);
        Assert.Contains("Windows PowerShell 5.1", article, StringComparison.Ordinal);
        Assert.Empty(Regex.Matches(article, @"(?im)[A-Z]:\\"));
    }

    [Fact]
    public void TechnicalArticleEvidenceMatchesTheRuntimeScreenshotAndConsumerSource()
    {
        using JsonDocument document = ReadJson(
            "samples",
            "assets",
            "tensorrtexec-refitted-plan-package-consumer-article-runtime-evidence.json");
        JsonElement root = document.RootElement;
        JsonElement assets = root.GetProperty("assets");
        JsonElement runtime = root.GetProperty("runtimeValidation");

        Assert.Equal(
            "tensorrtexec-refitted-plan-package-consumer-technical-article-runtime-evidence",
            root.GetProperty("recordKind").GetString());
        Assert.Equal("local-package-consumer-runtime", root.GetProperty("proofClassification").GetString());
        Assert.True(runtime.GetProperty("packageReferenceOnly").GetBoolean());
        Assert.False(runtime.GetProperty("usesProjectReference").GetBoolean());
        Assert.True(runtime.GetProperty("enqueueCompleted").GetBoolean());
        Assert.Equal(7, runtime.GetProperty("predictedIndex").GetInt32());
        Assert.Equal(0, runtime.GetProperty("referenceMismatchCount").GetInt32());
        Assert.Equal(53, runtime.GetProperty("strictPassedCount").GetInt32());
        Assert.True(runtime.GetProperty("passed").GetBoolean());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("performsPublish").GetBoolean());

        string screenshotPath = Path.Combine(
            RepositoryPaths.Root,
            assets.GetProperty("runtimeScreenshotPath").GetString()!.Replace('/', Path.DirectorySeparatorChar));
        string programPath = Path.Combine(
            RepositoryPaths.Root,
            assets.GetProperty("consumerProgramPath").GetString()!.Replace('/', Path.DirectorySeparatorChar));
        string runnerPath = Path.Combine(
            RepositoryPaths.Root,
            assets.GetProperty("runnerPath").GetString()!.Replace('/', Path.DirectorySeparatorChar));
        string validatorPath = Path.Combine(
            RepositoryPaths.Root,
            assets.GetProperty("validatorPath").GetString()!.Replace('/', Path.DirectorySeparatorChar));
        Assert.Equal(assets.GetProperty("runtimeScreenshotSha256").GetString(), ComputeSha256(screenshotPath));
        Assert.Equal(assets.GetProperty("consumerProgramSha256").GetString(), ComputeSha256(programPath));
        Assert.Equal(64, assets.GetProperty("runnerSha256").GetString()!.Length);
        Assert.Equal(64, assets.GetProperty("validatorSha256").GetString()!.Length);
        Assert.True(File.Exists(runnerPath));
        Assert.True(File.Exists(validatorPath));
    }

    [Fact]
    public void ApplicationAndSampleEntryPointsAvoidDuplicateOrMachineSpecificGuidance()
    {
        string applications = ReadSource("applications", "README.md");
        string samples = ReadSource("samples", "README.md");
        string sampleAssets = ReadSource("samples", "assets", "README.md");
        string consumer = ReadSource("tests", "fixtures", "package-consumers", "RefittedPlan.PackageConsumer", "README.md");

        Assert.Equal(1, applications.Split('\n').Count(line => line.TrimEnd('\r') == "# Applications"));
        Assert.Contains("tensorrtexec-refitted-plan-local-package-consumer.md", applications, StringComparison.Ordinal);
        Assert.Contains("<workspace-root>/models", samples, StringComparison.Ordinal);
        Assert.Contains("<workspace-root>/models", sampleAssets, StringComparison.Ordinal);
        Assert.Contains("repository-external workspace", consumer, StringComparison.Ordinal);
        Assert.Empty(Regex.Matches(applications + samples + sampleAssets + consumer, @"(?im)[A-Z]:\\"));
    }

    private static JsonDocument ReadJson(params string[] parts)
    {
        string path = Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray());
        Assert.True(File.Exists(path), "Required evidence file is missing: " + path);
        return JsonDocument.Parse(File.ReadAllText(path));
    }

    private static string ReadSource(params string[] parts)
    {
        return File.ReadAllText(Path.Combine(new[] { RepositoryPaths.Root }.Concat(parts).ToArray()));
    }

    private static string ComputeSha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }
}

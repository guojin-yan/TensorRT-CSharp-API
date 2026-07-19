using System.Text.Json;
using System.Text.RegularExpressions;
using System.Xml.Linq;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class SampleLayoutTests
{
    private static readonly Regex ProgramClassPattern = new(
        @"(?m)^\s*(?:internal|public)?\s*(?:static\s+)?(?:sealed\s+)?(?:partial\s+)?class\s+Program\b",
        RegexOptions.CultureInvariant);

    [Fact]
    public void SamplesDirectoryContainsUserFacingProjectsOnly()
    {
        string[] sampleDirectories = GetSampleProjectDirectories();

        Assert.DoesNotContain(sampleDirectories, name => name.Contains("SmokeRunner", StringComparison.OrdinalIgnoreCase));

        string samplesRoot = Path.Combine(RepositoryPaths.Root, "samples");
        foreach (string sampleDirectory in sampleDirectories)
        {
            string directory = Path.Combine(samplesRoot, sampleDirectory);
            string[] projects = Directory.GetFiles(directory, "*.csproj", SearchOption.TopDirectoryOnly);

            Assert.Single(projects);
        }
    }

    [Fact]
    public void SamplesReadmeListsEveryUserFacingProject()
    {
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "README.md"));
        string[] sampleDirectories = GetSampleProjectDirectories();

        Assert.NotEmpty(sampleDirectories);
        foreach (string sampleDirectory in sampleDirectories)
        {
            Assert.Contains($"`{sampleDirectory}`", readme, StringComparison.Ordinal);
        }

        Assert.DoesNotContain("SmokeRunner", readme, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void SmokeReadmeListsEveryValidationProject()
    {
        string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "smoke", "README.md"));
        string[] smokeDirectories = Directory.GetDirectories(Path.Combine(RepositoryPaths.Root, "smoke"))
            .Where(directory => Directory.GetFiles(directory, "*.csproj", SearchOption.TopDirectoryOnly).Length == 1)
            .Select(Path.GetFileName)
            .Where(name => !string.IsNullOrWhiteSpace(name))
            .Select(name => name!)
            .OrderBy(static name => name, StringComparer.Ordinal)
            .ToArray();

        Assert.NotEmpty(smokeDirectories);
        foreach (string smokeDirectory in smokeDirectories)
        {
            Assert.Contains($"`{smokeDirectory}`", readme, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void RootReadmesDoNotAdvertiseRemovedSamplePlaceholders()
    {
        foreach (string fileName in new[] { "README.md", "README.zh-CN.md" })
        {
            string readme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, fileName));

            Assert.DoesNotContain("CustomKernelPreprocess", readme, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("`Classification`", readme, StringComparison.Ordinal);
            Assert.Contains("`YoloVision`", readme, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ActiveSampleReferencesUseYoloVisionProjectName()
    {
        string root = RepositoryPaths.Root;
        string[] activeFiles =
        {
            Path.Combine(root, "samples", "README.md"),
            Path.Combine(root, "README.md"),
            Path.Combine(root, "README.zh-CN.md"),
            Path.Combine(root, "docs", "articles", "en", "getting-started.md"),
            Path.Combine(root, "docs", "articles", "en", "sample-runners.md"),
            Path.Combine(root, "docs", "articles", "zh-cn", "getting-started.md"),
            Path.Combine(root, "docs", "articles", "zh-cn", "sample-runners.md"),
            Path.Combine(root, "docs", "articles", "zh-cn", "technical-article-roadmap.md"),
            Path.Combine(root, "eng", "Export-UserAcceptanceSampleCatalog.ps1"),
            Path.Combine(root, "tests", "JYPPX.ProjectQuality.Tests", "JYPPX.ProjectQuality.Tests.csproj"),
            Path.Combine(root, "samples", "YoloVision", "YoloVision.csproj"),
        };
        string[] publicYoloVisionEntryFiles =
        {
            Path.Combine(root, "samples", "README.md"),
            Path.Combine(root, "README.md"),
            Path.Combine(root, "README.zh-CN.md"),
            Path.Combine(root, "docs", "articles", "en", "getting-started.md"),
            Path.Combine(root, "docs", "articles", "en", "sample-runners.md"),
            Path.Combine(root, "docs", "articles", "zh-cn", "getting-started.md"),
            Path.Combine(root, "docs", "articles", "zh-cn", "sample-runners.md"),
            Path.Combine(root, "docs", "articles", "zh-cn", "technical-article-roadmap.md"),
            Path.Combine(root, "eng", "Export-UserAcceptanceSampleCatalog.ps1"),
            Path.Combine(root, "samples", "YoloVision", "YoloVision.csproj"),
        };

        foreach (string activeFile in activeFiles)
        {
            string content = File.ReadAllText(activeFile);

            Assert.DoesNotContain("YoloDet", content, StringComparison.OrdinalIgnoreCase);
        }

        foreach (string activeFile in publicYoloVisionEntryFiles)
        {
            string content = File.ReadAllText(activeFile);

            Assert.Contains("YoloVision", content, StringComparison.Ordinal);
        }

        Assert.False(Directory.Exists(Path.Combine(root, "samples", "YoloDet")));
        Assert.True(Directory.Exists(Path.Combine(root, "samples", "YoloVision")));
        Assert.False(File.Exists(Path.Combine(root, "samples", "YoloVision", "YoloDet.csproj")));
        Assert.True(File.Exists(Path.Combine(root, "samples", "YoloVision", "YoloVision.csproj")));
    }

    [Fact]
    public void YoloVisionProjectIdentityIsStable()
    {
        string sampleRoot = Path.Combine(RepositoryPaths.Root, "samples", "YoloVision");
        string projectPath = Path.Combine(sampleRoot, "YoloVision.csproj");
        XDocument project = XDocument.Load(projectPath);

        string assemblyName = GetProjectProperty(project, "AssemblyName");
        string rootNamespace = GetProjectProperty(project, "RootNamespace");

        Assert.Equal("YoloVision", assemblyName);
        Assert.Equal("YoloVisionSample", rootNamespace);
        Assert.True(Directory.Exists(sampleRoot));
        Assert.False(Directory.Exists(Path.Combine(RepositoryPaths.Root, "samples", "YoloDet")));

        string[] staleBuildArtifacts = Directory.EnumerateFiles(sampleRoot, "*", SearchOption.AllDirectories)
            .Select(Path.GetFileName)
            .Where(name => !string.IsNullOrWhiteSpace(name))
            .Select(name => name!)
            .Where(static name =>
                name.StartsWith("YoloDet.", StringComparison.OrdinalIgnoreCase) ||
                name.StartsWith("YoloDet.csproj", StringComparison.OrdinalIgnoreCase))
            .OrderBy(static name => name, StringComparer.OrdinalIgnoreCase)
            .ToArray();

        Assert.Empty(staleBuildArtifacts);
    }

    [Fact]
    public void YoloVisionAssetEvidencePreservesTemplateBoundaryAndAcceptsStrictYoloXRuntimeProof()
    {
        string root = RepositoryPaths.Root;
        string templatePath = Path.Combine(root, "samples", "assets", "yolovision-assets.template.json");
        string examplePath = Path.Combine(root, "samples", "assets", "yolovision-yolox-s-example.json");
        using JsonDocument template = JsonDocument.Parse(File.ReadAllText(templatePath));
        using JsonDocument example = JsonDocument.Parse(File.ReadAllText(examplePath));

        AssertYoloVisionTemplateManifestBoundary(
            template.RootElement,
            expectedProofClassification: "template-only",
            expectedRecord: "models/yolovision-sample-run-evidence.json");
        Assert.Equal("YoloVision", example.RootElement.GetProperty("sampleName").GetString());
        Assert.Equal("real-model-runtime", example.RootElement.GetProperty("proofClassification").GetString());
        Assert.True(example.RootElement.GetProperty("isSmokePassed").GetBoolean());
        Assert.False(example.RootElement.GetProperty("isRedistributableInRepository").GetBoolean());
        Assert.Equal("yolox", example.RootElement.GetProperty("model").GetProperty("family").GetString());
        Assert.Equal("upstream-repository-license-hash-pinned", example.RootElement.GetProperty("model").GetProperty("licenseEvidence").GetString());
        JsonElement exampleEvidence = example.RootElement.GetProperty("evidence");
        Assert.Equal("real-model-runtime", exampleEvidence.GetProperty("sampleRunEvidenceValidatorState").GetString());
        Assert.True(exampleEvidence.GetProperty("sampleRunEvidenceCanPromoteRealModelRuntime").GetBoolean());
        Assert.Equal("passed", exampleEvidence.GetProperty("lastRunStatus").GetString());
        Assert.False(example.RootElement.GetProperty("boundary").GetProperty("isPackageConsumerRuntime").GetBoolean());

        string templateText = File.ReadAllText(templatePath);
        string exampleText = File.ReadAllText(examplePath);
        Assert.DoesNotContain("YoloDet", templateText, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("YoloDet", exampleText, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("dotnet run --project .\\\\samples\\\\YoloVision", templateText, StringComparison.Ordinal);
        Assert.Contains("dotnet run --project .\\\\samples\\\\YoloVision", exampleText, StringComparison.Ordinal);
    }

    [Fact]
    public void YoloVisionRealModelProofTemplatesKeepAssetHashesTasksAndPromotionBoundary()
    {
        string root = RepositoryPaths.Root;
        string readme = File.ReadAllText(Path.Combine(root, "samples", "YoloVision", "README.md"));
        string assetTemplate = File.ReadAllText(Path.Combine(root, "samples", "assets", "yolovision-assets.template.json"));
        string sidecarTemplate = File.ReadAllText(Path.Combine(root, "artifacts", "user-acceptance", "onnx-engine-build-evidence-sidecar.yolovision.template.json"));
        string runEvidenceTemplate = File.ReadAllText(Path.Combine(root, "artifacts", "user-acceptance", "sample-run-evidence-record.yolovision.template.json"));
        string combined = readme + assetTemplate + sidecarTemplate + runEvidenceTemplate;

        foreach (string marker in new[]
        {
            "--task det",
            "--task cls",
            "--task seg",
            "--task obb",
            "--task pose",
            "--task sem",
            "YOLO v5/v6/v7/v8/v9/v10/v11/v26/YOLOX/custom",
            "model SHA256",
            "labels SHA256",
            "image SHA256",
            "preprocessed tensor SHA256",
            "run log SHA256",
            "stdout/stderr summaries",
            "YoloVision Passed=True",
            "canPromoteRealModelRuntime"
        })
        {
            Assert.Contains(marker, combined, StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument asset = JsonDocument.Parse(assetTemplate);
        using JsonDocument sidecar = JsonDocument.Parse(sidecarTemplate);
        using JsonDocument runEvidence = JsonDocument.Parse(runEvidenceTemplate);

        Assert.False(asset.RootElement.GetProperty("isSmokePassed").GetBoolean());
        Assert.Equal("template-only", asset.RootElement.GetProperty("proofClassification").GetString());
        Assert.False(runEvidence.RootElement.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.Equal("template-only", runEvidence.RootElement.GetProperty("proofClassification").GetString());
        Assert.Equal("template-only", sidecar.RootElement.GetProperty("proofClassification").GetString());

        Assert.Contains(runEvidence.RootElement.GetProperty("promotionRules").EnumerateArray(), static item =>
            (item.GetString() ?? string.Empty).Contains("never package-consumer-runtime", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(sidecar.RootElement.GetProperty("promotionRules").EnumerateArray(), static item =>
            (item.GetString() ?? string.Empty).Contains("cannot be claimed by TensorRtExec/OnnxToEngine build reports", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void ProgramFilesDoNotUseTopLevelStatements()
    {
        string[] programFiles = Directory.EnumerateFiles(Path.Combine(RepositoryPaths.Root, "samples"), "Program.cs", SearchOption.AllDirectories)
            .Concat(Directory.EnumerateFiles(Path.Combine(RepositoryPaths.Root, "smoke"), "Program.cs", SearchOption.AllDirectories))
            .OrderBy(static path => path, StringComparer.Ordinal)
            .ToArray();

        Assert.NotEmpty(programFiles);

        foreach (string programFile in programFiles)
        {
            string source = File.ReadAllText(programFile);
            string firstSignificantLine = GetFirstSignificantLine(source);

            Assert.Matches(ProgramClassPattern, source);
            Assert.True(
                firstSignificantLine.StartsWith("namespace ", StringComparison.Ordinal) ||
                firstSignificantLine.Contains(" class Program", StringComparison.Ordinal),
                $"Program file appears to use top-level statements: {programFile}");
        }
    }

    [Fact]
    public void OnnxSampleSupportExposesMultiOutputSnapshotsForYoloVision()
    {
        string support = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "JYPPX.SampleSupport", "TensorRtOnnxSample.cs"));
        string yoloRunner = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "YoloSampleRunner.cs"));
        string yoloProgram = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "YoloVision", "Program.cs"));

        Assert.Contains("internal sealed class OnnxSampleOutputTensor", support, StringComparison.Ordinal);
        Assert.Contains("internal sealed class OnnxSampleMultiOutputResult", support, StringComparison.Ordinal);
        Assert.Contains("RunSingleFloatInputOutputs", support, StringComparison.Ordinal);
        Assert.Contains("ResolveOutputNames", support, StringComparison.Ordinal);
        Assert.Contains("CreateInputValuesForTesting", support, StringComparison.Ordinal);
        Assert.Contains("ReadFloatInputData", support, StringComparison.Ordinal);
        Assert.Contains("DecodeRuntimeOutputs", yoloRunner, StringComparison.Ordinal);
        Assert.Contains("YoloRuntimeOutputSet", yoloProgram, StringComparison.Ordinal);
        Assert.Contains("--input-data <path>", yoloProgram, StringComparison.Ordinal);
        Assert.Contains("--list-capabilities", yoloProgram, StringComparison.Ordinal);
        Assert.Contains("YoloCapabilityMatrix.FormatConsoleTable()", yoloProgram, StringComparison.Ordinal);
    }

    private static string GetFirstSignificantLine(string source)
    {
        foreach (string rawLine in source.Split(new[] { "\r\n", "\n" }, StringSplitOptions.None))
        {
            string line = rawLine.Trim();
            if (line.Length == 0 ||
                line.StartsWith("using ", StringComparison.Ordinal) ||
                line.StartsWith("//", StringComparison.Ordinal) ||
                line.StartsWith("#", StringComparison.Ordinal))
            {
                continue;
            }

            return line;
        }

        return string.Empty;
    }

    private static string GetProjectProperty(XDocument project, string propertyName)
    {
        return project.Root!
            .Elements("PropertyGroup")
            .Elements(propertyName)
            .Select(static element => element.Value)
            .Single();
    }

    private static void AssertYoloVisionTemplateManifestBoundary(JsonElement root, string expectedProofClassification, string expectedRecord)
    {
        Assert.Equal("YoloVision", root.GetProperty("sampleName").GetString());
        Assert.Equal(expectedProofClassification, root.GetProperty("proofClassification").GetString());
        Assert.False(root.GetProperty("isSmokePassed").GetBoolean());
        Assert.False(root.GetProperty("isRedistributableInRepository").GetBoolean());
        Assert.Equal("owner-required", root.GetProperty("model").GetProperty("licenseEvidence").GetString());
        Assert.Equal("owner-required", root.GetProperty("labels").GetProperty("licenseEvidence").GetString());
        Assert.Equal("owner-required", root.GetProperty("input").GetProperty("licenseEvidence").GetString());
        Assert.True(root.GetProperty("inputTensor").TryGetProperty("elementCountNumeric", out JsonElement elementCount));
        Assert.True(elementCount.GetInt32() >= 0);

        JsonElement evidence = root.GetProperty("evidence");
        Assert.Equal(expectedRecord, evidence.GetProperty("sampleRunEvidenceRecord").GetString());
        Assert.Equal(
            "artifacts/user-acceptance/sample-run-evidence-record-validation.json",
            evidence.GetProperty("sampleRunEvidenceValidation").GetString());
        Assert.Equal("owner-action-required", evidence.GetProperty("sampleRunEvidenceValidatorState").GetString());
        Assert.False(evidence.GetProperty("sampleRunEvidenceCanPromoteRealModelRuntime").GetBoolean());
        Assert.Contains(evidence.GetProperty("sampleRunEvidenceFailureReasons").EnumerateArray(), static item =>
            item.GetString()!.Contains("validator must pass", StringComparison.Ordinal));
        Assert.Contains(".\\applications\\TensorRtExec", evidence.GetProperty("buildOnlyCommand").GetString(), StringComparison.Ordinal);
        Assert.Contains(".\\samples\\YoloVision", evidence.GetProperty("runCommand").GetString(), StringComparison.Ordinal);
        Assert.Equal("not-run", evidence.GetProperty("lastRunStatus").GetString());
        Assert.Equal(string.Empty, evidence.GetProperty("lastRunLog").GetString());
        Assert.Equal(string.Empty, evidence.GetProperty("lastRunLogSha256").GetString());

        Assert.Contains(evidence.GetProperty("expectedEvidenceLines").EnumerateArray(), static item =>
            item.GetString() == "YoloVision Passed=True");
        Assert.Contains(root.GetProperty("evidenceClassifications").EnumerateArray(), static item =>
            item.GetString() == "real-model-runtime");
        Assert.Contains(root.GetProperty("evidenceClassifications").EnumerateArray(), static item =>
            item.GetString() == "package-consumer-runtime");
    }

    private static string[] GetSampleProjectDirectories()
    {
        return Directory.GetDirectories(Path.Combine(RepositoryPaths.Root, "samples"))
            .Where(directory => Directory.GetFiles(directory, "*.csproj", SearchOption.TopDirectoryOnly).Length == 1)
            .Select(Path.GetFileName)
            .Where(name => !string.IsNullOrWhiteSpace(name))
            .Select(name => name!)
            .Where(name => !string.Equals(name, "JYPPX.SampleSupport", StringComparison.Ordinal))
            .OrderBy(static name => name, StringComparer.Ordinal)
            .ToArray();
    }
}

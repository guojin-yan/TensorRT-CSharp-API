using System.Security.Cryptography;
using System.Text.Json;
using YoloVisionSample;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class YoloVisionSemanticSegmentationTests
{
    [Fact]
    public void PreprocessOptionsPreserveLegacyDefaultsAndParseImageNetNormalization()
    {
        Assert.Equal(new[] { 0.0f, 0.0f, 0.0f }, YoloPreprocessOptions.Default.Mean);
        Assert.Equal(new[] { 1.0f, 1.0f, 1.0f }, YoloPreprocessOptions.Default.StandardDeviation);
        Assert.Equal(64, YoloPreprocessOptions.Default.ContractSha256.Length);

        YoloModelProfile profile = YoloModelProfile.FromArgs(new[]
        {
            "--task", "sem",
            "--class-count", "21",
            "--input-shape", "1x3x320x320",
            "--resize", "stretch",
            "--mean", "0.485,0.456,0.406",
            "--std", "0.229,0.224,0.225"
        }, labelCount: 0);

        Assert.Equal(new[] { 0.485f, 0.456f, 0.406f }, profile.Preprocess.Mean);
        Assert.Equal(new[] { 0.229f, 0.224f, 0.225f }, profile.Preprocess.StandardDeviation);
        Assert.Equal("stretch", profile.Preprocess.ResizeMode);
        Assert.Equal(64, profile.Preprocess.ContractSha256.Length);
        Assert.Throws<ArgumentException>(() => YoloModelProfile.FromArgs(
            new[] { "--task", "sem", "--mean", "0.1,0.2" },
            labelCount: 21));
        Assert.Throws<ArgumentOutOfRangeException>(() => new YoloPreprocessOptions(
            "NCHW", "RGB", "stretch", 1.0f / 255.0f, true, false, "center", 0,
            new float[3], new[] { 1.0f, 0.0f, 1.0f }));
    }

    [Fact]
    public void ImagePreprocessorAppliesScaleMeanAndStandardDeviationPerOutputChannel()
    {
        string directory = Path.Combine(Path.GetTempPath(), "jyppx-yolovision-semantic-normalization", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        string imagePath = Path.Combine(directory, "pixel.ppm");
        string tensorPath = Path.Combine(directory, "pixel.fp32.bin");
        try
        {
            File.WriteAllText(imagePath, "P3\n1 1\n255\n255 128 0\n");
            YoloPreprocessOptions options = new YoloPreprocessOptions(
                "NCHW",
                "RGB",
                "stretch",
                1.0f / 255.0f,
                normalize: true,
                preserveAspectRatio: false,
                letterboxAlignment: "center",
                resizeShorterSide: 0,
                mean: new[] { 0.5f, 0.25f, 0.0f },
                standardDeviation: new[] { 0.5f, 0.25f, 2.0f });

            YoloImagePreprocessResult result = YoloImagePreprocessor.Preprocess(
                imagePath,
                tensorPath,
                new[] { 1, 3, 1, 1 },
                options);

            byte[] bytes = File.ReadAllBytes(tensorPath);
            float[] tensor = new float[3];
            Buffer.BlockCopy(bytes, 0, tensor, 0, bytes.Length);
            Assert.Equal(1.0f, tensor[0], precision: 6);
            Assert.Equal((128.0f / 255.0f - 0.25f) / 0.25f, tensor[1], precision: 6);
            Assert.Equal(0.0f, tensor[2], precision: 6);
            Assert.Equal(options.Mean, result.Mean);
            Assert.Equal(options.StandardDeviation, result.StandardDeviation);
            Assert.Equal(options.ContractSha256, result.PreprocessContractSha256);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void SemanticDecoderRequiresConfiguredUnambiguousClassDimension()
    {
        YoloModelProfile profile = SemanticProfile(classCount: 2);

        YoloSemanticMap nchw = YoloSampleRunner.DecodeSemanticMap(
            new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f },
            new[] { 1, 2, 1, 3 },
            profile);
        YoloSemanticMap nhwc = YoloSampleRunner.DecodeSemanticMap(
            new[] { 1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f },
            new[] { 1, 1, 3, 2 },
            profile);

        Assert.Equal(nchw.Values, nhwc.Values);
        Assert.Throws<ArgumentException>(() => YoloSampleRunner.DecodeSemanticMap(
            new float[4],
            new[] { 1, 2, 1, 2 },
            profile));
        Assert.Throws<ArgumentException>(() => YoloSampleRunner.DecodeSemanticMap(
            new float[3],
            new[] { 3, 1, 1 },
            profile));
        Assert.Throws<ArgumentException>(() => YoloSampleRunner.DecodeSemanticMap(
            new float[6],
            new[] { 1, 2, 1, 3 },
            SemanticProfile(classCount: 0)));
    }

    [Fact]
    public void SemanticMapRejectsNonFiniteLogitsAndProvidesDeterministicArgmaxHistogram()
    {
        YoloSemanticMap map = new YoloSemanticMap(
            classCount: 3,
            width: 2,
            height: 1,
            values: new[]
            {
                1.0f, 0.0f,
                2.0f, 3.0f,
                0.0f, 3.0f
            });

        Assert.Equal(1, map.GetClassIndex(0, 0));
        Assert.Equal(1, map.GetClassIndex(1, 0));
        Assert.Equal(new[] { 1, 1 }, map.GetClassIndexMap());
        Assert.Equal(new[] { 0, 2, 0 }, map.GetClassHistogram());
        Assert.Throws<ArgumentOutOfRangeException>(() => map.GetClassIndex(2, 0));
        Assert.Throws<ArgumentException>(() => new YoloSemanticMap(1, 1, 1, new[] { float.NaN }));
        Assert.Throws<ArgumentException>(() => YoloSampleRunner.DecodeSemanticMap(
            new[] { 0.0f, float.PositiveInfinity },
            new[] { 2, 1, 1 },
            SemanticProfile(classCount: 2)));
    }

    [Fact]
    public void SemanticArtifactWriterEmitsLittleEndianClassMapHashAndCompleteHistogram()
    {
        string directory = Path.Combine(Path.GetTempPath(), "jyppx-yolovision-semantic-artifacts", Guid.NewGuid().ToString("N"));
        try
        {
            YoloSemanticMap map = new YoloSemanticMap(
                2,
                2,
                2,
                new[]
                {
                    3.0f, 0.0f, 4.0f, 1.0f,
                    1.0f, 2.0f, 0.0f, 5.0f
                });
            string manifestPath = YoloSemanticMapArtifactWriter.Write(
                directory,
                YoloVisionResult.FromSemanticMap(map),
                new[] { "background", "foreground" });

            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifestPath));
            JsonElement root = document.RootElement;
            JsonElement artifact = root.GetProperty("classIndexArtifact");
            string classIndexPath = artifact.GetProperty("path").GetString()!;
            byte[] bytes = File.ReadAllBytes(classIndexPath);

            Assert.Equal(YoloSemanticMapArtifactWriter.SchemaVersion, root.GetProperty("schemaVersion").GetString());
            Assert.Equal(new[] { 0, 1, 0, 1 }, ReadInt32LittleEndian(bytes));
            Assert.Equal(16, artifact.GetProperty("byteLength").GetInt64());
            Assert.Equal(
                Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant(),
                artifact.GetProperty("sha256").GetString());
            Assert.Equal(new[] { 2, 2 }, root.GetProperty("classHistogram").EnumerateArray()
                .Select(item => item.GetProperty("pixelCount").GetInt32()).ToArray());
            Assert.False(root.GetProperty("boundary").GetProperty("isRuntimeProof").GetBoolean());
        }
        finally
        {
            if (Directory.Exists(directory))
            {
                Directory.Delete(directory, recursive: true);
            }
        }
    }

    [Fact]
    public void SemanticArtifactSchemaAndCliSurfaceRemainExplicit()
    {
        string schemaPath = Path.Combine(
            RepositoryPaths.Root,
            "applications", "YoloVision",
            "yolovision-semantic-map-artifacts.schema.json");
        using JsonDocument schema = JsonDocument.Parse(File.ReadAllText(schemaPath));
        Assert.Equal(
            YoloSemanticMapArtifactWriter.SchemaVersion,
            schema.RootElement.GetProperty("properties").GetProperty("schemaVersion").GetProperty("const").GetString());

        string program = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "YoloVision", "Program.cs"));
        Assert.Contains("--semantic-artifact-output-directory", program, StringComparison.Ordinal);
        Assert.Contains("--mean <r,g,b> --std <r,g,b>", program, StringComparison.Ordinal);
        Assert.DoesNotContain("--semantic-map-output-directory", program, StringComparison.Ordinal);
    }

    [Fact]
    public void OfficialLrasppManifestPinsAcquisitionExportAndExternalModelStorage()
    {
        string path = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "yolovision-torchvision-lraspp-official-assets.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        JsonElement root = document.RootElement;

        Assert.Equal(
            "yolovision-torchvision-lraspp-official-asset-acquisition-manifest",
            root.GetProperty("recordKind").GetString());
        Assert.Equal("v0.25.0", root.GetProperty("upstreamSourceTag").GetString());
        Assert.Equal(4, root.GetProperty("assets").GetArrayLength());
        Assert.Equal(
            "3cb94e561bdefe606ed7d1a2c4d0296409bec066f3a39a9fe9dabd72b23728f8",
            root.GetProperty("exportContract").GetProperty("onnxSha256").GetString());
        Assert.True(root.GetProperty("acquisition").GetProperty("modelRootOutsideGitRepository").GetBoolean());
        Assert.Contains(
            "/models/YoloVision/SemanticSegmentation/",
            root.GetProperty("acquisition").GetProperty("defaultModelRoot").GetString(),
            StringComparison.Ordinal);
        Assert.False(root.GetProperty("proofBoundary").GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("uploadsAssets").GetBoolean());

        string acquisition = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Acquire-TorchVisionLrasppOfficialAssets.ps1"));
        string reference = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "eng",
            "Invoke-YoloVisionSemanticReference.py"));
        Assert.Contains("download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth", acquisition, StringComparison.Ordinal);
        Assert.Contains("models\\YoloVision\\SemanticSegmentation", acquisition, StringComparison.Ordinal);
        Assert.Contains("--export-onnx", reference, StringComparison.Ordinal);
        Assert.Contains("lraspp_mobilenet_v3_large", reference, StringComparison.Ordinal);
        Assert.DoesNotContain("github upload", acquisition, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void OfficialLrasppRuntimeEvidenceClosesRawArgmaxAndControlledNegativeContracts()
    {
        string path = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "yolovision-torchvision-lraspp-real-model-runtime-evidence.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        JsonElement root = document.RootElement;
        JsonElement raw = root.GetProperty("runtimeReferenceValidation");
        JsonElement tensor = raw.GetProperty("tensors")[0];
        JsonElement semantic = root.GetProperty("semanticPostprocessValidation");
        JsonElement negative = root.GetProperty("controlledNegativeValidation");

        Assert.Equal("real-model-runtime", root.GetProperty("proofClassification").GetString());
        Assert.True(root.GetProperty("isSmokePassed").GetBoolean());
        Assert.True(raw.GetProperty("tf32Disabled").GetBoolean());
        Assert.True(raw.GetProperty("passed").GetBoolean());
        Assert.Equal(2_150_400, tensor.GetProperty("comparedValueCount").GetInt32());
        Assert.Equal(0, tensor.GetProperty("mismatchCount").GetInt32());
        Assert.Equal(102_400, semantic.GetProperty("pixelCount").GetInt32());
        Assert.Equal(0, semantic.GetProperty("mismatchCount").GetInt32());
        Assert.Equal(65_193, semantic.GetProperty("histogram")[0].GetProperty("pixelCount").GetInt32());
        Assert.Equal(37_207, semantic.GetProperty("histogram")[1].GetProperty("pixelCount").GetInt32());
        Assert.Equal(1, negative.GetProperty("exitCode").GetInt32());
        Assert.Equal(1, negative.GetProperty("mismatchCount").GetInt32());
        Assert.True(negative.GetProperty("failClosed").GetBoolean());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("publicRedistributionApproved").GetBoolean());
        Assert.False(root.GetProperty("proofBoundary").GetProperty("uploadsAssets").GetBoolean());
    }

    [Fact]
    public void StrictParityCanDisableTensorRtTf32WithoutChangingTheDefaultPath()
    {
        string source = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "_shared",
            "JYPPX.SampleSupport",
            "TensorRtOnnxSample.cs"));
        Assert.Contains("DisableTf32", source, StringComparison.Ordinal);
        Assert.Contains("--noTF32", source, StringComparison.Ordinal);
        Assert.Contains("config.SetFlag(TensorRtBuilderFlag.Tf32, false)", source, StringComparison.Ordinal);
        Assert.Contains("if (options.DisableTf32)", source, StringComparison.Ordinal);
    }

    private static YoloModelProfile SemanticProfile(int classCount)
    {
        return YoloModelProfile.FromArgs(new[]
        {
            "--task", "sem",
            "--class-count", classCount.ToString(System.Globalization.CultureInfo.InvariantCulture)
        }, labelCount: 0);
    }

    private static int[] ReadInt32LittleEndian(byte[] bytes)
    {
        Assert.Equal(0, bytes.Length % sizeof(int));
        int[] values = new int[bytes.Length / sizeof(int)];
        for (int index = 0; index < values.Length; index++)
        {
            values[index] = System.Buffers.Binary.BinaryPrimitives.ReadInt32LittleEndian(
                bytes.AsSpan(index * sizeof(int), sizeof(int)));
        }
        return values;
    }
}

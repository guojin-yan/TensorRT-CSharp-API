using System.Buffers.Binary;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using ClassificationSample;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class ClassificationRuntimeArtifactTests
{
    [Fact]
    public void P3P6AndBmpImagesUseTheSharedRgbDecoder()
    {
        string directory = CreateTempDirectory("classification-images");
        try
        {
            string p3 = Path.Combine(directory, "image-p3.ppm");
            string p6 = Path.Combine(directory, "image-p6.ppm");
            string bmp = Path.Combine(directory, "image.bmp");
            File.WriteAllText(p3, "P3\n2 1\n255\n255 0 0 0 255 0\n", Encoding.ASCII);
            File.WriteAllBytes(p6, Encoding.ASCII.GetBytes("P6\r\n2 1\r\n255\r\n").Concat(new byte[] { 255, 0, 0, 0, 255, 0 }).ToArray());
            File.WriteAllBytes(bmp, CreateBmp24(new byte[] { 255, 0, 0, 0, 255, 0 }, 2, 1));

            ClassificationPreprocessOptions options = StretchOptions("NCHW", "RGB", 1.0f);
            foreach (string imagePath in new[] { p3, p6, bmp })
            {
                string tensorPath = Path.Combine(directory, Path.GetFileName(imagePath) + ".bin");
                ClassificationImagePreprocessResult result = ClassificationImagePreprocessor.Preprocess(
                    imagePath, tensorPath, new[] { 1, 3, 1, 2 }, options);
                float[] values = ReadFloats(tensorPath);
                Assert.True(values.SequenceEqual(new[] { 255.0f, 0.0f, 0.0f, 255.0f, 0.0f, 0.0f }),
                    $"{Path.GetFileName(imagePath)}: {string.Join(",", values.Select(value => value.ToString(CultureInfo.InvariantCulture)))}");
                Assert.Equal(2, result.SourceWidth);
                Assert.Equal(1, result.SourceHeight);
                Assert.Equal(64, result.SourceSha256.Length);
                Assert.Equal(64, result.TensorSha256.Length);
            }

            string invalidP6 = Path.Combine(directory, "invalid-max.ppm");
            File.WriteAllBytes(invalidP6, Encoding.ASCII.GetBytes("P6\n1 1\n15\n").Concat(new byte[] { 16, 0, 0 }).ToArray());
            Assert.Throws<ArgumentException>(() => ClassificationImagePreprocessor.Preprocess(
                invalidP6, Path.Combine(directory, "invalid.bin"), new[] { 1, 3, 1, 1 }, options));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void PreprocessHonorsStretchCenterCropLayoutColorAndNormalization()
    {
        string directory = CreateTempDirectory("classification-preprocess");
        try
        {
            string imagePath = Path.Combine(directory, "image.ppm");
            File.WriteAllText(imagePath, "P3\n2 1\n255\n255 0 0 0 255 0\n", Encoding.ASCII);
            ClassificationPreprocessOptions nhwcBgr = new ClassificationPreprocessOptions(
                "stretch", 2, "NHWC", "BGR", 0.1f, new[] { 1.0f, 2.0f, 3.0f }, new[] { 2.0f, 4.0f, 5.0f });
            string tensorPath = Path.Combine(directory, "nhwc.bin");
            ClassificationImagePreprocessResult result = ClassificationImagePreprocessor.Preprocess(
                imagePath, tensorPath, new[] { 1, 1, 2, 3 }, nhwcBgr);

            Assert.Equal("NHWC", result.Options.TensorLayout);
            Assert.Equal("BGR", result.Options.ColorOrder);
            Assert.Equal(new[] { -0.5f, -0.5f, 4.5f, -0.5f, 5.875f, -0.6f }, ReadFloats(tensorPath));

            ClassificationPreprocessOptions crop = new ClassificationPreprocessOptions(
                "shorter-side-center-crop", 4, "NCHW", "RGB", 1.0f, new float[3], new[] { 1.0f, 1.0f, 1.0f });
            ClassificationImagePreprocessResult cropResult = ClassificationImagePreprocessor.Preprocess(
                imagePath, Path.Combine(directory, "crop.bin"), new[] { 1, 3, 2, 2 }, crop);
            Assert.Equal(8, cropResult.ResizedWidth);
            Assert.Equal(4, cropResult.ResizedHeight);
            Assert.Equal(3, cropResult.CropX);
            Assert.Equal(1, cropResult.CropY);

            string squarePath = Path.Combine(directory, "square.ppm");
            File.WriteAllText(squarePath, "P3\n2 2\n255\n255 0 0 0 255 0 0 0 255 255 255 255\n", Encoding.ASCII);
            ClassificationPreprocessOptions insufficientCrop = new ClassificationPreprocessOptions(
                "shorter-side-center-crop", 2, "NCHW", "RGB", 1.0f, new float[3], new[] { 1.0f, 1.0f, 1.0f });
            Assert.Throws<ArgumentException>(() => ClassificationImagePreprocessor.Preprocess(
                squarePath, Path.Combine(directory, "insufficient.bin"), new[] { 1, 3, 3, 2 }, insufficientCrop));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void PreprocessContractHashIsStableAndSensitiveToSemanticOptions()
    {
        ClassificationPreprocessOptions first = new ClassificationPreprocessOptions(
            "center-crop", 256, "channels-first", "rgb", 1.0f / 255.0f,
            new[] { 0.485f, 0.456f, 0.406f }, new[] { 0.229f, 0.224f, 0.225f });
        ClassificationPreprocessOptions second = new ClassificationPreprocessOptions(
            "shorter-side-center-crop", 256, "NCHW", "RGB", 1.0f / 255.0f,
            new[] { 0.485f, 0.456f, 0.406f }, new[] { 0.229f, 0.224f, 0.225f });
        ClassificationPreprocessOptions changed = new ClassificationPreprocessOptions(
            "shorter-side-center-crop", 256, "NCHW", "BGR", 1.0f / 255.0f,
            new[] { 0.485f, 0.456f, 0.406f }, new[] { 0.229f, 0.224f, 0.225f });

        Assert.Equal(first.ContractSha256, second.ContractSha256);
        Assert.Equal(64, first.ContractSha256.Length);
        Assert.NotEqual(first.ContractSha256, changed.ContractSha256);
        Assert.Contains("interpolation=bilinear-half-pixel", first.ToCanonicalString(), StringComparison.Ordinal);
    }

    [Fact]
    public void SoftmaxAndTopKAreDeterministicForTies()
    {
        float[] probabilities = ClassificationOutputProcessor.Transform(new[] { 2.0f, 2.0f, 0.0f }, ClassificationScoreTransform.Softmax);
        Assert.InRange(probabilities.Sum(), 0.99999f, 1.00001f);
        Assert.True(probabilities[0] > probabilities[2]);

        IReadOnlyList<ClassificationPrediction> top = ClassificationOutputProcessor.GetTopK(
            new[] { 0.5f, 0.5f, 0.2f }, new[] { "zero", "one", "two" }, 2);
        Assert.Equal(new[] { 0, 1 }, top.Select(item => item.Index).ToArray());
        Assert.Equal(new[] { "zero", "one" }, top.Select(item => item.Label).ToArray());
        Assert.Throws<ArgumentException>(() => ClassificationOutputProcessor.Transform(new[] { float.NaN }, ClassificationScoreTransform.Softmax));
    }

    [Fact]
    public void VisualizationEmbedsTheRealImageAndTopKPredictions()
    {
        string directory = CreateTempDirectory("classification-visualization");
        try
        {
            string imagePath = Path.Combine(directory, "image.bmp");
            File.WriteAllBytes(imagePath, CreateBmp24(new byte[] { 255, 0, 0, 0, 255, 0 }, 2, 1));
            ClassificationImagePreprocessResult preprocess = ClassificationImagePreprocessor.Preprocess(
                imagePath,
                Path.Combine(directory, "input.bin"),
                new[] { 1, 3, 1, 2 },
                StretchOptions("NCHW", "RGB", 1.0f));
            string outputPath = Path.Combine(directory, "classification.svg");

            ClassificationVisualizationWriter.Write(
                outputPath,
                imagePath,
                preprocess,
                new[]
                {
                    new ClassificationPrediction(1, "dog & friend", 0.75f),
                    new ClassificationPrediction(2, "second", 0.25f)
                });

            string svg = File.ReadAllText(outputPath);
            Assert.Contains("data-source-image=\"true\"", svg, StringComparison.Ordinal);
            Assert.Contains("data:image/bmp;base64,", svg, StringComparison.Ordinal);
            Assert.Contains("Top-2 predictions from the real TensorRT execution", svg, StringComparison.Ordinal);
            Assert.Contains("1. dog &amp; friend", svg, StringComparison.Ordinal);
            Assert.Contains("0.7500", svg, StringComparison.Ordinal);

            string mismatched = Path.Combine(directory, "mismatched.bmp");
            File.WriteAllBytes(mismatched, CreateBmp24(new byte[] { 255, 0, 0 }, 1, 1));
            Assert.Throws<ArgumentException>(() => ClassificationVisualizationWriter.Write(
                Path.Combine(directory, "mismatched.svg"),
                mismatched,
                preprocess,
                new[] { new ClassificationPrediction(0, "zero", 1.0f) }));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void ReferenceValidationChecksProvenanceAndValuesSeparately()
    {
        string directory = CreateTempDirectory("classification-reference");
        try
        {
            ClassificationReferenceContext context = CreateContext();
            float[] actual = { 1.0f, 2.0f, 3.0f };
            string referencePath = Path.Combine(directory, "reference.json");
            WriteReference(referencePath, context, actual);

            ClassificationReferenceValidationResult passed = ClassificationOutputProcessor.ValidateReference(
                referencePath, context, actual, 0.0f, 0.0f, "reject", "exact");
            Assert.True(passed.Requested);
            Assert.True(passed.Completed);
            Assert.True(passed.Passed);
            Assert.Equal(3, passed.ComparedElementCount);

            ClassificationReferenceValidationResult mismatch = ClassificationOutputProcessor.ValidateReference(
                referencePath, context, new[] { 1.0f, 2.0f, 3.5f }, 0.0f, 0.0f, "reject", "exact");
            Assert.True(mismatch.Completed);
            Assert.False(mismatch.Passed);
            Assert.Equal(1, mismatch.MismatchCount);
            Assert.Equal(2, mismatch.FirstMismatchIndex);

            ClassificationReferenceContext wrongTensor = new ClassificationReferenceContext(
                "wrong", context.Shape, context.ValueKind, context.ModelSha256, context.InputTensorSha256,
                context.PreprocessContractSha256, context.OutputTensorContractSha256, context.LabelsSha256, context.TaskSemanticsSha256);
            ClassificationReferenceValidationResult metadata = ClassificationOutputProcessor.ValidateReference(
                referencePath, wrongTensor, actual, 0.0f, 0.0f, "reject", "exact");
            Assert.False(metadata.Completed);
            Assert.False(metadata.Passed);
            Assert.Contains("tensorName", metadata.Diagnostic, StringComparison.Ordinal);

            ClassificationReferenceContext wrongShape = new ClassificationReferenceContext(
                context.TensorName, new[] { 3 }, context.ValueKind, context.ModelSha256, context.InputTensorSha256,
                context.PreprocessContractSha256, context.OutputTensorContractSha256, context.LabelsSha256, context.TaskSemanticsSha256);
            AssertMetadataMismatch(referencePath, wrongShape, actual, "shape");

            WriteReference(referencePath, context, new[] { 1.0f, 2.0f });
            AssertMetadataMismatch(referencePath, context, actual, "value count");

            WriteReference(referencePath, context, actual);
            ClassificationReferenceContext wrongFingerprint = new ClassificationReferenceContext(
                context.TensorName, context.Shape, context.ValueKind, Hash('f'), context.InputTensorSha256,
                context.PreprocessContractSha256, context.OutputTensorContractSha256, context.LabelsSha256, context.TaskSemanticsSha256);
            AssertMetadataMismatch(referencePath, wrongFingerprint, actual, "modelSha256");
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void ReferenceSpecialValuePoliciesAreExplicit()
    {
        string directory = CreateTempDirectory("classification-special-values");
        try
        {
            ClassificationReferenceContext context = CreateContext();
            string path = Path.Combine(directory, "special.json");
            WriteReference(path, context, new[] { float.NaN, float.PositiveInfinity, 1.0f });

            byte[] withoutBom = File.ReadAllBytes(path);
            File.WriteAllBytes(path, new byte[] { 0xef, 0xbb, 0xbf }.Concat(withoutBom).ToArray());

            ClassificationReferenceValidationResult rejectNaN = ClassificationOutputProcessor.ValidateReference(
                path, context, new[] { float.NaN, float.PositiveInfinity, 1.0f }, 0.0f, 0.0f, "reject", "exact");
            Assert.True(rejectNaN.Completed);
            Assert.False(rejectNaN.Passed);
            Assert.Equal(1, rejectNaN.MismatchCount);

            ClassificationReferenceValidationResult allowNaN = ClassificationOutputProcessor.ValidateReference(
                path, context, new[] { float.NaN, float.PositiveInfinity, 1.0f }, 0.0f, 0.0f, "equal", "exact");
            Assert.True(allowNaN.Passed);

            ClassificationReferenceValidationResult rejectInfinity = ClassificationOutputProcessor.ValidateReference(
                path, context, new[] { float.NaN, float.PositiveInfinity, 1.0f }, 0.0f, 0.0f, "equal", "reject");
            Assert.False(rejectInfinity.Passed);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void ClassificationSchemasAndCliKeepTheProofBoundaryVisible()
    {
        string sampleDirectory = Path.Combine(RepositoryPaths.Root, "samples", "Classification");
        string outputSchema = File.ReadAllText(Path.Combine(sampleDirectory, "classification-output.schema.json"));
        string referenceSchema = File.ReadAllText(Path.Combine(sampleDirectory, "classification-reference.schema.json"));
        string program = File.ReadAllText(Path.Combine(sampleDirectory, "Program.cs"));
        string writer = File.ReadAllText(Path.Combine(sampleDirectory, "ClassificationOutputArtifacts.cs"));
        Assert.Contains("classification-output.v1", outputSchema, StringComparison.Ordinal);
        Assert.Contains("classification-reference.schema.json", referenceSchema, StringComparison.Ordinal);
        Assert.Contains("isIndependentFrameworkGolden", outputSchema, StringComparison.Ordinal);
        Assert.Contains("\"const\": false", outputSchema, StringComparison.Ordinal);
        Assert.Contains("--image", program, StringComparison.Ordinal);
        Assert.Contains("--input <path>            Raw byte tensor", program, StringComparison.Ordinal);
        Assert.Contains("--reference-output", program, StringComparison.Ordinal);
        Assert.Contains("options.InputPath", program, StringComparison.Ordinal);
        Assert.Contains("options.InputDataPath", program, StringComparison.Ordinal);
        Assert.Contains("ParseOptionalSha256", program, StringComparison.Ordinal);
        Assert.Contains("Classification Passed=", program, StringComparison.Ordinal);
        Assert.Contains("isPackageConsumerRuntimeProof = false", writer, StringComparison.Ordinal);
        Assert.Contains("not an Owner-accepted golden", writer, StringComparison.Ordinal);
    }

    private static ClassificationPreprocessOptions StretchOptions(string layout, string colorOrder, float scale)
    {
        return new ClassificationPreprocessOptions(layout == "NCHW" ? "stretch" : "stretch", 2, layout, colorOrder, scale, new float[3], new[] { 1.0f, 1.0f, 1.0f });
    }

    private static ClassificationReferenceContext CreateContext()
    {
        return new ClassificationReferenceContext(
            "output", new[] { 1, 3 }, "logits", Hash('a'), Hash('b'), Hash('c'),
            ClassificationOutputProcessor.ComputeOutputTensorContractSha256("output", new[] { 1, 3 }, "logits"),
            Hash('d'), ClassificationOutputProcessor.ComputeTaskSemanticsSha256(ClassificationScoreTransform.Raw, 2, Hash('d')));
    }

    private static void AssertMetadataMismatch(
        string referencePath,
        ClassificationReferenceContext context,
        float[] actual,
        string expectedDiagnostic)
    {
        ClassificationReferenceValidationResult result = ClassificationOutputProcessor.ValidateReference(
            referencePath, context, actual, 0.0f, 0.0f, "reject", "exact");
        Assert.True(result.Requested);
        Assert.False(result.Completed);
        Assert.False(result.Passed);
        Assert.Contains(expectedDiagnostic, result.Diagnostic, StringComparison.Ordinal);
    }

    private static void WriteReference(string path, ClassificationReferenceContext context, float[] values)
    {
        var document = new
        {
            schemaVersion = 1,
            tensorName = context.TensorName,
            shape = context.Shape,
            values,
            valueKind = context.ValueKind,
            modelSha256 = context.ModelSha256,
            inputTensorSha256 = context.InputTensorSha256,
            preprocessContractSha256 = context.PreprocessContractSha256,
            outputTensorContractSha256 = context.OutputTensorContractSha256,
            labelsSha256 = context.LabelsSha256,
            taskSemanticsSha256 = context.TaskSemanticsSha256,
            sourceClassification = "independent-framework-candidate"
        };
        File.WriteAllText(path, JsonSerializer.Serialize(document, new JsonSerializerOptions
        {
            WriteIndented = true,
            NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals
        }) + Environment.NewLine, new UTF8Encoding(false));
    }

    private static string Hash(char value) => new string(value, 64);

    private static float[] ReadFloats(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        float[] values = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    private static string CreateTempDirectory(string name)
    {
        string path = Path.Combine(Path.GetTempPath(), "jyppx-" + name + "-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(path);
        return path;
    }

    private static byte[] CreateBmp24(byte[] rgb, int width, int height)
    {
        int rowStride = ((width * 24 + 31) / 32) * 4;
        int pixelOffset = 54;
        byte[] bytes = new byte[pixelOffset + rowStride * height];
        bytes[0] = (byte)'B';
        bytes[1] = (byte)'M';
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(2, 4), bytes.Length);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(10, 4), pixelOffset);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(14, 4), 40);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(18, 4), width);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(22, 4), height);
        BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(26, 2), 1);
        BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(28, 2), 24);
        for (int y = 0; y < height; y++)
        {
            int sourceY = height - 1 - y;
            int row = pixelOffset + y * rowStride;
            for (int x = 0; x < width; x++)
            {
                int source = (sourceY * width + x) * 3;
                int target = row + x * 3;
                bytes[target] = rgb[source + 2];
                bytes[target + 1] = rgb[source + 1];
                bytes[target + 2] = rgb[source];
            }
        }
        return bytes;
    }
}

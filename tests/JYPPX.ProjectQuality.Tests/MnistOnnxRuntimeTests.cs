using System.Text;
using JYPPX.TensorRtSharp.Tools;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class MnistOnnxRuntimeTests
{
    [Fact]
    public void PgmReader_parses_comments_and_applies_official_preprocessing()
    {
        string path = Path.Combine(Path.GetTempPath(), "jyppx-mnist-" + Guid.NewGuid().ToString("N") + ".pgm");
        try
        {
            byte[] header = Encoding.ASCII.GetBytes("P5\n# test asset\n3 1\n255\n");
            byte[] bytes = new byte[header.Length + 3];
            Buffer.BlockCopy(header, 0, bytes, 0, header.Length);
            bytes[^3] = 0;
            bytes[^2] = 128;
            bytes[^1] = 255;
            File.WriteAllBytes(path, bytes);

            MnistPgmImage image = MnistPgmReader.Read(path);
            float[] tensor = MnistPgmReader.ToTensorInput(image);

            Assert.Equal(3, image.Width);
            Assert.Equal(1, image.Height);
            Assert.Equal(255, image.MaxValue);
            Assert.Equal(new byte[] { 0, 128, 255 }, image.Pixels);
            Assert.Equal(1.0f, tensor[0], 6);
            Assert.Equal(1.0f - (128.0f / 255.0f), tensor[1], 6);
            Assert.Equal(0.0f, tensor[2], 6);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void OutputClassifier_uses_stable_softmax_and_argmax()
    {
        float[] logits = new float[10];
        logits[7] = 10.0f;

        MnistClassification classification = MnistOutputClassifier.Classify(logits);

        Assert.Equal(7, classification.PredictedDigit);
        Assert.True(classification.Confidence > 0.99f);
        Assert.Equal(1.0f, classification.Probabilities.Sum(), 5);
    }

    [Fact]
    public void OnnxToEngine_exposes_explicit_mnist_runner_without_weakening_generic_boundary()
    {
        string root = RepositoryPaths.Root;
        string program = File.ReadAllText(Path.Combine(root, "samples", "OnnxToEngine", "Program.cs"));
        string result = File.ReadAllText(Path.Combine(
            root, "src", "JYPPX.TensorRtSharp.Tools", "Runtime", "MnistOnnxRuntimeResult.cs"));
        string genericService = File.ReadAllText(Path.Combine(
            root, "src", "JYPPX.TensorRtSharp.Tools", "Build", "OnnxEngineBuildService.cs"));

        Assert.Contains("SampleCommandLine.HasSwitch(args, \"--mnist\")", program, StringComparison.Ordinal);
        Assert.Contains("\"real-model-runtime\"", result, StringComparison.Ordinal);
        Assert.Contains("IsPackageConsumerRuntimeProof => false", result, StringComparison.Ordinal);
        Assert.Contains("Generic external-model inference requires explicit binding/output semantics.", genericService, StringComparison.Ordinal);
    }
}

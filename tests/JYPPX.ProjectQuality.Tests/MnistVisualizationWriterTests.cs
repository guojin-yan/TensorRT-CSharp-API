using System.Text;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Tools;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class MnistVisualizationWriterTests
{
    [Fact]
    public void WriterDrawsTheActualPgmPredictionAndProbabilities()
    {
        string root = Path.Combine(Path.GetTempPath(), "jyppx-mnist-visual-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            string inputPath = Path.Combine(root, "digit-7.pgm");
            string outputPath = Path.Combine(root, "result.svg");
            byte[] header = Encoding.ASCII.GetBytes("P5\n28 28\n255\n");
            byte[] pixels = Enumerable.Repeat((byte)255, 28 * 28).ToArray();
            pixels[(8 * 28) + 12] = 0;
            File.WriteAllBytes(inputPath, header.Concat(pixels).ToArray());

            float[] probabilities = new float[10];
            probabilities[7] = 0.975f;
            probabilities[2] = 0.025f;
            MnistOnnxRuntimeResult result = new MnistOnnxRuntimeResult(
                success: true,
                skipped: false,
                state: "mnist-real-model-runtime",
                tensorRtLine: TensorRtApiLine.TensorRt10,
                modelPath: "mnist.onnx",
                inputPath,
                enginePath: "mnist.plan",
                modelSha256: "model",
                inputSha256: "input",
                preprocessedInputSha256: "tensor",
                engineSha256: "engine",
                parsed: true,
                engineSaved: true,
                engineFileRoundTrip: true,
                inferenceRan: true,
                outputMatch: true,
                expectedDigit: 7,
                predictedDigit: 7,
                confidence: 0.975f,
                minimumConfidence: 0.5f,
                inputTensorName: "Input3",
                inputShape: new[] { 1, 1, 28, 28 },
                inputDataType: "Float",
                outputTensorName: "Plus214_Output_0",
                outputShape: new[] { 1, 10 },
                outputDataType: "Float",
                logits: new float[10],
                probabilities,
                elapsedMilliseconds: 1.0f,
                skipReason: string.Empty,
                normalizedCommandLine: "--mnist",
                environment: null,
                logLines: Array.Empty<string>());

            MnistVisualizationWriter.Write(outputPath, result);

            string svg = File.ReadAllText(outputPath);
            Assert.Contains("INPUT · 28 × 28 PGM", svg, StringComparison.Ordinal);
            Assert.Contains(">7</text>", svg, StringComparison.Ordinal);
            Assert.Contains("97.500%", svg, StringComparison.Ordinal);
            Assert.Contains("preprocess 1 - pixel / 255", svg, StringComparison.Ordinal);

            probabilities[7] = float.NaN;
            Assert.Throws<InvalidDataException>(() => MnistVisualizationWriter.Write(outputPath, result));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void CommandLineDocumentsAndCallsTheVisualizationWriter()
    {
        string program = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "OnnxToEngine", "Program.cs"));
        Assert.Contains("--visualization", program, StringComparison.Ordinal);
        Assert.Contains("MnistVisualizationWriter.Write", program, StringComparison.Ordinal);
    }
}

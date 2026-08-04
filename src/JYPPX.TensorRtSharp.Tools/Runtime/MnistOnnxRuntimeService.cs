using System;
using System.Collections.Generic;
using System.IO;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

/// <summary>
/// Executes the ONNX Model Zoo MNIST model with explicit model semantics.
/// 使用明确模型语义执行 ONNX Model Zoo MNIST 模型。
/// </summary>
public sealed partial class MnistOnnxRuntimeService
{
    public MnistOnnxRuntimeResult Execute(MnistOnnxRuntimeOptions options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        ValidateOptions(options);
        string modelPath = Path.GetFullPath(options.OnnxPath);
        string inputPath = Path.GetFullPath(options.InputPgmPath);
        string enginePath = string.IsNullOrWhiteSpace(options.SaveEnginePath)
            ? Path.Combine(Path.GetTempPath(), $"jyppx-mnist-{Guid.NewGuid():N}.plan")
            : Path.GetFullPath(options.SaveEnginePath);
        bool deleteEngine = string.IsNullOrWhiteSpace(options.SaveEnginePath);
        byte[] model = File.ReadAllBytes(modelPath);
        byte[] inputFile = File.ReadAllBytes(inputPath);
        MnistPgmImage image = MnistPgmReader.Read(inputPath);
        float[] inputValues = MnistPgmReader.ToTensorInput(image);
        byte[] preprocessedInput = ToBytes(inputValues);
        List<string> log = new List<string>
        {
            $"MnistOnnxRuntime TensorRtLine={(int)options.TensorRtLine} Model={Path.GetFileName(modelPath)} Input={Path.GetFileName(inputPath)} ExpectedDigit={options.ExpectedDigit}",
            $"MnistPreprocess Width={image.Width} Height={image.Height} Formula=1-pixel/255 ElementCount={inputValues.Length}",
            $"MnistHashes Model={ComputeSha256(model)} Input={ComputeSha256(inputFile)} PreprocessedInput={ComputeSha256(preprocessedInput)}"
        };

        TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
        TensorRtAdapterInfo adapter = TensorRtToolSupport.SelectAdapter(snapshot, options.TensorRtLine);
        log.Add($"MnistPreflight TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion} Runtime={adapter.RuntimeCreationSupported} Builder={adapter.BuilderCreationSupported}");
        if (!adapter.RuntimeCreationSupported || !adapter.BuilderCreationSupported)
        {
            MnistOnnxRuntimeResult skipped = new MnistOnnxRuntimeResult(
                success: true,
                skipped: true,
                state: "dependency-unavailable",
                options.TensorRtLine,
                modelPath,
                inputPath,
                enginePath,
                ComputeSha256(model),
                ComputeSha256(inputFile),
                ComputeSha256(preprocessedInput),
                engineSha256: string.Empty,
                parsed: false,
                engineSaved: false,
                engineFileRoundTrip: false,
                inferenceRan: false,
                outputMatch: false,
                options.ExpectedDigit,
                predictedDigit: -1,
                confidence: 0.0f,
                options.MinimumConfidence,
                inputTensorName: string.Empty,
                inputShape: Array.Empty<int>(),
                inputDataType: string.Empty,
                outputTensorName: string.Empty,
                outputShape: Array.Empty<int>(),
                outputDataType: string.Empty,
                logits: Array.Empty<float>(),
                probabilities: Array.Empty<float>(),
                elapsedMilliseconds: null,
                skipReason: adapter.StatusMessage,
                normalizedCommandLine: options.ToCommandLine(),
                environment: null,
                log);
            MnistOnnxRuntimeDiagnostics.WriteArtifacts(skipped, options);
            return skipped;
        }

        string? engineDirectory = Path.GetDirectoryName(enginePath);
        if (!string.IsNullOrWhiteSpace(engineDirectory))
        {
            Directory.CreateDirectory(engineDirectory);
        }

        try
        {
            using TensorRtLogger logger = new TensorRtLogger(options.TensorRtLine);
            using TensorRtRuntime runtime = new TensorRtRuntime(logger);
            using TensorRtBuilder builder = new TensorRtBuilder(logger);
            using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
            using CudaStream stream = new CudaStream();
            config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, options.WorkspaceBytes);
            config.SetProfileStream(stream);
            config.SetEngineCapability(TensorRtEngineCapability.Standard);

            using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
            using TensorRtOnnxParser parser = new TensorRtOnnxParser(logger, network);
            if (!parser.Parse(model, Path.GetFileName(modelPath)))
            {
                throw new InvalidOperationException(parser.GetErrorSummary());
            }

            using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
            hostMemory.SaveToFile(enginePath);
            using TensorRtEngine engine = runtime.DeserializeFromFile(enginePath);
            using TensorRtExecutionContext context = engine.CreateExecutionContext();
            using TensorRtInferenceBindings bindings = new TensorRtInferenceBindings(engine, context, profileIndex: 0);

            TensorRtEngineTensorBinding input = SingleTensor(bindings.Report.GetInputs(), "input");
            TensorRtEngineTensorBinding output = SingleTensor(bindings.Report.GetOutputs(), "output");
            EnsureFloatDeviceTensor(input, "input");
            EnsureFloatDeviceTensor(output, "output");

            TensorRtDims inputShape = ResolveShape(context, input);
            TensorRtDims outputShape = ResolveShape(context, output);
            int inputElementCount = ElementCount(inputShape);
            int outputElementCount = ElementCount(outputShape);
            if (inputElementCount != inputValues.Length)
            {
                throw new InvalidOperationException(
                    $"MNIST input element count mismatch. Tensor={inputElementCount} Image={inputValues.Length}.");
            }

            if (outputElementCount != 10)
            {
                throw new InvalidOperationException($"MNIST output must contain ten logits. Actual={outputElementCount}.");
            }

            bindings.CopyInputFromHost(input.Name, inputValues, inputShape);
            bindings.AllocateDeviceBuffer(output.Name, outputShape, checked(outputElementCount * sizeof(float)));
            bindings.BindAll();

            TensorRtInferenceExecutionSummary executionSummary = null!;
            float elapsedMilliseconds = stream.MeasureElapsedTime(cudaStream =>
            {
                executionSummary = bindings.EnqueueAsync(cudaStream, synchronize: false, runShapeInference: false);
            });
            stream.Synchronize();

            float[] logits = bindings.ReadOutputSingles(output.Name, outputElementCount);
            MnistClassification classification = MnistOutputClassifier.Classify(logits);
            bool outputMatch = classification.PredictedDigit == options.ExpectedDigit &&
                classification.Confidence >= options.MinimumConfidence;
            string engineSha256 = ComputeFileSha256(enginePath);
            MnistRuntimeEnvironment environment = MnistRuntimeEnvironment.Capture(snapshot);

            log.Add($"MnistBindings Input={input.Name}:{input.DataType}:{inputShape} Output={output.Name}:{output.DataType}:{outputShape}");
            log.Add($"MnistExecution {executionSummary} ElapsedMs={elapsedMilliseconds:0.###}");
            log.Add($"MnistClassification Expected={options.ExpectedDigit} Predicted={classification.PredictedDigit} Confidence={classification.Confidence:0.000000} Minimum={options.MinimumConfidence:0.000000} OutputMatch={outputMatch}");
            log.Add($"MnistEngine Path={Path.GetFileName(enginePath)} Sha256={engineSha256} Bytes={new FileInfo(enginePath).Length}");

            MnistOnnxRuntimeResult result = new MnistOnnxRuntimeResult(
                success: outputMatch,
                skipped: false,
                state: outputMatch ? "mnist-real-model-runtime" : "mnist-output-mismatch",
                options.TensorRtLine,
                modelPath,
                inputPath,
                enginePath,
                ComputeSha256(model),
                ComputeSha256(inputFile),
                ComputeSha256(preprocessedInput),
                engineSha256,
                parsed: true,
                engineSaved: true,
                engineFileRoundTrip: true,
                inferenceRan: true,
                outputMatch,
                options.ExpectedDigit,
                classification.PredictedDigit,
                classification.Confidence,
                options.MinimumConfidence,
                input.Name,
                inputShape.Values,
                input.DataType.ToString(),
                output.Name,
                outputShape.Values,
                output.DataType.ToString(),
                logits,
                classification.Probabilities,
                elapsedMilliseconds,
                skipReason: string.Empty,
                normalizedCommandLine: options.ToCommandLine(),
                environment,
                log);
            MnistOnnxRuntimeDiagnostics.WriteArtifacts(result, options, preprocessedInput);
            return result;
        }
        finally
        {
            if (deleteEngine && File.Exists(enginePath))
            {
                File.Delete(enginePath);
            }
        }
    }
}

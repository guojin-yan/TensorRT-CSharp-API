using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.ExceptionServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed partial class OnnxEngineBuildService
{
    private static bool CanAttemptGenericExternalRuntime(OnnxEngineBuildOptions options)
    {
        return options.RuntimeOptions.NoDataTransfers ||
            !string.IsNullOrWhiteSpace(options.RuntimeOptions.LoadInputs) ||
            options.RuntimeOptions.RequestsOutputCapture;
    }

    private static OnnxEngineRuntimeExecution? TryRunGenericFloatEngineFromFile(
        OnnxEngineBuildOptions options,
        string enginePath,
        int profileIndex,
        List<string> log,
        string statePrefix)
    {
        try
        {
            using TensorRtLogger logger = new TensorRtLogger(options.TensorRtLine);
            using TensorRtRuntime runtime = new TensorRtRuntime(logger);
            ConfigureRuntimeForEnginePolicies(runtime, options, log, statePrefix + "Runtime");
            using TensorRtEngine engine = runtime.DeserializeFromFile(enginePath);
            return TryRunGenericFloatEngine(engine, options, profileIndex, log, statePrefix);
        }
        catch (Exception exception) when (exception is NotSupportedException || exception is ArgumentException || exception is InvalidOperationException || exception is TensorRtException || exception is BridgeProbeException || exception is DllNotFoundException || exception is BadImageFormatException || exception is FileNotFoundException)
        {
            log.Add($"{statePrefix}BoundedRuntime Attempted=True Succeeded=False Reason={exception.GetType().Name}:{exception.Message}");
            return null;
        }
    }

    private static OnnxEngineRuntimeExecution TryRunGenericFloatEngine(
        TensorRtEngine engine,
        OnnxEngineBuildOptions options,
        int profileIndex,
        List<string> log,
        string statePrefix,
        bool validateRefittableState = true)
    {
        ApplyEngineRuntimePolicies(engine, options, log, validateRefittableState);
        if (options.RuntimeOptions.NoDataTransfers && options.RuntimeOptions.RequestsReferenceValidation)
        {
            throw new ArgumentException("--referenceOutputs cannot be used with --noDataTransfers because output readback is disabled.");
        }

        int safeProfileIndex = Math.Max(0, profileIndex);
        int executionContextCount = options.RuntimeOptions.InfStreams ?? options.Streams;
        List<OnnxEngineBenchmarkWorker> workers = new List<OnnxEngineBenchmarkWorker>(executionContextCount);
        try
        {
            OnnxEngineBenchmarkWorker firstWorker = new OnnxEngineBenchmarkWorker(engine, safeProfileIndex, options.RuntimeOptions.UseSpinWait);
            workers.Add(firstWorker);

            IReadOnlyList<TensorRtEngineTensorBinding> inputs = firstWorker.Bindings.Report.GetInputs();
            IReadOnlyList<TensorRtEngineTensorBinding> outputs = firstWorker.Bindings.Report.GetOutputs();
            if (inputs.Count == 0)
            {
                throw new NotSupportedException("Generic bounded runtime requires at least one input tensor.");
            }

            if (outputs.Count == 0)
            {
                throw new NotSupportedException("Generic bounded runtime requires at least one output tensor.");
            }

            foreach (TensorRtEngineTensorBinding input in inputs)
            {
                if (input.DataType != TensorRtDataType.Float)
                {
                    throw new NotSupportedException($"Generic bounded runtime supports float input tensors only. Input '{input.Name}' is {input.DataType}.");
                }
            }

            foreach (TensorRtEngineTensorBinding output in outputs)
            {
                if (output.DataType != TensorRtDataType.Float)
                {
                    throw new NotSupportedException($"Generic bounded runtime supports float output tensors only. Output '{output.Name}' is {output.DataType}.");
                }
            }

            Dictionary<string, string> inputFiles = ParseLoadInputs(options.RuntimeOptions.LoadInputs);
            HashSet<string> engineInputNames = new HashSet<string>(inputs.Select(static input => input.Name), StringComparer.Ordinal);
            string[] unknownInputMappings = inputFiles.Keys.Where(name => !engineInputNames.Contains(name)).OrderBy(static name => name, StringComparer.Ordinal).ToArray();
            if (unknownInputMappings.Length > 0)
            {
                throw new ArgumentException("--loadInputs contains mappings for unknown input tensors: " + string.Join(", ", unknownInputMappings));
            }

            List<OnnxEngineRuntimeInput> runtimeInputs = new List<OnnxEngineRuntimeInput>(inputs.Count);
            for (int inputIndex = 0; inputIndex < inputs.Count; inputIndex++)
            {
                TensorRtEngineTensorBinding input = inputs[inputIndex];
                TensorRtDims runtimeShape = ResolveRuntimeInputShape(input, options);
                int inputElementCount = CountElements(runtimeShape);
                float[] inputValues = options.RuntimeOptions.NoDataTransfers
                    ? Array.Empty<float>()
                    : CreateRuntimeInputValues(input.Name, inputElementCount, inputFiles, inputIndex);
                bool setInputShape = ShouldSetInputShape(input, runtimeShape, options);
                string sourceClassification = options.RuntimeOptions.NoDataTransfers
                    ? "no-data-transfers"
                    : inputFiles.ContainsKey(input.Name)
                        ? "load-input-file"
                        : "deterministic-generated";
                string sourcePath = inputFiles.TryGetValue(input.Name, out string? mappedPath) ? mappedPath : string.Empty;
                OnnxEngineRuntimeInputArtifact inputArtifact = options.RuntimeOptions.NoDataTransfers
                    ? new OnnxEngineRuntimeInputArtifact(
                        input.Name,
                        runtimeShape.Values,
                        inputElementCount,
                        Array.Empty<float>(),
                        Array.Empty<byte>(),
                        sourceClassification,
                        sourcePath)
                    : new OnnxEngineRuntimeInputArtifact(
                        input.Name,
                        runtimeShape.Values,
                        inputValues,
                        sourceClassification,
                        sourcePath);
                runtimeInputs.Add(new OnnxEngineRuntimeInput(input, runtimeShape, inputValues, setInputShape, inputArtifact));
            }

            firstWorker.Configure(runtimeInputs, outputs, options.RuntimeOptions.NoDataTransfers);
            for (int workerIndex = 1; workerIndex < executionContextCount; workerIndex++)
            {
                OnnxEngineBenchmarkWorker worker = new OnnxEngineBenchmarkWorker(engine, safeProfileIndex, options.RuntimeOptions.UseSpinWait);
                workers.Add(worker);
                worker.Configure(runtimeInputs, outputs, options.RuntimeOptions.NoDataTransfers);
            }

            OnnxEngineBenchmarkRun benchmark = RunBoundedBenchmark(workers, options);
            float elapsedMilliseconds = benchmark.TimingSamplesMilliseconds[0];

            List<OnnxEngineRuntimeOutputTensor> capturedOutputs = new List<OnnxEngineRuntimeOutputTensor>(outputs.Count);
            if (!options.RuntimeOptions.NoDataTransfers)
            {
                foreach (TensorRtEngineTensorBinding output in outputs)
                {
                    TensorRtInferenceBuffer outputBuffer = firstWorker.Bindings.Buffers[output.Name];
                    TensorRtDims outputShape = outputBuffer.RuntimeShape ?? throw new InvalidOperationException($"Output '{output.Name}' does not have a concrete runtime shape.");
                    int outputElementCount = CountElements(outputShape);
                    float[] outputValues = firstWorker.Bindings.ReadOutputSingles(output.Name, outputElementCount);
                    capturedOutputs.Add(new OnnxEngineRuntimeOutputTensor(output.Name, outputShape.Values, outputValues));
                }
            }

            IReadOnlyList<OnnxEngineRuntimeOutputArtifact> outputArtifacts = capturedOutputs
                .Select(static output => new OnnxEngineRuntimeOutputArtifact(output.Name, output.Shape, output.Values))
                .ToArray();
            if (options.RuntimeOptions.DumpOutput)
            {
                foreach (OnnxEngineRuntimeOutputArtifact output in outputArtifacts)
                {
                    string preview = string.Join(",", output.Preview.Select(static value =>
                        value.ToString("R", System.Globalization.CultureInfo.InvariantCulture)));
                    log.Add(
                        $"{statePrefix}DumpOutput Tensor={output.TensorName} Shape={FormatShape(output.Shape)} " +
                        $"Elements={output.ElementCount} Bytes={output.ByteLength} Sha256={output.Sha256} " +
                        $"Preview=[{preview}] EvidenceBoundary=bounded-output-capture-not-reference-validation");
                }
            }

            OnnxEngineRuntimeOutputTensor? primaryOutput = capturedOutputs.Count == 0 ? null : capturedOutputs[0];
            bool identityOutputMatch = primaryOutput != null &&
                runtimeInputs.Count == 1 &&
                capturedOutputs.Count == 1 &&
                primaryOutput.Values.Length == runtimeInputs[0].Values.Length &&
                ValuesEqual(runtimeInputs[0].Values, primaryOutput.Values);
            OnnxEngineReferenceValidationArtifact referenceValidation = ValidateReferenceOutputs(
                capturedOutputs,
                options.RuntimeOptions);
            bool outputMatch = options.RuntimeOptions.RequestsReferenceValidation
                ? referenceValidation.Completed && referenceValidation.Passed
                : identityOutputMatch;
            string primaryOutputSummary = primaryOutput == null
                ? "not-read-back"
                : $"{primaryOutput.Name}:{FormatShape(primaryOutput.Shape)}";
            string inputSource = options.RuntimeOptions.NoDataTransfers
                ? "no-data-transfers"
                : string.IsNullOrWhiteSpace(options.RuntimeOptions.LoadInputs)
                    ? "deterministic-generated"
                    : "loadInputs";
            log.Add($"{statePrefix}BoundedRuntime Attempted=True Succeeded=True Inputs={runtimeInputs.Count} InputSource={inputSource} Outputs={capturedOutputs.Count} PrimaryOutput={primaryOutputSummary} ElapsedMs={elapsedMilliseconds:0.###} IdentityOutputMatch={identityOutputMatch} ReferenceOutputValidated={referenceValidation.Completed && referenceValidation.Passed} NoDataTransfers={options.RuntimeOptions.NoDataTransfers}");
            log.Add($"{statePrefix}BoundedRuntime InputTensors=" + string.Join("; ", runtimeInputs.Select(static item => $"{item.Binding.Name}:{FormatShape(item.Shape.Values)}:{item.Artifact.ElementCount}:{item.Artifact.SourceClassification}")));
            log.Add(options.RuntimeOptions.NoDataTransfers
                ? $"{statePrefix}BoundedRuntime OutputReadback=False Reason=noDataTransfers"
                : $"{statePrefix}BoundedRuntime OutputTensors=" + string.Join("; ", capturedOutputs.Select(static item => $"{item.Name}:{FormatShape(item.Shape)}:{item.Values.Length}")));
            if (referenceValidation.Requested)
            {
                log.Add($"ReferenceOutputValidation Requested=True Completed={referenceValidation.Completed} Passed={referenceValidation.Passed} Tensors={referenceValidation.TensorComparisons.Count} AbsTolerance={referenceValidation.AbsoluteTolerance:R} RelTolerance={referenceValidation.RelativeTolerance:R} NaNPolicy={referenceValidation.NaNPolicy} InfinityPolicy={referenceValidation.InfinityPolicy}");
                foreach (OnnxEngineReferenceTensorComparisonArtifact comparison in referenceValidation.TensorComparisons)
                {
                    log.Add($"ReferenceOutputTensor Tensor={comparison.TensorName} Passed={comparison.Passed} Compared={comparison.ComparedElementCount} Mismatches={comparison.MismatchCount} FirstMismatch={comparison.FirstMismatchIndex} MaxAbs={comparison.MaximumAbsoluteError:R} MaxRel={comparison.MaximumRelativeError:R} Diagnostic={comparison.Diagnostic}");
                }
                foreach (string diagnostic in referenceValidation.Diagnostics)
                {
                    log.Add("ReferenceOutputDiagnostic " + diagnostic);
                }
            }
            log.Add(
                $"RuntimeBenchmark Contexts={workers.Count} MeasurementRounds={benchmark.MeasurementRoundsExecuted} " +
                $"InferenceIterations={benchmark.TimingSamplesMilliseconds.Count} WarmUpIterations={benchmark.WarmUpIterationsExecuted} " +
                $"WarmUpElapsedMs={benchmark.WarmUpElapsedMilliseconds:0.###} MeasurementElapsedMs={benchmark.MeasurementElapsedMilliseconds:0.###} " +
                $"IterationsRequested={options.Iterations} DurationSecondsRequested={options.DurationSeconds} " +
                $"StreamsRequested={options.Streams} InfStreamsRequested={options.RuntimeOptions.InfStreams?.ToString() ?? ""} " +
                $"ThreadsApplied={benchmark.ThreadsExecuted} SpinWaitApplied={benchmark.UseSpinWaitApplied} " +
                $"NoDataTransfersApplied={options.RuntimeOptions.NoDataTransfers} CudaGraphApplied={benchmark.UseCudaGraphApplied} " +
                $"IdleTimeApplied={benchmark.IdleTimeMillisecondsApplied} SleepTimeApplied=0");
            if (!string.IsNullOrWhiteSpace(benchmark.UseCudaGraphFallbackReason))
            {
                log.Add($"RuntimeBenchmark CudaGraphFallbackReason={benchmark.UseCudaGraphFallbackReason}");
            }

            OnnxEngineRuntimeArtifactData artifactData = options.RuntimeOptions.NoDataTransfers
                ? OnnxEngineRuntimeArtifactData.CreateBenchmarkOnly(
                    runtimeInputs.Select(static input => input.Artifact).ToArray(),
                    benchmark.LastExecutionSummary.ToString(),
                    benchmark.TimingSamplesMilliseconds)
                : OnnxEngineRuntimeArtifactData.CreateRuntimeEvidence(
                    runtimeInputs.Select(static input => input.Artifact).ToArray(),
                    outputArtifacts,
                    referenceValidation,
                    benchmark.LastExecutionSummary.ToString(),
                    benchmark.TimingSamplesMilliseconds);

            return new OnnxEngineRuntimeExecution(
                inferenceRan: true,
                outputMatch,
                identityOutputMatch,
                outputValidated: referenceValidation.Completed && referenceValidation.Passed,
                safeProfileIndex,
                elapsedMilliseconds,
                OnnxEngineBenchmarkSummary.CreateExecuted(
                    benchmark.TimingSamplesMilliseconds,
                    options,
                    benchmark.MeasurementRoundsExecuted,
                    benchmark.WarmUpIterationsExecuted,
                    benchmark.WarmUpElapsedMilliseconds,
                    benchmark.MeasurementElapsedMilliseconds,
                    workers.Count,
                    benchmark.ThreadsExecuted,
                    benchmark.UseSpinWaitApplied,
                    benchmark.UseCudaGraphApplied,
                    benchmark.UseCudaGraphFallbackReason,
                    benchmark.MeasurementRoundsPerContext),
                artifactData);
        }
        finally
        {
            for (int index = workers.Count - 1; index >= 0; index--)
            {
                workers[index].Dispose();
            }
        }
    }

    private static void ApplyEngineRuntimePolicies(
        TensorRtEngine engine,
        OnnxEngineBuildOptions options,
        List<string> log,
        bool validateRefittableState)
    {
        TrtexecLikeDeploymentOptions deployment = options.DeploymentOptions;
        if (deployment.Refit && validateRefittableState)
        {
            bool refittable = engine.IsRefittable;
            log.Add($"TrtexecEnginePolicy Name=Refit Applied=True Requested=True Readback={refittable} ReadbackMatch={refittable}");
            if (!refittable)
            {
                throw new InvalidOperationException("TensorRT built an engine that is not refittable after --refit was applied.");
            }
        }
        else if (deployment.Refit)
        {
            log.Add("TrtexecEnginePolicy Name=Refit RuntimeRevalidation=False PersistenceCommitted=True Reason=full-weight-reload-does-not-require-refittable-state");
        }

        if (!deployment.WeightStreamingBudget.IsSpecified)
        {
            return;
        }

        if (options.TensorRtLine == TensorRtApiLine.TensorRt8)
        {
            log.Add(
                $"TrtexecDeploymentControl Name=WeightStreamingBudget Applied=False Requested={deployment.WeightStreamingBudget.ArgumentValue} " +
                "VersionGuard=TRT8 Reason=weight-streaming-runtime-api-is-not-available");
            return;
        }

        long streamableWeights = engine.StreamableWeightsSizeInBytes;
        long automaticBudget = engine.WeightStreamingAutomaticBudgetInBytes;
        long requestedBudget = deployment.WeightStreamingBudget.ResolveBytes(streamableWeights, automaticBudget);
        bool accepted = engine.SetWeightStreamingBudgetV2(requestedBudget);
        long readbackBudget = engine.WeightStreamingBudgetV2InBytes;
        long scratchBytes = engine.WeightStreamingScratchMemorySizeInBytes;
        bool readbackMatch = accepted && readbackBudget == requestedBudget;
        log.Add(
            $"TrtexecDeploymentControl Name=WeightStreamingBudget Applied={accepted} " +
            $"Requested={deployment.WeightStreamingBudget.ArgumentValue} Mode={deployment.WeightStreamingBudget.Kind} " +
            $"ResolvedBytes={requestedBudget} StreamableWeightsBytes={streamableWeights} AutomaticBudgetBytes={automaticBudget} " +
            $"Readback={readbackBudget} ScratchBytes={scratchBytes} ReadbackMatch={readbackMatch}");
        if (!readbackMatch)
        {
            throw new InvalidOperationException(
                $"TensorRT rejected or changed the requested weight-streaming budget. Requested={requestedBudget}, Readback={readbackBudget}.");
        }
    }

    private static void ConfigureRuntimeForEnginePolicies(
        TensorRtRuntime runtime,
        OnnxEngineBuildOptions options,
        List<string> log,
        string source)
    {
        if (!options.DeploymentOptions.VersionCompatible)
        {
            return;
        }

        runtime.EngineHostCodeAllowed = true;
        bool readback = runtime.EngineHostCodeAllowed;
        log.Add(
            $"TrtexecRuntimePolicy Name=EngineHostCodeAllowed Applied=True Requested=True " +
            $"Readback={readback} ReadbackMatch={readback} Source={source}");
        if (!readback)
        {
            throw new InvalidOperationException("TensorRT runtime did not enable host code for a version-compatible engine.");
        }
    }

    private sealed class OnnxEngineRuntimeExecution
    {
        public OnnxEngineRuntimeExecution(
            bool inferenceRan,
            bool outputMatch,
            bool identityOutputMatch,
            bool outputValidated,
            int profileIndex,
            float elapsedMilliseconds,
            OnnxEngineBenchmarkSummary benchmarkSummary,
            OnnxEngineRuntimeArtifactData artifactData)
        {
            InferenceRan = inferenceRan;
            OutputMatch = outputMatch;
            IdentityOutputMatch = identityOutputMatch;
            OutputValidated = outputValidated;
            ProfileIndex = profileIndex;
            ElapsedMilliseconds = elapsedMilliseconds;
            BenchmarkSummary = benchmarkSummary ?? OnnxEngineBenchmarkSummary.Empty;
            ArtifactData = artifactData ?? OnnxEngineRuntimeArtifactData.Empty;
        }

        public bool InferenceRan { get; }

        public bool OutputMatch { get; }

        public bool IdentityOutputMatch { get; }

        public bool OutputValidated { get; }

        public int ProfileIndex { get; }

        public float ElapsedMilliseconds { get; }

        public OnnxEngineBenchmarkSummary BenchmarkSummary { get; }

        public OnnxEngineRuntimeArtifactData ArtifactData { get; }
    }

    private sealed class OnnxEngineRuntimeInput
    {
        public OnnxEngineRuntimeInput(
            TensorRtEngineTensorBinding binding,
            TensorRtDims shape,
            float[] values,
            bool setInputShape,
            OnnxEngineRuntimeInputArtifact artifact)
        {
            Binding = binding ?? throw new ArgumentNullException(nameof(binding));
            Shape = shape ?? throw new ArgumentNullException(nameof(shape));
            Values = values ?? Array.Empty<float>();
            SetInputShape = setInputShape;
            Artifact = artifact ?? throw new ArgumentNullException(nameof(artifact));
        }

        public TensorRtEngineTensorBinding Binding { get; }

        public TensorRtDims Shape { get; }

        public float[] Values { get; }

        public bool SetInputShape { get; }

        public OnnxEngineRuntimeInputArtifact Artifact { get; }
    }

    private sealed class OnnxEngineRuntimeOutputTensor
    {
        public OnnxEngineRuntimeOutputTensor(string name, IReadOnlyList<int> shape, float[] values)
        {
            Name = name ?? string.Empty;
            Shape = shape ?? Array.Empty<int>();
            Values = values ?? Array.Empty<float>();
        }

        public string Name { get; }

        public IReadOnlyList<int> Shape { get; }

        public float[] Values { get; }
    }

}

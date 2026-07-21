using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Tools;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class TensorRtExecBoundedBenchmarkSchedulerTests
{
    [Fact]
    public void RuntimeEvidenceKeepsSyntheticAndPackageProofBoundaries()
    {
        string path = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "interface-coverage",
            "trtexec-bounded-benchmark-scheduler-runtime-evidence.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        JsonElement root = document.RootElement;
        JsonElement concurrency = root.GetProperty("concurrencyRun");
        JsonElement duration = root.GetProperty("durationRun");
        JsonElement boundary = root.GetProperty("proofBoundary");

        Assert.Equal(2, concurrency.GetProperty("executionContextsCreated").GetInt32());
        Assert.Equal(6, concurrency.GetProperty("rawTimingSampleCount").GetInt32());
        Assert.Equal(3, concurrency.GetProperty("averagedTimingSampleCount").GetInt32());
        Assert.True(concurrency.GetProperty("outputMatch").GetBoolean());
        Assert.True(duration.GetProperty("measurementElapsedMilliseconds").GetDouble() >= 1000);
        Assert.True(boundary.GetProperty("isBoundedSchedulerRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("isRealModelRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("canPublishPublicly").GetBoolean());
        Assert.Contains(root.GetProperty("unappliedControls").EnumerateArray(), static item => item.GetString() == "sleepTime");
        Assert.All(root.GetProperty("binaryHashes").EnumerateArray(), static item => Assert.Matches("^[A-F0-9]{64}$", item.GetProperty("sha256").GetString()));
    }

    [Fact]
    public void RuntimeControlsEvidenceRecordsAppliedMechanicsAndNoTransferBoundary()
    {
        string path = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "interface-coverage",
            "trtexec-runtime-controls-runtime-evidence.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        JsonElement root = document.RootElement;
        JsonElement graphRun = root.GetProperty("threadsSpinCudaGraphRun");
        JsonElement noTransferRun = root.GetProperty("noDataTransfersRun");
        JsonElement boundary = root.GetProperty("proofBoundary");

        Assert.Equal(2, graphRun.GetProperty("threadsExecuted").GetInt32());
        Assert.Equal(new[] { 3, 3 }, graphRun.GetProperty("measurementRoundsPerContext").EnumerateArray().Select(static item => item.GetInt32()).ToArray());
        Assert.True(graphRun.GetProperty("useSpinWaitApplied").GetBoolean());
        Assert.True(graphRun.GetProperty("useCudaGraphApplied").GetBoolean());
        Assert.Equal(string.Empty, graphRun.GetProperty("useCudaGraphFallbackReason").GetString());
        Assert.True(graphRun.GetProperty("outputMatch").GetBoolean());
        Assert.True(noTransferRun.GetProperty("hasBenchmarkExecutionEvidence").GetBoolean());
        Assert.True(noTransferRun.GetProperty("noDataTransfersApplied").GetBoolean());
        Assert.Equal(0, noTransferRun.GetProperty("inputHostToDeviceCopies").GetInt32());
        Assert.Equal(0, noTransferRun.GetProperty("outputDeviceToHostCopies").GetInt32());
        Assert.False(noTransferRun.GetProperty("outputMatch").GetBoolean());
        Assert.False(noTransferRun.GetProperty("hasRawBindingProof").GetBoolean());
        Assert.Contains(root.GetProperty("unappliedControls").EnumerateArray(), static item => item.GetString() == "sleepTime");
        Assert.False(boundary.GetProperty("isRealModelRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("isPackageConsumerRuntimeProof").GetBoolean());
        Assert.False(boundary.GetProperty("canPublishPublicly").GetBoolean());
        Assert.All(root.GetProperty("binaryHashes").EnumerateArray(), static item => Assert.Matches("^[A-F0-9]{64}$", item.GetProperty("sha256").GetString()));
    }

    [Fact]
    public void SchedulerUsesIndependentOwnersAndMinimumIterationDurationContract()
    {
        string service = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp.Tools",
            "OnnxEngineBuildService.cs"));
        string artifactWriter = File.ReadAllText(Path.Combine(
            RepositoryPaths.Root,
            "src",
            "JYPPX.TensorRtSharp.Tools",
            "OnnxEngineRuntimeArtifactWriter.cs"));

        Assert.Contains("executionContextCount = options.RuntimeOptions.InfStreams ?? options.Streams", service, StringComparison.Ordinal);
        Assert.Contains("new OnnxEngineBenchmarkWorker(engine, safeProfileIndex, options.RuntimeOptions.UseSpinWait)", service, StringComparison.Ordinal);
        Assert.Contains("context = engine.CreateExecutionContext()", service, StringComparison.Ordinal);
        Assert.Contains("stream = new CudaStream(CudaStreamCreationFlags.NonBlocking)", service, StringComparison.Ordinal);
        Assert.Contains("_context = context", service, StringComparison.Ordinal);
        Assert.Contains("Stream = stream", service, StringComparison.Ordinal);
        Assert.Contains("while (measurementRounds < options.Iterations || measurementStopwatch.Elapsed < minimumDuration)", service, StringComparison.Ordinal);
        Assert.Contains("while (measurementRounds < options.Iterations || stopwatch.Elapsed < minimumDuration)", service, StringComparison.Ordinal);
        Assert.Contains("while (warmUpStopwatch.ElapsedMilliseconds < options.WarmUpMilliseconds)", service, StringComparison.Ordinal);
        Assert.Contains("Thread.Sleep(options.RuntimeOptions.IdleTimeMilliseconds!.Value)", service, StringComparison.Ordinal);
        Assert.Contains("worker.StartTiming()", service, StringComparison.Ordinal);
        Assert.Contains("worker.CompleteTiming(options.RuntimeOptions.UseSpinWait)", service, StringComparison.Ordinal);
        Assert.Contains("new Thread(() =>", service, StringComparison.Ordinal);
        Assert.Contains("CudaDevice.SetCurrent(deviceOrdinal)", service, StringComparison.Ordinal);
        Assert.Contains("measurementStart.Wait()", service, StringComparison.Ordinal);
        Assert.Contains("while (!_stopEvent.IsReady())", service, StringComparison.Ordinal);
        Assert.Contains("Thread.SpinWait(64)", service, StringComparison.Ordinal);
        Assert.Contains("Stream.BeginCapture(CudaStreamCaptureMode.ThreadLocal)", service, StringComparison.Ordinal);
        Assert.Contains("graphExec = graph.Instantiate()", service, StringComparison.Ordinal);
        Assert.Contains("_cudaGraphExec.Launch(Stream)", service, StringComparison.Ordinal);
        Assert.Contains("Bindings.AllocateDeviceBuffer(input.Name, runtimeShape)", service, StringComparison.Ordinal);
        Assert.Contains("if (!options.RuntimeOptions.NoDataTransfers)", service, StringComparison.Ordinal);
        Assert.Contains("CreateBenchmarkOnly", service, StringComparison.Ordinal);
        Assert.Contains("OnnxEngineBenchmarkSummary.CreateExecuted", service, StringComparison.Ordinal);
        Assert.Contains("runtime-benchmark-executed-no-data-transfers", artifactWriter, StringComparison.Ordinal);
        Assert.Contains("HasBenchmarkExecutionEvidence", artifactWriter, StringComparison.Ordinal);
    }

    [Fact]
    public void ExecutedBenchmarkReportSeparatesAppliedAndUnappliedControls()
    {
        TrtexecLikeOptions parsed = TrtexecLikeParser.Parse(new[]
        {
            "--iterations", "2",
            "--warmUp", "5",
            "--duration", "0",
            "--streams", "1",
            "--infStreams", "2",
            "--avgRuns", "2",
            "--percentile", "90",
            "--idleTime", "1",
            "--sleepTime", "3",
            "--threads",
            "--useSpinWait",
            "--noDataTransfers",
            "--useCudaGraph",
            "--dumpOutput",
            "--exportOutput", "no-transfer-output.json",
            "--dumpRawBindingsToFile", "no-transfer-output.raw"
        });
        TrtexecLikeRuntimeOptions runtimeOptions = parsed.RuntimeOptions;
        OnnxEngineBenchmarkSummary summary = new OnnxEngineBenchmarkSummary(
            new[] { 1.0f, 1.2f, 0.9f, 1.1f },
            avgRunsRequested: 2,
            avgRunsExecuted: 4,
            percentileRequested: 90,
            percentileElapsedMilliseconds: 1.2f,
            threadsRequested: 1,
            threadsExecuted: 2,
            noDataTransfersRequested: true,
            noDataTransfersApplied: true,
            useSpinWaitRequested: true,
            sleepTimeMillisecondsRequested: 3,
            sleepTimeMillisecondsApplied: 0,
            idleTimeMillisecondsRequested: 1,
            idleTimeMillisecondsApplied: 1,
            benchmarkBoundary: "benchmark-executed-bounded-runtime; test evidence boundary.",
            iterationsRequested: 2,
            measurementRoundsExecuted: 2,
            inferenceIterationsExecuted: 4,
            warmUpMillisecondsRequested: 5,
            warmUpElapsedMilliseconds: 5.5,
            warmUpIterationsExecuted: 6,
            durationSecondsRequested: 0,
            measurementElapsedMilliseconds: 4.5,
            streamsRequested: 1,
            infStreamsRequested: 2,
            executionContextsCreated: 2,
            concurrentStreamsExecuted: 2,
            useSpinWaitApplied: true,
            useCudaGraphRequested: true,
            useCudaGraphApplied: false,
            useCudaGraphFallbackReason: "worker-0:CudaException:capture unsupported",
            measurementRoundsPerContext: new[] { 2, 2 });
        OnnxEngineBuildResult result = new OnnxEngineBuildResult(
            success: true,
            skipped: false,
            state: "identity-roundtrip",
            tensorRtLine: TensorRtApiLine.TensorRt10,
            modelSource: "embedded-dynamic-identity",
            enginePath: "model.plan",
            parsed: true,
            engineSaved: true,
            engineFileRoundTrip: true,
            inferenceRan: true,
            outputMatch: true,
            profileIndex: 0,
            elapsedMilliseconds: 1.0f,
            skipReason: string.Empty,
            normalizedCommandLine: parsed.ToArgumentLine(),
            diagnostics: Array.Empty<string>(),
            logLines: new[] { "RuntimeBenchmark Contexts=2 MeasurementRounds=2" },
            runtimeOptions: runtimeOptions,
            benchmarkSummary: summary);

        using JsonDocument report = JsonDocument.Parse(OnnxEngineBuildDiagnostics.ToJson(result));
        JsonElement benchmark = report.RootElement.GetProperty("BenchmarkSummary");
        JsonElement status = report.RootElement.GetProperty("OptionImplementationStatus");
        string[] applied = status.GetProperty("AppliedOptions").EnumerateArray().Select(static item => item.GetString()!).ToArray();
        string[] parseOnly = status.GetProperty("ParseOnlyOptions").EnumerateArray().Select(static item => item.GetString()!).ToArray();

        Assert.Equal(2, benchmark.GetProperty("MeasurementRoundsExecuted").GetInt32());
        Assert.Equal(4, benchmark.GetProperty("InferenceIterationsExecuted").GetInt32());
        Assert.Equal(2, benchmark.GetProperty("ExecutionContextsCreated").GetInt32());
        Assert.Equal(2, benchmark.GetProperty("AveragedTimingSampleCount").GetInt32());
        Assert.Equal(new[] { 2, 2 }, benchmark.GetProperty("MeasurementRoundsPerContext").EnumerateArray().Select(static item => item.GetInt32()).ToArray());
        Assert.True(benchmark.GetProperty("UseSpinWaitApplied").GetBoolean());
        Assert.False(benchmark.GetProperty("UseCudaGraphApplied").GetBoolean());
        Assert.Contains("capture unsupported", benchmark.GetProperty("UseCudaGraphFallbackReason").GetString(), StringComparison.Ordinal);
        Assert.Contains("--iterations", applied);
        Assert.Contains("--warmUp", applied);
        Assert.Contains("--duration", applied);
        Assert.Contains("--infStreams", applied);
        Assert.Contains("--idleTime", applied);
        Assert.Contains("--avgRuns", applied);
        Assert.Contains("--percentile", applied);
        Assert.Contains("--threads", applied);
        Assert.Contains("--useSpinWait", applied);
        Assert.Contains("--noDataTransfers", applied);
        Assert.Contains("--streams", parseOnly);
        Assert.Contains("--sleepTime", parseOnly);
        Assert.Contains("--useCudaGraph", parseOnly);
        Assert.Contains("--dumpOutput", parseOnly);
        Assert.Contains("--exportOutput", parseOnly);
        Assert.Contains("--dumpRawBindingsToFile", parseOnly);
        Assert.DoesNotContain("--threads", parseOnly);
        Assert.DoesNotContain("--useSpinWait", parseOnly);
        Assert.DoesNotContain("--noDataTransfers", parseOnly);
        Assert.DoesNotContain("--dumpOutput", applied);
        Assert.DoesNotContain("--exportOutput", applied);
        Assert.DoesNotContain("--dumpRawBindingsToFile", applied);
        Assert.DoesNotContain("--infStreams", parseOnly);
    }
}

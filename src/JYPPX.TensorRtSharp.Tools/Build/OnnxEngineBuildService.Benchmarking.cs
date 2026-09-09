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
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed partial class OnnxEngineBuildService
{
    private static OnnxEngineBenchmarkRun RunBoundedBenchmark(
        IReadOnlyList<OnnxEngineBenchmarkWorker> workers,
        OnnxEngineBuildOptions options)
    {
        bool useCudaGraphApplied = TryEnableCudaGraphs(workers, options.UseCudaGraph, out string cudaGraphFallbackReason);
#if JYPPX_PUBLIC_STABLE_4_0_0
        int sleepTimeMillisecondsApplied = 0;
#else
        int sleepTimeMillisecondsApplied = options.RuntimeOptions.SleepTimeMilliseconds.GetValueOrDefault();
#endif
        using BenchmarkStartDelayGate? startDelayGate = BenchmarkStartDelayGate.Create(
            workers,
            sleepTimeMillisecondsApplied);
        return options.RuntimeOptions.UseThreads
            ? RunThreadedBoundedBenchmark(workers, options, sleepTimeMillisecondsApplied, useCudaGraphApplied, cudaGraphFallbackReason)
            : RunSingleThreadBoundedBenchmark(workers, options, sleepTimeMillisecondsApplied, useCudaGraphApplied, cudaGraphFallbackReason);
    }

    private static bool TryEnableCudaGraphs(
        IReadOnlyList<OnnxEngineBenchmarkWorker> workers,
        bool requested,
        out string fallbackReason)
    {
        fallbackReason = string.Empty;
        if (!requested)
        {
            return false;
        }

        for (int index = 0; index < workers.Count; index++)
        {
            if (workers[index].TryEnableCudaGraph(out string workerReason))
            {
                continue;
            }

            foreach (OnnxEngineBenchmarkWorker worker in workers)
            {
                worker.DisableCudaGraph();
            }

            fallbackReason = $"worker-{index}:{workerReason}";
            return false;
        }

        return true;
    }

    private static OnnxEngineBenchmarkRun RunSingleThreadBoundedBenchmark(
        IReadOnlyList<OnnxEngineBenchmarkWorker> workers,
        OnnxEngineBuildOptions options,
        int sleepTimeMillisecondsApplied,
        bool useCudaGraphApplied,
        string cudaGraphFallbackReason)
    {
        int warmUpIterations = 0;
        Stopwatch warmUpStopwatch = Stopwatch.StartNew();
        while (warmUpStopwatch.ElapsedMilliseconds < options.WarmUpMilliseconds)
        {
            foreach (OnnxEngineBenchmarkWorker worker in workers)
            {
                worker.Enqueue();
            }

            foreach (OnnxEngineBenchmarkWorker worker in workers)
            {
                worker.RecordCompletion();
            }

            foreach (OnnxEngineBenchmarkWorker worker in workers)
            {
                worker.WaitForCompletion(options.RuntimeOptions.UseSpinWait);
            }

            warmUpIterations += workers.Count;
        }
        warmUpStopwatch.Stop();

        List<float> timingSamples = new List<float>();
        int measurementRounds = 0;
        TensorRtInferenceExecutionSummary? lastExecutionSummary = null;
        Stopwatch measurementStopwatch = Stopwatch.StartNew();
        TimeSpan minimumDuration = TimeSpan.FromSeconds(options.DurationSeconds);
        while (measurementRounds < options.Iterations || measurementStopwatch.Elapsed < minimumDuration)
        {
            if (measurementRounds > 0 && options.RuntimeOptions.IdleTimeMilliseconds.GetValueOrDefault() > 0)
            {
                Thread.Sleep(options.RuntimeOptions.IdleTimeMilliseconds!.Value);
            }

            foreach (OnnxEngineBenchmarkWorker worker in workers)
            {
                worker.StartTiming();
                lastExecutionSummary = worker.Enqueue();
                worker.StopTiming();
            }

            foreach (OnnxEngineBenchmarkWorker worker in workers)
            {
                timingSamples.Add(worker.CompleteTiming(options.RuntimeOptions.UseSpinWait));
            }

            measurementRounds++;
        }
        measurementStopwatch.Stop();

        if (lastExecutionSummary == null || timingSamples.Count == 0)
        {
            throw new InvalidOperationException("Bounded benchmark did not execute any inference iterations.");
        }

        return new OnnxEngineBenchmarkRun(
            timingSamples,
            lastExecutionSummary,
            measurementRounds,
            warmUpIterations,
            warmUpStopwatch.Elapsed.TotalMilliseconds,
            measurementStopwatch.Elapsed.TotalMilliseconds,
            sleepTimeMillisecondsApplied,
            measurementRounds > 1 ? options.RuntimeOptions.IdleTimeMilliseconds.GetValueOrDefault() : 0,
            threadsExecuted: 1,
            useSpinWaitApplied: options.RuntimeOptions.UseSpinWait,
            useCudaGraphApplied,
            cudaGraphFallbackReason,
            Enumerable.Repeat(measurementRounds, workers.Count).ToArray());
    }

    private static OnnxEngineBenchmarkRun RunThreadedBoundedBenchmark(
        IReadOnlyList<OnnxEngineBenchmarkWorker> workers,
        OnnxEngineBuildOptions options,
        int sleepTimeMillisecondsApplied,
        bool useCudaGraphApplied,
        string cudaGraphFallbackReason)
    {
        int deviceOrdinal = CudaDevice.Current;
        OnnxEngineWorkerRun?[] runs = new OnnxEngineWorkerRun?[workers.Count];
        Exception?[] failures = new Exception?[workers.Count];
        Thread[] threads = new Thread[workers.Count];
        using CountdownEvent ready = new CountdownEvent(workers.Count);
        using CountdownEvent warmUpCompleted = new CountdownEvent(workers.Count);
        using ManualResetEventSlim start = new ManualResetEventSlim(false);
        using ManualResetEventSlim measurementStart = new ManualResetEventSlim(false);

        for (int index = 0; index < workers.Count; index++)
        {
            int workerIndex = index;
            threads[index] = new Thread(() =>
            {
                bool readySignaled = false;
                bool warmUpSignaled = false;
                try
                {
                    CudaDevice.SetCurrent(deviceOrdinal);
                    ready.Signal();
                    readySignaled = true;
                    start.Wait();

                    OnnxEngineWorkerWarmUp warmUp = RunWorkerWarmUp(workers[workerIndex], options);
                    warmUpCompleted.Signal();
                    warmUpSignaled = true;
                    measurementStart.Wait();
                    runs[workerIndex] = RunWorkerMeasurement(workers[workerIndex], options, warmUp);
                }
                catch (Exception exception)
                {
                    failures[workerIndex] = exception;
                }
                finally
                {
                    if (!readySignaled)
                    {
                        ready.Signal();
                    }

                    if (!warmUpSignaled)
                    {
                        warmUpCompleted.Signal();
                    }
                }
            })
            {
                IsBackground = true,
                Name = $"TensorRtExec-worker-{index}"
            };
            threads[index].Start();
        }

        ready.Wait();
        start.Set();
        warmUpCompleted.Wait();
        measurementStart.Set();
        foreach (Thread thread in threads)
        {
            thread.Join();
        }

        Exception? failure = failures.FirstOrDefault(static item => item != null);
        if (failure != null)
        {
            ExceptionDispatchInfo.Capture(failure).Throw();
        }

        OnnxEngineWorkerRun[] completedRuns = runs.Select(static item => item ?? throw new InvalidOperationException("A TensorRtExec benchmark worker completed without a result.")).ToArray();
        float[] timingSamples = completedRuns.SelectMany(static item => item.TimingSamplesMilliseconds).ToArray();
        if (timingSamples.Length == 0)
        {
            throw new InvalidOperationException("Bounded benchmark did not execute any inference iterations.");
        }

        return new OnnxEngineBenchmarkRun(
            timingSamples,
            completedRuns[completedRuns.Length - 1].LastExecutionSummary,
            completedRuns.Min(static item => item.MeasurementRoundsExecuted),
            completedRuns.Sum(static item => item.WarmUpIterationsExecuted),
            completedRuns.Max(static item => item.WarmUpElapsedMilliseconds),
            completedRuns.Max(static item => item.MeasurementElapsedMilliseconds),
            sleepTimeMillisecondsApplied,
            completedRuns.Any(static item => item.IdleTimeMillisecondsApplied > 0)
                ? options.RuntimeOptions.IdleTimeMilliseconds.GetValueOrDefault()
                : 0,
            threadsExecuted: workers.Count,
            useSpinWaitApplied: options.RuntimeOptions.UseSpinWait,
            useCudaGraphApplied,
            cudaGraphFallbackReason,
            completedRuns.Select(static item => item.MeasurementRoundsExecuted).ToArray());
    }

    private static OnnxEngineWorkerWarmUp RunWorkerWarmUp(
        OnnxEngineBenchmarkWorker worker,
        OnnxEngineBuildOptions options)
    {
        int iterations = 0;
        Stopwatch stopwatch = Stopwatch.StartNew();
        while (stopwatch.ElapsedMilliseconds < options.WarmUpMilliseconds)
        {
            worker.Enqueue();
            worker.RecordCompletion();
            worker.WaitForCompletion(options.RuntimeOptions.UseSpinWait);
            iterations++;
        }

        stopwatch.Stop();
        return new OnnxEngineWorkerWarmUp(iterations, stopwatch.Elapsed.TotalMilliseconds);
    }

    private static OnnxEngineWorkerRun RunWorkerMeasurement(
        OnnxEngineBenchmarkWorker worker,
        OnnxEngineBuildOptions options,
        OnnxEngineWorkerWarmUp warmUp)
    {
        List<float> timingSamples = new List<float>();
        int measurementRounds = 0;
        TensorRtInferenceExecutionSummary? lastExecutionSummary = null;
        Stopwatch stopwatch = Stopwatch.StartNew();
        TimeSpan minimumDuration = TimeSpan.FromSeconds(options.DurationSeconds);
        while (measurementRounds < options.Iterations || stopwatch.Elapsed < minimumDuration)
        {
            if (measurementRounds > 0 && options.RuntimeOptions.IdleTimeMilliseconds.GetValueOrDefault() > 0)
            {
                Thread.Sleep(options.RuntimeOptions.IdleTimeMilliseconds!.Value);
            }

            worker.StartTiming();
            lastExecutionSummary = worker.Enqueue();
            worker.StopTiming();
            timingSamples.Add(worker.CompleteTiming(options.RuntimeOptions.UseSpinWait));
            measurementRounds++;
        }

        stopwatch.Stop();
        if (lastExecutionSummary == null || timingSamples.Count == 0)
        {
            throw new InvalidOperationException("Bounded benchmark worker did not execute any inference iterations.");
        }

        return new OnnxEngineWorkerRun(
            timingSamples,
            lastExecutionSummary,
            measurementRounds,
            warmUp.IterationsExecuted,
            warmUp.ElapsedMilliseconds,
            stopwatch.Elapsed.TotalMilliseconds,
            measurementRounds > 1 ? options.RuntimeOptions.IdleTimeMilliseconds.GetValueOrDefault() : 0);
    }

    private sealed class OnnxEngineBenchmarkWorker : IDisposable
    {
        private readonly TensorRtExecutionContext _context;
        private readonly CudaEvent _startEvent;
        private readonly CudaEvent _stopEvent;
        private CudaGraph? _cudaGraph;
        private CudaGraphExec? _cudaGraphExec;
        private TensorRtInferenceExecutionSummary? _cudaGraphExecutionSummary;

        public OnnxEngineBenchmarkWorker(TensorRtEngine engine, int profileIndex, bool useSpinWait)
        {
            CudaStream? stream = null;
            TensorRtExecutionContext? context = null;
            TensorRtInferenceBindings? bindings = null;
            CudaEvent? startEvent = null;
            try
            {
                stream = new CudaStream(CudaStreamCreationFlags.NonBlocking);
                context = engine.CreateExecutionContext();
                bindings = new TensorRtInferenceBindings(engine, context, profileIndex);
                CudaEventCreationFlags eventFlags = useSpinWait
                    ? CudaEventCreationFlags.Default
                    : CudaEventCreationFlags.BlockingSync;
                startEvent = new CudaEvent(eventFlags);
                CudaEvent stopEvent = new CudaEvent(eventFlags);
                Stream = stream;
                _context = context;
                Bindings = bindings;
                _startEvent = startEvent;
                _stopEvent = stopEvent;
            }
            catch
            {
                startEvent?.Dispose();
                bindings?.Dispose();
                context?.Dispose();
                stream?.Dispose();
                throw;
            }
        }

        public CudaStream Stream { get; }

        public TensorRtInferenceBindings Bindings { get; }

        public void Configure(
            IReadOnlyList<OnnxEngineRuntimeInput> inputs,
            IReadOnlyList<TensorRtEngineTensorBinding> outputs,
            bool noDataTransfers)
        {
            foreach (OnnxEngineRuntimeInput input in inputs)
            {
                if (input.SetInputShape)
                {
                    Bindings.SetInputShape(input.Binding.Name, input.Shape);
                }
            }

            foreach (OnnxEngineRuntimeInput input in inputs)
            {
                if (noDataTransfers)
                {
                    Bindings.AllocateDeviceBuffer(input.Binding.Name, input.Shape);
                }
                else
                {
                    Bindings.CopyInputFromHost(input.Binding.Name, input.Values, input.Shape);
                }
            }

            _ = Bindings.GetReadiness(runShapeInference: true);
            foreach (TensorRtEngineTensorBinding output in outputs)
            {
                Bindings.AllocateDeviceBuffer(output.Name);
            }

            Bindings.BindAll();
        }

        public TensorRtInferenceExecutionSummary Enqueue()
        {
            if (_cudaGraphExec != null)
            {
                _cudaGraphExec.Launch(Stream);
                return _cudaGraphExecutionSummary ?? throw new InvalidOperationException("CUDA graph execution summary is unavailable.");
            }

            return Bindings.EnqueueAsync(Stream, synchronize: false, runShapeInference: false);
        }

        public bool TryEnableCudaGraph(out string fallbackReason)
        {
            CudaGraph? graph = null;
            CudaGraphExec? graphExec = null;
            bool captureActive = false;
            try
            {
                _ = Bindings.EnqueueAsync(Stream, synchronize: false, runShapeInference: false);
                Stream.Synchronize();

                Stream.BeginCapture(CudaStreamCaptureMode.ThreadLocal);
                captureActive = true;
                TensorRtInferenceExecutionSummary executionSummary = Bindings.EnqueueAsync(Stream, synchronize: false, runShapeInference: false);
                graph = Stream.EndCapture();
                captureActive = false;
                graphExec = graph.Instantiate();

                _cudaGraph = graph;
                _cudaGraphExec = graphExec;
                _cudaGraphExecutionSummary = executionSummary;
                fallbackReason = string.Empty;
                return true;
            }
            catch (Exception exception) when (IsCudaGraphFallbackException(exception))
            {
                if (captureActive)
                {
                    try
                    {
                        using CudaGraph abandonedGraph = Stream.EndCapture();
                    }
                    catch (Exception cleanupException) when (IsCudaGraphFallbackException(cleanupException))
                    {
                    }
                }

                graphExec?.Dispose();
                graph?.Dispose();
                if (Stream.CaptureStatus != CudaStreamCaptureStatus.None)
                {
                    throw new InvalidOperationException("CUDA graph capture failed and the worker stream did not return to a reusable state.", exception);
                }

                fallbackReason = $"{exception.GetType().Name}:{SanitizeDiagnostic(exception.Message)}";
                return false;
            }
        }

        public void DisableCudaGraph()
        {
            _cudaGraphExec?.Dispose();
            _cudaGraphExec = null;
            _cudaGraph?.Dispose();
            _cudaGraph = null;
            _cudaGraphExecutionSummary = null;
        }

        public void StartTiming()
        {
            _startEvent.Record(Stream);
        }

        public void StopTiming()
        {
            _stopEvent.Record(Stream);
        }

        public void RecordCompletion()
        {
            _stopEvent.Record(Stream);
        }

        public void WaitForCompletion(bool useSpinWait)
        {
            if (useSpinWait)
            {
                while (!_stopEvent.IsReady())
                {
                    Thread.SpinWait(64);
                }

                return;
            }

            _stopEvent.Synchronize();
        }

        public float CompleteTiming(bool useSpinWait)
        {
            WaitForCompletion(useSpinWait);
            return _stopEvent.ElapsedTimeSince(_startEvent);
        }

        public void Dispose()
        {
            try
            {
                Stream.Synchronize();
            }
            catch (CudaException)
            {
            }

            DisableCudaGraph();
            _stopEvent.Dispose();
            _startEvent.Dispose();
            Bindings.Dispose();
            _context.Dispose();
            Stream.Dispose();
        }

        private static bool IsCudaGraphFallbackException(Exception exception)
        {
            return exception is CudaException ||
                exception is TensorRtException ||
                exception is NotSupportedException ||
                exception is InvalidOperationException ||
                exception is BridgeProbeException ||
                exception is DllNotFoundException ||
                exception is BadImageFormatException ||
                exception is FileNotFoundException;
        }

        private static string SanitizeDiagnostic(string value)
        {
            return (value ?? string.Empty).Replace('\r', ' ').Replace('\n', ' ').Trim();
        }
    }

    private sealed class BenchmarkStartDelayGate : IDisposable
    {
        private readonly CudaStream _delayStream;
        private readonly CudaEvent _readyEvent;

        private BenchmarkStartDelayGate(
            IReadOnlyList<OnnxEngineBenchmarkWorker> workers,
            int milliseconds)
        {
            CudaStream? delayStream = null;
            CudaEvent? readyEvent = null;
            try
            {
                delayStream = new CudaStream(CudaStreamCreationFlags.NonBlocking);
                readyEvent = new CudaEvent(CudaEventCreationFlags.DisableTiming);
                EnqueueStartDelay(delayStream, milliseconds);
                readyEvent.Record(delayStream);
                foreach (OnnxEngineBenchmarkWorker worker in workers)
                {
                    worker.Stream.WaitFor(readyEvent);
                }

                _delayStream = delayStream;
                _readyEvent = readyEvent;
            }
            catch
            {
                try
                {
                    delayStream?.Synchronize();
                }
                catch (CudaException)
                {
                }

                readyEvent?.Dispose();
                delayStream?.Dispose();
                throw;
            }
        }

        public static BenchmarkStartDelayGate? Create(
            IReadOnlyList<OnnxEngineBenchmarkWorker> workers,
            int milliseconds)
        {
            return milliseconds > 0
                ? new BenchmarkStartDelayGate(workers, milliseconds)
                : null;
        }

        private static void EnqueueStartDelay(CudaStream stream, int milliseconds)
        {
#if JYPPX_PUBLIC_STABLE_4_0_0
            throw new NotSupportedException(
                "The stable 4.0.0 package does not expose the stream-ordered delay API.");
#else
            CudaStream delayStream = stream;
            delayStream.EnqueueDelay(milliseconds);
#endif
        }

        public void Dispose()
        {
            try
            {
                _delayStream.Synchronize();
            }
            finally
            {
                _readyEvent.Dispose();
                _delayStream.Dispose();
            }
        }
    }

    private sealed class OnnxEngineBenchmarkRun
    {
        public OnnxEngineBenchmarkRun(
            IReadOnlyList<float> timingSamplesMilliseconds,
            TensorRtInferenceExecutionSummary lastExecutionSummary,
            int measurementRoundsExecuted,
            int warmUpIterationsExecuted,
            double warmUpElapsedMilliseconds,
            double measurementElapsedMilliseconds,
            int sleepTimeMillisecondsApplied,
            int idleTimeMillisecondsApplied,
            int threadsExecuted,
            bool useSpinWaitApplied,
            bool useCudaGraphApplied,
            string useCudaGraphFallbackReason,
            IReadOnlyList<int> measurementRoundsPerContext)
        {
            TimingSamplesMilliseconds = timingSamplesMilliseconds;
            LastExecutionSummary = lastExecutionSummary;
            MeasurementRoundsExecuted = measurementRoundsExecuted;
            WarmUpIterationsExecuted = warmUpIterationsExecuted;
            WarmUpElapsedMilliseconds = warmUpElapsedMilliseconds;
            MeasurementElapsedMilliseconds = measurementElapsedMilliseconds;
            SleepTimeMillisecondsApplied = sleepTimeMillisecondsApplied;
            IdleTimeMillisecondsApplied = idleTimeMillisecondsApplied;
            ThreadsExecuted = threadsExecuted;
            UseSpinWaitApplied = useSpinWaitApplied;
            UseCudaGraphApplied = useCudaGraphApplied;
            UseCudaGraphFallbackReason = useCudaGraphFallbackReason ?? string.Empty;
            MeasurementRoundsPerContext = measurementRoundsPerContext ?? Array.Empty<int>();
        }

        public IReadOnlyList<float> TimingSamplesMilliseconds { get; }

        public TensorRtInferenceExecutionSummary LastExecutionSummary { get; }

        public int MeasurementRoundsExecuted { get; }

        public int WarmUpIterationsExecuted { get; }

        public double WarmUpElapsedMilliseconds { get; }

        public double MeasurementElapsedMilliseconds { get; }

        public int SleepTimeMillisecondsApplied { get; }

        public int IdleTimeMillisecondsApplied { get; }

        public int ThreadsExecuted { get; }

        public bool UseSpinWaitApplied { get; }

        public bool UseCudaGraphApplied { get; }

        public string UseCudaGraphFallbackReason { get; }

        public IReadOnlyList<int> MeasurementRoundsPerContext { get; }
    }

    private sealed class OnnxEngineWorkerWarmUp
    {
        public OnnxEngineWorkerWarmUp(int iterationsExecuted, double elapsedMilliseconds)
        {
            IterationsExecuted = iterationsExecuted;
            ElapsedMilliseconds = elapsedMilliseconds;
        }

        public int IterationsExecuted { get; }

        public double ElapsedMilliseconds { get; }
    }

    private sealed class OnnxEngineWorkerRun
    {
        public OnnxEngineWorkerRun(
            IReadOnlyList<float> timingSamplesMilliseconds,
            TensorRtInferenceExecutionSummary lastExecutionSummary,
            int measurementRoundsExecuted,
            int warmUpIterationsExecuted,
            double warmUpElapsedMilliseconds,
            double measurementElapsedMilliseconds,
            int idleTimeMillisecondsApplied)
        {
            TimingSamplesMilliseconds = timingSamplesMilliseconds;
            LastExecutionSummary = lastExecutionSummary;
            MeasurementRoundsExecuted = measurementRoundsExecuted;
            WarmUpIterationsExecuted = warmUpIterationsExecuted;
            WarmUpElapsedMilliseconds = warmUpElapsedMilliseconds;
            MeasurementElapsedMilliseconds = measurementElapsedMilliseconds;
            IdleTimeMillisecondsApplied = idleTimeMillisecondsApplied;
        }

        public IReadOnlyList<float> TimingSamplesMilliseconds { get; }

        public TensorRtInferenceExecutionSummary LastExecutionSummary { get; }

        public int MeasurementRoundsExecuted { get; }

        public int WarmUpIterationsExecuted { get; }

        public double WarmUpElapsedMilliseconds { get; }

        public double MeasurementElapsedMilliseconds { get; }

        public int IdleTimeMillisecondsApplied { get; }
    }

}

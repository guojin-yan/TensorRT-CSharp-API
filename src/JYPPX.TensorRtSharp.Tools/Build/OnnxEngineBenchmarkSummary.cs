using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed class OnnxEngineBenchmarkSummary
{
    public OnnxEngineBenchmarkSummary(
        IReadOnlyList<float> timingSamplesMilliseconds,
        int? avgRunsRequested,
        int avgRunsExecuted,
        float? percentileRequested,
        float? percentileElapsedMilliseconds,
        int? threadsRequested,
        int threadsExecuted,
        bool noDataTransfersRequested,
        bool noDataTransfersApplied,
        bool useSpinWaitRequested,
        int? sleepTimeMillisecondsRequested,
        int sleepTimeMillisecondsApplied,
        int? idleTimeMillisecondsRequested,
        int idleTimeMillisecondsApplied,
        string benchmarkBoundary,
        int iterationsRequested = 0,
        int measurementRoundsExecuted = 0,
        int inferenceIterationsExecuted = 0,
        int warmUpMillisecondsRequested = 0,
        double warmUpElapsedMilliseconds = 0,
        int warmUpIterationsExecuted = 0,
        int durationSecondsRequested = 0,
        double measurementElapsedMilliseconds = 0,
        int streamsRequested = 0,
        int? infStreamsRequested = null,
        int executionContextsCreated = 0,
        int concurrentStreamsExecuted = 0,
        bool useSpinWaitApplied = false,
        bool useCudaGraphRequested = false,
        bool useCudaGraphApplied = false,
        string useCudaGraphFallbackReason = "",
        IReadOnlyList<int>? measurementRoundsPerContext = null)
    {
        TimingSamplesMilliseconds = timingSamplesMilliseconds ?? Array.Empty<float>();
        AveragedTimingSamplesMilliseconds = AverageWindows(TimingSamplesMilliseconds, avgRunsRequested);
        AvgRunsRequested = avgRunsRequested;
        AvgRunsExecuted = avgRunsExecuted;
        PercentileRequested = percentileRequested;
        PercentileElapsedMilliseconds = percentileElapsedMilliseconds;
        ThreadsRequested = threadsRequested;
        ThreadsExecuted = threadsExecuted;
        NoDataTransfersRequested = noDataTransfersRequested;
        NoDataTransfersApplied = noDataTransfersApplied;
        UseSpinWaitRequested = useSpinWaitRequested;
        SleepTimeMillisecondsRequested = sleepTimeMillisecondsRequested;
        SleepTimeMillisecondsApplied = sleepTimeMillisecondsApplied;
        IdleTimeMillisecondsRequested = idleTimeMillisecondsRequested;
        IdleTimeMillisecondsApplied = idleTimeMillisecondsApplied;
        BenchmarkBoundary = benchmarkBoundary ?? string.Empty;
        IterationsRequested = iterationsRequested;
        MeasurementRoundsExecuted = measurementRoundsExecuted;
        InferenceIterationsExecuted = inferenceIterationsExecuted;
        WarmUpMillisecondsRequested = warmUpMillisecondsRequested;
        WarmUpElapsedMilliseconds = warmUpElapsedMilliseconds;
        WarmUpIterationsExecuted = warmUpIterationsExecuted;
        DurationSecondsRequested = durationSecondsRequested;
        MeasurementElapsedMilliseconds = measurementElapsedMilliseconds;
        StreamsRequested = streamsRequested;
        InfStreamsRequested = infStreamsRequested;
        ExecutionContextsCreated = executionContextsCreated;
        ConcurrentStreamsExecuted = concurrentStreamsExecuted;
        UseSpinWaitApplied = useSpinWaitApplied;
        UseCudaGraphRequested = useCudaGraphRequested;
        UseCudaGraphApplied = useCudaGraphApplied;
        UseCudaGraphFallbackReason = useCudaGraphFallbackReason ?? string.Empty;
        MeasurementRoundsPerContext = measurementRoundsPerContext ?? Array.Empty<int>();
    }

    public static OnnxEngineBenchmarkSummary Empty { get; } = new OnnxEngineBenchmarkSummary(
        Array.Empty<float>(),
        avgRunsRequested: null,
        avgRunsExecuted: 0,
        percentileRequested: null,
        percentileElapsedMilliseconds: null,
        threadsRequested: null,
        threadsExecuted: 0,
        noDataTransfersRequested: false,
        noDataTransfersApplied: false,
        useSpinWaitRequested: false,
        sleepTimeMillisecondsRequested: null,
        sleepTimeMillisecondsApplied: 0,
        idleTimeMillisecondsRequested: null,
        idleTimeMillisecondsApplied: 0,
        benchmarkBoundary: "benchmark-skipped; no runtime timing samples were produced.");

    public IReadOnlyList<float> TimingSamplesMilliseconds { get; }

    public int TimingSampleCount => TimingSamplesMilliseconds.Count;

    public IReadOnlyList<float> AveragedTimingSamplesMilliseconds { get; }

    public int AveragedTimingSampleCount => AveragedTimingSamplesMilliseconds.Count;

    public float? AverageElapsedMilliseconds => TimingSampleCount == 0 ? null : TimingSamplesMilliseconds.Sum() / TimingSampleCount;

    public float? MinElapsedMilliseconds => TimingSampleCount == 0 ? null : TimingSamplesMilliseconds.Min();

    public float? MaxElapsedMilliseconds => TimingSampleCount == 0 ? null : TimingSamplesMilliseconds.Max();

    public int? AvgRunsRequested { get; }

    public int AvgRunsExecuted { get; }

    public float? PercentileRequested { get; }

    public float? PercentileElapsedMilliseconds { get; }

    public int? ThreadsRequested { get; }

    public int ThreadsExecuted { get; }

    public bool NoDataTransfersRequested { get; }

    public bool NoDataTransfersApplied { get; }

    public bool UseSpinWaitRequested { get; }

    /// <summary>Gets whether CUDA event polling was applied. 获取是否实际应用了 CUDA event 轮询。</summary>
    public bool UseSpinWaitApplied { get; }

    /// <summary>Gets whether CUDA graph execution was requested. 获取是否请求了 CUDA graph 执行。</summary>
    public bool UseCudaGraphRequested { get; }

    /// <summary>Gets whether every worker applied CUDA graph launch. 获取是否所有 worker 都应用了 CUDA graph launch。</summary>
    public bool UseCudaGraphApplied { get; }

    /// <summary>Gets the controlled CUDA graph fallback diagnostic. 获取 CUDA graph 受控回退诊断。</summary>
    public string UseCudaGraphFallbackReason { get; }

    public int? SleepTimeMillisecondsRequested { get; }

    public int SleepTimeMillisecondsApplied { get; }

    public int? IdleTimeMillisecondsRequested { get; }

    public int IdleTimeMillisecondsApplied { get; }

    public string BenchmarkBoundary { get; }

    public int IterationsRequested { get; }

    public int MeasurementRoundsExecuted { get; }

    public int InferenceIterationsExecuted { get; }

    public int WarmUpMillisecondsRequested { get; }

    public double WarmUpElapsedMilliseconds { get; }

    public int WarmUpIterationsExecuted { get; }

    public int DurationSecondsRequested { get; }

    public double MeasurementElapsedMilliseconds { get; }

    public int StreamsRequested { get; }

    public int? InfStreamsRequested { get; }

    public int ExecutionContextsCreated { get; }

    public int ConcurrentStreamsExecuted { get; }

    /// <summary>Gets executed measurement rounds for each context. 获取每个 context 实际执行的测量轮次。</summary>
    public IReadOnlyList<int> MeasurementRoundsPerContext { get; }

    public static OnnxEngineBenchmarkSummary Create(
        IReadOnlyList<float> timingSamplesMilliseconds,
        TrtexecLikeRuntimeOptions runtimeOptions,
        bool inferenceRan,
        bool outputMatch)
    {
        IReadOnlyList<float> samples = timingSamplesMilliseconds ?? Array.Empty<float>();
        int executedSamples = samples.Count;
        int threadsExecuted = executedSamples == 0 ? 0 : 1;
        string boundary = inferenceRan && outputMatch && executedSamples > 0
            ? "benchmark-executed-synthetic-input; avgRuns timing samples apply only to the embedded identity sample, threads are not parallelized, noDataTransfers does not suppress required sample copies, and the result is not real-model or package-consumer proof."
            : "benchmark-skipped; no runtime timing samples were produced.";

        return new OnnxEngineBenchmarkSummary(
            samples,
            runtimeOptions.AvgRuns,
            executedSamples,
            runtimeOptions.Percentile,
            PercentileOrNull(samples, runtimeOptions.Percentile),
            runtimeOptions.Threads,
            threadsExecuted,
            runtimeOptions.NoDataTransfers,
            noDataTransfersApplied: false,
            runtimeOptions.UseSpinWait,
            runtimeOptions.SleepTimeMilliseconds,
            sleepTimeMillisecondsApplied: 0,
            runtimeOptions.IdleTimeMilliseconds,
            idleTimeMillisecondsApplied: 0,
            boundary);
    }

    internal static OnnxEngineBenchmarkSummary CreateExecuted(
        IReadOnlyList<float> timingSamplesMilliseconds,
        OnnxEngineBuildOptions options,
        int measurementRoundsExecuted,
        int warmUpIterationsExecuted,
        double warmUpElapsedMilliseconds,
        double measurementElapsedMilliseconds,
        int sleepTimeMillisecondsApplied,
        int executionContextsCreated,
        int threadsExecuted,
        bool useSpinWaitApplied,
        bool useCudaGraphApplied,
        string useCudaGraphFallbackReason,
        IReadOnlyList<int> measurementRoundsPerContext)
    {
        IReadOnlyList<float> samples = timingSamplesMilliseconds ?? Array.Empty<float>();
        TrtexecLikeRuntimeOptions runtimeOptions = options.RuntimeOptions;
        int idleApplied = measurementRoundsExecuted > 1
            ? runtimeOptions.IdleTimeMilliseconds ?? 0
            : 0;
        string graphBoundary = options.UseCudaGraph
            ? (useCudaGraphApplied
                ? "CUDA graph capture/instantiate/launch was applied."
                : $"CUDA graph capture fell back to direct enqueue ({useCudaGraphFallbackReason}).")
            : "CUDA graph was not requested.";
        string boundary =
            "benchmark-executed-bounded-runtime; iterations, warmUp, duration, streams/infStreams, avgRuns statistics, percentile, stream-ordered sleepTime, idleTime, requested host threads, spin-wait completion, and noDataTransfers are backed by actual scheduler behavior; " +
            graphBoundary +
            " sleepTime uses one bridge-owned CUDA host function followed by an event fan-out to every inference stream; noDataTransfers suppresses tensor readback and therefore cannot establish output correctness; tensor correctness and package-consumer proof require separate model-specific evidence.";

        return new OnnxEngineBenchmarkSummary(
            samples,
            runtimeOptions.AvgRuns,
            runtimeOptions.AvgRuns.HasValue ? Math.Min(runtimeOptions.AvgRuns.Value, samples.Count) : 0,
            runtimeOptions.Percentile,
            PercentileOrNull(samples, runtimeOptions.Percentile),
            runtimeOptions.Threads,
            threadsExecuted,
            runtimeOptions.NoDataTransfers,
            noDataTransfersApplied: runtimeOptions.NoDataTransfers,
            runtimeOptions.UseSpinWait,
            runtimeOptions.SleepTimeMilliseconds,
            sleepTimeMillisecondsApplied,
            runtimeOptions.IdleTimeMilliseconds,
            idleTimeMillisecondsApplied: idleApplied,
            boundary,
            iterationsRequested: options.Iterations,
            measurementRoundsExecuted,
            inferenceIterationsExecuted: samples.Count,
            warmUpMillisecondsRequested: options.WarmUpMilliseconds,
            warmUpElapsedMilliseconds,
            warmUpIterationsExecuted,
            durationSecondsRequested: options.DurationSeconds,
            measurementElapsedMilliseconds,
            streamsRequested: options.Streams,
            infStreamsRequested: runtimeOptions.InfStreams,
            executionContextsCreated,
            concurrentStreamsExecuted: executionContextsCreated,
            useSpinWaitApplied,
            useCudaGraphRequested: options.UseCudaGraph,
            useCudaGraphApplied,
            useCudaGraphFallbackReason,
            measurementRoundsPerContext);
    }

    private static float? PercentileOrNull(IReadOnlyList<float> values, float? percentile)
    {
        if (values.Count == 0 || !percentile.HasValue)
        {
            return null;
        }

        float[] sorted = values.OrderBy(static value => value).ToArray();
        int index = (int)Math.Ceiling((percentile.Value / 100.0f) * sorted.Length) - 1;
        index = Math.Max(0, Math.Min(sorted.Length - 1, index));
        return sorted[index];
    }

    private static IReadOnlyList<float> AverageWindows(IReadOnlyList<float> values, int? windowSize)
    {
        if (values.Count == 0 || !windowSize.HasValue)
        {
            return Array.Empty<float>();
        }

        int size = Math.Max(1, windowSize.Value);
        List<float> averages = new List<float>((values.Count + size - 1) / size);
        for (int start = 0; start < values.Count; start += size)
        {
            int count = Math.Min(size, values.Count - start);
            float total = 0;
            for (int offset = 0; offset < count; offset++)
            {
                total += values[start + offset];
            }

            averages.Add(total / count);
        }

        return averages;
    }
}

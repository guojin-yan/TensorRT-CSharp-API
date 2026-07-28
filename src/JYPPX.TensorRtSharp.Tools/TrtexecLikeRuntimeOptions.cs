using System;
using System.Collections.Generic;
using System.Globalization;

namespace JYPPX.TensorRtSharp.Tools;

public sealed class TrtexecLikeRuntimeOptions
{
    public TrtexecLikeRuntimeOptions(
        bool noDataTransfers,
        bool useSpinWait,
        int? threads,
        int? avgRuns,
        float? percentile,
        int? sleepTimeMilliseconds,
        int? idleTimeMilliseconds,
        int? infStreams,
        string loadInputs,
        bool dumpOutput,
        string dumpRawBindingsToFile,
        string exportOutputPath,
        string exportTimesPath,
        string exportProfilePath,
        string saveProfilePath)
    {
        if (threads.HasValue && threads.Value <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(threads), "Thread count must be positive.");
        }
        if (avgRuns.HasValue && avgRuns.Value <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(avgRuns), "Average-run window must be positive.");
        }
        if (percentile.HasValue && (percentile.Value < 0 || percentile.Value > 100))
        {
            throw new ArgumentOutOfRangeException(nameof(percentile), "Percentile must be between 0 and 100.");
        }
        if (sleepTimeMilliseconds.HasValue && sleepTimeMilliseconds.Value < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sleepTimeMilliseconds), "Sleep time must be non-negative.");
        }
        if (idleTimeMilliseconds.HasValue && idleTimeMilliseconds.Value < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(idleTimeMilliseconds), "Idle time must be non-negative.");
        }
        if (infStreams.HasValue && infStreams.Value <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(infStreams), "Inference stream count must be positive.");
        }
        NoDataTransfers = noDataTransfers;
        UseSpinWait = useSpinWait;
        Threads = threads;
        AvgRuns = avgRuns;
        Percentile = percentile;
        SleepTimeMilliseconds = sleepTimeMilliseconds;
        IdleTimeMilliseconds = idleTimeMilliseconds;
        InfStreams = infStreams;
        LoadInputs = loadInputs ?? string.Empty;
        DumpOutput = dumpOutput;
        DumpRawBindingsToFile = dumpRawBindingsToFile ?? string.Empty;
        ExportOutputPath = exportOutputPath ?? string.Empty;
        ExportTimesPath = exportTimesPath ?? string.Empty;
        ExportProfilePath = exportProfilePath ?? string.Empty;
        SaveProfilePath = saveProfilePath ?? string.Empty;
    }

    public static TrtexecLikeRuntimeOptions Default { get; } = new TrtexecLikeRuntimeOptions(
        false,
        false,
        null,
        null,
        null,
        null,
        null,
        null,
        string.Empty,
        false,
        string.Empty,
        string.Empty,
        string.Empty,
        string.Empty,
        string.Empty);

    public bool NoDataTransfers { get; }

    public bool UseSpinWait { get; }

    public int? Threads { get; }

    /// <summary>Gets whether official boolean thread mode was requested. 获取是否请求了官方布尔线程模式。</summary>
    public bool UseThreads => Threads.HasValue;

    public int? AvgRuns { get; }

    public float? Percentile { get; }

    public int? SleepTimeMilliseconds { get; }

    public int? IdleTimeMilliseconds { get; }

    public int? InfStreams { get; }

    public string LoadInputs { get; }

    public bool DumpOutput { get; }

    public string DumpRawBindingsToFile { get; }

    public string ExportOutputPath { get; }

    public string ExportTimesPath { get; }

    public string ExportProfilePath { get; }

    public string SaveProfilePath { get; }

    public bool RequestsOutputCapture =>
        DumpOutput ||
        !string.IsNullOrWhiteSpace(DumpRawBindingsToFile) ||
        !string.IsNullOrWhiteSpace(ExportOutputPath);

    public bool HasRuntimeDiagnostics =>
        NoDataTransfers ||
        UseSpinWait ||
        Threads.HasValue ||
        AvgRuns.HasValue ||
        Percentile.HasValue ||
        SleepTimeMilliseconds.HasValue ||
        IdleTimeMilliseconds.HasValue ||
        InfStreams.HasValue ||
        !string.IsNullOrWhiteSpace(LoadInputs) ||
        DumpOutput ||
        !string.IsNullOrWhiteSpace(DumpRawBindingsToFile) ||
        !string.IsNullOrWhiteSpace(ExportOutputPath) ||
        !string.IsNullOrWhiteSpace(ExportTimesPath) ||
        !string.IsNullOrWhiteSpace(ExportProfilePath) ||
        !string.IsNullOrWhiteSpace(SaveProfilePath);

    public IReadOnlyList<string> ToArgumentSegments()
    {
        List<string> args = new List<string>();
        AddSwitch(args, "--noDataTransfers", NoDataTransfers);
        AddSwitch(args, "--useSpinWait", UseSpinWait);
        AddSwitch(args, "--threads", UseThreads);
        Add(args, "--avgRuns", FormatNullable(AvgRuns));
        Add(args, "--percentile", Percentile.HasValue ? Percentile.Value.ToString(CultureInfo.InvariantCulture) : string.Empty);
        Add(args, "--sleepTime", FormatNullable(SleepTimeMilliseconds));
        Add(args, "--idleTime", FormatNullable(IdleTimeMilliseconds));
        Add(args, "--infStreams", FormatNullable(InfStreams));
        Add(args, "--loadInputs", LoadInputs);
        AddSwitch(args, "--dumpOutput", DumpOutput);
        Add(args, "--dumpRawBindingsToFile", DumpRawBindingsToFile);
        Add(args, "--exportOutput", ExportOutputPath);
        Add(args, "--exportTimes", ExportTimesPath);
        Add(args, "--exportProfile", ExportProfilePath);
        Add(args, "--saveProfile", SaveProfilePath);
        return args;
    }

    public IReadOnlyList<string> ToDiagnostics()
    {
        List<string> diagnostics = new List<string>();
        if (!HasRuntimeDiagnostics)
        {
            return diagnostics;
        }

        diagnostics.Add("Runtime benchmark/output options are recorded and supported controls are conditionally applied by the bounded runtime; tensor correctness still requires output readback plus model-specific validation.");
        AddDiagnostic(diagnostics, "NoDataTransfers", NoDataTransfers);
        AddDiagnostic(diagnostics, "UseSpinWait", UseSpinWait);
        AddDiagnostic(diagnostics, "Threads", Threads);
        AddDiagnostic(diagnostics, "AvgRuns", AvgRuns);
        AddDiagnostic(diagnostics, "Percentile", Percentile);
        AddDiagnostic(diagnostics, "SleepTimeMs", SleepTimeMilliseconds);
        AddDiagnostic(diagnostics, "IdleTimeMs", IdleTimeMilliseconds);
        AddDiagnostic(diagnostics, "InfStreams", InfStreams);
        AddDiagnostic(diagnostics, "LoadInputs", LoadInputs);
        AddDiagnostic(diagnostics, "DumpOutput", DumpOutput);
        AddDiagnostic(diagnostics, "DumpRawBindingsToFile", DumpRawBindingsToFile);
        AddDiagnostic(diagnostics, "ExportOutput", ExportOutputPath);
        AddDiagnostic(diagnostics, "ExportTimes", ExportTimesPath);
        AddDiagnostic(diagnostics, "ExportProfile", ExportProfilePath);
        AddDiagnostic(diagnostics, "SaveProfile", SaveProfilePath);
        return diagnostics;
    }

    private static void Add(List<string> args, string name, string value)
    {
        if (!string.IsNullOrWhiteSpace(value))
        {
            args.Add(name);
            args.Add(QuoteIfNeeded(value));
        }
    }

    private static void AddSwitch(List<string> args, string name, bool enabled)
    {
        if (enabled)
        {
            args.Add(name);
        }
    }

    private static void AddDiagnostic(List<string> diagnostics, string name, string value)
    {
        if (!string.IsNullOrWhiteSpace(value))
        {
            diagnostics.Add(name + "=" + value);
        }
    }

    private static void AddDiagnostic(List<string> diagnostics, string name, bool enabled)
    {
        if (enabled)
        {
            diagnostics.Add(name + "=True");
        }
    }

    private static void AddDiagnostic(List<string> diagnostics, string name, int? value)
    {
        if (value.HasValue)
        {
            diagnostics.Add(name + "=" + value.Value.ToString(CultureInfo.InvariantCulture));
        }
    }

    private static void AddDiagnostic(List<string> diagnostics, string name, float? value)
    {
        if (value.HasValue)
        {
            diagnostics.Add(name + "=" + value.Value.ToString(CultureInfo.InvariantCulture));
        }
    }

    private static string FormatNullable(int? value)
    {
        return value.HasValue ? value.Value.ToString(CultureInfo.InvariantCulture) : string.Empty;
    }

    private static string QuoteIfNeeded(string value)
    {
        return value.IndexOf(' ') >= 0 ? "\"" + value + "\"" : value;
    }
}

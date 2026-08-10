using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using JYPPX.TensorRtSharp;

namespace CallbackLifecycleSample;

internal sealed class CallbackRecords
{
    private readonly ConcurrentQueue<LogRecord> _logs = new ConcurrentQueue<LogRecord>();
    private readonly ConcurrentDictionary<string, byte> _progressPhases = new ConcurrentDictionary<string, byte>(StringComparer.Ordinal);
    private readonly ConcurrentDictionary<string, byte> _profileLayers = new ConcurrentDictionary<string, byte>(StringComparer.Ordinal);
    private long _progressEventCount;
    private long _profileEventCount;
    private long _profileInvalidMetadataCount;
    private double _profileTotalMilliseconds;

    public void RecordLog(TensorRtLogSeverity severity, string message)
    {
        _logs.Enqueue(new LogRecord(severity.ToString(), message ?? string.Empty));
    }

    public bool RecordProgress(TensorRtProgressMonitorEvent progressEvent)
    {
        Interlocked.Increment(ref _progressEventCount);
        if (!string.IsNullOrWhiteSpace(progressEvent.PhaseName))
        {
            _progressPhases.TryAdd(progressEvent.PhaseName, 0);
        }

        return true;
    }

    public void RecordProfile(string layerName, float milliseconds)
    {
        Interlocked.Increment(ref _profileEventCount);
        if (string.IsNullOrWhiteSpace(layerName) || !float.IsFinite(milliseconds) || milliseconds < 0.0f)
        {
            Interlocked.Increment(ref _profileInvalidMetadataCount);
            return;
        }

        _profileLayers.TryAdd(layerName, 0);
        lock (_profileLayers)
        {
            _profileTotalMilliseconds += milliseconds;
        }
    }

    public long LogCount => _logs.Count;

    public long ProgressEventCount => Interlocked.Read(ref _progressEventCount);

    public int DistinctProgressPhaseCount => _progressPhases.Count;

    public long ProfileEventCount => Interlocked.Read(ref _profileEventCount);

    public int DistinctProfileLayerCount => _profileLayers.Count;

    public bool ProfileMetadataCopied =>
        Interlocked.Read(ref _profileInvalidMetadataCount) == 0 && _profileLayers.Count > 0;

    public double ProfileTotalMilliseconds
    {
        get
        {
            lock (_profileLayers)
            {
                return _profileTotalMilliseconds;
            }
        }
    }

    public object[] GetLogPreview()
    {
        return _logs.Take(5).Select(static record => (object)new
        {
            severity = record.Severity,
            message = record.Message
        }).ToArray();
    }

    private sealed record LogRecord(string Severity, string Message);
}

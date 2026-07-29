using System;
using System.Collections.Generic;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    private static TensorRtErrorRecorderSnapshot ReadOwnerErrorRecorderSnapshot(
        TensorRtApiLine line,
        NativeTensorRtErrorRecorderSnapshotInfo info,
        Func<int, NativeTensorRtErrorRecordInfo> readError)
    {
        bool hasRecorder = info.HasRecorder != 0;
        int errorCount = Math.Max(0, info.ErrorCount);
        if (!hasRecorder || errorCount == 0)
        {
            return BridgeInfoMapper.ToManaged(line, info, Array.Empty<TensorRtErrorRecord>());
        }

        List<TensorRtErrorRecord> records = new List<TensorRtErrorRecord>(errorCount);
        for (int index = 0; index < errorCount; index++)
        {
            records.Add(BridgeInfoMapper.ToManaged(readError(index)));
        }

        return BridgeInfoMapper.ToManaged(line, info, records);
    }
}

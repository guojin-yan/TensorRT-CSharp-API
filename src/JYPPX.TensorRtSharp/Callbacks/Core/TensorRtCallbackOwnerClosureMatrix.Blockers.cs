using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp;

public static partial class TensorRtCallbackOwnerClosureMatrix
{
    private static string[] BuildMatrixBlockers(IEnumerable<TensorRtCallbackOwnerClosureMatrixRow> rows)
    {
        List<string> blockers = new List<string>();
        foreach (TensorRtCallbackOwnerClosureMatrixRow row in rows)
        {
            foreach (string blocker in row.BlockedPrerequisites)
            {
                AddBlocker(blockers, row.OwnerFamily + ": " + blocker);
            }
        }

        return blockers.ToArray();
    }

    private static void AddBlockerIfFalse(List<string> blockers, bool condition, string blocker)
    {
        if (!condition)
        {
            AddBlocker(blockers, blocker);
        }
    }

    private static void AddBlocker(List<string> blockers, string blocker)
    {
        if (!string.IsNullOrWhiteSpace(blocker) && !blockers.Contains(blocker))
        {
            blockers.Add(blocker);
        }
    }
}

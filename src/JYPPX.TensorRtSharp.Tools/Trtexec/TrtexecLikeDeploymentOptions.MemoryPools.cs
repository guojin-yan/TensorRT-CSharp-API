using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed partial class TrtexecLikeDeploymentOptions
{
    private string MemoryPoolSizesToArgument()
    {
        return MemoryPoolSizes.Count == 0
            ? string.Empty
            : string.Join(",", MemoryPoolSizes.Select(static item => item.ToArgumentSegment()));
    }
}

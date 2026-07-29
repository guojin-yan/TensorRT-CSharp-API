using System;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

internal sealed partial class TensorRtOutputAllocatorRuntimeGate
{
    internal TensorRtOutputAllocatorRuntimeGateResult RunInternalNotifyShapeRuntimeGate(TensorRtOutputAllocatorRuntimeGateRequest request)
    {
        return RunInternalRuntimeGate(NotifyShapeOperation, request);
    }

    internal TensorRtOutputAllocatorRuntimeGateResult RunInternalReallocateOutputRuntimeGate(TensorRtOutputAllocatorRuntimeGateRequest request)
    {
        return RunInternalRuntimeGate(ReallocateOutputOperation, request);
    }

}

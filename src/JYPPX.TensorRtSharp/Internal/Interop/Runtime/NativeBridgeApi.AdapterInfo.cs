using System;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static NativeTensorRtAdapterInfo GetAdapterInfo(TensorRtApiLine line)
    {
        return GetAdapterInfoCore(GetBindings(line));
    }

}

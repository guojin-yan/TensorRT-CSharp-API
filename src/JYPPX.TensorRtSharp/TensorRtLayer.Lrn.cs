using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtLayer
{
    public int GetLrnWindowSize()
    {
        return NativeBridgeApi.GetLrnWindowSize(Line, _handle);
    }

    public void SetLrnWindowSize(int windowSize)
    {
        if (windowSize < 1 || windowSize > 15 || windowSize % 2 == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(windowSize), "LRN window size must be an odd value in [1, 15].");
        }

        NativeBridgeApi.SetLrnWindowSize(Line, _handle, windowSize);
    }

    public float GetLrnAlpha()
    {
        return NativeBridgeApi.GetLrnAlpha(Line, _handle);
    }

    public void SetLrnAlpha(float alpha)
    {
        NativeBridgeApi.SetLrnAlpha(Line, _handle, alpha);
    }

    public float GetLrnBeta()
    {
        return NativeBridgeApi.GetLrnBeta(Line, _handle);
    }

    public void SetLrnBeta(float beta)
    {
        NativeBridgeApi.SetLrnBeta(Line, _handle, beta);
    }

    public float GetLrnK()
    {
        return NativeBridgeApi.GetLrnK(Line, _handle);
    }

    public void SetLrnK(float k)
    {
        NativeBridgeApi.SetLrnK(Line, _handle, k);
    }
}

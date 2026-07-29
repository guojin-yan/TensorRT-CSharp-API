using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public sealed partial class CudaArray
{
    /// <summary>
    /// Materializes a 2D region as a managed byte array. 将二维区域物化为托管字节数组。
    /// </summary>
    public byte[] ToArray2D(int pitch, int widthBytes, int height)
    {
        Validate2DRegion(pitch, widthBytes, height, nameof(pitch));
        byte[] data = new byte[checked(pitch * height)];
        CopyTo2D(data, pitch, widthBytes, height);
        return data;
    }

    /// <summary>
    /// Materializes a 3D region as a managed byte array. 将三维区域物化为托管字节数组。
    /// </summary>
    public byte[] ToArray3D(int pitch, int pitchedWidthBytes, int pitchedHeight, int widthBytes, int height, int depth)
    {
        Validate3DRegion(pitch, pitchedWidthBytes, pitchedHeight, widthBytes, height, depth, nameof(pitch));
        byte[] data = new byte[checked(pitch * pitchedHeight * depth)];
        CopyTo3D(data, pitch, pitchedWidthBytes, pitchedHeight, widthBytes, height, depth);
        return data;
    }

}

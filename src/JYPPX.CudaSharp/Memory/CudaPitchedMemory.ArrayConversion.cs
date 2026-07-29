using System;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public sealed partial class CudaPitchedMemory
{
    /// <summary>
    /// Copies this pitched allocation to a newly allocated logical 2D host buffer.
    /// 将当前 pitched 分配复制到新建的逻辑 2D 主机缓冲区。
    /// </summary>
    /// <param name="destinationPitch">The byte pitch of each host row. 主机每行 pitch，单位为字节。</param>
    /// <returns>The copied host buffer. 复制得到的主机缓冲区。</returns>
    public byte[] ToArray2D(int destinationPitch)
    {
        ValidateHostPitch(destinationPitch, WidthInBytes, nameof(destinationPitch));
        byte[] data = new byte[checked(destinationPitch * Height)];
        CopyTo2D(data, destinationPitch);
        return data;
    }

    /// <summary>
    /// Copies this pitched allocation to a newly allocated logical 3D host buffer.
    /// 将当前 pitched 分配复制到新建的逻辑 3D 主机缓冲区。
    /// </summary>
    /// <param name="destinationPitch">The byte pitch of each host row. 主机每行 pitch，单位为字节。</param>
    /// <param name="height">The height of one slice. 单个 slice 的高度。</param>
    /// <param name="depth">The number of slices. slice 数量。</param>
    /// <returns>The copied host buffer. 复制得到的主机缓冲区。</returns>
    public byte[] ToArray3D(int destinationPitch, int height, int depth)
    {
        Validate3DRegion(destinationPitch, WidthInBytes, height, depth, nameof(destinationPitch));
        Validate3DExtent(WidthInBytes, height, depth);
        byte[] data = new byte[checked(destinationPitch * height * depth)];
        CopyTo3D(data, destinationPitch, WidthInBytes, height, depth);
        return data;
    }

}

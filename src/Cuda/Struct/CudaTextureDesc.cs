using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Cuda.Struct
{


    /**
     * CUDA texture descriptor
     */
    public struct CudaTextureDesc
    {
        /**
         * Texture address mode for up to 3 dimensions
         */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        cudaTextureAddressMode[] addressMode;
        /**
         * Texture filter mode
         */
        cudaTextureFilterMode filterMode;
        /**
         * Texture read mode
         */
        cudaTextureReadMode readMode;
        /**
         * Perform sRGB->linear conversion during texture read
         */
        int sRGB;
        /**
         * Texture Border Color
         */
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
        float[] borderColor;
        /**
         * Indicates whether texture reads are normalized or not
         */
        int normalizedCoords;
        /**
         * Limit to the anisotropy ratio
         */
        uint maxAnisotropy;
        /**
         * Mipmap filter mode
         */
        cudaTextureFilterMode mipmapFilterMode;
        /**
         * Offset applied to the supplied mipmap level
         */
        float mipmapLevelBias;
        /**
         * Lower end of the mipmap level range to clamp access to
         */
        float minMipmapLevelClamp;
        /**
         * Upper end of the mipmap level range to clamp access to
         */
        float maxMipmapLevelClamp;
        /**
         * Disable any trilinear filtering optimizations.
         */
        int disableTrilinearOptimization;
    };
}

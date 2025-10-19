using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    [StructLayout(LayoutKind.Sequential)]
    public struct Dims
    {
        /// <summary>
        /// The rank (number of dimensions).
        /// </summary>
        public int nbDims;

        /// <summary>
        /// The extent of each dimension.
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
        public long[] d;

        public Dims()
        {
            nbDims = 0;
            d = new long[8];
        }
        public Dims(int leng, int[] data)
        {
            d = new long[8];
            nbDims = leng;
            Array.Copy(data, d, Math.Min(leng, 8));

        }
        public Dims(params int[] data)
        {
            d = new long[8];
            nbDims = data.Length;
            Array.Copy(data, d, Math.Min(nbDims, 8));
        }
        public long GetDimension(int index)
        {
            if (index >= 0 && index < nbDims && index < 8)
            {
                return d[index];
            }
            return 0;
        }

        public void SetDimension(int index, int value)
        {
            if (index >= 0 && index < 8)
            {
                if (index >= nbDims)
                {
                    nbDims = index + 1;
                }
                d[index] = value;
            }
        }


        public long GetElementProduct()
        {
            long count = 1;
            for (int i = 0; i < nbDims; i++)
            {
                count *= d[i];
            }
            return count;
        }
        public override string ToString()
        {
            if (nbDims <= 0 || d == null || d.Length < nbDims)
            {
                return "EmptyDims";
            }
            var builder = new System.Text.StringBuilder();
            builder.Append('[');

            for (int i = 0; i < nbDims; i++)
            {
                builder.Append(d[i]);
                if (i < nbDims - 1)
                {
                    builder.Append(", ");
                }
            }

            builder.Append(']');
            return builder.ToString();
        }
    }
}
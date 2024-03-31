﻿using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorRtSharp
{
    public struct Dims
    {
        /// <summary>
        /// The maximum rank (number of dimensions) supported for a tensor.
        /// </summary>
        public static int MAX_DIMS = 8;
        /// <summary>
        /// The rank (number of dimensions).
        /// </summary>
        int nbDims;
        /// <summary>
        /// The extent of each dimension.
        /// </summary>
        List<int> d = new List<int>();

        public Dims(int leng, int[] data) 
        {
            nbDims = leng;
            for (int i = 0; i < nbDims; i++)
            {
                d.Add(data[i]);
            }
        }
        public Dims(params int[] data) 
        {
            nbDims = data.Length;
            for (int i = 0; i < nbDims; i++) 
            {
                d.Add(data[i]);
            }
        }

        /// <summary>
        /// Obtain the product of all dimensions of the shape.
        /// </summary>
        /// <returns>The product of all dimensions of the shape.</returns>
        public int prod() 
        {
            int p = 1;
            foreach (int i in d)
            {
                p = p * i;
            }
            return p;
        }
    }
}
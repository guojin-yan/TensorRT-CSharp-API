using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    internal class 
    {
    }

    //!
    //! \enum ReduceOperation
    //!
    //! \brief Enumerates the reduce operations that may be performed by a Reduce layer.
    //!
    //! The table shows the result of reducing across an empty volume of a given type.
    //!
    //! Operation | kFLOAT and kHALF  | kINT32  | kINT8
    //! --------- | ----------------- | ------- | -----
    //! kSUM      | 0                 | 0       | 0
    //! kPROD     | 1                 | 1       | 1
    //! kMAX      | negative infinity | INT_MIN | -128
    //! kMIN      | positive infinity | INT_MAX | 127
    //! kAVG      | NaN               | 0       | -128
    //!
    //! The current version of TensorRT usually performs reduction for kINT8 via kFLOAT or kHALF.
    //! The kINT8 values show the quantized representations of the floating-point values.
    //!
    public enum TrtReduceOperation : int
    {
        kSUM = 0,
        kPROD = 1,
        kMAX = 2,
        kMIN = 3,
        kAVG = 4
    };
}

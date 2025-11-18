using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    //!
    //! \enum FillOperation
    //!
    //! \brief Enumerates the tensor fill operations that may performed by a fill layer.
    //!
    //! \see IFillLayer
    //!
    public enum TrtFillOperation : int
    {
        //! Compute each value via an affine function of its indices.
        //! For example, suppose the parameters for the IFillLayer are:
        //!
        //! * Dimensions = [3,4]
        //! * Alpha = 1
        //! * Beta = [100,10]
        //!
        //! Element [i,j] of the output is Alpha + Beta[0]*i + Beta[1]*j.
        //! Thus the output matrix is:
        //!
        //!      1  11  21  31
        //!    101 111 121 131
        //!    201 211 221 231
        //!
        //! A static beta b is implicitly a 1D tensor, i.e. Beta = [b].
        kLINSPACE = 0,

        //! Randomly draw values from a uniform distribution.
        kRANDOM_UNIFORM = 1,

        //! Randomly draw values from a normal distribution.
        kRANDOM_NORMAL = 2
    };
}

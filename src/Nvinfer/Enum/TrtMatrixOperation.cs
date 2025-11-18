using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    //!
    //! \enum MatrixOperation
    //!
    //! \brief Enumerates the operations that may be performed on a tensor
    //!        by IMatrixMultiplyLayer before multiplication.
    //!
    public enum TrtMatrixOperation : int
    {
        //! Treat x as a matrix if it has two dimensions, or as a collection of
        //! matrices if x has more than two dimensions, where the last two dimensions
        //! are the matrix dimensions. x must have at least two dimensions.
        kNONE = 0,

        //! Like kNONE, but transpose the matrix dimensions.
        kTRANSPOSE = 1,

        //! Treat x as a vector if it has one dimension, or as a collection of
        //! vectors if x has more than one dimension. x must have at least one dimension.
        //!
        //! The first input tensor with dimensions [M,K] used with MatrixOperation::kVECTOR is equivalent to a tensor
        //! with dimensions [M, 1, K] with MatrixOperation::kNONE, i.e. is treated as M row vectors of length K,
        //! or dimensions [M, K, 1] with MatrixOperation::kTRANSPOSE.
        //!
        //! The second input tensor with dimensions [M,K] used with MatrixOperation::kVECTOR is equivalent to a tensor
        //! with dimensions [M, K, 1] with MatrixOperation::kNONE, i.e. is treated as M column vectors of length K,
        //! or dimensions [M, 1, K] with MatrixOperation::kTRANSPOSE.
        kVECTOR = 2,
    };
}

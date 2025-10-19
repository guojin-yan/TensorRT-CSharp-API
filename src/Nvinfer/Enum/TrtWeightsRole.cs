using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    //!
    //! \enum WeightsRole
    //!
    //! \brief How a layer uses particular Weights.
    //!
    //! The power weights of an IScaleLayer are omitted.  Refitting those is not supported.
    //!
    public enum TrtWeightsRole : int
    {
        kKERNEL = 0,   //!< kernel for IConvolutionLayer or IDeconvolutionLayer
        kBIAS = 1,     //!< bias for IConvolutionLayer or IDeconvolutionLayer
        kSHIFT = 2,    //!< shift part of IScaleLayer
        kSCALE = 3,    //!< scale part of IScaleLayer
        kCONSTANT = 4, //!< weights for IConstantLayer
        kANY = 5,      //!< Any other weights role
    };
}

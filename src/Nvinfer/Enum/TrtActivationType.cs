using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    //!
    //! \enum ActivationType
    //!
    //! \brief Enumerates the types of activation to perform in an activation layer.
    //!
    public enum TrtActivationType : int
    {
        kRELU = 0,              //!< Rectified linear activation.
        kSIGMOID = 1,           //!< Sigmoid activation.
        kTANH = 2,              //!< TanH activation.
        kLEAKY_RELU = 3,        //!< LeakyRelu activation: x>=0 ? x : alpha * x.
        kELU = 4,               //!< Elu activation: x>=0 ? x : alpha * (exp(x) - 1).
        kSELU = 5,              //!< Selu activation: x>0 ? beta * x : beta * (alpha*exp(x) - alpha)
        kSOFTSIGN = 6,          //!< Softsign activation: x / (1+|x|)
        kSOFTPLUS = 7,          //!< Parametric softplus activation: alpha*log(exp(beta*x)+1)
        kCLIP = 8,              //!< Clip activation: max(alpha, min(beta, x))
        kHARD_SIGMOID = 9,      //!< Hard sigmoid activation: max(0, min(1, alpha*x+beta))
        kSCALED_TANH = 10,      //!< Scaled tanh activation: alpha*tanh(beta*x)
        kTHRESHOLDED_RELU = 11, //!< Thresholded ReLU activation: x>alpha ? x : 0
        kGELU_ERF = 12,         //!< GELU erf activation: 0.5 * x * (1 + erf(sqrt(0.5) * x))
        kGELU_TANH = 13         //!< GELU tanh activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (0.044715F * pow(x, 3) + x)))
    };
}

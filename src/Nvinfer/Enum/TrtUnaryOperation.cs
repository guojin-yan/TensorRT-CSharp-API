using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    //!
    //! \enum UnaryOperation
    //!
    //! \brief Enumerates the unary operations that may be performed by a Unary layer.
    //!
    //! Operations kNOT must have inputs of DataType::kBOOL.
    //!
    //! Operation kSIGN and kABS must have inputs of floating-point type, DataType::kINT8, DataType::kINT32 or
    //! DataType::kINT64.
    //!
    //! Operation kISINF must have inputs of floating-point type.
    //!
    //! All other operations must have inputs of floating-point type.
    //!
    //! \see IUnaryLayer
    //!
    public enum TrtUnaryOperation : int
    {
        kEXP = 0,    //!< Exponentiation.
        kLOG = 1,    //!< Log (base e).
        kSQRT = 2,   //!< Square root.
        kRECIP = 3,  //!< Reciprocal.
        kABS = 4,    //!< Absolute value.
        kNEG = 5,    //!< Negation.
        kSIN = 6,    //!< Sine.
        kCOS = 7,    //!< Cosine.
        kTAN = 8,    //!< Tangent.
        kSINH = 9,   //!< Hyperbolic sine.
        kCOSH = 10,  //!< Hyperbolic cosine.
        kASIN = 11,  //!< Inverse sine.
        kACOS = 12,  //!< Inverse cosine.
        kATAN = 13,  //!< Inverse tangent.
        kASINH = 14, //!< Inverse hyperbolic sine.
        kACOSH = 15, //!< Inverse hyperbolic cosine.
        kATANH = 16, //!< Inverse hyperbolic tangent.
        kCEIL = 17,  //!< Ceiling.
        kFLOOR = 18, //!< Floor.
        kERF = 19,   //!< Gauss error function.
        kNOT = 20,   //!< Logical NOT.
        kSIGN = 21,  //!< Sign, If input > 0, output 1; if input < 0, output -1; if input == 0, output 0.
        kROUND = 22, //!< Round to nearest even for floating-point data type.
        kISINF = 23, //!< Return true if input value equals +/- infinity for floating-point data type.
        kISNAN = 24, //!< Return true if input value is a NaN for floating-point data type.
    };

}

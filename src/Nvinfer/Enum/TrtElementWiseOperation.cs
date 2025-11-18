using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    //!
    //! \enum ElementWiseOperation
    //!
    //! \brief Enumerates the binary operations that may be performed by an ElementWise layer.
    //!
    //! Operations kAND, kOR, and kXOR must have inputs of DataType::kBOOL.
    //!
    //! Operation kPOW must have inputs of floating-point type or DataType::kINT8.
    //!
    //! All other operations must have inputs of floating-point type, DataType::kINT8, DataType::kINT32, or
    //! DataType::kINT64.
    //!
    //! \see IElementWiseLayer
    //!
    public enum TrtElementWiseOperation : int
    {
        kSUM = 0,       //!< Sum of the two elements.
        kPROD = 1,      //!< Product of the two elements.
        kMAX = 2,       //!< Maximum of the two elements.
        kMIN = 3,       //!< Minimum of the two elements.
        kSUB = 4,       //!< Subtract the second element from the first.
        kDIV = 5,       //!< Divide the first element by the second.
        kPOW = 6,       //!< The first element to the power of the second element.
        kFLOOR_DIV = 7, //!< Floor division of the first element by the second.
        kAND = 8,       //!< Logical AND of two elements.
        kOR = 9,        //!< Logical OR of two elements.
        kXOR = 10,      //!< Logical XOR of two elements.
        kEQUAL = 11,    //!< Check if two elements are equal.
        kGREATER = 12,  //!< Check if element in first tensor is greater than corresponding element in second tensor.
        kLESS = 13      //!< Check if element in first tensor is less than corresponding element in second tensor.
    };
}

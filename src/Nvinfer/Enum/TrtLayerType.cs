using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    //!
    //! \enum LayerType
    //!
    //! \brief The type values of layer classes.
    //!
    //! \see ILayer::getType()
    //!
    public enum TrtLayerType
    {
        kCONVOLUTION = 0,         //!< Convolution layer.
        kCAST = 1,                //!< Cast layer
        kACTIVATION = 2,          //!< Activation layer.
        kPOOLING = 3,             //!< Pooling layer.
        kLRN = 4,                 //!< LRN layer.
        kSCALE = 5,               //!< Scale layer.
        kSOFTMAX = 6,             //!< SoftMax layer.
        kDECONVOLUTION = 7,       //!< Deconvolution layer.
        kCONCATENATION = 8,       //!< Concatenation layer.
        kELEMENTWISE = 9,         //!< Elementwise layer.
        kPLUGIN = 10,             //!< Plugin layer.
        kUNARY = 11,              //!< UnaryOp operation Layer.
        kPADDING = 12,            //!< Padding layer.
        kSHUFFLE = 13,            //!< Shuffle layer.
        kREDUCE = 14,             //!< Reduce layer.
        kTOPK = 15,               //!< TopK layer.
        kGATHER = 16,             //!< Gather layer.
        kMATRIX_MULTIPLY = 17,    //!< Matrix multiply layer.
        kRAGGED_SOFTMAX = 18,     //!< Ragged softmax layer.
        kCONSTANT = 19,           //!< Constant layer.
        kIDENTITY = 20,           //!< Identity layer.
        kPLUGIN_V2 = 21,          //!< PluginV2 layer.
        kSLICE = 22,              //!< Slice layer.
        kSHAPE = 23,              //!< Shape layer.
        kPARAMETRIC_RELU = 24,    //!< Parametric ReLU layer.
        kRESIZE = 25,             //!< Resize Layer.
        kTRIP_LIMIT = 26,         //!< Loop Trip limit layer
        kRECURRENCE = 27,         //!< Loop Recurrence layer
        kITERATOR = 28,           //!< Loop Iterator layer
        kLOOP_OUTPUT = 29,        //!< Loop output layer
        kSELECT = 30,             //!< Select layer.
        kFILL = 31,               //!< Fill layer
        kQUANTIZE = 32,           //!< Quantize layer
        kDEQUANTIZE = 33,         //!< Dequantize layer
        kCONDITION = 34,          //!< Condition layer
        kCONDITIONAL_INPUT = 35,  //!< Conditional Input layer
        kCONDITIONAL_OUTPUT = 36, //!< Conditional Output layer
        kSCATTER = 37,            //!< Scatter layer
        kEINSUM = 38,             //!< Einsum layer
        kASSERTION = 39,          //!< Assertion layer
        kONE_HOT = 40,            //!< OneHot layer
        kNON_ZERO = 41,           //!< NonZero layer
        kGRID_SAMPLE = 42,        //!< Grid sample layer
        kNMS = 43,                //!< NMS layer
        kREVERSE_SEQUENCE = 44,   //!< Reverse sequence layer
        kNORMALIZATION = 45,      //!< Normalization layer
        kPLUGIN_V3 = 46,          //!< PluginV3 layer.
        kSQUEEZE = 47,            //!< Squeeze Layer.
        kUNSQUEEZE = 48,          //!< Unsqueeze Layer.
        kCUMULATIVE = 49,         //!< Cumulative layer.
        kDYNAMIC_QUANTIZE = 50,    //!< Dynamic Quantize layer.
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 层类型枚举，定义了各种层类的类型值
    /// Layer type enumeration, defining the type values of layer classes
    /// </summary>
    /// <remarks>
    /// \see ILayer::getType()
    /// </remarks>
    public enum TrtLayerType
    {
        /// <summary>
        /// 卷积层
        /// Convolution layer
        /// </summary>
        kCONVOLUTION = 0,         //!< Convolution layer.

        /// <summary>
        /// 类型转换层
        /// Cast layer
        /// </summary>
        kCAST = 1,                //!< Cast layer

        /// <summary>
        /// 激活函数层
        /// Activation layer
        /// </summary>
        kACTIVATION = 2,          //!< Activation layer.

        /// <summary>
        /// 池化层
        /// Pooling layer
        /// </summary>
        kPOOLING = 3,             //!< Pooling layer.

        /// <summary>
        /// 局部响应归一化层
        /// LRN layer
        /// </summary>
        kLRN = 4,                 //!< LRN layer.

        /// <summary>
        /// 缩放层
        /// Scale layer
        /// </summary>
        kSCALE = 5,               //!< Scale layer.

        /// <summary>
        /// Softmax层
        /// SoftMax layer
        /// </summary>
        kSOFTMAX = 6,             //!< SoftMax layer.

        /// <summary>
        /// 反卷积层
        /// Deconvolution layer
        /// </summary>
        kDECONVOLUTION = 7,       //!< Deconvolution layer.

        /// <summary>
        /// 连接层
        /// Concatenation layer
        /// </summary>
        kCONCATENATION = 8,       //!< Concatenation layer.

        /// <summary>
        /// 元素级操作层
        /// Elementwise layer
        /// </summary>
        kELEMENTWISE = 9,         //!< Elementwise layer.

        /// <summary>
        /// 插件层
        /// Plugin layer
        /// </summary>
        kPLUGIN = 10,             //!< Plugin layer.

        /// <summary>
        /// 一元操作层
        /// UnaryOp operation Layer
        /// </summary>
        kUNARY = 11,              //!< UnaryOp operation Layer.

        /// <summary>
        /// 填充层
        /// Padding layer
        /// </summary>
        kPADDING = 12,            //!< Padding layer.

        /// <summary>
        /// 数据重排层
        /// Shuffle layer
        /// </summary>
        kSHUFFLE = 13,            //!< Shuffle layer.

        /// <summary>
        /// 归约层
        /// Reduce layer
        /// </summary>
        kREDUCE = 14,             //!< Reduce layer.

        /// <summary>
        /// TopK层
        /// TopK layer
        /// </summary>
        kTOPK = 15,               //!< TopK layer.

        /// <summary>
        /// 聚合层
        /// Gather layer
        /// </summary>
        kGATHER = 16,             //!< Gather layer.

        /// <summary>
        /// 矩阵乘法层
        /// Matrix multiply layer
        /// </summary>
        kMATRIX_MULTIPLY = 17,    //!< Matrix multiply layer.

        /// <summary>
        /// 不规则Softmax层
        /// Ragged softmax layer
        /// </summary>
        kRAGGED_SOFTMAX = 18,     //!< Ragged softmax layer.

        /// <summary>
        /// 常量层
        /// Constant layer
        /// </summary>
        kCONSTANT = 19,           //!< Constant layer.

        /// <summary>
        /// 恒等层
        /// Identity layer
        /// </summary>
        kIDENTITY = 20,           //!< Identity layer.

        /// <summary>
        /// 版本2插件层
        /// PluginV2 layer
        /// </summary>
        kPLUGIN_V2 = 21,          //!< PluginV2 layer.

        /// <summary>
        /// 切片层
        /// Slice layer
        /// </summary>
        kSLICE = 22,              //!< Slice layer.

        /// <summary>
        /// 形状操作层
        /// Shape layer
        /// </summary>
        kSHAPE = 23,              //!< Shape layer.

        /// <summary>
        /// 参数化ReLU层
        /// Parametric ReLU layer
        /// </summary>
        kPARAMETRIC_RELU = 24,    //!< Parametric ReLU layer.

        /// <summary>
        /// 缩放层
        /// Resize Layer
        /// </summary>
        kRESIZE = 25,             //!< Resize Layer.

        /// <summary>
        /// 循环迭代限制层
        /// Loop Trip limit layer
        /// </summary>
        kTRIP_LIMIT = 26,         //!< Loop Trip limit layer

        /// <summary>
        /// 循环递归层
        /// Loop Recurrence layer
        /// </summary>
        kRECURRENCE = 27,         //!< Loop Recurrence layer

        /// <summary>
        /// 迭代器层
        /// Loop Iterator layer
        /// </summary>
        kITERATOR = 28,           //!< Loop Iterator layer

        /// <summary>
        /// 循环输出层
        /// Loop output layer
        /// </summary>
        kLOOP_OUTPUT = 29,        //!< Loop output layer

        /// <summary>
        /// 选择层
        /// Select layer
        /// </summary>
        kSELECT = 30,             //!< Select layer.

        /// <summary>
        /// 填充层
        /// Fill layer
        /// </summary>
        kFILL = 31,               //!< Fill layer

        /// <summary>
        /// 量化层
        /// Quantize layer
        /// </summary>
        kQUANTIZE = 32,           //!< Quantize layer

        /// <summary>
        /// 反量化层
        /// Dequantize layer
        /// </summary>
        kDEQUANTIZE = 33,         //!< Dequantize layer

        /// <summary>
        /// 条件层
        /// Condition layer
        /// </summary>
        kCONDITION = 34,          //!< Condition layer

        /// <summary>
        /// 条件输入层
        /// Conditional Input layer
        /// </summary>
        kCONDITIONAL_INPUT = 35,  //!< Conditional Input layer

        /// <summary>
        /// 条件输出层
        /// Conditional Output layer
        /// </summary>
        kCONDITIONAL_OUTPUT = 36, //!< Conditional Output layer

        /// <summary>
        /// 散射层
        /// Scatter layer
        /// </summary>
        kSCATTER = 37,            //!< Scatter layer

        /// <summary>
        /// Einstein求和层
        /// Einsum layer
        /// </summary>
        kEINSUM = 38,             //!< Einsum layer

        /// <summary>
        /// 断言层
        /// Assertion layer
        /// </summary>
        kASSERTION = 39,          //!< Assertion layer

        /// <summary>
        /// OneHot层
        /// OneHot layer
        /// </summary>
        kONE_HOT = 40,            //!< OneHot layer

        /// <summary>
        /// 非零元素层
        /// NonZero layer
        /// </summary>
        kNON_ZERO = 41,           //!< NonZero layer

        /// <summary>
        /// 网格采样层
        /// Grid sample layer
        /// </summary>
        kGRID_SAMPLE = 42,        //!< Grid sample layer

        /// <summary>
        /// 非极大值抑制层
        /// NMS layer
        /// </summary>
        kNMS = 43,                //!< NMS layer

        /// <summary>
        /// 序列反转层
        /// Reverse sequence layer
        /// </summary>
        kREVERSE_SEQUENCE = 44,   //!< Reverse sequence layer

        /// <summary>
        /// 归一化层
        /// Normalization layer
        /// </summary>
        kNORMALIZATION = 45,      //!< Normalization layer

        /// <summary>
        /// 版本3插件层
        /// PluginV3 layer
        /// </summary>
        kPLUGIN_V3 = 46,          //!< PluginV3 layer.

        /// <summary>
        /// 压缩层
        /// Squeeze Layer
        /// </summary>
        kSQUEEZE = 47,            //!< Squeeze Layer.

        /// <summary>
        /// 解压缩层
        /// Unsqueeze Layer
        /// </summary>
        kUNSQUEEZE = 48,          //!< Unsqueeze Layer.

        /// <summary>
        /// 累计层
        /// Cumulative layer
        /// </summary>
        kCUMULATIVE = 49,         //!< Cumulative layer.

        /// <summary>
        /// 动态量化层
        /// Dynamic Quantize layer
        /// </summary>
        kDYNAMIC_QUANTIZE = 50,    //!< Dynamic Quantize layer.
    }

}

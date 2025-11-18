using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 定义了在激活层中可以执行的激活函数类型。<br/>
    /// Enumerates the types of activation functions that can be performed in an activation layer.
    /// </summary>
    public enum TrtActivationType : int
    {
        /// <summary>
        /// 整流线性单元。<br/>
        /// Rectified Linear Unit.
        /// </summary>
        kRELU = 0,

        /// <summary>
        /// Sigmoid 激活函数。<br/>
        /// Sigmoid activation function.
        /// </summary>
        kSIGMOID = 1,

        /// <summary>
        /// TanH 激活函数。<br/>
        /// TanH activation function.
        /// </summary>
        kTANH = 2,

        /// <summary>
        /// Leaky ReLU 激活函数。<br/>
        /// LeakyReLU activation function.
        /// <remarks>
        /// 公式为: <c>x>=0 ? x : alpha * x</c><br/>
        /// Formula: <c>x>=0 ? x : alpha * x</c>
        /// </remarks>
        /// </summary>
        kLEAKY_RELU = 3,

        /// <summary>
        /// ELU (Exponential Linear Unit) 激活函数。<br/>
        /// ELU (Exponential Linear Unit) activation function.
        /// <remarks>
        /// 公式为: <c>x>=0 ? x : alpha * (exp(x) - 1)</c><br/>
        /// Formula: <c>x>=0 ? x : alpha * (exp(x) - 1)</c>
        /// </remarks>
        /// </summary>
        kELU = 4,

        /// <summary>
        /// SELU (Scaled Exponential Linear Unit) 激活函数。<br/>
        /// SELU (Scaled Exponential Linear Unit) activation function.
        /// <remarks>
        /// 公式为: <c>x>0 ? beta * x : beta * (alpha*exp(x) - alpha)</c><br/>
        /// Formula: <c>x>0 ? beta * x : beta * (alpha*exp(x) - alpha)</c>
        /// </remarks>
        /// </summary>
        kSELU = 5,

        /// <summary>
        /// Softsign 激活函数。<br/>
        /// Softsign activation function.
        /// <remarks>
        /// 公式为: <c>x / (1+|x|)</c><br/>
        /// Formula: <c>x / (1+|x|)</c>
        /// </remarks>
        /// </summary>
        kSOFTSIGN = 6,

        /// <summary>
        /// 参数化 Softplus 激活函数。<br/>
        /// Parametric softplus activation function.
        /// <remarks>
        /// 公式为: <c>alpha*log(exp(beta*x)+1)</c><br/>
        /// Formula: <c>alpha*log(exp(beta*x)+1)</c>
        /// </remarks>
        /// </summary>
        kSOFTPLUS = 7,

        /// <summary>
        /// 裁剪激活函数。<br/>
        /// Clip activation function.
        /// <remarks>
        /// 公式为: <c>max(alpha, min(beta, x))</c><br/>
        /// Formula: <c>max(alpha, min(beta, x))</c>
        /// </remarks>
        /// </summary>
        kCLIP = 8,

        /// <summary>
        /// Hard Sigmoid 激活函数。<br/>
        /// Hard sigmoid activation function.
        /// <remarks>
        /// 公式为: <c>max(0, min(1, alpha*x+beta))</c><br/>
        /// Formula: <c>max(0, min(1, alpha*x+beta))</c>
        /// </remarks>
        /// </summary>
        kHARD_SIGMOID = 9,

        /// <summary>
        /// 缩放的 TanH 激活函数。<br/>
        /// Scaled tanh activation function.
        /// <remarks>
        /// 公式为: <c>alpha*tanh(beta*x)</c><br/>
        /// Formula: <c>alpha*tanh(beta*x)</c>
        /// </remarks>
        /// </summary>
        kSCALED_TANH = 10,

        /// <summary>
        /// 带阈值的 ReLU 激活函数。<br/>
        /// Thresholded ReLU activation function.
        /// <remarks>
        /// 公式为: <c>x>alpha ? x : 0</c><br/>
        /// Formula: <c>x>alpha ? x : 0</c>
        /// </remarks>
        /// </summary>
        kTHRESHOLDED_RELU = 11,

        /// <summary>
        /// GELU (Gaussian Error Linear Unit) 激活函数，使用误差函数 计算。<br/>
        /// GELU (Gaussian Error Linear Unit) activation function, computed using the error function (erf).
        /// <remarks>
        /// 公式为: <c>0.5 * x * (1 + erf(sqrt(0.5) * x))</c><br/>
        /// Formula: <c>0.5 * x * (1 + erf(sqrt(0.5) * x))</c>
        /// </remarks>
        /// </summary>
        kGELU_ERF = 12,

        /// <summary>
        /// GELU (Gaussian Error Linear Unit) 激活函数，使用 TanH 的近似计算。<br/>
        /// GELU (Gaussian Error Linear Unit) activation function, computed using a TanH approximation.
        /// <remarks>
        /// 公式为: <c>0.5 * x * (1 + tanh(sqrt(2/pi) * (0.044715F * pow(x, 3) + x)))</c><br/>
        /// Formula: <c>0.5 * x * (1 + tanh(sqrt(2/pi) * (0.044715F * pow(x, 3) + x)))</c>
        /// </remarks>
        /// </summary>
        kGELU_TANH = 13
    };

}

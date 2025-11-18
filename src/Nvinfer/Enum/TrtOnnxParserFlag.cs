using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 定义了用于控制 ONNX 解析器行为的标志。<br/>
    /// Defines flags to control the behavior of the ONNX parser.
    /// </summary>
    public enum TrtOnnxParserFlag : int
    {
        /// <summary>
        /// 对 InstanceNormalization 节点使用 TensorRT 原生层实现。<br/>
        /// Use TensorRT's native layer implementation for InstanceNormalization nodes.
        /// </summary>
        /// <remarks>
        /// 启用此标志后，解析器在处理 ONNX 模型时，会优先使用 TensorRT 的原生层实现，而不是插件（plugin）实现来处理 InstanceNormalization 节点。<br/>
        /// When enabled, the parser will prioritize using TensorRT's native layer implementation over the plugin implementation for InstanceNormalization nodes when parsing the ONNX model.
        /// <para>
        /// 此标志在构建版本兼容或硬件兼容的引擎时是必需的。该标志默认为开启状态 (ON)。<br/>
        /// This flag is required when building version-compatible or hardware-compatible engines. This flag is ON by default.
        /// </para>
        /// </remarks>
        kNATIVE_INSTANCENORM = 0,

        /// <summary>
        /// 启用 DLA 的 UINT8 量化和非对称量化支持。<br/>
        /// Enable UINT8 and asymmetric quantization support for DLA.
        /// </summary>
        /// <remarks>
        /// 启用此标志后，解析器会允许将 UINT8 作为量化数据类型，并支持在 Quantize 和 Dequantize 节点中使用具有非零零点值 的非对称量化。<br/>
        /// When enabled, the parser allows UINT8 as a quantization data type and supports asymmetric quantization with non-zero zero-point values in Quantize and Dequantize nodes.
        /// <para>
        /// 使用此选项生成的引擎必须以 DLA 版本 >= 3.16 为目标进行构建。该标志默认为关闭状态 (OFF)。<br/>
        /// The resulting engine must be built targeting DLA version >= 3.16. This flag is OFF by default.
        /// </para>
        /// </remarks>
        kENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA = 1,
    };

}

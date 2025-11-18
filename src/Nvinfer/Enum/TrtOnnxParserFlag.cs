using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    public enum TrtOnnxParserFlag : int
    {
        //! Parse the ONNX model into the INetworkDefinition with the intention of using TensorRT's native layer
        //! implementation over the plugin implementation for InstanceNormalization nodes.
        //! This flag is required when building version-compatible or hardware-compatible engines.
        //! This flag is set to be ON by default.
        kNATIVE_INSTANCENORM = 0,
        //! Enable UINT8 as a quantization data type and asymmetric quantization with non-zero zero-point values
        //! in Quantize and Dequantize nodes. This flag is set to be OFF by default.
        //! The resulting engine must be built targeting DLA version >= 3.16.
        kENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA = 1,
    };
}

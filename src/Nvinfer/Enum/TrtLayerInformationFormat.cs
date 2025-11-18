using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 定义了引擎检查器打印层信息时所使用的格式。<br/>
    /// The format in which the IEngineInspector prints the layer information.
    /// </summary>
    /// <seealso cref="TrtEngineInspector.GetLayerInformation()"/>
    /// <seealso cref="TrtEngineInspector.GetEngineInformation()"/>
    public enum TrtLayerInformationFormat : int
    {
        /// <summary>
        /// 每个图层的信息打印在一行内。<br/>
        /// Print layer information in one line per layer.
        /// </summary>
        kONELINE = 0,

        /// <summary>
        /// 以 JSON 格式打印层信息。<br/>
        /// Print layer information in JSON format.
        /// </summary>
        kJSON = 1
    };

}

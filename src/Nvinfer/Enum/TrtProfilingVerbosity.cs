using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 性能分析详细程度枚举，定义了在NVTX注释和IEngineInspector中暴露的层信息详细程度列表
    /// Profiling verbosity enumeration, defining the list of verbosity levels of layer information exposed in NVTX annotations and in IEngineInspector
    /// </summary>
    /// <remarks>
    /// \see IBuilderConfig::setProfilingVerbosity(),
    ///      IBuilderConfig::getProfilingVerbosity(),
    ///      IEngineInspector
    /// </remarks>
    public enum TrtProfilingVerbosity : int
    {
        /// <summary>
        /// 仅打印层名称，这是默认设置
        /// Print only the layer names. This is the default setting.
        /// </summary>
        kLAYER_NAMES_ONLY = 0, //!< Print only the layer names. This is the default setting.

        /// <summary>
        /// 不打印任何层信息
        /// Do not print any layer information.
        /// </summary>
        kNONE = 1,             //!< Do not print any layer information.

        /// <summary>
        /// 打印详细的层信息，包括层名称和层参数
        /// Print detailed layer information including layer names and layer parameters.
        /// </summary>
        kDETAILED = 2,         //!< Print detailed layer information including layer names and layer parameters.

    }

}

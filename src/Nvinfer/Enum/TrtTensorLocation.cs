using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 张量位置枚举，定义了张量数据存储的位置（设备或主机）
    /// The location for tensor data storage, device or host
    /// </summary>
    public enum TrtTensorLocation : int
    {
        /// <summary>
        /// 张量数据存储在设备上（如GPU）
        /// Data stored on device (e.g. GPU)
        /// </summary>
        kDEVICE = 0,

        /// <summary>
        /// 张量数据存储在主机上
        /// Data stored on host
        /// </summary>
        kHOST = 1,
    };

}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 设备类型枚举，指定此层/网络将在哪种设备上执行
    /// Device type enumeration, specifying the device that this layer/network will execute on
    /// </summary>
    public enum TrtDeviceType : int
    {
        /// <summary>
        /// GPU设备
        /// GPU Device
        /// </summary>
        kGPU = 0,

        /// <summary>
        /// DLA核心
        /// DLA Core
        /// </summary>
        kDLA = 1,
    }

}

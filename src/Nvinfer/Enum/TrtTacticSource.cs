using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 寻找位置枚举，控制IStreamReaderV2的寻找模式
    /// Seek position enumeration, controls the seek mode of IStreamReaderV2
    /// </summary>
    public enum TrtTacticSource : int
    {
        /// <summary>
        /// 从文件开头开始
        /// From the beginning of the file
        /// </summary>
        //! From the beginning of the file.
        kSET = 0,

        /// <summary>
        /// 从文件的当前位置开始
        /// From the current position of the file
        /// </summary>
        //! From the current position of the file.
        kCUR = 1,

        /// <summary>
        /// 从文件末尾开始
        /// From the tail of the file
        /// </summary>
        //! From the tail of the file.
        kEND = 2,
    };

}

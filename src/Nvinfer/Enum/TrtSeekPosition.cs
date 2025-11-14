using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 查找位置枚举，定义在文件中的查找位置类型
    /// Seek position enumeration, defining the position type when seeking in a file
    /// </summary>
    public enum TrtSeekPosition : int
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
    }

}

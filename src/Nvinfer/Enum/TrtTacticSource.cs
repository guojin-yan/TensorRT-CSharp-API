using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    //!
    //! \enum SeekPosition
    //! \brief Controls the seek mode of IStreamReaderV2.
    //!
    public enum TrtTacticSource : int
    {
        //! From the beginning of the file.
        kSET = 0,

        //! From the current position of the file.
        kCUR = 1,

        //! From the tail of the file.
        kEND = 2,
    };
}

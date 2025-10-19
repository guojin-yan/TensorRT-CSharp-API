using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    //!
    //! \enum OptProfileSelector
    //!
    //! \brief When setting or querying optimization profile parameters (such as shape tensor inputs or dynamic dimensions),
    //!        select whether we are interested in the minimum, optimum, or maximum values for these parameters.
    //!        The minimum and maximum specify the permitted range that is supported at runtime, while the optimum value
    //!        is used for the kernel selection. This should be the "typical" value that is expected to occur at runtime.
    //!
    //! \see IOptimizationProfile::setDimensions(), IOptimizationProfile::setShapeValuesV2(), IOptimizationProfile::setShapeValues()
    //!
    public enum TrtOptProfileSelector : int
    {
        kMIN = 0, //!< This is used to set or get the minimum permitted value for dynamic dimensions etc.
        kOPT = 1, //!< This is used to set or get the value that is used in the optimization (kernel selection).
        kMAX = 2  //!< This is used to set or get the maximum permitted value for dynamic dimensions etc.
    };
}

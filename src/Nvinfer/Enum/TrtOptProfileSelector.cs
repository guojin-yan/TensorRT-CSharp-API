using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 优化配置选择器枚举，用于在设置或查询优化配置参数（如形状张量输入或动态维度）时，
    /// 选择我们是否对这些参数的最小值、最佳值或最大值感兴趣。
    /// 最小值和最大值指定了运行时支持的允许范围，而最佳值用于内核选择。
    /// 这应该是运行时预期的"典型"值。
    /// Optimization profile selector enumeration, used when setting or querying optimization profile parameters (such as shape tensor inputs or dynamic dimensions),
    /// select whether we are interested in the minimum, optimum, or maximum values for these parameters.
    /// The minimum and maximum specify the permitted range that is supported at runtime, while the optimum value
    /// is used for the kernel selection. This should be the "typical" value that is expected to occur at runtime.
    /// </summary>
    /// <remarks>
    /// \see IOptimizationProfile::setDimensions(), IOptimizationProfile::setShapeValuesV2(), IOptimizationProfile::setShapeValues()
    /// </remarks>
    public enum TrtOptProfileSelector : int
    {
        /// <summary>
        /// 用于设置或获取动态维度等的最小允许值
        /// This is used to set or get the minimum permitted value for dynamic dimensions etc.
        /// </summary>
        kMIN = 0, //!< This is used to set or get the minimum permitted value for dynamic dimensions etc.

        /// <summary>
        /// 用于在优化（内核选择）中使用的值
        /// This is used to set or get the value that is used in the optimization (kernel selection)
        /// </summary>
        kOPT = 1, //!< This is used to set or get the value that is used in the optimization (kernel selection)

        /// <summary>
        /// 用于设置或获取动态维度等的最大允许值
        /// This is used to set or get the maximum permitted value for dynamic dimensions etc.
        /// </summary>
        kMAX = 2  //!< This is used to set or get the maximum permitted value for dynamic dimensions etc.
    };

}

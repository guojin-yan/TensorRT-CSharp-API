using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 网络定义创建标志枚举，用于在创建网络时指定不可变的网络属性列表
    /// Network definition creation flag enumeration, used with createNetworkV2() to specify immutable properties of the network
    /// </summary>
    /// <remarks>
    /// \see IBuilder::createNetworkV2
    /// </remarks>
    public enum TrtNetworkDefinitionCreationFlag
    {
        /// <summary>
        /// 已忽略，因为在TensorRT 10.0中网络总是"显式批次"的
        /// Ignored because networks are always "explicit batch" in TensorRT 10.0
        /// </summary>
        /// <remarks>
        /// \deprecated 在TensorRT 10.0中已弃用
        /// \deprecated Deprecated in TensorRT 10.0
        /// </remarks>
        //! Ignored because networks are always "explicit batch" in TensorRT 10.0.
        //!
        //! \deprecated Deprecated in TensorRT 10.0.
        kEXPLICIT_BATCH = 0,

        /// <summary>
        /// 将网络标记为强类型
        /// 网络中的每个张量都遵循类型推理规则和输入/操作注释来定义数据类型。不允许设置层精度和层输出类型，
        /// 网络输出类型将基于输入类型和类型推理规则进行推断。
        /// Mark the network to be strongly typed.
        /// Every tensor in the network has a data type defined in the network following only type inference rules and the
        /// inputs/operator annotations. Setting layer precision and layer output types is not allowed, and the network
        /// output types will be inferred based on the input types and the type inference rules.
        /// </summary>
        kSTRONGLY_TYPED = 1,

        /// <summary>
        /// 如果设置，对于同时具有AOT和JIT实现的Python插件，将使用JIT实现。
        /// 任何特定的插件JIT/AOT规范可能会覆盖此设置。
        /// 不能与NetworkDefinitionCreationFlag::kPREFER_AOT_PYTHON_PLUGINS一起使用。
        /// If set, for a Python plugin with both AOT and JIT implementations, the JIT implementation will be used.
        /// Any plugin-specific JIT/AOT specification may override this.
        /// Cannot be used in conjunction with NetworkDefinitionCreationFlag::kPREFER_AOT_PYTHON_PLUGINS.
        /// </summary>
        kPREFER_JIT_PYTHON_PLUGINS = 2,

        /// <summary>
        /// 如果设置，对于同时具有AOT和JIT实现的Python插件，将使用AOT实现。
        /// 任何特定的插件JIT/AOT规范可能会覆盖此设置。
        /// 不能与NetworkDefinitionCreationFlag::kPREFER_JIT_PYTHON_PLUGINS一起使用。
        /// If set, for a Python plugin with both AOT and JIT implementations, the AOT implementation will be used.
        /// Any plugin-specific JIT/AOT specification may override this.
        /// Cannot be used in conjunction with NetworkDefinitionCreationFlag::kPREFER_JIT_PYTHON_PLUGINS.
        /// </summary>
        kPREFER_AOT_PYTHON_PLUGINS = 3,
    }


}

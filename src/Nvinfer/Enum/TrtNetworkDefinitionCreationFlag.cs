using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    //!
    //! \enum NetworkDefinitionCreationFlag
    //!
    //! \brief List of immutable network properties expressed at network creation time.
    //! NetworkDefinitionCreationFlag is used with createNetworkV2() to specify immutable properties of the network.
    //!
    //! \see IBuilder::createNetworkV2
    //!
    public enum TrtNetworkDefinitionCreationFlag
    {  

        //! Ignored because networks are always "explicit batch" in TensorRT 10.0.
        //!
        //! \deprecated Deprecated in TensorRT 10.0.
        kEXPLICIT_BATCH = 0,

        //! Mark the network to be strongly typed.
        //! Every tensor in the network has a data type defined in the network following only type inference rules and the
        //! inputs/operator annotations. Setting layer precision and layer output types is not allowed, and the network
        //! output types will be inferred based on the input types and the type inference rules.
        kSTRONGLY_TYPED = 1,
        //! If set, for a Python plugin with both AOT and JIT implementations, the JIT implementation will be used.
        //! Any plugin-specific JIT/AOT specification may override this.
        //! Cannot be used in conjunction with NetworkDefinitionCreationFlag::kPREFER_AOT_PYTHON_PLUGINS.
        kPREFER_JIT_PYTHON_PLUGINS = 2,

        //! If set, for a Python plugin with both AOT and JIT implementations, the AOT implementation will be used.
        //! Any plugin-specific JIT/AOT specification may override this.
        //! Cannot be used in conjunction with NetworkDefinitionCreationFlag::kPREFER_JIT_PYTHON_PLUGINS.
        kPREFER_AOT_PYTHON_PLUGINS = 3,
    }

}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    //!
    //! \enum EngineCapability
    //!
    //! \brief List of supported engine capability flows.
    //!
    //! \details The EngineCapability determines the restrictions of a network during build time and what runtime
    //! it targets. When BuilderFlag::kSAFETY_SCOPE is not set (by default), EngineCapability::kSTANDARD does not provide
    //! any restrictions on functionality and the resulting serialized engine can be executed with TensorRT's standard
    //! runtime APIs in the nvinfer1 namespace. EngineCapability::kSAFETY provides a restricted subset of network
    //! operations that are safety certified and the resulting serialized engine can be executed with TensorRT's safe
    //! runtime APIs in the nvinfer1::safe namespace. EngineCapability::kDLA_STANDALONE provides a restricted subset of
    //! network operations that are DLA compatible and the resulting serialized engine can be executed using standalone
    //! DLA runtime APIs. See sampleCudla for an example of integrating cuDLA APIs with TensorRT APIs.
    //!
    public enum TrtEngineCapability : int
    {
        //!
        //! Standard: TensorRT flow without targeting the safety runtime.
        //! This flow supports both DeviceType::kGPU and DeviceType::kDLA.
        //!
        kSTANDARD = 0,

        //!
        //! Safety: TensorRT flow with restrictions targeting the safety runtime.
        //! See safety documentation for list of supported layers and formats.
        //! This flow supports only DeviceType::kGPU.
        //!
        //! This flag is only supported in NVIDIA Drive(R) products.
        kSAFETY = 1,

        //!
        //! DLA Standalone: TensorRT flow with restrictions targeting external, to TensorRT, DLA runtimes.
        //! See DLA documentation for list of supported layers and formats.
        //! This flow supports only DeviceType::kDLA.
        //!
        kDLA_STANDALONE = 2,
    }

}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 引擎能力枚举，列出支持的引擎能力流程
    /// Engine capability enumeration, listing the supported engine capability flows
    /// </summary>
    /// <remarks>
    /// EngineCapability决定了构建时网络的限制以及运行时目标。当未设置BuilderFlag::kSAFETY_SCOPE时（默认情况下），
    /// EngineCapability::kSTANDARD不提供任何功能限制，并且序列化的引擎可以使用nvinfer1命名空间中的TensorRT标准运行时API执行。
    /// EngineCapability::kSAFETY提供经过安全认证的网络操作受限子集，并且序列化的引擎可以使用nvinfer1::safe命名空间中的TensorRT安全运行时API执行。
    /// EngineCapability::kDLA_STANDALONE提供与DLA兼容的网络操作受限子集，并且序列化的引擎可以使用独立DLA运行时API执行。
    /// 有关集成cuDLA API与TensorRT API的示例，请参见sampleCudla。
    /// \details The EngineCapability determines the restrictions of a network during build time and what runtime
    /// it targets. When BuilderFlag::kSAFETY_SCOPE is not set (by default), EngineCapability::kSTANDARD does not provide
    /// any restrictions on functionality and the resulting serialized engine can be executed with TensorRT's standard
    /// runtime APIs in the nvinfer1 namespace. EngineCapability::kSAFETY provides a restricted subset of network
    /// operations that are safety certified and the resulting serialized engine can be executed with TensorRT's safe
    /// runtime APIs in the nvinfer1::safe namespace. EngineCapability::kDLA_STANDALONE provides a restricted subset of
    /// network operations that are DLA compatible and the resulting serialized engine can be executed using standalone
    /// DLA runtime APIs. See sampleCudla for an example of integrating cuDLA APIs with TensorRT APIs.
    /// </remarks>
    public enum TrtEngineCapability : int
    {
        /// <summary>
        /// 标准：不针对安全运行时的TensorRT流程
        /// Standard: TensorRT flow without targeting the safety runtime
        /// </summary>
        /// <remarks>
        /// 此流程支持DeviceType::kGPU和DeviceType::kDLA
        /// This flow supports both DeviceType::kGPU and DeviceType::kDLA
        /// </remarks>
        kSTANDARD = 0,

        /// <summary>
        /// 安全：针对安全运行时的TensorRT流程，带有一定限制
        /// Safety: TensorRT flow with restrictions targeting the safety runtime
        /// </summary>
        /// <remarks>
        /// 有关支持的层和格式列表，请参阅安全文档
        /// See safety documentation for list of supported layers and formats
        /// 此流程仅支持DeviceType::kGPU
        /// This flow supports only DeviceType::kGPU
        /// <br/>
        /// 此标志仅在NVIDIA Drive(R)产品中支持
        /// This flag is only supported in NVIDIA Drive(R) products
        /// </remarks>
        kSAFETY = 1,

        /// <summary>
        /// DLA独立：针对外部TensorRT的DLA运行时的TensorRT流程，带有一定限制
        /// DLA Standalone: TensorRT flow with restrictions targeting external, to TensorRT, DLA runtimes
        /// </summary>
        /// <remarks>
        /// 有关支持的层和格式列表，请参阅DLA文档
        /// See DLA documentation for list of supported layers and formats
        /// 此流程仅支持DeviceType::kDLA
        /// This flow supports only DeviceType::kDLA
        /// </remarks>
        kDLA_STANDALONE = 2,
    }


}

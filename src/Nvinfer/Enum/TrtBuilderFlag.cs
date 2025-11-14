using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// 构建器标志枚举，定义了从网络定义创建引擎时构建器可以启用的有效模式列表
    /// Builder flag enumeration, defining the list of valid modes that the builder can enable when creating an engine from a network definition
    /// </summary>
    /// <remarks>
    /// 有关详细信息，请参阅 IBuilderConfig::setFlags() 和 IBuilderConfig::getFlags()
    /// See IBuilderConfig::setFlags() and IBuilderConfig::getFlags() for more information
    /// </remarks>
    /// <see cref="IBuilderConfig::setFlags()"/>
    /// <see cref="IBuilderConfig::getFlags()"/>
    public enum TrtBuilderFlag : int
    {
        /// <summary>
        /// 启用FP16层选择，支持FP32回退
        /// Enable FP16 layer selection, with FP32 fallback
        /// </summary>
        /// <remarks>
        /// 已在TensorRT 10.12中弃用。已被强类型取代。
        /// Deprecated in TensorRT 10.12. Superseded by strong typing.
        /// </remarks>
        kFP16 = 0,

        /// <summary>
        /// 启用Int8层选择，如果同时指定了kFP16，则支持FP32回退和FP16回退
        /// Enable Int8 layer selection, with FP32 fallback with FP16 fallback if kFP16 also specified
        /// </summary>
        /// <remarks>
        /// 已在TensorRT 10.12中弃用。已被强类型取代。
        /// Deprecated in TensorRT 10.12. Superseded by strong typing.
        /// </remarks>
        kINT8 = 1,

        /// <summary>
        /// 通过在每层之后同步来启用层调试
        /// Enable debugging of layers via synchronizing after every layer
        /// </summary>
        kDEBUG = 2,

        /// <summary>
        /// 如果层无法在DLA上执行，则启用在GPU上执行标记的层
        /// Enable layers marked to execute on GPU if layer cannot execute on DLA
        /// </summary>
        kGPU_FALLBACK = 3,

        /// <summary>
        /// 启用构建可重调优引擎
        /// Enable building a refittable engine
        /// </summary>
        kREFIT = 4,

        /// <summary>
        /// 禁用相同层之间重用计时信息
        /// Disable reuse of timing information across identical layers
        /// </summary>
        kDISABLE_TIMING_CACHE = 5,

        /// <summary>
        /// 允许（但不要求）使用TF32类型为DataType::kFLOAT的张量进行计算
        /// Allow (but not require) computations on tensors of type DataType::kFLOAT to use TF32
        /// </summary>
        /// <remarks>
        /// TF32通过在乘法之前将输入舍入到10位尾数来计算内积，但使用23位尾数累加总和。默认启用。
        /// TF32 computes inner products by rounding the inputs to 10-bit mantissas before
        /// multiplying, but accumulates the sum using 23-bit mantissas. Enabled by default.
        /// </remarks>
        kTF32 = 6,

        /// <summary>
        /// 允许构建器在权重具有适当稀疏性时检查权重并使用优化函数
        /// Allow the builder to examine weights and use optimized functions when weights have suitable sparsity
        /// </summary>
        kSPARSE_WEIGHTS = 7,

        /// <summary>
        /// 更改EngineCapability::kSTANDARD流程中的允许参数，以匹配EngineCapability::kSAFETY对DeviceType::kGPU
        /// 和EngineCapability::kDLA_STANDALONE对DeviceType::kDLA情况检查的限制。如果构建时未设置EngineCapability::kSAFETY，则强制此标志为true。
        /// Change the allowed parameters in the EngineCapability::kSTANDARD flow to
        /// match the restrictions that EngineCapability::kSAFETY check against for DeviceType::kGPU
        /// and EngineCapability::kDLA_STANDALONE check against the DeviceType::kDLA case. This flag
        /// is forced to true if EngineCapability::kSAFETY at build time if it is unset.
        /// </summary>
        /// <remarks>
        /// 此标志仅在NVIDIA Drive(R)产品中支持
        /// This flag is only supported in NVIDIA Drive(R) products.
        /// </remarks>
        kSAFETY_SCOPE = 8,

        /// <summary>
        /// 要求层以指定的精度执行。否则构建将失败。
        /// Require that layers execute in specified precisions. Build fails otherwise.
        /// </summary>
        /// <remarks>
        /// 已在TensorRT 10.12中弃用。已被强类型取代。
        /// Deprecated in TensorRT 10.12. Superseded by strong typing.
        /// </remarks>
        kOBEY_PRECISION_CONSTRAINTS = 9,

        /// <summary>
        /// 优先考虑层以指定的精度执行。如果构建会失败，则回退（并发出警告）到其他精度。
        /// Prefer that layers execute in specified precisions.
        /// Fall back (with warning) to another precision if build would otherwise fail.
        /// </summary>
        /// <remarks>
        /// 已在TensorRT 10.12中弃用。已被强类型取代。
        /// Deprecated in TensorRT 10.12. Superseded by strong typing.
        /// </remarks>
        kPREFER_PRECISION_CONSTRAINTS = 10,

        /// <summary>
        /// 要求不会在层和网络I/O张量之间插入重设格式，ITensor::setAllowedFormat已为此调用。
        /// 如果功能性正确需要重设格式，构建将失败。
        /// Require that no reformats be inserted between a layer and a network I/O tensor
        /// for which ITensor::setAllowedFormats was called.
        /// Build fails if a reformat is required for functional correctness.
        /// </summary>
        /// <remarks>
        /// 已在TensorRT 10.7中弃用。不需要的API。
        /// Deprecated in TensorRT 10.7. Unneeded API.
        /// </remarks>
        kDIRECT_IO = 11,

        /// <summary>
        /// 如果IAlgorithmSelector::selectAlgorithms返回空算法集则失败。
        /// Fail if IAlgorithmSelector::selectAlgorithms returns an empty set of algorithms.
        /// </summary>
        /// <remarks>
        /// 已在TensorRT 10.10中弃用。由于IAlgorithmSelector弃用，不需要该API。
        /// Deprecated in TensorRT 10.10. Unneeded API due to IAlgorithmSelector deprecation.
        /// </remarks>
        kREJECT_EMPTY_ALGORITHMS = 12,

        /// <summary>
        /// 限制为精简运行时操作，以实现计划的版本前向兼容性。
        /// Restrict to lean runtime operators to provide version forward compatibility
        /// for the plan.
        /// </summary>
        /// <remarks>
        /// 此标志仅支持NVIDIA Volta及更高版本的GPU。
        /// 此标志在NVIDIA Drive(R)产品中不支持。
        /// This flag is only supported by NVIDIA Volta and later GPUs.
        /// This flag is not supported in NVIDIA Drive(R) products.
        /// </remarks>
        kVERSION_COMPATIBLE = 13,

        /// <summary>
        /// 当启用版本前向兼容性时，从计划中排除精简运行时。默认情况下，此标志未设置，因此精简运行时将包含在计划中。
        /// Exclude lean runtime from the plan when version forward compatability is enabled.
        /// By default, this flag is unset, so the lean runtime will be included in the plan.
        /// </summary>
        /// <remarks>
        /// 如果未设置BuilderFlag::kVERSION_COMPATIBLE，则忽略此标志的值。
        /// If BuilderFlag::kVERSION_COMPATIBLE is not set then the value of this flag will be ignored.
        /// </remarks>
        kEXCLUDE_LEAN_RUNTIME = 14,

        /// <summary>
        /// 启用具有FP8输入/输出的插件。
        /// Enable plugins with FP8 input/output.
        /// </summary>
        /// <remarks>
        /// 当启用HardwareCompatibilityLevel::kAMPERE_PLUS时不支持此标志。
        /// This flag is not supported when HardwareCompatibilityLevel::kAMPERE_PLUS is enabled.
        /// 有关更多信息，请参阅HardwareCompatibilityLevel
        /// \see HardwareCompatibilityLevel
        /// 已在TensorRT 10.12中弃用。已被强类型取代。
        /// Deprecated in TensorRT 10.12. Superseded by strong typing.
        /// </remarks>
        kFP8 = 15,

        /// <summary>
        /// 当计时缓存中不存在正在计时的策略时发出错误。
        /// 此标志仅在IBuilderConfig具有关联的ITimingCache时有效。
        /// Emit error when a tactic being timed is not present in the timing cache.
        /// This flag has an effect only when IBuilderConfig has an associated ITimingCache.
        /// </summary>
        kERROR_ON_TIMING_CACHE_MISS = 16,

        /// <summary>
        /// 启用DataType::kBF16层选择，支持FP32回退。
        /// 此标志仅支持NVIDIA Ampere及更高版本的GPU。
        /// Enable DataType::kBF16 layer selection, with FP32 fallback.
        /// This flag is only supported by NVIDIA Ampere and later GPUs.
        /// </summary>
        /// <remarks>
        /// 已在TensorRT 10.12中弃用。已被强类型取代。
        /// Deprecated in TensorRT 10.12. Superseded by strong typing.
        /// </remarks>
        kBF16 = 17,

        /// <summary>
        /// 在引擎构建期间禁用JIT编译结果缓存。
        /// 默认情况下，JIT编译的代码将序列化为计时缓存的一部分，这可能会显著增加缓存大小。
        /// 设置此标志可防止代码被序列化。仅当未设置BuilderFlag::DISABLE_TIMING_CACHE时，此标志才有效。
        /// Disable caching of JIT-compilation results during engine build.
        /// By default, JIT-compiled code will be serialized as part of the timing cache, which may significantly increase
        /// the cache size. Setting this flag prevents the code from being serialized. This flag has an effect only when
        /// BuilderFlag::DISABLE_TIMING_CACHE is not set.
        /// </summary>
        kDISABLE_COMPILATION_CACHE = 18,

        /// <summary>
        /// 从引擎计划文件中剥离可重调优权重。
        /// Strip the refittable weights from the engine plan file.
        /// </summary>
        kSTRIP_PLAN = 19,

        /// <summary>
        /// 已在TensorRT 10.0中弃用。已被kSTRIP_PLAN取代。
        /// Deprecated in TensorRT 10.0. Superseded by kSTRIP_PLAN.
        /// </summary>
        kWEIGHTLESS = kSTRIP_PLAN,

        /// <summary>
        /// 在假设重调优权重将与构建时提供的权重相同的情况下创建可重调优引擎。
        /// 生成的引擎将与不可重调优引擎具有相同的性能。所有可重调优权重都可以通过重调优API进行重调优，
        /// 但如果重调优权重与构建时权重不同，则行为未定义。与'kSTRIP_PLAN'一起使用时，这将导致一个小的计划文件，
        /// 之后通过重调优提供权重。这使得可以将一组权重与不同的推理后端或TensorRT计划用于多个GPU架构一起使用。
        /// Create a refittable engine under the assumption that the refit weights will be identical to those provided at
        /// build time. The resulting engine will have the same performance as a non-refittable one. All refittable weights
        /// can be refitted through the refit API, but if the refit weights are not identical to the build-time weights,
        /// behavior is undefined. When used alongside 'kSTRIP_PLAN', this flag will result in a small plan file for which
        /// weights are later supplied via refitting. This enables use of a single set of weights with different inference
        /// backends, or with TensorRT plans for multiple GPU architectures.
        /// </summary>
        kREFIT_IDENTICAL = 20,

        /// <summary>
        /// 为当前引擎启用权重流式传输。
        /// 主机的权重流式传输允许执行不适合GPU内存的模型，使TensorRT能够智能地将网络权重从CPU DRAM流式传输。
        /// 请参阅ICudaEngine::getMinimumWeightStreamingBudget了解启用此标志时的默认内存预算。
        /// 启用此功能会改变IRuntime::deserializeCudaEngine的行为，在GPU内存上分配整个网络的权重，
        /// 而不是在GPU内存上。然后，ICudaEngine::createExecutionContext将确定在CPU和GPU之间
        /// 权重的最佳分割并相应放置权重。
        /// \brief Enable weight streaming for the current engine.
        /// Weight streaming from the host enables execution of models that do not fit
        /// in GPU memory by allowing TensorRT to intelligently stream network weights
        /// from the CPU DRAM. Please see ICudaEngine::getMinimumWeightStreamingBudget
        /// for the default memory budget when this flag is enabled.
        /// Enabling this feature changes the behavior of
        /// IRuntime::deserializeCudaEngine to allocate the entire network's weights
        /// on the CPU DRAM instead of GPU memory. Then,
        /// ICudaEngine::createExecutionContext will determine the optimal split of
        /// weights between the CPU and GPU and place weights accordingly.
        /// </summary>
        /// <remarks>
        /// 未来的TensorRT版本可能会默认启用此标志。
        /// Future TensorRT versions may enable this flag by default.
        /// 启用此标志可能会略微增加构建时间。
        /// Enabling this flag may marginally increase build time.
        /// 启用此功能将显著增加ICudaEngine::createExecutionContext的延迟。
        /// Enabling this feature will significantly increase the latency of
        /// ICudaEngine::createExecutionContext.
        /// 有关更多信息，请参阅IRuntime::deserializeCudaEngine、ICudaEngine::getMinimumWeightStreamingBudget和ICudaEngine::setWeightStreamingBudget
        /// \see IRuntime::deserializeCudaEngine,
        ///      ICudaEngine::getMinimumWeightStreamingBudget,
        ///      ICudaEngine::setWeightStreamingBudget
        /// </remarks>
        kWEIGHT_STREAMING = 21,

        /// <summary>
        /// 启用具有INT4输入/输出的插件。
        /// Enable plugins with INT4 input/output.
        /// </summary>
        /// <remarks>
        /// 已在TensorRT 10.12中弃用。已被强类型取代。
        /// Deprecated in TensorRT 10.12. Superseded by strong typing.
        /// </remarks>
        kINT4 = 22,

        /// <summary>
        /// 启用构建可重调优引擎并提供细粒度控制。这允许使用INetworkDefinition::markWeightsRefittable
        /// 和INetworkDefinition::unmarkWeightsRefittable控制哪些权重是可重调优的。启用此标志时，
        /// 默认情况下所有权重都是不可重调优的。此标志不能与kREFIT或kREFIT_IDENTICAL一起使用。
        /// Enable building a refittable engine and provide fine-grained control. This allows
        /// control over which weights are refittable or not using INetworkDefinition::markWeightsRefittable and
        /// INetworkDefinition::unmarkWeightsRefittable. By default, all weights are non-refittable when this flag is
        /// enabled. This flag cannot be used together with kREFIT or kREFIT_IDENTICAL.
        /// </summary>
        kREFIT_INDIVIDUAL = 23,

        /// <summary>
        /// 禁用浮点优化：0*x => 0，x-x => 0，或x/x => 1。当x是NaN或Inf时，这些恒等式不成立，
        /// 因此可能会隐藏NaN的产生或传播。此标志通常与kSPARSE_WEIGHTS结合使用。
        /// 有三种有效的稀疏性配置：
        /// 1. 禁用所有稀疏性。kSPARSE_WEIGHTS和kSTRICT_NANS都未设置
        /// 2. 仅在不影响NaN的产生/传播的地方启用稀疏性。kSPARSE_WEIGHTS和kSTRICT_NANS都设置
        /// 3. 启用所有稀疏性。设置了kSPARSE_WEIGHTS，未设置kSTRICT_NANS
        /// Disable floating-point optimizations: 0*x => 0, x-x => 0, or x/x => 1. These identities are
        /// not true when x is a NaN or Inf, and thus might hide propagation or generation of NaNs. This flag is typically
        /// used in combination with kSPARSE_WEIGHTS.
        /// There are three valid sparsity configurations.
        /// 1. Disable all sparsity. Both kSPARSE_WEIGHTS and kSTRICT_NANS are unset
        /// 2. Enable sparsity only where it does not affect propagation/generation of NaNs. Both kSPARSE_WEIGHTS and
        /// kSTRICT_NANS are set
        /// 3. Enable all sparsity. kSPARSE_WEIGHTS is set and kSTRICT_NANS is unset
        /// </summary>
        kSTRICT_NANS = 24,

        /// <summary>
        /// 在构建期间启用内存监视器。
        /// Enable memory monitor during build time.
        /// </summary>
        kMONITOR_MEMORY = 25,

        /// <summary>
        /// 启用具有FP4输入/输出的插件。
        /// Enable plugins with FP4 input/output.
        /// </summary>
        /// <remarks>
        /// 已在TensorRT 10.12中弃用。已被强类型取代。
        /// Deprecated in TensorRT 10.12. Superseded by strong typing.
        /// </remarks>
        kFP4 = 26,

        /// <summary>
        /// 启用可编辑计时缓存。
        /// Enable editable timing cache.
        /// </summary>
        BuilderFlag_kEDITABLE_TIMING_CACHE = 27,

        /// <summary>
        /// 启用分配独立性。当设置BuilderFlag::kDISTRIBUTIVE_INDEPENDENCE且层的输出将轴i记录为分配轴时，
        /// 该层的行为就像沿轴i的每次评估都是使用相同操作完成的。
        /// 分配轴的定义如下：
        /// 对于IMatrixMultiplyLayer：所有不是向量或矩阵维度的轴都是分配轴。
        /// 对于执行归约的层：所有非归约轴都是分配轴。
        /// 对于执行einsum的层：设n是最左边的归约轴。n左侧的轴是分配轴。
        /// Enable distributive independence.
        /// When BuilderFlag::kDISTRIBUTIVE_INDEPENDENCE is set and a layer documents axis i of an output as a distributive
        /// axis, then the layer behaves exactly as if each evaluation across axis i was done using identical operations.
        /// The definition of distributive axis is as follows:
        /// For IMatrixMultiplyLayer:
        /// All axes that are not one of the vector or matrix dimensions are distributive axes.
        /// For layers that perform reduction:
        /// All non-reduction axes are distributive axes.
        /// For layers that perform einsum:
        /// Let n be the leftmost reduction axis. The axes to the left of n are distributive axes.
        /// </summary>
        kDISTRIBUTIVE_INDEPENDENCE = 28,
    }

}

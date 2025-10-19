using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Nvinfer
{
    //!
    //! \enum BuilderFlag
    //!
    //! \brief List of valid modes that the builder can enable when creating an engine from a network definition.
    //!
    //! \see IBuilderConfig::setFlags(), IBuilderConfig::getFlags()
    //!
    public enum TrtBuilderFlag : int
    {
        //! Enable FP16 layer selection, with FP32 fallback.
        //! \deprecated Deprecated in TensorRT 10.12. Superseded by strong typing.
        kFP16 = 0,

        //! Enable Int8 layer selection, with FP32 fallback with FP16 fallback if kFP16 also specified.
        //! \deprecated Deprecated in TensorRT 10.12. Superseded by strong typing.
        kINT8 = 1,

        //! Enable debugging of layers via synchronizing after every layer.
        kDEBUG = 2,

        //! Enable layers marked to execute on GPU if layer cannot execute on DLA.
        kGPU_FALLBACK = 3,

        //! Enable building a refittable engine.
        kREFIT = 4,

        //! Disable reuse of timing information across identical layers.
        kDISABLE_TIMING_CACHE = 5,

        //! Allow (but not require) computations on tensors of type DataType::kFLOAT to use TF32.
        //! TF32 computes inner products by rounding the inputs to 10-bit mantissas before
        //! multiplying, but accumulates the sum using 23-bit mantissas. Enabled by default.
        kTF32 = 6,

        //! Allow the builder to examine weights and use optimized functions when weights have suitable sparsity.
        kSPARSE_WEIGHTS = 7,

        //! Change the allowed parameters in the EngineCapability::kSTANDARD flow to
        //! match the restrictions that EngineCapability::kSAFETY check against for DeviceType::kGPU
        //! and EngineCapability::kDLA_STANDALONE check against the DeviceType::kDLA case. This flag
        //! is forced to true if EngineCapability::kSAFETY at build time if it is unset.
        //!
        //! This flag is only supported in NVIDIA Drive(R) products.
        kSAFETY_SCOPE = 8,

        //! Require that layers execute in specified precisions. Build fails otherwise.
        //! \deprecated Deprecated in TensorRT 10.12. Superseded by strong typing.
        kOBEY_PRECISION_CONSTRAINTS = 9,

        //! Prefer that layers execute in specified precisions.
        //! Fall back (with warning) to another precision if build would otherwise fail.
        //! \deprecated Deprecated in TensorRT 10.12. Superseded by strong typing.
        kPREFER_PRECISION_CONSTRAINTS = 10,

        //! Require that no reformats be inserted between a layer and a network I/O tensor
        //! for which ITensor::setAllowedFormats was called.
        //! Build fails if a reformat is required for functional correctness.
        //! \deprecated Deprecated in TensorRT 10.7. Unneeded API.
        kDIRECT_IO = 11,

        //! Fail if IAlgorithmSelector::selectAlgorithms returns an empty set of algorithms.
        //! \deprecated Deprecated in TensorRT 10.10. Unneeded API due to IAlgorithmSelector deprecation.
        kREJECT_EMPTY_ALGORITHMS = 12,

        //! Restrict to lean runtime operators to provide version forward compatibility
        //! for the plan.
        //!
        //! This flag is only supported by NVIDIA Volta and later GPUs.
        //! This flag is not supported in NVIDIA Drive(R) products.
        kVERSION_COMPATIBLE = 13,

        //! Exclude lean runtime from the plan when version forward compatability is enabled.
        //! By default, this flag is unset, so the lean runtime will be included in the plan.
        //!
        //! If BuilderFlag::kVERSION_COMPATIBLE is not set then the value of this flag will be ignored.
        kEXCLUDE_LEAN_RUNTIME = 14,

        //! Enable plugins with FP8 input/output.
        //!
        //! This flag is not supported when HardwareCompatibilityLevel::kAMPERE_PLUS is enabled.
        //!
        //! \see HardwareCompatibilityLevel
        //! \deprecated Deprecated in TensorRT 10.12. Superseded by strong typing.
        kFP8 = 15,

        //! Emit error when a tactic being timed is not present in the timing cache.
        //! This flag has an effect only when IBuilderConfig has an associated ITimingCache.
        kERROR_ON_TIMING_CACHE_MISS = 16,

        //! Enable DataType::kBF16 layer selection, with FP32 fallback.
        //! This flag is only supported by NVIDIA Ampere and later GPUs.
        //! \deprecated Deprecated in TensorRT 10.12. Superseded by strong typing.
        kBF16 = 17,

        //! Disable caching of JIT-compilation results during engine build.
        //! By default, JIT-compiled code will be serialized as part of the timing cache, which may significantly increase
        //! the cache size. Setting this flag prevents the code from being serialized. This flag has an effect only when
        //! BuilderFlag::DISABLE_TIMING_CACHE is not set.
        kDISABLE_COMPILATION_CACHE = 18,

        //! Strip the refittable weights from the engine plan file.
        kSTRIP_PLAN = 19,

        //! \deprecated Deprecated in TensorRT 10.0. Superseded by kSTRIP_PLAN.
        kWEIGHTLESS = kSTRIP_PLAN,

        //! Create a refittable engine under the assumption that the refit weights will be identical to those provided at
        //! build time. The resulting engine will have the same performance as a non-refittable one. All refittable weights
        //! can be refitted through the refit API, but if the refit weights are not identical to the build-time weights,
        //! behavior is undefined. When used alongside 'kSTRIP_PLAN', this flag will result in a small plan file for which
        //! weights are later supplied via refitting. This enables use of a single set of weights with different inference
        //! backends, or with TensorRT plans for multiple GPU architectures.
        kREFIT_IDENTICAL = 20,

        //!
        //! \brief Enable weight streaming for the current engine.
        //!
        //! Weight streaming from the host enables execution of models that do not fit
        //! in GPU memory by allowing TensorRT to intelligently stream network weights
        //! from the CPU DRAM. Please see ICudaEngine::getMinimumWeightStreamingBudget
        //! for the default memory budget when this flag is enabled.
        //!
        //! Enabling this feature changes the behavior of
        //! IRuntime::deserializeCudaEngine to allocate the entire network's weights
        //! on the CPU DRAM instead of GPU memory. Then,
        //! ICudaEngine::createExecutionContext will determine the optimal split of
        //! weights between the CPU and GPU and place weights accordingly.
        //!
        //! Future TensorRT versions may enable this flag by default.
        //!
        //! \warning Enabling this flag may marginally increase build time.
        //!
        //! \warning Enabling this feature will significantly increase the latency of
        //!          ICudaEngine::createExecutionContext.
        //!
        //! \see IRuntime::deserializeCudaEngine,
        //!      ICudaEngine::getMinimumWeightStreamingBudget,
        //!      ICudaEngine::setWeightStreamingBudget
        //!
        kWEIGHT_STREAMING = 21,

        //! Enable plugins with INT4 input/output.
        //! \deprecated Deprecated in TensorRT 10.12. Superseded by strong typing.
        kINT4 = 22,

        //! Enable building a refittable engine and provide fine-grained control. This allows
        //! control over which weights are refittable or not using INetworkDefinition::markWeightsRefittable and
        //! INetworkDefinition::unmarkWeightsRefittable. By default, all weights are non-refittable when this flag is
        //! enabled. This flag cannot be used together with kREFIT or kREFIT_IDENTICAL.
        kREFIT_INDIVIDUAL = 23,

        //!  Disable floating-point optimizations: 0*x => 0, x-x => 0, or x/x => 1. These identities are
        //!  not true when x is a NaN or Inf, and thus might hide propagation or generation of NaNs. This flag is typically
        //!  used in combination with kSPARSE_WEIGHTS.
        //!  There are three valid sparsity configurations.
        //!  1. Disable all sparsity. Both kSPARSE_WEIGHTS and kSTRICT_NANS are unset
        //!  2. Enable sparsity only where it does not affect propagation/generation of NaNs. Both kSPARSE_WEIGHTS and
        //!  kSTRICT_NANS are set
        //!  3. Enable all sparsity. kSPARSE_WEIGHTS is set and kSTRICT_NANS is unset
        kSTRICT_NANS = 24,

        //! Enable memory monitor during build time.
        kMONITOR_MEMORY = 25,

        //! Enable plugins with FP4 input/output.
        //! \deprecated Deprecated in TensorRT 10.12. Superseded by strong typing.
        kFP4 = 26,

        //! Enable editable timing cache.
        BuilderFlag_kEDITABLE_TIMING_CACHE = 27,

        //! Enable distributive independence.
        //! When BuilderFlag::kDISTRIBUTIVE_INDEPENDENCE is set and a layer documents axis i of an output as a distributive
        //! axis, then the layer behaves exactly as if each evaluation across axis i was done using identical operations.
        //! The definition of distributive axis is as follows:
        //! For IMatrixMultiplyLayer:
        //! All axes that are not one of the vector or matrix dimensions are distributive axes.
        //! For layers that perform reduction:
        //! All non-reduction axes are distributive axes.
        //! For layers that perform einsum:
        //! Let n be the leftmost reduction axis. The axes to the left of n are distributive axes.
        kDISTRIBUTIVE_INDEPENDENCE = 28,

    }
}

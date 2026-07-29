using System;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Selects TensorRT builder preview features. Some values are valid only on specific TensorRT major versions.
/// 选择 TensorRT builder 预览特性；部分取值仅在特定 TensorRT 大版本中有效。
/// </summary>
public enum TensorRtPreviewFeature
{
    /// <summary>
    /// TensorRT 8.x dynamic-shape compiler preview feature. Deprecated by NVIDIA.
    /// TensorRT 8.x 动态 shape 编译器预览特性，NVIDIA 已标记为弃用。
    /// </summary>
    FasterDynamicShapes0805 = 0,

    /// <summary>
    /// TensorRT 10.x profile-sharing feature. Deprecated and always enabled by NVIDIA.
    /// TensorRT 10.x profile 共享特性；NVIDIA 已弃用并默认始终启用。
    /// </summary>
    ProfileSharing0806Trt10 = 0,

    /// <summary>
    /// TensorRT 8.x option that disables external library tactics for TensorRT core.
    /// TensorRT 8.x 中禁用 TensorRT core 外部库 tactics 的选项。
    /// </summary>
    DisableExternalTacticSourcesForCore0805 = 1,

    /// <summary>
    /// TensorRT 8.x profile-sharing feature. In TensorRT 10.x use <see cref="ProfileSharing0806Trt10"/> instead.
    /// TensorRT 8.x profile 共享特性；在 TensorRT 10.x 中请使用 <see cref="ProfileSharing0806Trt10"/>。
    /// </summary>
    ProfileSharing0806 = 2,

    /// <summary>
    /// TensorRT 10.x plugin I/O aliasing preview feature.
    /// TensorRT 10.x plugin I/O 别名预览特性。
    /// </summary>
    AliasedPluginIo1003 = 1,

    /// <summary>
    /// TensorRT 10.x runtime activation-memory resize preview feature.
    /// TensorRT 10.x 运行时 activation 内存 resize 预览特性。
    /// </summary>
    RuntimeActivationResize1010 = 2
}

/// <summary>
/// Represents TensorRT TensorRtDeviceType values.
/// 表示 TensorRT TensorRtDeviceType 枚举值。
/// </summary>
public enum TensorRtDeviceType
{
    /// <summary>
    /// Represents the Gpu value of TensorRtDeviceType.
    /// 表示 TensorRtDeviceType 的 Gpu 取值。
    /// </summary>
    Gpu = 0,
    /// <summary>
    /// Represents the Dla value of TensorRtDeviceType.
    /// 表示 TensorRtDeviceType 的 Dla 取值。
    /// </summary>
    Dla = 1
}

/// <summary>
/// Selects TensorRT 11 tiling optimization effort.
/// 选择 TensorRT 11 tiling 优化搜索强度。
/// </summary>
public enum TensorRtTilingOptimizationLevel
{
    /// <summary>
    /// Disable tiling optimization.
    /// 禁用 tiling 优化。
    /// </summary>
    None = 0,

    /// <summary>
    /// Use a fast heuristic strategy.
    /// 使用快速启发式策略。
    /// </summary>
    Fast = 1,

    /// <summary>
    /// Use a moderate mixed heuristic/profiling strategy.
    /// 使用中等强度的启发式与 profiling 混合策略。
    /// </summary>
    Moderate = 2,

    /// <summary>
    /// Use the most exhaustive tiling search.
    /// 使用更完整的 tiling 搜索。
    /// </summary>
    Full = 3
}

/// <summary>
/// Selects TensorRT 8/10 quantization calibration behavior.
/// 选择 TensorRT 8/10 quantization calibration 行为。
/// </summary>
public enum TensorRtQuantizationFlag
{
    /// <summary>
    /// Run the INT8 calibration pass before layer fusion.
    /// 在 layer fusion 前运行 INT8 calibration pass。
    /// </summary>
    CalibrateBeforeFusion = 0
}

/// <summary>
/// Represents a TensorRT 8/10 quantization flag bitmask.
/// 表示 TensorRT 8/10 quantization flag 位掩码。
/// </summary>
[Flags]
public enum TensorRtQuantizationFlags : uint
{
    /// <summary>
    /// No quantization flags are enabled.
    /// 不启用任何 quantization flag。
    /// </summary>
    None = 0,

    /// <summary>
    /// Run the INT8 calibration pass before layer fusion.
    /// 在 layer fusion 前运行 INT8 calibration pass。
    /// </summary>
    CalibrateBeforeFusion = 1u << (int)TensorRtQuantizationFlag.CalibrateBeforeFusion
}

/// <summary>
/// Represents TensorRT TensorRtTacticSources values.
/// 表示 TensorRT TensorRtTacticSources 枚举值。
/// </summary>
[Flags]
public enum TensorRtTacticSources : uint
{
    /// <summary>
    /// Represents the None value of TensorRtTacticSources.
    /// 表示 TensorRtTacticSources 的 None 取值。
    /// </summary>
    None = 0,
    /// <summary>
    /// Represents the CuBlas value of TensorRtTacticSources.
    /// 表示 TensorRtTacticSources 的 CuBlas 取值。
    /// </summary>
    CuBlas = 1u << 0,
    /// <summary>
    /// Represents the CuBlasLt value of TensorRtTacticSources.
    /// 表示 TensorRtTacticSources 的 CuBlasLt 取值。
    /// </summary>
    CuBlasLt = 1u << 1,
    /// <summary>
    /// Represents the CuDnn value of TensorRtTacticSources.
    /// 表示 TensorRtTacticSources 的 CuDnn 取值。
    /// </summary>
    CuDnn = 1u << 2,
    /// <summary>
    /// Represents the EdgeMaskConvolutions value of TensorRtTacticSources.
    /// 表示 TensorRtTacticSources 的 EdgeMaskConvolutions 取值。
    /// </summary>
    EdgeMaskConvolutions = 1u << 3,
    /// <summary>
    /// Represents the JitConvolutions value of TensorRtTacticSources.
    /// 表示 TensorRtTacticSources 的 JitConvolutions 取值。
    /// </summary>
    JitConvolutions = 1u << 4
}

/// <summary>
/// Represents TensorRT TensorRtBuilderFlag values.
/// 表示 TensorRT TensorRtBuilderFlag 枚举值。
/// </summary>
public enum TensorRtBuilderFlag
{
    /// <summary>
    /// Represents the Fp16 value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 Fp16 取值。
    /// </summary>
    Fp16 = 0,
    /// <summary>
    /// Represents the Int8 value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 Int8 取值。
    /// </summary>
    Int8 = 1,
    /// <summary>
    /// Represents the Debug value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 Debug 取值。
    /// </summary>
    Debug = 2,
    /// <summary>
    /// Represents the GpuFallback value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 GpuFallback 取值。
    /// </summary>
    GpuFallback = 3,
    /// <summary>
    /// Represents the Refit value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 Refit 取值。
    /// </summary>
    Refit = 4,
    /// <summary>
    /// Represents the DisableTimingCache value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 DisableTimingCache 取值。
    /// </summary>
    DisableTimingCache = 5,
    /// <summary>
    /// Represents the Tf32 value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 Tf32 取值。
    /// </summary>
    Tf32 = 6,
    /// <summary>
    /// Represents the SparseWeights value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 SparseWeights 取值。
    /// </summary>
    SparseWeights = 7,
    /// <summary>
    /// Represents the SafetyScope value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 SafetyScope 取值。
    /// </summary>
    SafetyScope = 8,
    /// <summary>
    /// Represents the ObeyPrecisionConstraints value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 ObeyPrecisionConstraints 取值。
    /// </summary>
    ObeyPrecisionConstraints = 9,
    /// <summary>
    /// Represents the PreferPrecisionConstraints value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 PreferPrecisionConstraints 取值。
    /// </summary>
    PreferPrecisionConstraints = 10,
    /// <summary>
    /// Represents the DirectIO value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 DirectIO 取值。
    /// </summary>
    DirectIO = 11,
    /// <summary>
    /// Represents the RejectEmptyAlgorithms value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 RejectEmptyAlgorithms 取值。
    /// </summary>
    RejectEmptyAlgorithms = 12,
    /// <summary>
    /// Represents the VersionCompatible value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 VersionCompatible 取值。
    /// </summary>
    VersionCompatible = 13,
    /// <summary>
    /// Represents the ExcludeLeanRuntime value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 ExcludeLeanRuntime 取值。
    /// </summary>
    ExcludeLeanRuntime = 14,
    /// <summary>
    /// Represents the Fp8 value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 Fp8 取值。
    /// </summary>
    Fp8 = 15,
    /// <summary>
    /// Represents the ErrorOnTimingCacheMiss value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 ErrorOnTimingCacheMiss 取值。
    /// </summary>
    ErrorOnTimingCacheMiss = 16,
    /// <summary>
    /// Represents the Bf16 value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 Bf16 取值。
    /// </summary>
    Bf16 = 17,
    /// <summary>
    /// Represents the DisableCompilationCache value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 DisableCompilationCache 取值。
    /// </summary>
    DisableCompilationCache = 18,
    /// <summary>
    /// Represents the StripPlan value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 StripPlan 取值。
    /// </summary>
    StripPlan = 19,
    /// <summary>
    /// Represents the RefitIdentical value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 RefitIdentical 取值。
    /// </summary>
    RefitIdentical = 20,
    /// <summary>
    /// Represents the WeightStreaming value of TensorRtBuilderFlag.
    /// 表示 TensorRtBuilderFlag 的 WeightStreaming 取值。
    /// </summary>
    WeightStreaming = 21
}

/// <summary>
/// Represents a TensorRT builder flag bitmask.
/// 表示 TensorRT builder flag 位掩码。
/// </summary>
[Flags]
public enum TensorRtBuilderFlags : uint
{
    /// <summary>
    /// No builder flags are enabled.
    /// 不启用任何 builder flag。
    /// </summary>
    None = 0,
    /// <summary>
    /// Represents the Fp16 value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 Fp16 取值。
    /// </summary>
    Fp16 = 1u << (int)TensorRtBuilderFlag.Fp16,
    /// <summary>
    /// Represents the Int8 value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 Int8 取值。
    /// </summary>
    Int8 = 1u << (int)TensorRtBuilderFlag.Int8,
    /// <summary>
    /// Represents the Debug value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 Debug 取值。
    /// </summary>
    Debug = 1u << (int)TensorRtBuilderFlag.Debug,
    /// <summary>
    /// Represents the GpuFallback value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 GpuFallback 取值。
    /// </summary>
    GpuFallback = 1u << (int)TensorRtBuilderFlag.GpuFallback,
    /// <summary>
    /// Represents the Refit value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 Refit 取值。
    /// </summary>
    Refit = 1u << (int)TensorRtBuilderFlag.Refit,
    /// <summary>
    /// Represents the DisableTimingCache value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 DisableTimingCache 取值。
    /// </summary>
    DisableTimingCache = 1u << (int)TensorRtBuilderFlag.DisableTimingCache,
    /// <summary>
    /// Represents the Tf32 value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 Tf32 取值。
    /// </summary>
    Tf32 = 1u << (int)TensorRtBuilderFlag.Tf32,
    /// <summary>
    /// Represents the SparseWeights value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 SparseWeights 取值。
    /// </summary>
    SparseWeights = 1u << (int)TensorRtBuilderFlag.SparseWeights,
    /// <summary>
    /// Represents the SafetyScope value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 SafetyScope 取值。
    /// </summary>
    SafetyScope = 1u << (int)TensorRtBuilderFlag.SafetyScope,
    /// <summary>
    /// Represents the ObeyPrecisionConstraints value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 ObeyPrecisionConstraints 取值。
    /// </summary>
    ObeyPrecisionConstraints = 1u << (int)TensorRtBuilderFlag.ObeyPrecisionConstraints,
    /// <summary>
    /// Represents the PreferPrecisionConstraints value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 PreferPrecisionConstraints 取值。
    /// </summary>
    PreferPrecisionConstraints = 1u << (int)TensorRtBuilderFlag.PreferPrecisionConstraints,
    /// <summary>
    /// Represents the DirectIO value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 DirectIO 取值。
    /// </summary>
    DirectIO = 1u << (int)TensorRtBuilderFlag.DirectIO,
    /// <summary>
    /// Represents the RejectEmptyAlgorithms value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 RejectEmptyAlgorithms 取值。
    /// </summary>
    RejectEmptyAlgorithms = 1u << (int)TensorRtBuilderFlag.RejectEmptyAlgorithms,
    /// <summary>
    /// Represents the VersionCompatible value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 VersionCompatible 取值。
    /// </summary>
    VersionCompatible = 1u << (int)TensorRtBuilderFlag.VersionCompatible,
    /// <summary>
    /// Represents the ExcludeLeanRuntime value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 ExcludeLeanRuntime 取值。
    /// </summary>
    ExcludeLeanRuntime = 1u << (int)TensorRtBuilderFlag.ExcludeLeanRuntime,
    /// <summary>
    /// Represents the Fp8 value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 Fp8 取值。
    /// </summary>
    Fp8 = 1u << (int)TensorRtBuilderFlag.Fp8,
    /// <summary>
    /// Represents the ErrorOnTimingCacheMiss value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 ErrorOnTimingCacheMiss 取值。
    /// </summary>
    ErrorOnTimingCacheMiss = 1u << (int)TensorRtBuilderFlag.ErrorOnTimingCacheMiss,
    /// <summary>
    /// Represents the Bf16 value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 Bf16 取值。
    /// </summary>
    Bf16 = 1u << (int)TensorRtBuilderFlag.Bf16,
    /// <summary>
    /// Represents the DisableCompilationCache value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 DisableCompilationCache 取值。
    /// </summary>
    DisableCompilationCache = 1u << (int)TensorRtBuilderFlag.DisableCompilationCache,
    /// <summary>
    /// Represents the StripPlan value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 StripPlan 取值。
    /// </summary>
    StripPlan = 1u << (int)TensorRtBuilderFlag.StripPlan,
    /// <summary>
    /// Represents the RefitIdentical value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 RefitIdentical 取值。
    /// </summary>
    RefitIdentical = 1u << (int)TensorRtBuilderFlag.RefitIdentical,
    /// <summary>
    /// Represents the WeightStreaming value of TensorRtBuilderFlags.
    /// 表示 TensorRtBuilderFlags 的 WeightStreaming 取值。
    /// </summary>
    WeightStreaming = 1u << (int)TensorRtBuilderFlag.WeightStreaming
}

/// <summary>
/// Represents TensorRT TensorRtMemoryPoolType values.
/// 表示 TensorRT TensorRtMemoryPoolType 枚举值。
/// </summary>
public enum TensorRtMemoryPoolType
{
    /// <summary>
    /// Represents the Workspace value of TensorRtMemoryPoolType.
    /// 表示 TensorRtMemoryPoolType 的 Workspace 取值。
    /// </summary>
    Workspace = 0,
    /// <summary>
    /// Represents the DlaManagedSram value of TensorRtMemoryPoolType.
    /// 表示 TensorRtMemoryPoolType 的 DlaManagedSram 取值。
    /// </summary>
    DlaManagedSram = 1,
    /// <summary>
    /// Represents the DlaLocalDram value of TensorRtMemoryPoolType.
    /// 表示 TensorRtMemoryPoolType 的 DlaLocalDram 取值。
    /// </summary>
    DlaLocalDram = 2,
    /// <summary>
    /// Represents the DlaGlobalDram value of TensorRtMemoryPoolType.
    /// 表示 TensorRtMemoryPoolType 的 DlaGlobalDram 取值。
    /// </summary>
    DlaGlobalDram = 3,
    /// <summary>
    /// Represents the TacticDram value of TensorRtMemoryPoolType.
    /// 表示 TensorRtMemoryPoolType 的 TacticDram 取值。
    /// </summary>
    TacticDram = 4,
    /// <summary>
    /// Represents the TacticSharedMemory value of TensorRtMemoryPoolType.
    /// 表示 TensorRtMemoryPoolType 的 TacticSharedMemory 取值。
    /// </summary>
    TacticSharedMemory = 5
}

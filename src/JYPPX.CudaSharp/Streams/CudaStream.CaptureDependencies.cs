using System;
using System.Collections.Generic;
using JYPPX.CudaSharp.Internal.Handles;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

public sealed partial class CudaStream
{
    /// <summary>
    /// Adds or replaces dependencies through CUDA's per-thread stream variant.
    /// 通过 CUDA per-thread stream 变体追加或替换依赖。
    /// </summary>
    public void UpdateCaptureDependenciesPtzs(
        IReadOnlyList<CudaGraphNode> dependencies,
        CudaStreamCaptureDependencyMode mode = CudaStreamCaptureDependencyMode.Add)
    {
        if (dependencies == null)
        {
            throw new ArgumentNullException(nameof(dependencies));
        }

        if (mode != CudaStreamCaptureDependencyMode.Add && mode != CudaStreamCaptureDependencyMode.Replace)
        {
            throw new ArgumentOutOfRangeException(nameof(mode));
        }

        NativeCudaApi.UpdateStreamCaptureDependenciesPtzs(_handle, dependencies, mode);
    }

    /// <summary>
    /// Adds or replaces dependencies with copied CUDA graph edge data.
    /// 使用复制型 CUDA graph edge data 追加或替换依赖。
    /// </summary>
    public void UpdateCaptureDependenciesV2(
        IReadOnlyList<CudaGraphNodeDependency> dependencies,
        CudaStreamCaptureDependencyMode mode = CudaStreamCaptureDependencyMode.Add)
    {
        if (dependencies == null)
        {
            throw new ArgumentNullException(nameof(dependencies));
        }

        if (mode != CudaStreamCaptureDependencyMode.Add && mode != CudaStreamCaptureDependencyMode.Replace)
        {
            throw new ArgumentOutOfRangeException(nameof(mode));
        }

        NativeCudaApi.UpdateStreamCaptureDependenciesV2(_handle, dependencies, mode);
    }

    /// <summary>
    /// Adds or replaces the dependency set used by the next operation in an active stream capture.
    /// 在 active stream capture 中追加或替换下一项操作使用的 dependency set。
    /// </summary>
    public void UpdateCaptureDependencies(
        IReadOnlyList<CudaGraphNode> dependencies,
        CudaStreamCaptureDependencyMode mode = CudaStreamCaptureDependencyMode.Add)
    {
        if (dependencies == null)
        {
            throw new ArgumentNullException(nameof(dependencies));
        }

        if (mode != CudaStreamCaptureDependencyMode.Add && mode != CudaStreamCaptureDependencyMode.Replace)
        {
            throw new ArgumentOutOfRangeException(nameof(mode));
        }

        NativeCudaApi.UpdateStreamCaptureDependencies(_handle, dependencies, mode);
    }

}

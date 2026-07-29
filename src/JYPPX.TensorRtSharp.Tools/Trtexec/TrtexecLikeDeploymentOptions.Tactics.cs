using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using JYPPX.TensorRtSharp;

namespace JYPPX.TensorRtSharp.Tools;

public sealed partial class TrtexecLikeDeploymentOptions
{
    /// <summary>
    /// Applies the normalized trtexec tactic-source additions and removals to a default TensorRT mask.
    /// 将规范化的 trtexec tactic source 增删项应用到 TensorRT 默认掩码。
    /// </summary>
    /// <param name="defaultSources">The default tactic-source mask reported by TensorRT. TensorRT 返回的默认 tactic source 掩码。</param>
    /// <returns>The requested tactic-source mask. 请求的 tactic source 掩码。</returns>
    public TensorRtTacticSources ResolveTacticSources(TensorRtTacticSources defaultSources)
    {
        TensorRtTacticSources resolved = defaultSources;
        foreach (string item in TacticSources.Split(new[] { ',', ';' }, StringSplitOptions.RemoveEmptyEntries))
        {
            string trimmed = item.Trim();
            if (trimmed.Length < 2 || (trimmed[0] != '+' && trimmed[0] != '-'))
            {
                throw new ArgumentException("Tactic source entries must begin with + or -.", nameof(TacticSources));
            }

            TensorRtTacticSources source = trimmed.Substring(1)
                .Replace("-", string.Empty, StringComparison.Ordinal)
                .Replace("_", string.Empty, StringComparison.Ordinal)
                .ToLowerInvariant() switch
            {
                "cublas" => TensorRtTacticSources.CuBlas,
                "cublaslt" => TensorRtTacticSources.CuBlasLt,
                "cudnn" => TensorRtTacticSources.CuDnn,
                "edgemaskconvolutions" or "edgemask" => TensorRtTacticSources.EdgeMaskConvolutions,
                "jitconvolutions" or "jit" => TensorRtTacticSources.JitConvolutions,
                _ => throw new ArgumentException($"Unsupported tactic source '{trimmed}'.", nameof(TacticSources))
            };

            resolved = trimmed[0] == '+' ? resolved | source : resolved & ~source;
        }

        return resolved;
    }
}

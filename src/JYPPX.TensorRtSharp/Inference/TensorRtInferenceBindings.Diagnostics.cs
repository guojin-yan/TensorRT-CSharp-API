using System.Text;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtInferenceBindings
{
    /// <summary>
    /// Creates a readable multi-line summary for deployment diagnostics.
    /// 创建用于部署诊断的多行可读摘要。
    /// </summary>
    /// <returns>A summary string. 摘要字符串。</returns>
    public string Describe()
    {
        ThrowIfDisposed();
        StringBuilder builder = new StringBuilder();
        builder.Append("TensorRtInferenceBindings profile=").Append(ProfileIndex).Append(" engine=").AppendLine(Report.EngineName);
        foreach (TensorRtInferenceBuffer buffer in _buffers.Values)
        {
            builder.Append("  ").AppendLine(buffer.ToString());
        }

        return builder.ToString();
    }
}

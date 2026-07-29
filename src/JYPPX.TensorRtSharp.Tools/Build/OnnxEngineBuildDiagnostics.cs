using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using JYPPX.Shared.Interop;

namespace JYPPX.TensorRtSharp.Tools;

public static partial class OnnxEngineBuildDiagnostics
{
    public static void WriteReport(OnnxEngineBuildResult result, string reportPath)
    {
        if (result == null)
        {
            throw new ArgumentNullException(nameof(result));
        }

        if (string.IsNullOrWhiteSpace(reportPath))
        {
            return;
        }

        string? directory = Path.GetDirectoryName(Path.GetFullPath(reportPath));
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        string extension = Path.GetExtension(reportPath);
        string content = string.Equals(extension, ".md", StringComparison.OrdinalIgnoreCase)
            ? ToMarkdown(result)
            : ToJson(result);
        File.WriteAllText(reportPath, content);
    }

}

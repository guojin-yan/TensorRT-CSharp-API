using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace JYPPX.TensorRtSharp.Tools;

public static partial class OnnxEngineRuntimeArtifactWriter
{
    private static readonly JsonSerializerOptions JsonOptions = new JsonSerializerOptions
    {
        WriteIndented = true,
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals
    };

    public static void WriteArtifacts(OnnxEngineBuildResult result, OnnxEngineRuntimeArtifactData? data = null)
    {
        if (result == null)
        {
            throw new ArgumentNullException(nameof(result));
        }

        TrtexecLikeRuntimeOptions options = result.RuntimeOptions;
        if (!options.HasRuntimeDiagnostics)
        {
            return;
        }

        OnnxEngineRuntimeArtifactData artifactData = data ?? OnnxEngineRuntimeArtifactData.Empty;
        if (!string.IsNullOrWhiteSpace(options.ExportTimesPath))
        {
            WriteJson(options.ExportTimesPath, CreateTimesArtifact(result, artifactData));
        }

        if (!string.IsNullOrWhiteSpace(options.ExportOutputPath))
        {
            WriteJson(options.ExportOutputPath, CreateOutputArtifact(result, artifactData));
        }

        if (!string.IsNullOrWhiteSpace(options.ExportProfilePath))
        {
            WriteJson(options.ExportProfilePath, CreateProfileArtifact(result, artifactData));
        }

        string engineReadbackPath = GetEngineReadbackArtifactPath(options);
        if (!string.IsNullOrWhiteSpace(engineReadbackPath))
        {
            WriteJson(engineReadbackPath, CreateEngineReadbackArtifact(result, artifactData));
        }

        if (!string.IsNullOrWhiteSpace(options.SaveProfilePath))
        {
            WriteText(options.SaveProfilePath, CreateProfileText(result, artifactData));
        }

        if (!string.IsNullOrWhiteSpace(options.DumpRawBindingsToFile))
        {
            WriteRawBindingsOrBoundary(options.DumpRawBindingsToFile, result, artifactData);
        }
    }

    private static string ComputeSha256(byte[] bytes)
    {
        if (bytes == null || bytes.Length == 0)
        {
            return string.Empty;
        }

        using SHA256 sha256 = SHA256.Create();
        byte[] hash = sha256.ComputeHash(bytes);
        StringBuilder builder = new StringBuilder(hash.Length * 2);
        foreach (byte item in hash)
        {
            builder.Append(item.ToString("x2", CultureInfo.InvariantCulture));
        }

        return builder.ToString();
    }

}

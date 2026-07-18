using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace YoloVisionSample;

/// <summary>
/// Pointer-free result of an offline YoloVision configuration preflight.
/// YoloVision 离线配置预检的无指针结果。
/// </summary>
public sealed class YoloVisionPreflightResult
{
    public YoloVisionPreflightResult(
        string state,
        bool hasBlockers,
        bool hasOwnerAction,
        string normalizedCommandSha256,
        string json)
    {
        State = state ?? string.Empty;
        HasBlockers = hasBlockers;
        HasOwnerAction = hasOwnerAction;
        NormalizedCommandSha256 = normalizedCommandSha256 ?? string.Empty;
        Json = json ?? string.Empty;
    }

    public string State { get; }

    public bool HasBlockers { get; }

    public bool HasOwnerAction { get; }

    public string NormalizedCommandSha256 { get; }

    public string Json { get; }
}

/// <summary>
/// Creates an offline, non-executing YoloVision preflight report.
/// 创建不执行 TensorRT 的 YoloVision 离线预检报告。
/// </summary>
public static class YoloVisionPreflightReport
{
    public const string SchemaVersion = "yolovision-preflight.v1";

    private static readonly JsonSerializerOptions SerializerOptions = new JsonSerializerOptions
    {
        WriteIndented = true
    };

    public static YoloVisionPreflightResult Create(
        string[] args,
        YoloModelProfile profile,
        string modelPath,
        string labelsPath,
        string inputPath,
        string inputDataPath,
        string imagePath,
        IReadOnlyList<string> labels,
        YoloMultiOutputMetadata? metadata)
    {
        if (args == null)
        {
            throw new ArgumentNullException(nameof(args));
        }

        if (profile == null)
        {
            throw new ArgumentNullException(nameof(profile));
        }

        labels ??= Array.Empty<string>();
        bool strict = HasSwitch(args, "--strict-preflight");
        List<PreflightCheck> checks = new List<PreflightCheck>();

        AssetSnapshot model = Snapshot(modelPath, required: true);
        AssetSnapshot labelAsset = Snapshot(labelsPath, required: false);
        string inputSourcePath = FirstNonEmpty(imagePath, inputDataPath, inputPath);
        AssetSnapshot input = Snapshot(inputSourcePath, required: false);

        int inputSourceCount = new[] { imagePath, inputDataPath, inputPath }
            .Count(value => !string.IsNullOrWhiteSpace(value));
        checks.Add(new PreflightCheck(
            "input-source-exclusive",
            inputSourceCount <= 1 ? "info" : "blocker",
            inputSourceCount <= 1,
            inputSourceCount <= 1
                ? "At most one of --image, --input-data, and --input is selected."
                : "Choose exactly one input source: --image, --input-data, or --input."));

        AddAssetCheck(
            checks,
            "model-path",
            model,
            strict,
            "Provide an ONNX model path and make sure the file exists before runtime execution.");

        if (!string.IsNullOrWhiteSpace(labelsPath))
        {
            AddAssetCheck(
                checks,
                "labels-path",
                labelAsset,
                strict,
                "The supplied labels path must point to an existing file.");
        }
        else
        {
            checks.Add(new PreflightCheck(
                "labels-path",
                profile.TaskType == YoloTaskType.SemanticSegmentation
                    ? "info"
                    : strict
                    ? "blocker"
                    : "owner-action-required",
                profile.TaskType == YoloTaskType.SemanticSegmentation,
                "No labels path was supplied; owner evidence should provide labels or an explicit class map."));
        }

        if (string.IsNullOrWhiteSpace(inputSourcePath))
        {
            checks.Add(new PreflightCheck(
                "input-source",
                strict ? "blocker" : "owner-action-required",
                false,
                "No external image or tensor was supplied; runtime will use synthetic input unless an owner adds one."));
        }
        else
        {
            AddAssetCheck(
                checks,
                "input-source",
                input,
                strict,
                "The supplied image or tensor path must point to an existing file.");
        }

        bool classCountKnown = profile.Postprocess.ClassCount > 0 || labels.Count > 0;
        checks.Add(new PreflightCheck(
            "class-count-or-labels",
            classCountKnown ? "info" : strict ? "blocker" : "owner-action-required",
            classCountKnown,
            classCountKnown
                ? $"Class count is available ({(profile.Postprocess.ClassCount > 0 ? profile.Postprocess.ClassCount : labels.Count)})."
                : "Provide --class-count or a labels file before claiming model-specific output evidence."));

        bool auxiliaryMetadataRequired = profile.TaskType == YoloTaskType.Segmentation ||
                                         profile.TaskType == YoloTaskType.OrientedBoundingBox ||
                                         profile.TaskType == YoloTaskType.Pose;
        if (auxiliaryMetadataRequired)
        {
            checks.Add(new PreflightCheck(
                "task-output-metadata",
                metadata == null ? (strict ? "blocker" : "owner-action-required") : "info",
                metadata != null,
                metadata == null
                    ? $"Task '{ToTaskAlias(profile.TaskType)}' requires explicit auxiliary output metadata before multi-output decoding."
                    : $"Task '{ToTaskAlias(profile.TaskType)}' has explicit auxiliary output metadata."));
        }
        else
        {
            checks.Add(new PreflightCheck(
                "task-output-metadata",
                "info",
                true,
                $"Task '{ToTaskAlias(profile.TaskType)}' uses its primary output role unless the owner declares additional roles."));
        }

        string outputRoleMap = GetStringArgument(args, "--output-role-map", string.Empty);
        if (!string.IsNullOrWhiteSpace(outputRoleMap))
        {
            checks.Add(new PreflightCheck(
                "output-role-map",
                "info",
                true,
                "Explicit output role mappings are recorded for owner review."));
        }

        string normalizedCommandLine = NormalizeCommandLine(args);
        string normalizedCommandSha256 = ComputeSha256(Encoding.UTF8.GetBytes(normalizedCommandLine));
        bool hasBlockers = checks.Any(item => !item.passed && string.Equals(item.severity, "blocker", StringComparison.Ordinal));
        bool hasOwnerAction = checks.Any(item => !item.passed && !string.Equals(item.severity, "info", StringComparison.Ordinal));
        string state = hasBlockers
            ? "invalid"
            : hasOwnerAction
            ? "owner-action-required"
            : "ready-for-runtime-precheck";

        object payload = new
        {
            schemaVersion = SchemaVersion,
            sample = "YoloVision",
            state,
            strict,
            normalizedCommandLine,
            normalizedCommandSha256,
            profile = new
            {
                family = ToFamilyAlias(profile.Family),
                task = ToTaskAlias(profile.TaskType),
                inputName = profile.InputName,
                outputName = profile.OutputName,
                inputShape = profile.InputShape,
                preprocess = new
                {
                    tensorLayout = profile.Preprocess.TensorLayout,
                    colorOrder = profile.Preprocess.ColorOrder,
                    resize = profile.Preprocess.ResizeMode,
                    scale = profile.Preprocess.Scale,
                    normalize = profile.Preprocess.Normalize,
                    preserveAspectRatio = profile.Preprocess.PreserveAspectRatio
                },
                postprocess = new
                {
                    layout = profile.Postprocess.Layout.ToString(),
                    hasObjectness = profile.Postprocess.HasObjectness,
                    classCount = profile.Postprocess.ClassCount,
                    confidence = profile.Postprocess.ConfidenceThreshold,
                    iouThreshold = profile.Postprocess.IouThreshold,
                    topK = profile.Postprocess.TopK,
                    applyNms = profile.Postprocess.ApplyNms,
                    nmsMode = profile.Postprocess.NmsMode.ToString()
                }
            },
            assets = new
            {
                model = ToJsonAsset(model),
                labels = ToJsonAsset(new AssetSnapshot(labelAsset.Path, labelAsset.Exists, profile.TaskType != YoloTaskType.SemanticSegmentation, labelAsset.Sha256)),
                input = ToJsonAsset(input),
                sourceKind = string.IsNullOrWhiteSpace(inputSourcePath)
                    ? "synthetic-pattern"
                    : !string.IsNullOrWhiteSpace(imagePath)
                    ? "image-not-preprocessed"
                    : !string.IsNullOrWhiteSpace(inputDataPath)
                    ? "external-float-tensor"
                    : "external-byte-tensor"
            },
            output = new
            {
                explicitRoleMap = outputRoleMap,
                metadataDeclared = metadata != null,
                metadata = metadata == null
                    ? null
                    : new
                    {
                        maskCoefficientCount = metadata.MaskCoefficientCount,
                        poseKeypointCount = metadata.PoseKeypointCount,
                        poseKeypointStride = metadata.PoseKeypointStride,
                        obbAngleInDegrees = metadata.ObbAngleInDegrees,
                        auxiliaryChannelStart = metadata.AuxiliaryChannelStart,
                        auxiliaryLayout = metadata.AuxiliaryLayout.ToString()
                    }
            },
            checks,
            execution = new
            {
                tensorRtRuntimeProbed = false,
                onnxParserInvoked = false,
                engineBuildInvoked = false,
                inferenceInvoked = false
            },
            boundary = new
            {
                proofClassification = "precheck",
                isRuntimeProof = false,
                isRealModelRuntimeProof = false,
                isPackageConsumerRuntimeProof = false,
                canPromoteRealModelRuntime = false,
                canPromotePackageConsumerRuntime = false,
                note = "Offline configuration and asset preflight only; no TensorRT, ONNX parser, engine build, plugin load, or inference was executed."
            }
        };

        return new YoloVisionPreflightResult(
            state,
            hasBlockers,
            hasOwnerAction,
            normalizedCommandSha256,
            JsonSerializer.Serialize(payload, SerializerOptions));
    }

    public static void Write(string outputPath, YoloVisionPreflightResult result)
    {
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Preflight report path must not be empty.", nameof(outputPath));
        }

        if (result == null)
        {
            throw new ArgumentNullException(nameof(result));
        }

        string fullPath = Path.GetFullPath(outputPath);
        string? directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllText(fullPath, result.Json, Encoding.UTF8);
    }

    private static object ToJsonAsset(AssetSnapshot asset)
    {
        return new
        {
            path = asset.Path,
            exists = asset.Exists,
            required = asset.Required,
            sha256 = asset.Sha256
        };
    }

    private static void AddAssetCheck(
        List<PreflightCheck> checks,
        string id,
        AssetSnapshot asset,
        bool strict,
        string message)
    {
        if (asset.Exists)
        {
            checks.Add(new PreflightCheck(id, "info", true, $"{id} exists and has SHA256 {asset.Sha256}."));
            return;
        }

        checks.Add(new PreflightCheck(
            id,
            strict ? "blocker" : "owner-action-required",
            false,
            message));
    }

    private static AssetSnapshot Snapshot(string path, bool required)
    {
        string fullPath = string.IsNullOrWhiteSpace(path) ? string.Empty : Path.GetFullPath(path);
        bool exists = !string.IsNullOrWhiteSpace(fullPath) && File.Exists(fullPath);
        return new AssetSnapshot(fullPath, exists, required, exists ? ComputeFileSha256(fullPath) : string.Empty);
    }

    private static string ComputeFileSha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return ComputeSha256(stream);
    }

    private static string ComputeSha256(Stream stream)
    {
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    private static string ComputeSha256(byte[] bytes)
    {
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }

    private static string NormalizeCommandLine(IReadOnlyList<string> args)
    {
        return string.Join(" ", args.Select(QuoteArgument));
    }

    private static string QuoteArgument(string value)
    {
        value ??= string.Empty;
        if (value.Length == 0 || value.Any(char.IsWhiteSpace) || value.Contains('"'))
        {
            return "\"" + value.Replace("\\", "\\\\", StringComparison.Ordinal).Replace("\"", "\\\"", StringComparison.Ordinal) + "\"";
        }

        return value;
    }

    private static string FirstNonEmpty(params string[] values)
    {
        foreach (string value in values)
        {
            if (!string.IsNullOrWhiteSpace(value))
            {
                return value;
            }
        }

        return string.Empty;
    }

    private static string GetStringArgument(IReadOnlyList<string> args, string name, string defaultValue)
    {
        for (int index = 0; index < args.Count - 1; index++)
        {
            if (string.Equals(args[index], name, StringComparison.OrdinalIgnoreCase))
            {
                return args[index + 1];
            }
        }

        return defaultValue;
    }

    private static bool HasSwitch(IReadOnlyList<string> args, string name)
    {
        for (int index = 0; index < args.Count; index++)
        {
            if (string.Equals(args[index], name, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }

    private static string ToTaskAlias(YoloTaskType task)
    {
        return task switch
        {
            YoloTaskType.Detection => "det",
            YoloTaskType.Classification => "cls",
            YoloTaskType.Segmentation => "seg",
            YoloTaskType.OrientedBoundingBox => "obb",
            YoloTaskType.Pose => "pose",
            YoloTaskType.SemanticSegmentation => "sem",
            _ => task.ToString()
        };
    }

    private static string ToFamilyAlias(YoloModelFamily family)
    {
        return family switch
        {
            YoloModelFamily.Custom => "custom",
            YoloModelFamily.YoloV5 => "v5",
            YoloModelFamily.YoloV6 => "v6",
            YoloModelFamily.YoloV7 => "v7",
            YoloModelFamily.YoloV8 => "v8",
            YoloModelFamily.YoloV9 => "v9",
            YoloModelFamily.YoloV10 => "v10",
            YoloModelFamily.YoloV11 => "v11",
            YoloModelFamily.YoloV26 => "v26",
            _ => family.ToString()
        };
    }

    private sealed record PreflightCheck(string id, string severity, bool passed, string message);

    private sealed record AssetSnapshot(string Path, bool Exists, bool Required, string Sha256);
}

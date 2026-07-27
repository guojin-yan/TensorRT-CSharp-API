using System;
using System.Collections.Generic;
using System.Globalization;

namespace YoloVisionSample;

public static class YoloRuntimeOutputRoleResolver
{
    private static readonly (YoloOutputTensorRole Role, string[] Options)[] ExplicitOutputOptions =
    {
        (YoloOutputTensorRole.Detection, new[] { "--detection-output", "--box-output", "--boxes-output" }),
        (YoloOutputTensorRole.Classification, new[] { "--classification-output", "--class-output", "--logits-output" }),
        (YoloOutputTensorRole.SemanticMap, new[] { "--semantic-output", "--semantic-map-output" }),
        (YoloOutputTensorRole.MaskPrototypes, new[] { "--mask-prototypes-output", "--prototype-output", "--prototypes-output" }),
        (YoloOutputTensorRole.MaskCoefficients, new[] { "--mask-coefficients-output", "--coefficients-output" }),
        (YoloOutputTensorRole.ObbAngles, new[] { "--obb-angle-output", "--angle-output", "--angles-output" }),
        (YoloOutputTensorRole.PoseKeypoints, new[] { "--pose-keypoints-output", "--keypoint-output", "--keypoints-output" })
    };

    public static YoloOutputTensorRole ResolveRole(string outputName, YoloTaskType taskType, string[] args, bool isPrimary)
    {
        if (args == null)
        {
            throw new ArgumentNullException(nameof(args));
        }

        Dictionary<string, YoloOutputTensorRole> explicitMap = ParseExplicitRoleMap(args);
        if (!string.IsNullOrWhiteSpace(outputName) && explicitMap.TryGetValue(outputName, out YoloOutputTensorRole mappedRole))
        {
            return mappedRole;
        }

        foreach ((YoloOutputTensorRole role, string[] options) in ExplicitOutputOptions)
        {
            foreach (string option in options)
            {
                string configuredName = GetStringArgument(args, option, string.Empty);
                if (!string.IsNullOrWhiteSpace(configuredName) &&
                    string.Equals(configuredName, outputName, StringComparison.OrdinalIgnoreCase))
                {
                    return role;
                }
            }
        }

        YoloOutputTensorRole? inferredRole = InferRoleFromName(outputName);
        if (inferredRole.HasValue)
        {
            return inferredRole.Value;
        }

        return isPrimary ? GetPrimaryOutputRole(taskType) : YoloOutputTensorRole.Detection;
    }

    public static YoloOutputTensorRole GetPrimaryOutputRole(YoloTaskType taskType)
    {
        return taskType switch
        {
            YoloTaskType.Classification => YoloOutputTensorRole.Classification,
            YoloTaskType.SemanticSegmentation => YoloOutputTensorRole.SemanticMap,
            _ => YoloOutputTensorRole.Detection
        };
    }

    public static YoloMultiOutputMetadata? CreateMetadata(string[] args, YoloTaskType taskType)
    {
        if (args == null)
        {
            throw new ArgumentNullException(nameof(args));
        }

        int? auxiliaryChannelStart = GetOptionalNonNegativeIntArgument(args, "--aux-channel-start");
        YoloOutputLayout auxiliaryLayout = YoloOutputLayoutInference.Parse(GetStringArgument(args, "--aux-layout", "auto"));

        if (taskType == YoloTaskType.Segmentation)
        {
            int maskCoefficientCount = GetPositiveIntArgument(args, "--mask-coefficient-count", 0);
            if (maskCoefficientCount <= 0)
            {
                maskCoefficientCount = GetPositiveIntArgument(args, "--mask-coefficients", 0);
            }

            float maskThreshold = GetFloatArgument(args, "--mask-threshold", YoloSegmentationMask.DefaultThreshold, 0.0f, 1.0f);
            return maskCoefficientCount > 0
                ? YoloMultiOutputMetadata.ForSegmentation(maskCoefficientCount, auxiliaryChannelStart, auxiliaryLayout, maskThreshold)
                : null;
        }

        if (taskType == YoloTaskType.Pose)
        {
            int keypointCount = GetPositiveIntArgument(args, "--keypoint-count", 0);
            if (keypointCount <= 0)
            {
                keypointCount = GetPositiveIntArgument(args, "--pose-keypoint-count", 0);
            }

            if (keypointCount <= 0)
            {
                return null;
            }

            int keypointStride = GetPositiveIntArgument(args, "--keypoint-stride", 3);
            return YoloMultiOutputMetadata.ForPose(keypointCount, keypointStride, auxiliaryChannelStart, auxiliaryLayout);
        }

        if (taskType == YoloTaskType.OrientedBoundingBox && DeclaresAuxiliaryRole(args, YoloOutputTensorRole.ObbAngles))
        {
            bool angleInDegrees = HasSwitch(args, "--angle-degrees") ||
                                  HasSwitch(args, "--obb-angle-degrees") ||
                                  GetBooleanArgument(args, "--angle-in-degrees", defaultValue: false);
            if (HasSwitch(args, "--angle-radians") || HasSwitch(args, "--obb-angle-radians"))
            {
                angleInDegrees = false;
            }

            return YoloMultiOutputMetadata.ForObb(angleInDegrees, auxiliaryChannelStart, auxiliaryLayout);
        }

        return null;
    }

    private static bool DeclaresAuxiliaryRole(string[] args, YoloOutputTensorRole role)
    {
        foreach ((YoloOutputTensorRole candidateRole, string[] options) in ExplicitOutputOptions)
        {
            if (candidateRole != role)
            {
                continue;
            }

            foreach (string option in options)
            {
                if (!string.IsNullOrWhiteSpace(GetStringArgument(args, option, string.Empty)))
                {
                    return true;
                }
            }
        }

        foreach (YoloOutputTensorRole mappedRole in ParseExplicitRoleMap(args).Values)
        {
            if (mappedRole == role)
            {
                return true;
            }
        }

        return false;
    }

    private static Dictionary<string, YoloOutputTensorRole> ParseExplicitRoleMap(string[] args)
    {
        Dictionary<string, YoloOutputTensorRole> map = new Dictionary<string, YoloOutputTensorRole>(StringComparer.OrdinalIgnoreCase);
        string text = GetStringArgument(args, "--output-role-map", string.Empty);
        if (string.IsNullOrWhiteSpace(text))
        {
            return map;
        }

        string[] entries = text.Split(new[] { ',', ';' }, StringSplitOptions.RemoveEmptyEntries);
        foreach (string entry in entries)
        {
            string[] parts = entry.Split(new[] { ':', '=' }, 2, StringSplitOptions.TrimEntries);
            if (parts.Length != 2 || string.IsNullOrWhiteSpace(parts[0]))
            {
                throw new ArgumentException($"Invalid --output-role-map entry '{entry}'. Expected outputName:role.");
            }

            map[parts[0]] = ParseRole(parts[1]);
        }

        return map;
    }

    private static YoloOutputTensorRole ParseRole(string value)
    {
        string normalized = Normalize(value);
        return normalized switch
        {
            "det" or "detect" or "detection" or "box" or "boxes" => YoloOutputTensorRole.Detection,
            "cls" or "class" or "classification" or "logit" or "logits" => YoloOutputTensorRole.Classification,
            "sem" or "semantic" or "semanticmap" => YoloOutputTensorRole.SemanticMap,
            "proto" or "prototype" or "prototypes" or "maskprototype" or "maskprototypes" => YoloOutputTensorRole.MaskPrototypes,
            "coeff" or "coeffs" or "coefficient" or "coefficients" or "maskcoefficient" or "maskcoefficients" => YoloOutputTensorRole.MaskCoefficients,
            "obbangle" or "obbangles" or "angle" or "angles" or "theta" => YoloOutputTensorRole.ObbAngles,
            "pose" or "keypoint" or "keypoints" or "posekeypoint" or "posekeypoints" or "kpt" or "kpts" => YoloOutputTensorRole.PoseKeypoints,
            _ => throw new ArgumentException($"Unsupported YOLO output role '{value}'.")
        };
    }

    private static YoloOutputTensorRole? InferRoleFromName(string outputName)
    {
        string normalized = Normalize(outputName);
        if (normalized.Length == 0)
        {
            return null;
        }

        if (normalized.Contains("proto", StringComparison.Ordinal) || normalized.Contains("maskprototype", StringComparison.Ordinal))
        {
            return YoloOutputTensorRole.MaskPrototypes;
        }

        if (normalized.Contains("keypoint", StringComparison.Ordinal) ||
            normalized.Contains("kpt", StringComparison.Ordinal) ||
            normalized.Contains("pose", StringComparison.Ordinal))
        {
            return YoloOutputTensorRole.PoseKeypoints;
        }

        if (normalized.Contains("angle", StringComparison.Ordinal) ||
            normalized.Contains("theta", StringComparison.Ordinal) ||
            normalized.Contains("obb", StringComparison.Ordinal))
        {
            return YoloOutputTensorRole.ObbAngles;
        }

        if (normalized.Contains("semantic", StringComparison.Ordinal))
        {
            return YoloOutputTensorRole.SemanticMap;
        }

        if (normalized.Contains("logit", StringComparison.Ordinal) ||
            normalized.Contains("prob", StringComparison.Ordinal) ||
            normalized.Contains("class", StringComparison.Ordinal))
        {
            return YoloOutputTensorRole.Classification;
        }

        if (normalized.Contains("box", StringComparison.Ordinal) ||
            normalized.Contains("detect", StringComparison.Ordinal))
        {
            return YoloOutputTensorRole.Detection;
        }

        return null;
    }

    private static string GetStringArgument(string[] args, string name, string defaultValue)
    {
        for (int index = 0; index < args.Length - 1; index++)
        {
            if (string.Equals(args[index], name, StringComparison.OrdinalIgnoreCase))
            {
                return args[index + 1];
            }
        }

        return defaultValue;
    }

    private static int GetPositiveIntArgument(string[] args, string name, int defaultValue)
    {
        string value = GetStringArgument(args, name, defaultValue.ToString(CultureInfo.InvariantCulture));
        return int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed) && parsed > 0 ? parsed : defaultValue;
    }

    private static int? GetOptionalNonNegativeIntArgument(string[] args, string name)
    {
        string value = GetStringArgument(args, name, string.Empty);
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        if (int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed) && parsed >= 0)
        {
            return parsed;
        }

        throw new ArgumentException($"{name} must be a non-negative integer.");
    }

    private static float GetFloatArgument(string[] args, string name, float defaultValue, float minimum, float maximum)
    {
        string value = GetStringArgument(args, name, string.Empty);
        if (string.IsNullOrWhiteSpace(value))
        {
            return defaultValue;
        }

        if (float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed) &&
            float.IsFinite(parsed) &&
            parsed >= minimum &&
            parsed <= maximum)
        {
            return parsed;
        }

        throw new ArgumentException($"{name} must be in [{minimum}, {maximum}].");
    }

    private static bool GetBooleanArgument(string[] args, string name, bool defaultValue)
    {
        string value = GetStringArgument(args, name, string.Empty);
        if (string.IsNullOrWhiteSpace(value))
        {
            return defaultValue;
        }

        if (bool.TryParse(value, out bool parsed))
        {
            return parsed;
        }

        throw new ArgumentException($"{name} must be true or false.");
    }

    private static bool HasSwitch(string[] args, string name)
    {
        for (int index = 0; index < args.Length; index++)
        {
            if (string.Equals(args[index], name, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }

    private static string Normalize(string value)
    {
        return (value ?? string.Empty)
            .Trim()
            .Replace("-", string.Empty, StringComparison.Ordinal)
            .Replace("_", string.Empty, StringComparison.Ordinal)
            .Replace("/", string.Empty, StringComparison.Ordinal)
            .ToLowerInvariant();
    }
}

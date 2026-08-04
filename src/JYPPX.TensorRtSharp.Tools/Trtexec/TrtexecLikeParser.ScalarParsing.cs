using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp.Tools;

public static partial class TrtexecLikeParser
{
    private static int ParsePositiveInt(string value, string argumentName)
    {
        if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed) || parsed <= 0)
        {
            throw new ArgumentException($"{argumentName} must be a positive integer.");
        }

        return parsed;
    }

    private static int ParseNonNegativeInt(string value, string argumentName)
    {
        if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed) || parsed < 0)
        {
            throw new ArgumentException($"{argumentName} must be a non-negative integer.");
        }

        return parsed;
    }

    private static int? ParseOptionalNonNegativeInt(string value, string argumentName)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        return ParseNonNegativeInt(value, argumentName);
    }

    private static int? ParseOptionalPositiveInt(string value, string argumentName)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        return ParsePositiveInt(value, argumentName);
    }

    private static float? ParseOptionalRangeFloat(string value, string argumentName, float minInclusive, float maxInclusive)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        if (!float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed) ||
            parsed < minInclusive ||
            parsed > maxInclusive)
        {
            throw new ArgumentException($"{argumentName} must be a number in the range [{minInclusive}, {maxInclusive}].");
        }

        return parsed;
    }

    private static TrtexecLikeReferenceNaNPolicy ParseReferenceNaNPolicy(string value)
    {
        return (value ?? string.Empty).Trim().ToLowerInvariant() switch
        {
            "reject" => TrtexecLikeReferenceNaNPolicy.Reject,
            "equal" => TrtexecLikeReferenceNaNPolicy.Equal,
            _ => throw new ArgumentException("--referenceNaNPolicy must be reject or equal.")
        };
    }

    private static TrtexecLikeReferenceInfinityPolicy ParseReferenceInfinityPolicy(string value)
    {
        return (value ?? string.Empty).Trim().ToLowerInvariant() switch
        {
            "exact" => TrtexecLikeReferenceInfinityPolicy.Exact,
            "reject" => TrtexecLikeReferenceInfinityPolicy.Reject,
            _ => throw new ArgumentException("--referenceInfinityPolicy must be exact or reject.")
        };
    }

    private static int ParseRangeInt(string value, string argumentName, int minInclusive, int maxInclusive)
    {
        if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed) ||
            parsed < minInclusive ||
            parsed > maxInclusive)
        {
            throw new ArgumentException($"{argumentName} must be an integer in the range [{minInclusive}, {maxInclusive}].");
        }

        return parsed;
    }
}

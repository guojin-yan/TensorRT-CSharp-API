using System;
using System.Collections.Generic;
using System.Linq;

namespace JYPPX.TensorRtSharp.Tools;

public sealed class EngineBuildProfile
{
    public EngineBuildProfile(IReadOnlyList<EngineBuildShape> minShapes, IReadOnlyList<EngineBuildShape> optShapes, IReadOnlyList<EngineBuildShape> maxShapes)
    {
        MinShapes = minShapes ?? Array.Empty<EngineBuildShape>();
        OptShapes = optShapes ?? Array.Empty<EngineBuildShape>();
        MaxShapes = maxShapes ?? Array.Empty<EngineBuildShape>();
        ValidateNames(MinShapes, OptShapes, MaxShapes);
    }

    public IReadOnlyList<EngineBuildShape> MinShapes { get; }

    public IReadOnlyList<EngineBuildShape> OptShapes { get; }

    public IReadOnlyList<EngineBuildShape> MaxShapes { get; }

    public bool IsEmpty => MinShapes.Count == 0 && OptShapes.Count == 0 && MaxShapes.Count == 0;

    public bool TryGetShapeTriple(string tensorName, out EngineBuildShape min, out EngineBuildShape opt, out EngineBuildShape max)
    {
        min = Find(MinShapes, tensorName);
        opt = Find(OptShapes, tensorName);
        max = Find(MaxShapes, tensorName);
        return min != null && opt != null && max != null;
    }

    public static EngineBuildProfile Parse(string minShapes, string optShapes, string maxShapes)
    {
        return new EngineBuildProfile(
            ParseShapeList(minShapes, "--minShapes"),
            ParseShapeList(optShapes, "--optShapes"),
            ParseShapeList(maxShapes, "--maxShapes"));
    }

    private static IReadOnlyList<EngineBuildShape> ParseShapeList(string value, string argumentName)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return Array.Empty<EngineBuildShape>();
        }

        return value.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries)
            .Select(item => EngineBuildShape.Parse(item.Trim(), argumentName))
            .ToArray();
    }

    private static EngineBuildShape Find(IReadOnlyList<EngineBuildShape> shapes, string tensorName)
    {
        foreach (EngineBuildShape shape in shapes)
        {
            if (string.Equals(shape.TensorName, tensorName, StringComparison.Ordinal))
            {
                return shape;
            }
        }

        return null!;
    }

    private static void ValidateNames(params IReadOnlyList<EngineBuildShape>[] lists)
    {
        string[] names = lists
            .SelectMany(static list => list)
            .Select(static shape => shape.TensorName)
            .Distinct(StringComparer.Ordinal)
            .ToArray();

        foreach (string name in names)
        {
            bool presentInAllNonEmptyLists = lists.All(list => list.Count == 0 || list.Any(shape => string.Equals(shape.TensorName, name, StringComparison.Ordinal)));
            if (!presentInAllNonEmptyLists)
            {
                throw new ArgumentException($"Shape profile for tensor '{name}' must appear in min/opt/max shape sets when those sets are provided.");
            }
        }
    }
}

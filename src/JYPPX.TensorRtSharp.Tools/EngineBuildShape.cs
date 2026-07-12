using System;
using System.Globalization;
using System.Linq;

namespace JYPPX.TensorRtSharp.Tools;

public sealed class EngineBuildShape
{
    public EngineBuildShape(string tensorName, int[] dimensions)
    {
        if (string.IsNullOrWhiteSpace(tensorName))
        {
            throw new ArgumentException("Tensor name must not be empty.", nameof(tensorName));
        }

        if (dimensions == null || dimensions.Length == 0)
        {
            throw new ArgumentException("Shape must contain at least one dimension.", nameof(dimensions));
        }

        if (dimensions.Any(static dimension => dimension <= 0))
        {
            throw new ArgumentException("Shape dimensions must be positive.", nameof(dimensions));
        }

        TensorName = tensorName;
        Dimensions = dimensions;
    }

    public string TensorName { get; }

    public int[] Dimensions { get; }

    public static EngineBuildShape Parse(string text, string argumentName)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            throw new ArgumentException($"{argumentName} must not be empty.");
        }

        string[] parts = text.Split(new[] { ':' }, 2);
        if (parts.Length != 2 || string.IsNullOrWhiteSpace(parts[0]) || string.IsNullOrWhiteSpace(parts[1]))
        {
            throw new ArgumentException($"{argumentName} must use tensor:dimxdim syntax.");
        }

        string[] tokens = parts[1].Split(new[] { 'x', 'X' }, StringSplitOptions.RemoveEmptyEntries);
        int[] dimensions = new int[tokens.Length];
        for (int index = 0; index < tokens.Length; index++)
        {
            if (!int.TryParse(tokens[index].Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int dimension) || dimension <= 0)
            {
                throw new ArgumentException($"{argumentName} must contain positive integer dimensions.");
            }

            dimensions[index] = dimension;
        }

        return new EngineBuildShape(parts[0].Trim(), dimensions);
    }

    public override string ToString()
    {
        return $"{TensorName}:{string.Join("x", Dimensions)}";
    }
}

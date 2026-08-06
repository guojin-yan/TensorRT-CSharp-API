using System;
using System.Linq;

namespace JYPPX.SampleSupport;

internal static class SampleCommandLine
{
    public static int GetIntArgument(string[] args, string name, int defaultValue)
    {
        string value = GetStringArgument(args, name, defaultValue.ToString());
        return int.TryParse(value, out int parsed) ? parsed : defaultValue;
    }

    public static int GetPositiveIntArgument(string[] args, string name, int defaultValue)
    {
        string value = GetStringArgument(args, name, defaultValue.ToString());
        return int.TryParse(value, out int parsed) && parsed > 0 ? parsed : defaultValue;
    }

    public static string GetStringArgument(string[] args, string name, string defaultValue)
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

    public static bool HasSwitch(string[] args, string name)
    {
        return args.Any(argument => string.Equals(argument, name, StringComparison.OrdinalIgnoreCase));
    }
}

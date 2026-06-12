using System.IO;

namespace JYPPX.ProjectQuality.Tests;

internal static class RepositoryPaths
{
    public static string Root
    {
        get
        {
            string directory = Directory.GetCurrentDirectory();
            while (!File.Exists(Path.Combine(directory, "TensorRtSharp.sln")))
            {
                string? parent = Directory.GetParent(directory)?.FullName;
                if (parent == null)
                {
                    throw new DirectoryNotFoundException("Unable to locate TensorRtSharp.sln from the test working directory.");
                }

                directory = parent;
            }

            return directory;
        }
    }
}

namespace JYPPX.ProjectQuality.Tests;

internal static class PowerShellHost
{
    public static string ResolveExecutable()
    {
        string? configuredPath = Environment.GetEnvironmentVariable("JYPPX_POWERSHELL_EXECUTABLE");
        if (!string.IsNullOrWhiteSpace(configuredPath) && File.Exists(configuredPath))
        {
            return configuredPath;
        }

        if (OperatingSystem.IsWindows())
        {
            string programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
            string installedPowerShell = Path.Combine(programFiles, "PowerShell", "7", "pwsh.exe");
            if (File.Exists(installedPowerShell))
            {
                return installedPowerShell;
            }
        }

        string[] commandNames = OperatingSystem.IsWindows()
            ? ["pwsh.exe", "pwsh", "powershell.exe", "powershell"]
            : ["pwsh", "pwsh-preview"];

        string pathValue = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
        foreach (string directory in pathValue.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            foreach (string commandName in commandNames)
            {
                string candidate = Path.Combine(directory.Trim('"'), commandName);
                if (File.Exists(candidate))
                {
                    return candidate;
                }
            }
        }

        if (OperatingSystem.IsWindows())
        {
            string systemRoot = Environment.GetFolderPath(Environment.SpecialFolder.Windows);
            string windowsPowerShell = Path.Combine(
                systemRoot,
                "System32",
                "WindowsPowerShell",
                "v1.0",
                "powershell.exe");
            if (File.Exists(windowsPowerShell))
            {
                return windowsPowerShell;
            }
        }

        throw new InvalidOperationException(
            "No PowerShell executable was found. Install PowerShell 7 (pwsh), install Windows PowerShell on Windows, or set JYPPX_POWERSHELL_EXECUTABLE.");
    }
}

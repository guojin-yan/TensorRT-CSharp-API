using System;
using System.Linq;
using System.Windows.Forms;
using TensorRtExecApp.Console;
using TensorRtExecApp.WinForms;

namespace TensorRtExecApp;

internal static class Program
{
    [STAThread]
    public static int Main(string[] args)
    {
        if (args.Any(static arg => string.Equals(arg, "--ui", StringComparison.OrdinalIgnoreCase)) ||
            args.Length == 0)
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm());
            return 0;
        }

        return TensorRtExecCommand.Run(args);
    }
}

using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Threading.Tasks;
using JYPPX.CudaSharp;

internal static class Program
{
    private const int AllocationSize = 64;
    private const byte ExporterValue = 0x2A;
    private const byte ImporterValue = 0x6D;

    private static int Main(string[] args)
    {
        try
        {
            if (HasSwitch(args, "--child"))
            {
                RunImporter(args);
            }
            else
            {
                RunExporter();
            }

            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"CudaIpcImportSmokeRunner Failed=True Type={exception.GetType().Name} Message={exception.Message}");
            return 1;
        }
    }

    private static void RunExporter()
    {
        if (CudaDevice.Count <= 0)
        {
            throw new InvalidOperationException("No CUDA device is available for the IPC import smoke.");
        }

        int device = CudaDevice.Current;
        bool ipcSupported = CudaDevice.GetBooleanAttribute(device, CudaDeviceAttribute.IpcEventSupport);
        if (!ipcSupported)
        {
            throw new InvalidOperationException("The selected CUDA device does not report IPC event support.");
        }

        using CudaMemory sourceMemory = new CudaMemory(AllocationSize);
        using CudaStream sourceStream = new CudaStream();
        using CudaEvent sourceEvent = new CudaEvent(
            CudaEventCreationFlags.Interprocess | CudaEventCreationFlags.DisableTiming);
        CudaIpcMemoryExportDescriptor memoryDescriptor = sourceMemory.ExportIpcDescriptor();
        CudaIpcExportToken eventToken = sourceEvent.ExportIpcToken();

        sourceMemory.FillAsync(ExporterValue, sourceStream);
        sourceEvent.Record(sourceStream);
        sourceStream.Synchronize();
        byte[] exporterInitialized = new byte[AllocationSize];
        sourceMemory.CopyTo(exporterInitialized);
        ValidateBytes(exporterInitialized, ExporterValue, "exporter initialization");

        ChildResult child = RunChild(device, eventToken, memoryDescriptor);
        if (child.ExitCode != 0)
        {
            throw new InvalidOperationException(
                $"IPC importer process failed with exit code {child.ExitCode}. stdout={child.StandardOutput} stderr={child.StandardError}");
        }

        sourceStream.Synchronize();
        byte[] result = new byte[AllocationSize];
        sourceMemory.CopyTo(result);
        ValidateBytes(result, ImporterValue, "exporter readback");

        Console.WriteLine(
            $"CudaIpcImportSmokeRunner Passed=True CrossProcess=True Device={device} " +
            $"EventImported=True MemoryImported=True ImportedClose=True SourceOwnerAlive=True " +
            $"IpcEventSupport={ipcSupported} BytesVerified={result.Length} " +
            $"ChildMarker={child.StandardOutput.Contains("ChildPassed=True", StringComparison.Ordinal)}");
    }

    private static void RunImporter(string[] args)
    {
        int device = int.Parse(GetRequiredArgument(args, "--device"));
        int size = int.Parse(GetRequiredArgument(args, "--size"));
        byte[] eventBytes = Convert.FromBase64String(
            Console.ReadLine() ?? throw new InvalidOperationException("CUDA IPC event token was not received."));
        byte[] memoryBytes = Convert.FromBase64String(
            Console.ReadLine() ?? throw new InvalidOperationException("CUDA IPC memory token was not received."));
        CudaDevice.SetCurrent(device);

        CudaIpcExportToken eventToken = CudaIpcExportToken.FromBytes(
            CudaIpcExportTokenKind.Event,
            eventBytes);
        CudaIpcMemoryExportDescriptor memoryDescriptor =
            CudaIpcMemoryExportDescriptor.FromBytes(memoryBytes, size);

        using CudaEvent importedEvent = CudaEvent.ImportIpcToken(eventToken);
        using CudaMemory importedMemory = CudaMemory.ImportIpcDescriptor(memoryDescriptor);
        if (!importedEvent.IsIpcImported || !importedMemory.IsIpcImported)
        {
            throw new InvalidOperationException("Imported CUDA IPC wrappers did not retain their owner classification.");
        }

        importedEvent.Synchronize();
        byte[] observed = new byte[size];
        importedMemory.CopyTo(observed);
        ValidateBytes(observed, ExporterValue, "importer readback");
        importedMemory.Fill(ImporterValue);

        try
        {
            using CudaStream stream = new CudaStream();
            importedMemory.FreeAsync(stream);
            throw new InvalidOperationException("Imported CUDA IPC memory unexpectedly allowed asynchronous free.");
        }
        catch (InvalidOperationException exception) when (exception.Message.Contains("cudaIpcCloseMemHandle", StringComparison.Ordinal))
        {
        }

        Console.WriteLine($"ChildPassed=True Device={device} BytesVerified={observed.Length} TokenContentsPrinted=False");
    }

    private static ChildResult RunChild(
        int device,
        CudaIpcExportToken eventToken,
        CudaIpcMemoryExportDescriptor memoryDescriptor)
    {
        string processPath = Environment.ProcessPath ?? throw new InvalidOperationException("Current process path is unavailable.");
        ProcessStartInfo startInfo = new()
        {
            FileName = processPath,
            UseShellExecute = false,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };

        if (string.Equals(Path.GetFileNameWithoutExtension(processPath), "dotnet", StringComparison.OrdinalIgnoreCase))
        {
            startInfo.ArgumentList.Add(Assembly.GetExecutingAssembly().Location);
        }

        startInfo.ArgumentList.Add("--child");
        AddArgument(startInfo, "--device", device.ToString());
        AddArgument(startInfo, "--size", memoryDescriptor.SizeInBytes.ToString());

        using Process process = Process.Start(startInfo) ?? throw new InvalidOperationException("Failed to start the CUDA IPC importer process.");
        process.StandardInput.WriteLine(Convert.ToBase64String(eventToken.ToArray()));
        process.StandardInput.WriteLine(Convert.ToBase64String(memoryDescriptor.Token.ToArray()));
        process.StandardInput.Close();
        Task<string> outputTask = process.StandardOutput.ReadToEndAsync();
        Task<string> errorTask = process.StandardError.ReadToEndAsync();
        if (!process.WaitForExit(120_000))
        {
            process.Kill(entireProcessTree: true);
            throw new TimeoutException("CUDA IPC importer process timed out.");
        }

        return new ChildResult(
            process.ExitCode,
            outputTask.GetAwaiter().GetResult().Trim(),
            errorTask.GetAwaiter().GetResult().Trim());
    }

    private static void AddArgument(ProcessStartInfo startInfo, string name, string value)
    {
        startInfo.ArgumentList.Add(name);
        startInfo.ArgumentList.Add(value);
    }

    private static bool HasSwitch(string[] args, string name) =>
        Array.Exists(args, value => string.Equals(value, name, StringComparison.OrdinalIgnoreCase));

    private static string GetRequiredArgument(string[] args, string name)
    {
        for (int index = 0; index + 1 < args.Length; index++)
        {
            if (string.Equals(args[index], name, StringComparison.OrdinalIgnoreCase))
            {
                return args[index + 1];
            }
        }

        throw new ArgumentException($"Required argument was not provided: {name}", nameof(args));
    }

    private static void ValidateBytes(byte[] values, byte expected, string stage)
    {
        for (int index = 0; index < values.Length; index++)
        {
            if (values[index] != expected)
            {
                throw new InvalidOperationException(
                    $"CUDA IPC {stage} mismatch at byte {index}: expected {expected}, actual {values[index]}.");
            }
        }
    }

    private readonly struct ChildResult
    {
        public ChildResult(int exitCode, string standardOutput, string standardError)
        {
            ExitCode = exitCode;
            StandardOutput = standardOutput;
            StandardError = standardError;
        }

        public int ExitCode { get; }
        public string StandardOutput { get; }
        public string StandardError { get; }
    }
}

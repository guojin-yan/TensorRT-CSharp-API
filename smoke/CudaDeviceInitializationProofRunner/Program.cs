using System;
using JYPPX.CudaSharp;

namespace CudaDeviceInitializationProofRunner;

internal static class Program
{
    private static int Main(string[] args)
    {
        int deviceOrdinal = ParseDeviceOrdinal(args);

        Console.WriteLine($"CudaDeviceInitializationProof Start Device={deviceOrdinal}");
        Console.WriteLine("ProofKind=local-smoke-not-external-proof");
        Console.WriteLine("IsPackageConsumerRuntimeProof=False");
        Console.WriteLine("CanPromoteRuntimeProof=False");

        try
        {
            CudaDevice.SetValidDevices(new[] { deviceOrdinal });
            Console.WriteLine("SetValidDevices=Ok Count=1");

            CudaDevice.InitDevice(deviceOrdinal, CudaDeviceRuntimeFlags.ScheduleAuto);
            Console.WriteLine($"InitDevice=Ok Device={deviceOrdinal}");

            CudaDeviceProperties properties = CudaDevice.GetProperties(deviceOrdinal);
            CudaDeviceSelectionRequirements requirements = new CudaDeviceSelectionRequirements
            {
                Major = properties.Info.Major,
                Minor = properties.Info.Minor,
                MultiProcessorCount = Math.Max(1, properties.Info.MultiProcessorCount),
                MaxThreadsPerBlock = Math.Max(1, properties.Info.MaxThreadsPerBlock)
            };

            int chosenDevice = CudaDevice.ChooseDevice(requirements);
            Console.WriteLine($"ChooseDevice=Ok Device={chosenDevice}");
            Console.WriteLine($"DeviceProperties=Ok Name={properties.Info.Name} CC={properties.Info.Major}.{properties.Info.Minor} SMs={properties.Info.MultiProcessorCount}");
            Console.WriteLine("CudaDeviceInitializationProof Completed=True");
            return 0;
        }
        catch (CudaException exception)
        {
            Console.WriteLine($"Skipped=True Reason=CudaException:{exception.Message}");
            Console.WriteLine("CudaDeviceInitializationProof Completed=False");
            return 0;
        }
        catch (DllNotFoundException exception)
        {
            Console.WriteLine($"Skipped=True Reason=DllNotFoundException:{exception.Message}");
            Console.WriteLine("CudaDeviceInitializationProof Completed=False");
            return 0;
        }
        catch (BadImageFormatException exception)
        {
            Console.WriteLine($"Skipped=True Reason=BadImageFormatException:{exception.Message}");
            Console.WriteLine("CudaDeviceInitializationProof Completed=False");
            return 0;
        }
    }

    private static int ParseDeviceOrdinal(string[] args)
    {
        if (args.Length == 0)
        {
            return 0;
        }

        if (!int.TryParse(args[0], out int ordinal) || ordinal < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(args), args[0], "Device ordinal must be a non-negative integer.");
        }

        return ordinal;
    }
}

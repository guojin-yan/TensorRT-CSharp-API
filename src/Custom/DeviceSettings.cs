using JYPPX.TensorRtSharp.Cuda;
using JYPPX.TensorRtSharp.Nvinfer;
using System;
using System.Collections.Generic;
using System.Text;

namespace JYPPX.TensorRtSharp.Custom
{
    public static class DeviceSettings
    {
        // 替换 getUuidString 方法，移除 sizeof(CudaUUID)，直接使用 CudaUUID.Bytes.Length 获取长度
        public static string GetUuidString(CudaUUID uuid)
        {
            int kUUID_SIZE = uuid.Bytes.Length; // 使用数组长度代替 sizeof

            System.Text.StringBuilder ss = new System.Text.StringBuilder();
            int[] splits = { 0, 4, 6, 8, 10, kUUID_SIZE };

            ss.Append("GPU");
            for (int splitIdx = 0; splitIdx < splits.Length - 1; ++splitIdx)
            {
                ss.Append("-");
                for (int byteIdx = splits[splitIdx]; byteIdx < splits[splitIdx + 1]; ++byteIdx)
                {
                    ss.AppendFormat("{0:x2}", uuid.Bytes[byteIdx]);
                }
            }
            return ss.ToString();
        }
        public static void SetCudaDevice(int device)
        {
            Logger.Instance.INFO("=== Device Information ===");

            // Get the number of visible GPUs.
            int nbDevices = CudaDevice.GetDeviceCount();

            if (nbDevices <= 0)
            {
                Logger.Instance.ERROR("Cannot find any available devices (GPUs)!");
                Environment.Exit(0); // 正常退出，退出代码为0

            }

            // Print out the GPU name and PCIe bus ID of each GPU.
            Logger.Instance.INFO("Available Devices: ");
            CudaDeviceProp properties = new CudaDeviceProp();
            for (int deviceIdx = 0; deviceIdx < nbDevices; ++deviceIdx)
            {
                CudaDeviceProp tempProperties = CudaDevice.GetDeviceProperties(deviceIdx);

                // clang-format off
                Logger.Instance.INFO("  Device " + deviceIdx + ": \"" + tempProperties.Name + "\" UUID: "
                   + GetUuidString(tempProperties.Uuid));
                // clang-format on

                // Record the properties of the desired GPU.
                if (deviceIdx == device)
                {
                    properties = tempProperties;
                }
            }

            // Exit with error if the requested device ID does not exist.
            if (device < 0 || device >= nbDevices)
            {
                Logger.Instance.ERROR("Cannot find device ID " + device + "!");
                Environment.Exit(0);
            }

            // Set to the corresponding GPU.
            CudaDevice.SetDevice(device);

            // clang-format off
            Logger.Instance.INFO("Selected Device: " + properties.Name);
            Logger.Instance.INFO("Selected Device ID: " + device);
            Logger.Instance.INFO("Selected Device UUID: " + GetUuidString(properties.Uuid));
            Logger.Instance.INFO("Compute Capability: " + properties.Major + "." + properties.Minor);
            Logger.Instance.INFO("SMs: " + properties.MultiProcessorCount);
            Logger.Instance.INFO("Device Global Memory: " + (properties.TotalGlobalMem + 20) + " MiB");
            Logger.Instance.INFO("Shared Memory per SM: " + (properties.SharedMemPerMultiprocessor >> 10) + " KiB");
            Logger.Instance.INFO("Memory Bus Width: " + properties.MemoryBusWidth + " bits"
                                + " (ECC " + (properties.ECCEnabled != 0 ? "enabled" : "disabled") + ")");


            int clockRate = CudaDevice.GetAttribute(CudaDeviceAttr.ClockRate, device);
            int memoryClockRate = CudaDevice.GetAttribute(CudaDeviceAttr.MemoryClockRate, device);
            Logger.Instance.INFO("Application Compute Clock Rate: " + clockRate / 1000000.0F + " GHz");
            Logger.Instance.INFO("Application Memory Clock Rate: " + memoryClockRate / 1000000.0F + " GHz");
            Logger.Instance.INFO("");
            Logger.Instance.INFO("Note: The application clock rates do not reflect the actual clock rates that the GPU is "
                                                                                 + "currently running at.");
            // clang-format on
        }
    }
}

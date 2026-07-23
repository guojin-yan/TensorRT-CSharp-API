using System;
using System.IO;
using System.Security.Cryptography;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

internal static class Program
{
    private static void Main(string[] args)
    {
        string binaryProtoPath = JYPPX.SampleSupport.SampleCommandLine.GetStringArgument(
            args,
            "--binary-proto",
            string.Empty);
        if (string.IsNullOrWhiteSpace(binaryProtoPath))
        {
            throw new ArgumentException("--binary-proto must point to a Caffe binaryproto file.");
        }

        binaryProtoPath = Path.GetFullPath(binaryProtoPath);
        TensorRtDependencyProbeReport dependencies = TensorRtEnvironmentProbe.ProbeNativeDependencies(TensorRtApiLine.TensorRt8);
        Console.WriteLine($"LegacyParserDependencyProbe BridgeInitialized={dependencies.BridgeInitialized} Candidates={dependencies.SearchPathCandidateCount} Diagnostics={dependencies.Diagnostics.Count}");

        try
        {
            TensorRtLegacyUffRequiredVersionSnapshot uff =
                TensorRtLegacyParserDiagnostics.GetUffRequiredVersion();
            TensorRtCaffeBinaryProtoSnapshot binaryProto =
                TensorRtLegacyParserDiagnostics.ReadCaffeBinaryProto(binaryProtoPath);

            byte[] firstCopy = binaryProto.Data;
            byte[] secondCopy = binaryProto.Data;
            bool independentCopies = !ReferenceEquals(firstCopy, secondCopy);
            if (firstCopy.Length > 0)
            {
                byte original = secondCopy[0];
                firstCopy[0] ^= 0xFF;
                independentCopies = independentCopies && secondCopy[0] == original && binaryProto.Data[0] == original;
            }

            bool nonTrt8Guard = false;
            try
            {
                TensorRtLegacyParserDiagnostics.GetUffRequiredVersion(TensorRtApiLine.TensorRt10);
            }
            catch (BridgeProbeException exception) when (exception.StatusCode == BridgeStatusCode.NotSupported)
            {
                nonTrt8Guard = true;
            }

            if (uff.Major < 0 || uff.Minor < 0 || uff.Patch < 0)
            {
                throw new InvalidOperationException("UFF required version contains a negative component.");
            }
            if (binaryProto.DataLength <= 0 || binaryProto.Dimensions.Rank != 4)
            {
                throw new InvalidOperationException("Caffe binaryproto did not produce a non-empty rank-4 copied snapshot.");
            }
            if (!independentCopies || !nonTrt8Guard || !uff.PointerFreeCopiedMetadata || !binaryProto.PointerFreeCopiedData)
            {
                throw new InvalidOperationException("Legacy parser copied-metadata ownership checks failed.");
            }

            Console.WriteLine($"UffRequiredVersion={uff} PointerFree={uff.PointerFreeCopiedMetadata} RetainsParser={uff.RetainsNativeParser} Shutdown={uff.CallsProcessGlobalProtobufShutdown}");
            Console.WriteLine($"CaffeBinaryProto={binaryProto} SHA256={ComputeSha256(secondCopy)} IndependentCopies={independentCopies} RetainsNative={binaryProto.RetainsNativeObject} PointerFree={binaryProto.PointerFreeCopiedData} Shutdown={binaryProto.CallsProcessGlobalProtobufShutdown}");
            Console.WriteLine($"NonTrt8Guard={nonTrt8Guard} Completed=True");
        }
        catch (Exception exception) when (IsSkippableDependencyException(exception))
        {
            Console.WriteLine($"Skipped=True Reason={exception.GetType().Name}:{exception.Message}");
        }
    }

    private static bool IsSkippableDependencyException(Exception exception)
    {
        if (exception is DllNotFoundException || exception is BadImageFormatException)
        {
            return true;
        }

        return exception is BridgeProbeException bridgeProbe &&
            (bridgeProbe.StatusCode == BridgeStatusCode.DependencyMissing ||
             bridgeProbe.StatusCode == BridgeStatusCode.NotSupported);
    }

    private static string ComputeSha256(byte[] data)
    {
        using SHA256 sha256 = SHA256.Create();
        return BitConverter.ToString(sha256.ComputeHash(data)).Replace("-", string.Empty);
    }
}

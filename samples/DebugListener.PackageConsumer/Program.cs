using System;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp;
using JYPPX.TensorRtSharp.Shared.Interop;

internal static class Program
{
    public static int Main(string[] args)
    {
        try
        {
            TensorRtApiLine line = ParseTensorRtLine(GetArgument(args, "--tensor-rt-line", "10"));
            string runtimePackageKey = GetArgument(args, "--runtime-package-key", "unspecified");
            if (!HasSwitch(args, "--debug-listener-runtime-smoke-only"))
            {
                throw new ArgumentException("The --debug-listener-runtime-smoke-only switch is required.");
            }

            TensorRtEnvironmentSnapshot environment = TensorRtEnvironmentProbe.GetCurrent();
            string expectedTensorRtMajor = ((int)line).ToString() + ".";
            if (!environment.RuntimeInfo.TensorRtAvailable ||
                !environment.RuntimeInfo.CudaToolkitAvailable ||
                !environment.BuildInfo.TensorRtVersion.StartsWith(expectedTensorRtMajor, StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    "The loaded bridge does not match the requested TensorRT line. TRT=" +
                    environment.BuildInfo.TensorRtVersion + " CUDA=" + environment.BuildInfo.CudaToolkitVersion);
            }

            Console.WriteLine("PackageReferenceOnly=True");
            Console.WriteLine("ProjectReference=False");
            Console.WriteLine("SourceTreeBinary=False");
            Console.WriteLine(
                "RuntimeEnvironment" +
                " TRT=" + environment.BuildInfo.TensorRtVersion +
                " CUDA=" + environment.BuildInfo.CudaToolkitVersion +
                " TensorRtAvailable=" + environment.RuntimeInfo.TensorRtAvailable +
                " CudaAvailable=" + environment.RuntimeInfo.CudaToolkitAvailable);

            RunRealDebugListenerRuntimeSmoke(line, runtimePackageKey);
            Console.WriteLine("DebugListenerPackageConsumer Passed=True Mode=DebugListenerRuntimeSmokeOnly");
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine("DebugListenerPackageConsumer Passed=False Error=" + exception);
            return 1;
        }
    }

    private static void RunRealDebugListenerRuntimeSmoke(TensorRtApiLine line, string runtimePackageKey)
    {
        if (line != TensorRtApiLine.TensorRt10 && line != TensorRtApiLine.TensorRt11)
        {
            throw new NotSupportedException("Debug listeners require TensorRT 10 or TensorRT 11.");
        }

        using TensorRtLogger logger = new(line);
        using TensorRtRuntime runtime = new(logger);
        using TensorRtBuilder builder = new(logger);
        using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
        config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 32UL * 1024UL * 1024UL);
        config.SetEngineCapability(TensorRtEngineCapability.Standard);
        config.SetHardwareCompatibilityLevel(TensorRtHardwareCompatibilityLevel.None);

        using TensorRtNetworkDefinition network =
            builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
        using TensorRtTensor input = network.AddInput(
            "debug_input",
            TensorRtDataType.Float,
            new TensorRtDims(new[] { 1, 4 }));
        using TensorRtLayer identity = network.AddIdentity(input);
        identity.Name = "debug_identity";
        using TensorRtTensor output = identity.GetOutput(0);
        output.Name = "debug_output";
        network.MarkOutput(output);
        if (!network.MarkDebugTensor(output) || !network.IsDebugTensor(output))
        {
            throw new InvalidOperationException("TensorRT did not retain the build-time debug tensor mark.");
        }

        using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
        using TensorRtEngine engine = runtime.Deserialize(hostMemory);
        using CudaStream stream = new();
        using CudaMemory inputBuffer = new(4 * sizeof(float));
        using CudaMemory outputBuffer = new(4 * sizeof(float));
        inputBuffer.Fill(0, 4 * sizeof(float));
        outputBuffer.Fill(0, 4 * sizeof(float));

        TensorRtDebugTensorMetadataSnapshot copiedMetadata = default;
        using TensorRtExecutionContext positiveContext = engine.CreateExecutionContext();
        using TensorRtDebugListenerCallbackOwner positiveOwner = new(
            line,
            metadata =>
            {
                copiedMetadata = metadata;
                return true;
            });
        positiveContext.SetTensorAddress("debug_input", inputBuffer);
        positiveContext.SetTensorAddress("debug_output", outputBuffer);
        positiveContext.SetDebugListener(positiveOwner);
        positiveContext.SetTensorDebugState("debug_output", true);
        positiveContext.EnqueueAsync(stream);
        stream.Synchronize();

        TensorRtDebugListenerRuntimeSnapshot positiveAttached = positiveOwner.GetRuntimeSnapshot();
        bool positiveCleared = positiveContext.ClearDebugListener();
        TensorRtDebugListenerRuntimeSnapshot positiveDetached = positiveOwner.GetRuntimeSnapshot();
        bool positivePassed =
            positiveAttached.IsAttached &&
            positiveAttached.IsRealCallbackRuntimeProof &&
            positiveAttached.InvocationCount > 0 &&
            positiveAttached.FailureCount == 0 &&
            positiveAttached.InFlightCallbackCount == 0 &&
            string.Equals(positiveAttached.TensorName, "debug_output", StringComparison.Ordinal) &&
            copiedMetadata.MetadataCopied &&
            string.Equals(copiedMetadata.TensorName, "debug_output", StringComparison.Ordinal) &&
            positiveCleared &&
            !positiveDetached.IsAttached &&
            positiveDetached.DetachCount > 0 &&
            !positiveContext.HasManagedDebugListener &&
            !positiveContext.HasDebugListener;
        if (!positivePassed)
        {
            throw new InvalidOperationException(
                "Debug listener positive runtime invariants failed. Attached=" + positiveAttached +
                " Detached=" + positiveDetached);
        }

        using TensorRtExecutionContext negativeContext = engine.CreateExecutionContext();
        using TensorRtDebugListenerCallbackOwner negativeOwner = new(line, _ => false);
        negativeContext.SetTensorAddress("debug_input", inputBuffer);
        negativeContext.SetTensorAddress("debug_output", outputBuffer);
        negativeContext.SetDebugListener(negativeOwner);
        negativeContext.SetTensorDebugState("debug_output", true);
        bool negativeEnqueueFailed = false;
        try
        {
            negativeContext.EnqueueAsync(stream);
            stream.Synchronize();
        }
        catch (TensorRtException)
        {
            negativeEnqueueFailed = true;
        }

        TensorRtDebugListenerRuntimeSnapshot negativeAttached = negativeOwner.GetRuntimeSnapshot();
        bool negativeCleared = negativeContext.ClearDebugListener();
        TensorRtDebugListenerRuntimeSnapshot negativeDetached = negativeOwner.GetRuntimeSnapshot();
        bool negativeCallbackRejected =
            negativeAttached.InvocationCount > 0 &&
            negativeAttached.FailureCount > 0 &&
            !negativeAttached.LastCallbackSucceeded &&
            !negativeAttached.IsRealCallbackRuntimeProof;
        bool negativePassed =
            negativeCallbackRejected &&
            negativeAttached.InFlightCallbackCount == 0 &&
            negativeCleared &&
            !negativeDetached.IsAttached &&
            !negativeContext.HasManagedDebugListener &&
            !negativeContext.HasDebugListener;
        if (!negativePassed)
        {
            throw new InvalidOperationException(
                "Debug listener rejection did not fail closed. Attached=" + negativeAttached +
                " Detached=" + negativeDetached + " EnqueueFailed=" + negativeEnqueueFailed);
        }

        Console.WriteLine("DebugListenerRuntimeSummary");
        Console.WriteLine($"  TensorRtLine={(int)line} RuntimePackageKey={runtimePackageKey}");
        Console.WriteLine($"  Callbacks={positiveAttached.InvocationCount} Failures={positiveAttached.FailureCount} InFlight={positiveAttached.InFlightCallbackCount}");
        Console.WriteLine($"  Tensor={positiveAttached.TensorName} Shape=[{string.Join(",", positiveAttached.ShapeDimensions)}] MetadataCopied={copiedMetadata.MetadataCopied}");
        Console.WriteLine($"  NativePointerExposed={positiveAttached.BorrowedPointerExposed} DetachCount={positiveDetached.DetachCount}");
        Console.WriteLine($"  RejectionCase=Passed EnqueueFailed={negativeEnqueueFailed} Callbacks={negativeAttached.InvocationCount} Failures={negativeAttached.FailureCount}");
        Console.WriteLine(
            "DebugListenerRealRuntime=Passed" +
            $" TensorRtLine={(int)line}" +
            $" RuntimePackageKey={runtimePackageKey}" +
            $" NativeVTableInstalled={positiveAttached.AttachCount > 0}" +
            $" ProcessDebugTensorInvoked={positiveAttached.InvocationCount > 0}" +
            $" InvocationCount={positiveAttached.InvocationCount}" +
            $" FailureCount={positiveAttached.FailureCount}" +
            $" InFlightCallbackCount={positiveAttached.InFlightCallbackCount}" +
            $" TensorName={positiveAttached.TensorName}" +
            $" Shape=[{string.Join(",", positiveAttached.ShapeDimensions)}]" +
            $" MetadataCopied={copiedMetadata.MetadataCopied}" +
            $" BorrowedPointerExposed={positiveAttached.BorrowedPointerExposed}" +
            $" DetachCount={positiveDetached.DetachCount}" +
            $" NegativeCallbackRejected={negativeCallbackRejected}" +
            $" NegativeEnqueueFailed={negativeEnqueueFailed}" +
            $" NegativeInvocationCount={negativeAttached.InvocationCount}" +
            $" NegativeFailureCount={negativeAttached.FailureCount}" +
            $" IsRealCallbackRuntimeProof={positiveAttached.IsRealCallbackRuntimeProof}");
    }

    private static bool HasSwitch(string[] args, string name)
    {
        return Array.Exists(args, value => string.Equals(value, name, StringComparison.OrdinalIgnoreCase));
    }

    private static string GetArgument(string[] args, string name, string fallback)
    {
        for (int index = 0; index + 1 < args.Length; index++)
        {
            if (string.Equals(args[index], name, StringComparison.OrdinalIgnoreCase))
            {
                return args[index + 1];
            }
        }

        return fallback;
    }

    private static TensorRtApiLine ParseTensorRtLine(string value)
    {
        return value switch
        {
            "8" => TensorRtApiLine.TensorRt8,
            "10" => TensorRtApiLine.TensorRt10,
            "11" => TensorRtApiLine.TensorRt11,
            _ => throw new ArgumentOutOfRangeException(nameof(value), value, "TensorRT line must be 8, 10, or 11.")
        };
    }
}

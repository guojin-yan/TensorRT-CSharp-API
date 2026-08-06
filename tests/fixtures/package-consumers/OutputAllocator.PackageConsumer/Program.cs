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
            if (!HasSwitch(args, "--output-allocator-runtime-smoke-only"))
            {
                throw new ArgumentException("The --output-allocator-runtime-smoke-only switch is required.");
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

            RunRealOutputAllocatorRuntimeSmoke(line, runtimePackageKey);
            Console.WriteLine("OutputAllocatorPackageConsumer Passed=True Mode=OutputAllocatorRuntimeSmokeOnly");
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine("OutputAllocatorPackageConsumer Passed=False Error=" + exception);
            return 1;
        }
    }

    private static void RunRealOutputAllocatorRuntimeSmoke(TensorRtApiLine line, string runtimePackageKey)
    {
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
            "allocator_input",
            TensorRtDataType.Float,
            new TensorRtDims(new[] { 1, 4 }));
        using TensorRtLayer identity = network.AddIdentity(input);
        identity.Name = "allocator_identity";
        using TensorRtTensor output = identity.GetOutput(0);
        output.Name = "allocator_output";
        network.MarkOutput(output);

        using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
        using TensorRtEngine engine = runtime.Deserialize(hostMemory);
        using CudaStream stream = new();
        using CudaMemory inputBuffer = new(4 * sizeof(float));
        inputBuffer.Fill(0, 4 * sizeof(float));

        TensorRtOutputAllocatorCallbackRequest acceptedRequest = default;
        bool acceptedRequestObserved = false;
        using TensorRtExecutionContext positiveContext = engine.CreateExecutionContext();
        using TensorRtOutputAllocatorCallbackOwner positiveOwner = new(
            line,
            request =>
            {
                if (request.Kind == TensorRtOutputAllocatorCallbackKind.ReallocateOutput)
                {
                    acceptedRequest = request;
                    acceptedRequestObserved = true;
                }

                return true;
            });
        positiveContext.SetTensorAddress("allocator_input", inputBuffer);
        positiveContext.SetOutputAllocator("allocator_output", positiveOwner);
        positiveContext.EnqueueAsync(stream);
        stream.Synchronize();

        TensorRtOutputAllocatorRuntimeSnapshot positiveAttached = positiveOwner.GetRuntimeSnapshot();
        bool positiveCleared = positiveContext.ClearOutputAllocator("allocator_output");
        TensorRtOutputAllocatorRuntimeSnapshot positiveDetached = positiveOwner.GetRuntimeSnapshot();
        bool positivePassed =
            positiveAttached.IsAttached &&
            positiveAttached.RealCallbackRuntime &&
            positiveAttached.ReallocateOutputCount > 0UL &&
            positiveAttached.AllocationCount > 0UL &&
            positiveAttached.FailureCount == 0UL &&
            positiveAttached.InFlightCallbackCount == 0UL &&
            string.Equals(positiveAttached.TensorName, "allocator_output", StringComparison.Ordinal) &&
            acceptedRequestObserved &&
            string.Equals(acceptedRequest.TensorName, "allocator_output", StringComparison.Ordinal) &&
            positiveCleared &&
            !positiveDetached.IsAttached &&
            positiveDetached.LiveAllocationCount == 0UL &&
            positiveDetached.LiveAllocationBytes == 0UL &&
            positiveDetached.ReleaseCount > 0UL &&
            !positiveContext.HasManagedOutputAllocator("allocator_output") &&
            !positiveContext.HasOutputAllocator("allocator_output");
        if (!positivePassed)
        {
            throw new InvalidOperationException(
                "Output allocator positive runtime invariants failed. Attached=" + positiveAttached +
                " Detached=" + positiveDetached);
        }

        TensorRtOutputAllocatorCallbackRequest rejectedRequest = default;
        bool rejectedRequestObserved = false;
        using TensorRtExecutionContext negativeContext = engine.CreateExecutionContext();
        using TensorRtOutputAllocatorCallbackOwner negativeOwner = new(
            line,
            request =>
            {
                if (request.Kind == TensorRtOutputAllocatorCallbackKind.ReallocateOutput)
                {
                    rejectedRequest = request;
                    rejectedRequestObserved = true;
                }

                return request.Kind != TensorRtOutputAllocatorCallbackKind.ReallocateOutput;
            });
        negativeContext.SetTensorAddress("allocator_input", inputBuffer);
        negativeContext.SetOutputAllocator("allocator_output", negativeOwner);
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

        TensorRtOutputAllocatorRuntimeSnapshot negativeAttached = negativeOwner.GetRuntimeSnapshot();
        bool negativeCleared = negativeContext.ClearOutputAllocator("allocator_output");
        TensorRtOutputAllocatorRuntimeSnapshot negativeDetached = negativeOwner.GetRuntimeSnapshot();
        bool negativePassed =
            negativeEnqueueFailed &&
            negativeAttached.ReallocateOutputCount > 0UL &&
            negativeAttached.AllocationCount == 0UL &&
            negativeAttached.FailureCount > 0UL &&
            !negativeAttached.LastAllocationSucceeded &&
            rejectedRequestObserved &&
            rejectedRequest.Kind == TensorRtOutputAllocatorCallbackKind.ReallocateOutput &&
            negativeCleared &&
            !negativeDetached.IsAttached &&
            negativeDetached.LiveAllocationCount == 0UL &&
            !negativeContext.HasManagedOutputAllocator("allocator_output");
        if (!negativePassed)
        {
            throw new InvalidOperationException(
                "Output allocator rejection did not fail closed. Attached=" + negativeAttached +
                " Detached=" + negativeDetached + " EnqueueFailed=" + negativeEnqueueFailed);
        }

        Console.WriteLine("OutputAllocatorRuntimeSummary");
        Console.WriteLine($"  TensorRtLine={(int)line} RuntimePackageKey={runtimePackageKey}");
        Console.WriteLine($"  Callbacks={positiveAttached.InvocationCount} NotifyShape={positiveAttached.NotifyShapeCount} Reallocate={positiveAttached.ReallocateOutputCount}");
        Console.WriteLine($"  Allocations={positiveAttached.AllocationCount} Releases={positiveDetached.ReleaseCount} LiveAllocations={positiveDetached.LiveAllocationCount}");
        Console.WriteLine($"  PeakLiveBytes={positiveAttached.PeakLiveAllocationBytes} NativePointerExposed={positiveAttached.NativePointerExposed}");
        Console.WriteLine($"  RejectionCase=Passed EnqueueFailed={negativeEnqueueFailed} Allocations={negativeAttached.AllocationCount} Failures={negativeAttached.FailureCount}");
        Console.WriteLine(
            "OutputAllocatorRealRuntime=Passed" +
            $" TensorRtLine={(int)line}" +
            $" RuntimePackageKey={runtimePackageKey}" +
            $" InvocationCount={positiveAttached.InvocationCount}" +
            $" NotifyShapeCount={positiveAttached.NotifyShapeCount}" +
            $" ReallocateOutputCount={positiveAttached.ReallocateOutputCount}" +
            $" AllocationCount={positiveAttached.AllocationCount}" +
            $" ReleaseCount={positiveDetached.ReleaseCount}" +
            $" LiveAllocationCount={positiveDetached.LiveAllocationCount}" +
            $" PeakLiveAllocationBytes={positiveAttached.PeakLiveAllocationBytes}" +
            $" PointerExposed={positiveAttached.NativePointerExposed}" +
            $" NegativeEnqueueFailed={negativeEnqueueFailed}" +
            $" NegativeAllocationCount={negativeAttached.AllocationCount}" +
            $" NegativeFailureCount={negativeAttached.FailureCount}" +
            $" RealCallbackRuntime={positiveAttached.RealCallbackRuntime}");
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

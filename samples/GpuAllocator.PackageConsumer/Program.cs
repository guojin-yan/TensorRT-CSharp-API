using System;
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
            if (!HasSwitch(args, "--gpu-allocator-runtime-smoke-only"))
            {
                throw new ArgumentException("The --gpu-allocator-runtime-smoke-only switch is required.");
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
            RunRealGpuAllocatorRuntimeSmoke(line, runtimePackageKey);
            Console.WriteLine("GpuAllocatorPackageConsumer Passed=True Mode=GpuAllocatorRuntimeSmokeOnly");
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine("GpuAllocatorPackageConsumer Passed=False Error=" + exception);
            return 1;
        }
    }

    private static void RunRealGpuAllocatorRuntimeSmoke(TensorRtApiLine line, string runtimePackageKey)
    {
        using TensorRtLogger logger = new(line);
        using TensorRtHostMemory hostMemory = BuildGpuAllocatorSmokePlan(line, logger);

        TensorRtGpuAllocatorRuntimeSnapshot runtimeAttached;
        TensorRtGpuAllocatorRuntimeSnapshot runtimeDetached;
        TensorRtGpuAllocatorRuntimeSnapshot runtimeReleased;
        using (TensorRtGpuAllocatorCallbackOwner runtimeOwner = new(line, static _ => true))
        using (TensorRtRuntime runtime = new(logger))
        {
            runtime.SetGpuAllocator(runtimeOwner);
            using TensorRtEngine engine = runtime.Deserialize(hostMemory);
            runtimeAttached = runtimeOwner.GetRuntimeSnapshot();
            runtime.ClearGpuAllocator();
            runtimeDetached = runtimeOwner.GetRuntimeSnapshot();
            engine.Dispose();
            runtimeReleased = runtimeOwner.GetRuntimeSnapshot();

            bool runtimePassed =
                runtimeAttached.IsAttached &&
                runtimeAttached.AttachmentTarget == TensorRtGpuAllocatorAttachmentTarget.Runtime &&
                !runtimeDetached.IsAttached &&
                runtimeDetached.AttachmentTarget == TensorRtGpuAllocatorAttachmentTarget.None &&
                !runtime.HasManagedGpuAllocator &&
                runtimeReleased.LiveAllocationCount == 0UL &&
                runtimeReleased.LiveAllocationBytes == 0UL &&
                runtimeReleased.CallbackFailureCount == 0UL &&
                runtimeReleased.CudaFailureCount == 0UL;
            if (!runtimePassed)
            {
                throw new InvalidOperationException(
                    "Runtime GPU allocator lifecycle failed. Attached=" + runtimeAttached +
                    " Detached=" + runtimeDetached + " Released=" + runtimeReleased);
            }
        }

        TensorRtGpuAllocatorRuntimeSnapshot builderAttached;
        TensorRtGpuAllocatorRuntimeSnapshot builderDetached;
        TensorRtGpuAllocatorRuntimeSnapshot builderReleased;
        using (TensorRtGpuAllocatorCallbackOwner builderOwner = new(line, static _ => true))
        using (TensorRtBuilder builder = new(logger))
        using (TensorRtBuilderConfig config = builder.CreateBuilderConfig())
        using (TensorRtNetworkDefinition network = CreateSmokeNetwork(builder, "gpu_builder"))
        {
            config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 32UL * 1024UL * 1024UL);
            builder.SetGpuAllocator(builderOwner);
            using TensorRtEngine engine = builder.BuildEngineWithConfig(network, config);
            builderAttached = builderOwner.GetRuntimeSnapshot();
            builder.ClearGpuAllocator();
            builderDetached = builderOwner.GetRuntimeSnapshot();
            engine.Dispose();
            builderReleased = builderOwner.GetRuntimeSnapshot();

            bool builderPassed =
                builderAttached.IsAttached &&
                builderAttached.AttachmentTarget == TensorRtGpuAllocatorAttachmentTarget.Builder &&
                builderAttached.RealCallbackRuntime &&
                !builderDetached.IsAttached &&
                !builder.HasManagedGpuAllocator &&
                builderReleased.LiveAllocationCount == 0UL &&
                builderReleased.LiveAllocationBytes == 0UL &&
                builderReleased.DeallocateCount + builderReleased.DeallocateAsyncCount > 0UL &&
                builderReleased.CallbackFailureCount == 0UL &&
                builderReleased.CudaFailureCount == 0UL;
            if (!builderPassed)
            {
                throw new InvalidOperationException(
                    "Builder GPU allocator lifecycle failed. Attached=" + builderAttached +
                    " Detached=" + builderDetached + " Released=" + builderReleased);
            }
        }

        (TensorRtGpuAllocatorRuntimeSnapshot rejectedSnapshot, bool rejectedBuildFailed) =
            RunBuilderFailureScenario(line, logger, static request => request.IsRelease, "gpu_rejected");
        if (!rejectedBuildFailed ||
            rejectedSnapshot.RejectedCount == 0UL ||
            rejectedSnapshot.CallbackFailureCount != 0UL ||
            rejectedSnapshot.CudaFailureCount != 0UL ||
            rejectedSnapshot.LiveAllocationCount != 0UL)
        {
            throw new InvalidOperationException(
                "Allocator rejection did not fail closed. Snapshot=" + rejectedSnapshot +
                " BuildFailed=" + rejectedBuildFailed);
        }

        (TensorRtGpuAllocatorRuntimeSnapshot exceptionSnapshot, bool exceptionBuildFailed) =
            RunBuilderFailureScenario(
                line,
                logger,
                static request => request.IsRelease
                    ? true
                    : throw new InvalidOperationException("controlled GPU allocator callback failure"),
                "gpu_exception");
        if (!exceptionBuildFailed ||
            exceptionSnapshot.CallbackFailureCount == 0UL ||
            exceptionSnapshot.LiveAllocationCount != 0UL ||
            exceptionSnapshot.CudaFailureCount != 0UL)
        {
            throw new InvalidOperationException(
                "Allocator exception did not fail closed. Snapshot=" + exceptionSnapshot +
                " BuildFailed=" + exceptionBuildFailed);
        }

        Console.WriteLine("GpuAllocatorRuntimeSummary");
        Console.WriteLine($"  TensorRtLine={(int)line} RuntimePackageKey={runtimePackageKey}");
        Console.WriteLine($"  RuntimeAttachLifecycle=Passed Callbacks={runtimeReleased.InvocationCount} LiveAllocations={runtimeReleased.LiveAllocationCount}");
        Console.WriteLine($"  BuilderCallbacks={builderReleased.InvocationCount} Allocate={builderReleased.AllocateCount} Reallocate={builderReleased.ReallocateCount} AllocateAsync={builderReleased.AllocateAsyncCount}");
        Console.WriteLine($"  BuilderDeallocate={builderReleased.DeallocateCount} DeallocateAsync={builderReleased.DeallocateAsyncCount}");
        Console.WriteLine($"  PeakLiveBytes={builderReleased.PeakLiveAllocationBytes} FinalLiveAllocations={builderReleased.LiveAllocationCount}");
        Console.WriteLine($"  RejectionCase=Passed BuildFailed={rejectedBuildFailed} RejectedCount={rejectedSnapshot.RejectedCount}");
        Console.WriteLine($"  ExceptionCase=Passed BuildFailed={exceptionBuildFailed} CallbackFailures={exceptionSnapshot.CallbackFailureCount}");
        Console.WriteLine($"  NativePointerExposed={builderReleased.NativePointerExposed} RealCallbackRuntime={builderReleased.RealCallbackRuntime}");
        Console.WriteLine(
            "GpuAllocatorRealRuntime=Passed" +
            $" TensorRtLine={(int)line}" +
            $" RuntimePackageKey={runtimePackageKey}" +
            $" RuntimeInvocationCount={runtimeReleased.InvocationCount}" +
            $" RuntimeAllocateCount={runtimeReleased.AllocateCount}" +
            $" RuntimeAllocateAsyncCount={runtimeReleased.AllocateAsyncCount}" +
            $" RuntimeDeallocateCount={runtimeReleased.DeallocateCount}" +
            $" RuntimeDeallocateAsyncCount={runtimeReleased.DeallocateAsyncCount}" +
            $" RuntimePeakLiveAllocationBytes={runtimeReleased.PeakLiveAllocationBytes}" +
            $" RuntimeLiveAllocationCount={runtimeReleased.LiveAllocationCount}" +
            $" BuilderInvocationCount={builderReleased.InvocationCount}" +
            $" BuilderAllocateCount={builderReleased.AllocateCount}" +
            $" BuilderReallocateCount={builderReleased.ReallocateCount}" +
            $" BuilderAllocateAsyncCount={builderReleased.AllocateAsyncCount}" +
            $" BuilderDeallocateCount={builderReleased.DeallocateCount}" +
            $" BuilderDeallocateAsyncCount={builderReleased.DeallocateAsyncCount}" +
            $" BuilderPeakLiveAllocationBytes={builderReleased.PeakLiveAllocationBytes}" +
            $" BuilderLiveAllocationCount={builderReleased.LiveAllocationCount}" +
            $" RejectedBuildFailed={rejectedBuildFailed}" +
            $" RejectedCount={rejectedSnapshot.RejectedCount}" +
            $" ExceptionBuildFailed={exceptionBuildFailed}" +
            $" ExceptionCallbackFailureCount={exceptionSnapshot.CallbackFailureCount}" +
            $" PointerExposed={runtimeReleased.NativePointerExposed}" +
            " RuntimeAttachLifecycle=True" +
            $" RealCallbackRuntime={builderReleased.RealCallbackRuntime}");
    }

    private static TensorRtHostMemory BuildGpuAllocatorSmokePlan(TensorRtApiLine line, TensorRtLogger logger)
    {
        using TensorRtBuilder builder = new(logger);
        using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
        using TensorRtNetworkDefinition network = CreateSmokeNetwork(builder, "gpu_runtime");
        config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 32UL * 1024UL * 1024UL);
        return builder.BuildSerializedNetwork(network, config);
    }

    private static TensorRtNetworkDefinition CreateSmokeNetwork(TensorRtBuilder builder, string prefix)
    {
        TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
        try
        {
            using TensorRtTensor input = network.AddInput(prefix + "_input", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 4 }));
            using TensorRtLayer identity = network.AddIdentity(input);
            identity.Name = prefix + "_identity";
            using TensorRtTensor output = identity.GetOutput(0);
            output.Name = prefix + "_output";
            network.MarkOutput(output);
            return network;
        }
        catch
        {
            network.Dispose();
            throw;
        }
    }

    private static (TensorRtGpuAllocatorRuntimeSnapshot Snapshot, bool BuildFailed) RunBuilderFailureScenario(
        TensorRtApiLine line,
        TensorRtLogger logger,
        TensorRtGpuAllocatorHandler handler,
        string prefix)
    {
        using TensorRtGpuAllocatorCallbackOwner owner = new(line, handler);
        using TensorRtBuilder builder = new(logger);
        using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
        using TensorRtNetworkDefinition network = CreateSmokeNetwork(builder, prefix);
        config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 32UL * 1024UL * 1024UL);
        builder.SetGpuAllocator(owner);
        bool buildFailed = false;
        try
        {
            using TensorRtEngine unexpected = builder.BuildEngineWithConfig(network, config);
        }
        catch (TensorRtException)
        {
            buildFailed = true;
        }

        TensorRtGpuAllocatorRuntimeSnapshot snapshot = owner.GetRuntimeSnapshot();
        builder.ClearGpuAllocator();
        return (snapshot, buildFailed);
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

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Reflection;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;

internal static class Program
{
    private static void Main(string[] args)
    {
        string requestedLine = JYPPX.SampleSupport.SampleCommandLine.GetStringArgument(args, "--tensor-rt-line", "auto");
        string runtimePackageKey = JYPPX.SampleSupport.SampleCommandLine.GetStringArgument(args, "--runtime-package-key", string.Empty);
        bool dependencyProbeOnly = JYPPX.SampleSupport.SampleCommandLine.HasSwitch(args, "--dependency-probe-only");
        bool debugListenerRuntimeSmokeOnly = JYPPX.SampleSupport.SampleCommandLine.HasSwitch(args, "--debug-listener-runtime-smoke-only");
        bool outputAllocatorRuntimeSmokeOnly = JYPPX.SampleSupport.SampleCommandLine.HasSwitch(args, "--output-allocator-runtime-smoke-only");
        string callbackStateGetterProbe = JYPPX.SampleSupport.SampleCommandLine.GetStringArgument(args, "--callback-state-getter-probe", string.Empty);
        bool enableDebugListenerRuntimeSmoke =
            debugListenerRuntimeSmokeOnly ||
            JYPPX.SampleSupport.SampleCommandLine.HasSwitch(args, "--enable-debug-listener-runtime-smoke");
        bool enableOutputAllocatorRuntimeSmoke =
            outputAllocatorRuntimeSmokeOnly ||
            JYPPX.SampleSupport.SampleCommandLine.HasSwitch(args, "--enable-output-allocator-runtime-smoke");

        Console.WriteLine($"CallbackAllocatorSafeControlsSmokeRunner TensorRtLineRequest={requestedLine} RuntimePackageKey={runtimePackageKey} DependencyProbeOnly={dependencyProbeOnly} EnableDebugListenerRuntimeSmoke={enableDebugListenerRuntimeSmoke} DebugListenerRuntimeSmokeOnly={debugListenerRuntimeSmokeOnly} EnableOutputAllocatorRuntimeSmoke={enableOutputAllocatorRuntimeSmoke} OutputAllocatorRuntimeSmokeOnly={outputAllocatorRuntimeSmokeOnly}");

        if (dependencyProbeOnly)
        {
            TensorRtApiLine probeLine = ResolveProbeLine(requestedLine);
            PrintDependencyProbe(probeLine);
            PrintSafeControlSurface(probeLine, enableRuntimeSmoke: false, runtimePackageKey: runtimePackageKey);
            Console.WriteLine("Skipped=True Reason=DependencyProbeOnly");
            return;
        }

        TensorRtEnvironmentSnapshot snapshot;
        try
        {
            snapshot = TensorRtEnvironmentProbe.GetCurrent();
        }
        catch (Exception exception) when (IsSkippableEnvironmentException(exception))
        {
            Console.WriteLine($"Skipped=True Reason=EnvironmentProbe:{exception.GetType().Name}:{exception.Message}");
            return;
        }

        TensorRtApiLine? line = ResolveTensorRtLine(snapshot, requestedLine);
        if (line == null)
        {
            Console.WriteLine("Skipped=True Reason=NoRequestedTensorRtAdapterAvailable");
            return;
        }

        TensorRtAdapterInfo adapter = GetAdapter(snapshot, line.Value);
        Console.WriteLine($"ResolvedTensorRtLine={(int)line.Value} TRT={snapshot.BuildInfo.TensorRtVersion} CUDA={snapshot.BuildInfo.CudaToolkitVersion}");
        Console.WriteLine($"Adapter Runtime={adapter.RuntimeCreationSupported} Builder={adapter.BuilderCreationSupported} Message={adapter.StatusMessage}");

        PrintDependencyProbe(line.Value);
        if (!outputAllocatorRuntimeSmokeOnly)
        {
            PrintSafeControlSurface(line.Value, enableDebugListenerRuntimeSmoke, runtimePackageKey);
        }

        if (!adapter.RuntimeCreationSupported || !adapter.BuilderCreationSupported)
        {
            Console.WriteLine($"Skipped=True Reason=AdapterNotReady:{adapter.StatusMessage}");
            return;
        }

        if (!string.IsNullOrWhiteSpace(callbackStateGetterProbe))
        {
            try
            {
                RunCallbackStateGetterProbe(line.Value, callbackStateGetterProbe);
                Console.WriteLine($"CallbackStateGetterProbe={callbackStateGetterProbe} Completed=True");
            }
            catch (Exception exception)
            {
                Console.WriteLine($"CallbackStateGetterProbe={callbackStateGetterProbe} Completed=False Exception={exception.GetType().Name}:{exception.Message}");
                Environment.ExitCode = 3;
            }

            return;
        }

        if (debugListenerRuntimeSmokeOnly)
        {
            try
            {
                RunRealDebugListenerRuntimeSmoke(line.Value, runtimePackageKey);
            }
            catch (Exception exception) when (IsSkippableEnvironmentException(exception))
            {
                Console.WriteLine($"DebugListenerRealRuntime=Skipped Reason={exception.GetType().Name}:{exception.Message}");
                Console.WriteLine($"Skipped=True Reason={exception.GetType().Name}:{exception.Message}");
                return;
            }

            Console.WriteLine("CallbackAllocatorSafeControlsSmokeRunner Passed=True Mode=DebugListenerRuntimeSmokeOnly");
            return;
        }

        if (outputAllocatorRuntimeSmokeOnly)
        {
            try
            {
                RunRealOutputAllocatorRuntimeSmoke(line.Value, runtimePackageKey);
            }
            catch (Exception exception) when (IsSkippableEnvironmentException(exception))
            {
                Console.WriteLine($"OutputAllocatorRealRuntime=Skipped Reason={exception.GetType().Name}:{exception.Message}");
                Console.WriteLine($"Skipped=True Reason={exception.GetType().Name}:{exception.Message}");
                return;
            }

            Console.WriteLine("CallbackAllocatorSafeControlsSmokeRunner Passed=True Mode=OutputAllocatorRuntimeSmokeOnly");
            return;
        }

        try
        {
            RunSafeControls(line.Value);
            if (enableDebugListenerRuntimeSmoke)
            {
                RunRealDebugListenerRuntimeSmoke(line.Value, runtimePackageKey);
            }
            if (enableOutputAllocatorRuntimeSmoke)
            {
                RunRealOutputAllocatorRuntimeSmoke(line.Value, runtimePackageKey);
            }
        }
        catch (Exception exception) when (IsSkippableEnvironmentException(exception))
        {
            Console.WriteLine($"Skipped=True Reason={exception.GetType().Name}:{exception.Message}");
            return;
        }

        Console.WriteLine("CallbackAllocatorSafeControlsSmokeRunner Passed=True");
    }

    private static void RunCallbackStateGetterProbe(TensorRtApiLine line, string probe)
    {
        if (line != TensorRtApiLine.TensorRt10 && line != TensorRtApiLine.TensorRt11)
        {
            throw new InvalidOperationException("Callback-state getter probes require TensorRT 10 or TensorRT 11.");
        }

        using TensorRtLogger logger = new TensorRtLogger(line);
        using TensorRtRuntime runtime = new TensorRtRuntime(logger);
        using TensorRtBuilder builder = new TensorRtBuilder(logger);
        using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
        config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 32UL * 1024UL * 1024UL);
        config.SetEngineCapability(TensorRtEngineCapability.Standard);
        config.SetHardwareCompatibilityLevel(TensorRtHardwareCompatibilityLevel.None);

        using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
        using TensorRtTensor input = network.AddInput("callback_probe_input", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 4 }));
        using TensorRtLayer identity = network.AddIdentity(input);
        using TensorRtTensor output = identity.GetOutput(0);
        output.Name = "callback_probe_output";
        network.MarkOutput(output);

        using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
        using TensorRtEngine engine = runtime.Deserialize(hostMemory);
        using TensorRtExecutionContext context = engine.CreateExecutionContext();

        Console.WriteLine($"CallbackStateGetterProbe={probe} Started=True TensorRtLine={(int)line}");
        switch (probe.ToLowerInvariant())
        {
            case "has-output-allocator":
                Console.WriteLine($"CallbackStateGetterProbeValue={context.HasOutputAllocator("callback_probe_output")}");
                break;
            case "has-temporary-storage-allocator":
                Console.WriteLine($"CallbackStateGetterProbeValue={context.HasTemporaryStorageAllocator}");
                break;
            case "has-debug-listener":
                Console.WriteLine($"CallbackStateGetterProbeValue={context.HasDebugListener}");
                break;
            case "output-allocator-interface-info":
                Console.WriteLine($"CallbackStateGetterProbeValue={ProbeOutputAllocatorInterfaceInfo(context)}");
                break;
            case "temporary-storage-allocator-interface-info":
                Console.WriteLine($"CallbackStateGetterProbeValue={ProbeTemporaryStorageAllocatorInterfaceInfo(context)}");
                break;
            case "debug-listener-interface-info":
                Console.WriteLine($"CallbackStateGetterProbeValue={ProbeDebugListenerInterfaceInfo(context)}");
                break;
            case "callback-state-snapshot":
                Console.WriteLine($"CallbackStateGetterProbeValue={context.GetCallbackStateSnapshot("callback_probe_output")}");
                break;
            case "try-callback-state-snapshot":
                bool complete = context.TryGetCallbackStateSnapshot(
                    "callback_probe_output",
                    out TensorRtExecutionContextCallbackStateSnapshot snapshot,
                    out string diagnostic);
                Console.WriteLine($"CallbackStateGetterProbeValue=Complete={complete};Snapshot={snapshot};Diagnostic={SanitizeSmokeValue(diagnostic)}");
                break;
            default:
                throw new ArgumentException($"Unknown callback-state getter probe '{probe}'.", nameof(probe));
        }
    }

    private static string ProbeOutputAllocatorInterfaceInfo(TensorRtExecutionContext context)
    {
        bool available = context.TryGetOutputAllocatorInterfaceInfo(
            "callback_probe_output",
            out TensorRtInterfaceInfo info,
            out string diagnostic);
        return $"Available={available};Info={FormatInterfaceInfo(info, diagnostic)}";
    }

    private static string ProbeTemporaryStorageAllocatorInterfaceInfo(TensorRtExecutionContext context)
    {
        bool available = context.TryGetTemporaryStorageAllocatorInterfaceInfo(
            out TensorRtInterfaceInfo info,
            out string diagnostic);
        return $"Available={available};Info={FormatInterfaceInfo(info, diagnostic)}";
    }

    private static string ProbeDebugListenerInterfaceInfo(TensorRtExecutionContext context)
    {
        bool available = context.TryGetDebugListenerInterfaceInfo(
            out TensorRtInterfaceInfo info,
            out string diagnostic);
        return $"Available={available};Info={FormatInterfaceInfo(info, diagnostic)}";
    }

    private static void RunRealDebugListenerRuntimeSmoke(TensorRtApiLine line, string runtimePackageKey)
    {
        if (line != TensorRtApiLine.TensorRt10 && line != TensorRtApiLine.TensorRt11)
        {
            Console.WriteLine("DebugListenerRealRuntime=Skipped Reason=RequiresTensorRt10Or11");
            return;
        }

        using TensorRtLogger logger = new TensorRtLogger(line);
        using TensorRtRuntime runtime = new TensorRtRuntime(logger);
        using TensorRtBuilder builder = new TensorRtBuilder(logger);
        using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
        config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 32UL * 1024UL * 1024UL);
        config.SetEngineCapability(TensorRtEngineCapability.Standard);
        config.SetHardwareCompatibilityLevel(TensorRtHardwareCompatibilityLevel.None);

        using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
        using TensorRtTensor input = network.AddInput("debug_input", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 4 }));
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
        using TensorRtExecutionContext context = engine.CreateExecutionContext();
        using CudaStream stream = new CudaStream();
        using CudaMemory inputBuffer = new CudaMemory(4 * sizeof(float));
        using CudaMemory outputBuffer = new CudaMemory(4 * sizeof(float));
        inputBuffer.Fill(0, 4 * sizeof(float));
        outputBuffer.Fill(0, 4 * sizeof(float));
        context.SetTensorAddress("debug_input", inputBuffer);
        context.SetTensorAddress("debug_output", outputBuffer);

        TensorRtDebugTensorMetadataSnapshot managedMetadata = default;
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner(
            line,
            metadata =>
            {
                managedMetadata = metadata;
                return true;
            });
        context.SetDebugListener(owner);
        context.SetTensorDebugState("debug_output", true);
        context.EnqueueAsync(stream);
        stream.Synchronize();

        TensorRtDebugListenerRuntimeSnapshot attached = owner.GetRuntimeSnapshot();
        bool cleared = context.ClearDebugListener();
        TensorRtDebugListenerRuntimeSnapshot detached = owner.GetRuntimeSnapshot();
        bool passed =
            attached.IsAttached &&
            attached.IsRealCallbackRuntimeProof &&
            attached.InvocationCount > 0 &&
            attached.FailureCount == 0 &&
            attached.InFlightCallbackCount == 0 &&
            string.Equals(attached.TensorName, "debug_output", StringComparison.Ordinal) &&
            managedMetadata.MetadataCopied &&
            string.Equals(managedMetadata.TensorName, "debug_output", StringComparison.Ordinal) &&
            cleared &&
            !detached.IsAttached &&
            detached.DetachCount > 0 &&
            !context.HasManagedDebugListener &&
            !context.HasDebugListener;
        if (!passed)
        {
            throw new InvalidOperationException(
                "Real TensorRT debug listener callback runtime smoke did not satisfy attach/invoke/detach invariants. " +
                "Attached=" + attached + " Detached=" + detached);
        }

        Console.WriteLine(
            "DebugListenerRealRuntime=Passed" +
            $" TensorRtLine={(int)line}" +
            $" RuntimePackageKey={runtimePackageKey}" +
            $" NativeVTableInstalled={attached.AttachCount > 0}" +
            $" ProcessDebugTensorInvoked={attached.InvocationCount > 0}" +
            $" InvocationCount={attached.InvocationCount}" +
            $" FailureCount={attached.FailureCount}" +
            $" InFlightCallbackCount={attached.InFlightCallbackCount}" +
            $" TensorName={attached.TensorName}" +
            $" Shape=[{string.Join(",", attached.ShapeDimensions)}]" +
            $" MetadataCopied={managedMetadata.MetadataCopied}" +
            $" BorrowedPointerExposed={attached.BorrowedPointerExposed}" +
            $" DetachCount={detached.DetachCount}" +
            $" IsRealCallbackRuntimeProof={attached.IsRealCallbackRuntimeProof}");
    }

    private static void RunRealOutputAllocatorRuntimeSmoke(TensorRtApiLine line, string runtimePackageKey)
    {
        using TensorRtLogger logger = new TensorRtLogger(line);
        using TensorRtRuntime runtime = new TensorRtRuntime(logger);
        using TensorRtBuilder builder = new TensorRtBuilder(logger);
        using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
        config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 32UL * 1024UL * 1024UL);
        config.SetEngineCapability(TensorRtEngineCapability.Standard);
        config.SetHardwareCompatibilityLevel(TensorRtHardwareCompatibilityLevel.None);

        using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
        using TensorRtTensor input = network.AddInput("allocator_input", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 4 }));
        using TensorRtLayer identity = network.AddIdentity(input);
        identity.Name = "allocator_identity";
        using TensorRtTensor output = identity.GetOutput(0);
        output.Name = "allocator_output";
        network.MarkOutput(output);

        using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
        using TensorRtEngine engine = runtime.Deserialize(hostMemory);
        using CudaStream stream = new CudaStream();
        using CudaMemory inputBuffer = new CudaMemory(4 * sizeof(float));
        inputBuffer.Fill(0, 4 * sizeof(float));

        TensorRtOutputAllocatorCallbackRequest acceptedReallocateRequest = default;
        bool acceptedReallocateObserved = false;
        using TensorRtExecutionContext positiveContext = engine.CreateExecutionContext();
        using TensorRtOutputAllocatorCallbackOwner positiveOwner = new TensorRtOutputAllocatorCallbackOwner(
            line,
            request =>
            {
                if (request.Kind == TensorRtOutputAllocatorCallbackKind.ReallocateOutput)
                {
                    acceptedReallocateRequest = request;
                    acceptedReallocateObserved = true;
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
            acceptedReallocateObserved &&
            string.Equals(acceptedReallocateRequest.TensorName, "allocator_output", StringComparison.Ordinal) &&
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
                "Real TensorRT output allocator positive runtime smoke did not satisfy attach/invoke/release/detach invariants. " +
                "Attached=" + positiveAttached + " Detached=" + positiveDetached);
        }

        TensorRtOutputAllocatorCallbackRequest rejectedRequest = default;
        bool rejectedReallocateObserved = false;
        using TensorRtExecutionContext negativeContext = engine.CreateExecutionContext();
        using TensorRtOutputAllocatorCallbackOwner negativeOwner = new TensorRtOutputAllocatorCallbackOwner(
            line,
            request =>
            {
                if (request.Kind == TensorRtOutputAllocatorCallbackKind.ReallocateOutput)
                {
                    rejectedRequest = request;
                    rejectedReallocateObserved = true;
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
            rejectedReallocateObserved &&
            rejectedRequest.Kind == TensorRtOutputAllocatorCallbackKind.ReallocateOutput &&
            negativeCleared &&
            !negativeDetached.IsAttached &&
            negativeDetached.LiveAllocationCount == 0UL &&
            !negativeContext.HasManagedOutputAllocator("allocator_output");
        if (!negativePassed)
        {
            throw new InvalidOperationException(
                "Real TensorRT output allocator rejection runtime smoke did not fail closed. " +
                "Attached=" + negativeAttached + " Detached=" + negativeDetached +
                " EnqueueFailed=" + negativeEnqueueFailed);
        }

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

    private static void RunSafeControls(TensorRtApiLine line)
    {
        using TensorRtLogger logger = new TensorRtLogger(line);
        using TensorRtRuntime runtime = new TensorRtRuntime(logger);
        Console.WriteLine($"RuntimeSafeControls {ProbeRuntime(runtime)}");

        using TensorRtBuilder builder = new TensorRtBuilder(logger);
        using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
        Console.WriteLine($"BuilderSafeControls {ProbeBuilder(builder)}");
        Console.WriteLine($"BuilderConfigSafeControls {ProbeBuilderConfig(line, config)}");

        if (line != TensorRtApiLine.TensorRt11)
        {
            Console.WriteLine("Skipped=True Reason=FullEngineContextCallbackAllocatorSafeControlsRequireTensorRt11");
            return;
        }

        config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 32UL * 1024UL * 1024UL);
        config.SetEngineCapability(TensorRtEngineCapability.Standard);
        config.SetHardwareCompatibilityLevel(TensorRtHardwareCompatibilityLevel.None);

        using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
        using TensorRtTensor input = network.AddInput("input", TensorRtDataType.Float, new TensorRtDims(new[] { 1, 4 }));
        using TensorRtLayer identity = network.AddIdentity(input);
        using TensorRtTensor output = identity.GetOutput(0);
        output.Name = "output";
        network.MarkOutput(output);

        Console.WriteLine($"NetworkSafeControls {ProbeNetwork(network)}");

        using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
        using TensorRtEngine engine = runtime.Deserialize(hostMemory);
        using TensorRtEngineInspector inspector = engine.CreateInspector();
        using TensorRtExecutionContext context = engine.CreateExecutionContext();

        Console.WriteLine($"EngineSafeControls {ProbeEngine(engine)}");
        Console.WriteLine($"InspectorSafeControls {ProbeInspector(inspector)}");
        Console.WriteLine($"ContextSafeControls {ProbeContext(context, "output")}");
    }

    private static string ProbeRuntime(TensorRtRuntime runtime)
    {
        TensorRtRuntimeDiagnosticSnapshot diagnosticSnapshot = runtime.GetDiagnosticSnapshot();
        TensorRtRuntimeDiagnosticSummary diagnosticSummary = diagnosticSnapshot.ToSummary();
        TensorRtPluginCreatorV3MetadataDesignGateResult pluginCreatorV3MetadataGate =
            TensorRtPluginCreatorV3MetadataDesignGate.EvaluateKnownSurface(runtime.Line);
        bool hasErrorRecorderBefore = runtime.HasErrorRecorder;
        bool hasLogger = runtime.HasLogger;
        bool snapshotAvailable = runtime.TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot);
        bool metadataAvailable = runtime.TryGetErrorRecorderVersionedMetadata(
            out TensorRtVersionedInterfaceMetadata metadata,
            out string metadataDiagnostic);
        TensorRtErrorRecorderSummary errorRecorderSummary = snapshot.ToSummary();
        runtime.ClearErrorRecorder();
        bool hasErrorRecorderAfter = runtime.HasErrorRecorder;
        runtime.ClearGpuAllocator();
        return $"Logger={hasLogger} ErrorRecorder={hasErrorRecorderBefore}->{hasErrorRecorderAfter} Snapshot={snapshotAvailable}/{snapshot.ErrorCount}/{snapshot.Records.Count}/Overflow={snapshot.HasOverflowed} VersionedMetadata={FormatVersionedMetadata(metadataAvailable, metadata, metadataDiagnostic)} ErrorRecorderSummary={errorRecorderSummary.HasRecorder}/{errorRecorderSummary.ErrorCount}/{errorRecorderSummary.CopiedErrorRecordCount}/{errorRecorderSummary.InterfaceInfoAvailable}/{errorRecorderSummary.CopiedRecordCountMatchesErrorCount} RuntimeDiagnosticSnapshot={diagnosticSnapshot.HasLogger}/{diagnosticSnapshot.HasErrorRecorder}/{diagnosticSnapshot.ErrorRecorder.ErrorCount}/{diagnosticSnapshot.Diagnostics.Count} RuntimeDiagnosticSummary={diagnosticSummary.HasLogger}/{diagnosticSummary.HasErrorRecorder}/{diagnosticSummary.ErrorCount}/{diagnosticSummary.CopiedErrorRecordCount}/{diagnosticSummary.DiagnosticCount}/RuntimeProof={diagnosticSummary.CanPromoteRuntimeProof} PluginCreatorV3MetadataDesignGate={pluginCreatorV3MetadataGate.DesignGateReady}/{pluginCreatorV3MetadataGate.CandidateMethodCount}/RuntimeProof={pluginCreatorV3MetadataGate.CanPromoteRuntimeProof} ClearGpuAllocator=True";
    }

    private static string ProbeBuilder(TensorRtBuilder builder)
    {
        bool hasErrorRecorderBefore = builder.HasErrorRecorder;
        bool hasLogger = builder.HasLogger;
        bool snapshotAvailable = builder.TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot);
        TensorRtErrorRecorderSummary errorRecorderSummary = snapshot.ToSummary();
        bool metadataAvailable = builder.TryGetErrorRecorderVersionedMetadata(
            out TensorRtVersionedInterfaceMetadata metadata,
            out string metadataDiagnostic);
        builder.ClearErrorRecorder();
        bool hasErrorRecorderAfter = builder.HasErrorRecorder;
        builder.ClearGpuAllocator();
        return $"Logger={hasLogger} ErrorRecorder={hasErrorRecorderBefore}->{hasErrorRecorderAfter} Snapshot={snapshotAvailable}/{snapshot.ErrorCount}/{snapshot.Records.Count}/Overflow={snapshot.HasOverflowed} ErrorRecorderSummary={errorRecorderSummary.HasRecorder}/{errorRecorderSummary.ErrorCount}/{errorRecorderSummary.CopiedErrorRecordCount}/{errorRecorderSummary.CopiedRecordCountMatchesErrorCount} VersionedMetadata={FormatVersionedMetadata(metadataAvailable, metadata, metadataDiagnostic)} ClearGpuAllocator=True";
    }

    private static string ProbeBuilderConfig(TensorRtApiLine line, TensorRtBuilderConfig config)
    {
        if (line == TensorRtApiLine.TensorRt8)
        {
            return "ProgressMonitor=Skipped/RequiresTensorRt10Or11";
        }

        bool hasProgressMonitorBefore = config.HasProgressMonitor;
        bool metadataAvailable = config.TryGetProgressMonitorVersionedMetadata(
            out TensorRtVersionedInterfaceMetadata metadata,
            out string metadataDiagnostic);
        config.ClearProgressMonitor();
        bool hasProgressMonitorAfter = config.HasProgressMonitor;
        return $"ProgressMonitor={hasProgressMonitorBefore}->{hasProgressMonitorAfter}/VersionedMetadata={FormatVersionedMetadata(metadataAvailable, metadata, metadataDiagnostic)}";
    }

    private static string ProbeNetwork(TensorRtNetworkDefinition network)
    {
        bool hasErrorRecorderBefore = network.HasErrorRecorder;
        bool snapshotAvailable = network.TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot);
        TensorRtErrorRecorderSummary errorRecorderSummary = snapshot.ToSummary();
        bool metadataAvailable = network.TryGetErrorRecorderVersionedMetadata(
            out TensorRtVersionedInterfaceMetadata metadata,
            out string metadataDiagnostic);
        network.ClearErrorRecorder();
        bool hasErrorRecorderAfter = network.HasErrorRecorder;
        return $"ErrorRecorder={hasErrorRecorderBefore}->{hasErrorRecorderAfter}/Snapshot={snapshotAvailable}/{snapshot.ErrorCount}/{snapshot.Records.Count}/Overflow={snapshot.HasOverflowed}/ErrorRecorderSummary={errorRecorderSummary.HasRecorder}/{errorRecorderSummary.ErrorCount}/{errorRecorderSummary.CopiedErrorRecordCount}/{errorRecorderSummary.CopiedRecordCountMatchesErrorCount}/VersionedMetadata={FormatVersionedMetadata(metadataAvailable, metadata, metadataDiagnostic)}";
    }

    private static string ProbeEngine(TensorRtEngine engine)
    {
        bool hasErrorRecorderBefore = engine.HasErrorRecorder;
        bool metadataAvailable = engine.TryGetErrorRecorderVersionedMetadata(
            out TensorRtVersionedInterfaceMetadata metadata,
            out string metadataDiagnostic);
        engine.ClearErrorRecorder();
        bool hasErrorRecorderAfter = engine.HasErrorRecorder;
        return $"ErrorRecorder={hasErrorRecorderBefore}->{hasErrorRecorderAfter}/VersionedMetadata={FormatVersionedMetadata(metadataAvailable, metadata, metadataDiagnostic)}";
    }

    private static string ProbeInspector(TensorRtEngineInspector inspector)
    {
        bool hasErrorRecorderBefore = inspector.HasErrorRecorder;
        bool snapshotAvailable = inspector.TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot);
        TensorRtErrorRecorderSummary errorRecorderSummary = snapshot.ToSummary();
        bool metadataAvailable = inspector.TryGetErrorRecorderVersionedMetadata(
            out TensorRtVersionedInterfaceMetadata metadata,
            out string metadataDiagnostic);
        inspector.ClearErrorRecorder();
        bool hasErrorRecorderAfter = inspector.HasErrorRecorder;
        return $"ErrorRecorder={hasErrorRecorderBefore}->{hasErrorRecorderAfter}/Snapshot={snapshotAvailable}/{snapshot.ErrorCount}/{snapshot.Records.Count}/Overflow={snapshot.HasOverflowed}/ErrorRecorderSummary={errorRecorderSummary.HasRecorder}/{errorRecorderSummary.ErrorCount}/{errorRecorderSummary.CopiedErrorRecordCount}/{errorRecorderSummary.CopiedRecordCountMatchesErrorCount}/VersionedMetadata={FormatVersionedMetadata(metadataAvailable, metadata, metadataDiagnostic)}";
    }

    private static string ProbeContext(TensorRtExecutionContext context, string outputTensorName)
    {
        bool hasErrorRecorderBefore = context.HasErrorRecorder;
        bool errorRecorderMetadataAvailable = context.TryGetErrorRecorderVersionedMetadata(
            out TensorRtVersionedInterfaceMetadata errorRecorderMetadata,
            out string errorRecorderMetadataDiagnostic);
        context.ClearErrorRecorder();
        bool hasErrorRecorderAfter = context.HasErrorRecorder;

        TensorRtExecutionContextRuntimeDiagnosticSnapshot runtimeSnapshot = context.GetRuntimeDiagnosticSnapshot(outputTensorName);
        TensorRtExecutionContextRuntimeDiagnosticSummary runtimeSummary = runtimeSnapshot.ToSummary();
        TensorRtExecutionContextCallbackAllocatorSafeControlSummary safeControlSummary =
            context.GetCallbackAllocatorSafeControlSummary(outputTensorName);
        bool hasOutputAllocatorBefore = context.HasOutputAllocator(outputTensorName);
        TensorRtExecutionContextCallbackStateSnapshot callbackSnapshot = context.GetCallbackStateSnapshot(outputTensorName);
        bool outputAllocatorInfoAvailable = context.TryGetOutputAllocatorInterfaceInfo(outputTensorName, out TensorRtInterfaceInfo outputAllocatorInfo, out string outputAllocatorInfoDiagnostic);
        bool outputAllocatorMetadataAvailable = context.TryGetOutputAllocatorVersionedMetadata(
            outputTensorName,
            out TensorRtVersionedInterfaceMetadata outputAllocatorMetadata,
            out string outputAllocatorMetadataDiagnostic);
        TensorRtExecutionContextCallbackStateSnapshot clearedCallbackSnapshot = context.ClearCallbackState(outputTensorName);
        bool outputAllocatorCleared = clearedCallbackSnapshot.OutputAllocatorCleared;
        bool directOutputAllocatorCleared = context.ClearOutputAllocator(outputTensorName);
        bool hasOutputAllocatorAfter = context.HasOutputAllocator(outputTensorName);

        bool hasTemporaryAllocatorBefore = context.HasTemporaryStorageAllocator;
        bool temporaryAllocatorInfoAvailable = context.TryGetTemporaryStorageAllocatorInterfaceInfo(out TensorRtInterfaceInfo temporaryAllocatorInfo, out string temporaryAllocatorInfoDiagnostic);
        bool temporaryAllocatorMetadataAvailable = context.TryGetTemporaryStorageAllocatorVersionedMetadata(
            out TensorRtVersionedInterfaceMetadata temporaryAllocatorMetadata,
            out string temporaryAllocatorMetadataDiagnostic);
        bool temporaryAllocatorCleared = clearedCallbackSnapshot.TemporaryStorageAllocatorCleared;
        bool directTemporaryAllocatorCleared = context.ClearTemporaryStorageAllocator();
        bool hasTemporaryAllocatorAfter = context.HasTemporaryStorageAllocator;

        bool hasDebugListenerBefore = context.HasDebugListener;
        bool debugListenerInfoAvailable = context.TryGetDebugListenerInterfaceInfo(out TensorRtInterfaceInfo debugListenerInfo, out string debugListenerInfoDiagnostic);
        bool debugListenerMetadataAvailable = context.TryGetDebugListenerVersionedMetadata(
            out TensorRtVersionedInterfaceMetadata debugListenerMetadata,
            out string debugListenerMetadataDiagnostic);
        bool debugListenerCleared = clearedCallbackSnapshot.DebugListenerCleared;
        bool directDebugListenerCleared = context.ClearDebugListener();
        bool hasDebugListenerAfter = context.HasDebugListener;

        bool hasProfilerBefore = context.HasNativeProfiler;
        context.ClearProfiler();
        bool hasProfilerAfter = context.HasNativeProfiler;
        bool hasManagedProfilerAfter = context.HasProfiler;

        return $"ErrorRecorder={hasErrorRecorderBefore}->{hasErrorRecorderAfter}/VersionedMetadata={FormatVersionedMetadata(errorRecorderMetadataAvailable, errorRecorderMetadata, errorRecorderMetadataDiagnostic)} " +
            $"OutputAllocator={hasOutputAllocatorBefore}->{hasOutputAllocatorAfter}/Cleared={outputAllocatorCleared}/DirectClear={directOutputAllocatorCleared}/Info={outputAllocatorInfoAvailable}:{FormatInterfaceInfo(outputAllocatorInfo, outputAllocatorInfoDiagnostic)}/VersionedMetadata={FormatVersionedMetadata(outputAllocatorMetadataAvailable, outputAllocatorMetadata, outputAllocatorMetadataDiagnostic)} " +
            $"TemporaryStorageAllocator={hasTemporaryAllocatorBefore}->{hasTemporaryAllocatorAfter}/Cleared={temporaryAllocatorCleared}/DirectClear={directTemporaryAllocatorCleared}/Info={temporaryAllocatorInfoAvailable}:{FormatInterfaceInfo(temporaryAllocatorInfo, temporaryAllocatorInfoDiagnostic)}/VersionedMetadata={FormatVersionedMetadata(temporaryAllocatorMetadataAvailable, temporaryAllocatorMetadata, temporaryAllocatorMetadataDiagnostic)} " +
            $"DebugListener={hasDebugListenerBefore}->{hasDebugListenerAfter}/Cleared={debugListenerCleared}/DirectClear={directDebugListenerCleared}/Info={debugListenerInfoAvailable}:{FormatInterfaceInfo(debugListenerInfo, debugListenerInfoDiagnostic)}/VersionedMetadata={FormatVersionedMetadata(debugListenerMetadataAvailable, debugListenerMetadata, debugListenerMetadataDiagnostic)} " +
            $"CallbackStateSnapshot={callbackSnapshot.LastOperation}->{clearedCallbackSnapshot.LastOperation}/OutputInfo={callbackSnapshot.OutputAllocatorInterfaceInfoAvailable}/TempInfo={callbackSnapshot.TemporaryStorageAllocatorInterfaceInfoAvailable}/DebugInfo={callbackSnapshot.DebugListenerInterfaceInfoAvailable} " +
            $"CallbackAllocatorSafeControlSummary={FormatCallbackAllocatorSafeControlSummary(safeControlSummary)} " +
            $"RuntimeDiagnosticSnapshot={runtimeSnapshot.OutputTensorName}/{runtimeSnapshot.HasErrorRecorder}/{runtimeSnapshot.HasOutputAllocator}/{runtimeSnapshot.IsOutputTensorAddressSet}/{runtimeSnapshot.HasTemporaryStorageAllocator}/{runtimeSnapshot.HasDebugListener}/{runtimeSnapshot.HasNativeProfiler}/{runtimeSnapshot.CallbackState.LastOperation}/{runtimeSnapshot.Diagnostics.Count} " +
            $"ExecutionContextRuntimeDiagnosticSummary={runtimeSummary.HasErrorRecorder}/{runtimeSummary.HasOutputAllocator}/{runtimeSummary.IsOutputTensorAddressSet}/{runtimeSummary.HasTemporaryStorageAllocator}/{runtimeSummary.HasDebugListener}/{runtimeSummary.HasNativeProfiler}/{runtimeSummary.CallbackStateLastStatus}/{runtimeSummary.DiagnosticCount} " +
            $"ProfilerNative={hasProfilerBefore}->{hasProfilerAfter}/Managed={hasManagedProfilerAfter}";
    }

    private static string FormatCallbackAllocatorSafeControlSummary(TensorRtExecutionContextCallbackAllocatorSafeControlSummary result)
    {
        return "execution-context-callback-allocator-safe-control-summary" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";Line={(int)result.Line}" +
            $";OutputTensorName={SanitizeSmokeValue(result.OutputTensorName)}" +
            $";HasOutputAllocator={result.HasOutputAllocator}" +
            $";HasTemporaryStorageAllocator={result.HasTemporaryStorageAllocator}" +
            $";HasDebugListener={result.HasDebugListener}" +
            $";OutputAllocatorInterfaceInfoAvailable={result.OutputAllocatorInterfaceInfoAvailable}" +
            $";TemporaryStorageAllocatorInterfaceInfoAvailable={result.TemporaryStorageAllocatorInterfaceInfoAvailable}" +
            $";DebugListenerInterfaceInfoAvailable={result.DebugListenerInterfaceInfoAvailable}" +
            $";CopiedInterfaceInfoCount={result.CopiedInterfaceInfoCount}" +
            $";DiagnosticCount={result.DiagnosticCount}" +
            $";PointerFreeSurfaceReady={result.PointerFreeSurfaceReady}" +
            $";CallbackInvocationAttempted={result.CallbackInvocationAttempted}" +
            $";IsRuntimeInvocationProofComplete={result.IsRuntimeInvocationProofComplete}" +
            $";OutputAllocatorDiagnostic={SanitizeSmokeValue(result.OutputAllocatorDiagnostic)}" +
            $";TemporaryStorageAllocatorDiagnostic={SanitizeSmokeValue(result.TemporaryStorageAllocatorDiagnostic)}" +
            $";DebugListenerDiagnostic={SanitizeSmokeValue(result.DebugListenerDiagnostic)}" +
            $";Summary={SanitizeSmokeValue(result.Summary)}";
    }

    private static string FormatInterfaceInfo(TensorRtInterfaceInfo interfaceInfo, string diagnostic)
    {
        return string.IsNullOrEmpty(interfaceInfo.Kind)
            ? diagnostic.Replace(' ', '_')
            : interfaceInfo.ToString();
    }

    private static string FormatVersionedMetadata(
        bool available,
        TensorRtVersionedInterfaceMetadata metadata,
        string diagnostic)
    {
        return available
            ? metadata.InterfaceInfo + ":" + metadata.ApiLanguage
            : "Unavailable:" + diagnostic.Replace(' ', '_');
    }

    private static void PrintDependencyProbe(TensorRtApiLine line)
    {
        TensorRtDependencyProbeReport dependencyProbe = TensorRtEnvironmentProbe.ProbeNativeDependencies(line);
        Console.WriteLine($"DependencyProbe Line={(int)line} BridgeInitialized={dependencyProbe.BridgeInitialized} Candidates={dependencyProbe.NativeBridgeCandidates.Count} Loaded={dependencyProbe.LoadedModuleCount} SearchPathCandidates={dependencyProbe.SearchPathCandidateCount} Diagnostics={dependencyProbe.Diagnostics.Count} Message={dependencyProbe.BridgeDiagnostic}");
    }

    private static void PrintSafeControlSurface(
        TensorRtApiLine line,
        bool enableRuntimeSmoke = false,
        string runtimePackageKey = "")
    {
        Console.WriteLine($"SafeControlSurfaceLine={(int)line}");
        Console.WriteLine("SafeControlSurface=allocator-debug-listener-safe-controls;callback-interface-info-safe-controls;execution-context-callback-state-snapshot;execution-context-callback-allocator-safe-control-summary;error-recorder-diagnostics-design-gate;dimension-expression-snapshot-design-gate;calibrator-metadata-design-gate;runtime-deserialization-boundary-precheck;runtime-deserialization-dependency-diagnostics;allocator-owner-dry-run-diagnostics;allocator-owner-native-dry-run-controls;allocator-owner-state-ledger-dry-run-controls;allocator-owner-internal-runtime-prototype;allocator-owner-ledger-safety-gate;output-allocator-internal-runtime-gate;output-allocator-callback-owner-design;output-allocator-attach-detach-design-gate;output-buffer-ownership-safety-gate;output-allocator-runtime-proof-precheck;debug-listener-callback-owner-design;debug-listener-attach-detach-design-gate;debug-listener-borrowed-tensor-safety-gate;debug-listener-attach-vtable-safety-gate;debug-listener-native-attach-nothrow-preflight;debug-listener-native-owner-address-design-gate;debug-listener-native-nothrow-vtable-design-gate;debug-listener-native-attach-entry-design-gate;debug-listener-native-detach-before-release-design-gate;debug-listener-native-owner-lifecycle-dry-run;debug-listener-native-attach-entry-runtime-scaffold;debug-listener-native-attach-entry-minimal-safety;debug-listener-native-owner-stable-identity;debug-listener-native-owner-noncopyable-storage;debug-listener-native-nothrow-destructor;debug-listener-native-owner-lifecycle-gate;debug-listener-native-attach-bridge-shape-gate;debug-listener-exception-status-mapping-gate;debug-listener-inflight-accounting-gate;debug-listener-native-nothrow-vtable-scaffold-gate;debug-listener-nothrow-vtable-callback-stub;debug-listener-borrowed-debug-tensor-metadata-runtime-gate;debug-listener-native-vtable-install-preflight;debug-listener-native-owner-vtable-install-experiment;debug-listener-runtime-proof-precheck;debug-listener-runtime-proof-attempt-preflight;debug-listener-real-non-null-attach-runtime-smoke;debug-listener-process-debug-tensor-callback-trampoline;callback-trampoline-shape;debug-listener-real-callback-runtime-proof;debug-listener-callback-proof-gap-report;callback-owner-closure-matrix;real-callback-runtime-blocked;attempted-no-invocation");
        Console.WriteLine("CallbackInterfaceInfoSafeControls=TryGetOutputAllocatorInterfaceInfo;TryGetTemporaryStorageAllocatorInterfaceInfo;TryGetDebugListenerInterfaceInfo");
        Console.WriteLine("ExecutionContextCallbackStateSnapshot=GetCallbackStateSnapshot;ClearCallbackState;TensorRtExecutionContextCallbackStateSnapshot;TryGetCallbackStateSnapshot;partial-state-diagnostics");
        Console.WriteLine("CallbackAllocatorSafeControlSummary=GetCallbackAllocatorSafeControlSummary;TensorRtExecutionContextCallbackAllocatorSafeControlSummary;copied-metadata-only;pointer-free;not-runtime-proof");
        Console.WriteLine("ExecutionContextRuntimeDiagnosticSnapshot=GetRuntimeDiagnosticSnapshot;TensorRtExecutionContextRuntimeDiagnosticSnapshot;pointer-free");
        Console.WriteLine("ExecutionContextRuntimeDiagnosticSummary=ToSummary;TensorRtExecutionContextRuntimeDiagnosticSummary;pointer-free;not-runtime-proof");
        Console.WriteLine("RuntimeDiagnosticSnapshot=GetDiagnosticSnapshot;TensorRtRuntimeDiagnosticSnapshot;pointer-free");
        Console.WriteLine("RuntimeDiagnosticSummary=ToSummary;TensorRtRuntimeDiagnosticSummary;pointer-free;not-runtime-proof");
        Console.WriteLine("ErrorRecorderSummary=ToSummary;TensorRtErrorRecorderSummary;copied-record-count;pointer-free;not-runtime-proof");
        Console.WriteLine("PluginCreatorV3MetadataDesignGate=EvaluateKnownSurface;TensorRtPluginCreatorV3MetadataDesignGateResult;design-gate-only;pointer-free;not-runtime-proof");
        Console.WriteLine("RefitterDiagnosticSnapshot=GetDiagnosticSnapshot;TensorRtRefitterDiagnosticSnapshot;copied-inventory;pointer-free");
        Console.WriteLine("RefitterDiagnosticSummary=ToSummary;TensorRtRefitterDiagnosticSummary;copied-inventory;pointer-free;not-runtime-proof");
        PrintErrorRecorderDiagnosticsDesignGate(line);
        PrintDimensionExpressionSnapshotDesignGate(line);
        PrintCalibratorMetadataDesignGate(line);
        PrintRuntimeDeserializationBoundaryPrecheck(line);
        PrintRuntimeDeserializationDependencyDiagnostics(line);
        PrintAllocatorOwnerDryRunDiagnostic(line);
        PrintAllocatorOwnerInternalRuntimePrototypeDiagnostic();
        PrintAllocatorOwnerLedgerSafetyGateDiagnostic(line);
        PrintOutputAllocatorInternalRuntimeGateDiagnostic();
        PrintOutputAllocatorCallbackOwnerDesignDiagnostic(line);
        PrintDebugListenerCallbackOwnerDesignDiagnostic(line, enableRuntimeSmoke, runtimePackageKey);
        PrintCallbackAllocatorReadinessSnapshot(line);
    }

    private static void PrintErrorRecorderDiagnosticsDesignGate(TensorRtApiLine line)
    {
        TensorRtErrorRecorderDiagnosticsDesignGateResult gate =
            TensorRtErrorRecorderDiagnosticsDesignGate.EvaluateKnownSurface(line);
        Console.WriteLine("ErrorRecorderDiagnosticsDesignGate=" + FormatErrorRecorderDiagnosticsDesignGate(gate));
    }

    private static void PrintDimensionExpressionSnapshotDesignGate(TensorRtApiLine line)
    {
        TensorRtDimensionExpressionSnapshotDesignGateResult gate =
            TensorRtDimensionExpressionSnapshotDesignGate.EvaluateKnownSurface(line);
        Console.WriteLine("DimensionExpressionSnapshotDesignGate=" + FormatDimensionExpressionSnapshotDesignGate(gate));
    }

    private static void PrintCalibratorMetadataDesignGate(TensorRtApiLine line)
    {
        TensorRtCalibratorMetadataDesignGateResult gate =
            TensorRtCalibratorMetadataDesignGate.EvaluateKnownSurface(line);
        Console.WriteLine("CalibratorMetadataDesignGate=" + FormatCalibratorMetadataDesignGate(gate));
    }

    private static void PrintRuntimeDeserializationBoundaryPrecheck(TensorRtApiLine line)
    {
        TensorRtRuntimeDeserializationBoundaryPrecheckResult precheck =
            TensorRtRuntimeDeserializationBoundaryPrecheck.EvaluateKnownSurface(line);
        Console.WriteLine("RuntimeDeserializationBoundaryPrecheck=" + FormatRuntimeDeserializationBoundaryPrecheck(precheck));
    }

    private static void PrintRuntimeDeserializationDependencyDiagnostics(TensorRtApiLine line)
    {
        TensorRtRuntimeDeserializationDependencyDiagnosticsResult diagnostics =
            TensorRtRuntimeDeserializationDependencyDiagnostics.EvaluateKnownSurface(line);
        Console.WriteLine("RuntimeDeserializationDependencyDiagnostics=" + FormatRuntimeDeserializationDependencyDiagnostics(diagnostics));
    }

    private static void PrintAllocatorOwnerDryRunDiagnostic(TensorRtApiLine line)
    {
        using TensorRtAllocatorCallbackOwner owner = new TensorRtAllocatorCallbackOwner(static request =>
            TensorRtAllocatorDryRunResult.Success($"dry-run:{request.Reason}:{request.Size}:{request.Alignment}"));

        TensorRtAllocatorDryRunResult result = owner.RunDryRunDiagnostic(new TensorRtAllocatorDryRunRequest(4096, 256, "smoke"));
        Console.WriteLine($"AllocatorOwnerDryRunDiagnostics={nameof(TensorRtAllocatorCallbackOwner)};{nameof(TensorRtAllocatorCallbackOwner.RunDryRunDiagnostic)};{nameof(TensorRtAllocatorCallbackOwner.CallbackInvocationCount)}={owner.CallbackInvocationCount};{nameof(TensorRtAllocatorCallbackOwner.CallbackFailureCount)}={owner.CallbackFailureCount};{nameof(TensorRtAllocatorCallbackOwner.IsAttached)}={owner.IsAttached};Result={result.Succeeded}:{result.Diagnostic}");

        try
        {
            TensorRtAllocatorNativeDryRunResult nativeResult = owner.RunNativeDryRunDiagnostic(line, new TensorRtAllocatorDryRunRequest(8192, 512, "smoke-native"));
            Console.WriteLine($"AllocatorOwnerNativeDryRunControls=allocator-owner-native-dry-run-controls;{nameof(TensorRtAllocatorCallbackOwner.RunNativeDryRunDiagnostic)};{nameof(TensorRtAllocatorNativeDryRunResult)};Line={(int)nativeResult.Line};Status={nativeResult.LastStatus};Invocations={nativeResult.InvocationCount};Failures={nativeResult.FailureCount};Attached={nativeResult.IsAttached};Size={nativeResult.LastSize};Alignment={nativeResult.LastAlignment};Succeeded={nativeResult.Succeeded};Diagnostic={nativeResult.Diagnostic}");

            TensorRtAllocatorOwnerStateDryRunResult stateResult = owner.RunNativeStateLedgerDryRunDiagnostic(line, new TensorRtAllocatorDryRunRequest(16384, 1024, "smoke-ledger"), "IGpuAllocator", 0);
            Console.WriteLine($"AllocatorOwnerStateLedgerDryRunControls=allocator-owner-state-ledger-dry-run-controls;{nameof(TensorRtAllocatorCallbackOwner.RunNativeStateLedgerDryRunDiagnostic)};{nameof(TensorRtAllocatorOwnerStateDryRunResult)};Line={(int)stateResult.Line};OwnerId={stateResult.OwnerId};Status={stateResult.LastStatus};Transitions={stateResult.StateTransitionCount};Allocations={stateResult.LedgerAllocationCount};Releases={stateResult.LedgerReleaseCount};Failures={stateResult.LedgerFailureCount};Attached={stateResult.IsAttached};Live={stateResult.HasLiveAllocation};Operation={stateResult.LastOperation};Succeeded={stateResult.Succeeded};Diagnostic={stateResult.Diagnostic}");
        }
        catch (Exception exception) when (IsSkippableEnvironmentException(exception))
        {
            Console.WriteLine($"AllocatorOwnerNativeDryRunControls=Skipped Reason={exception.GetType().Name}:{exception.Message.Replace(Environment.NewLine, " ")}");
            Console.WriteLine($"AllocatorOwnerStateLedgerDryRunControls=Skipped Reason={exception.GetType().Name}:{exception.Message.Replace(Environment.NewLine, " ")}");
        }
    }

    private static string FormatErrorRecorderDiagnosticsDesignGate(TensorRtErrorRecorderDiagnosticsDesignGateResult result)
    {
        return "error-recorder-diagnostics-design-gate" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";IsRuntimeExecutionEvidence={result.IsRuntimeExecutionEvidence}" +
            $";IsRuntimeExecutionProof={result.IsRuntimeExecutionProof}" +
            $";DiagnosticsKind={result.DiagnosticsKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";LineSupportsErrorRecorder={result.LineSupportsErrorRecorder}" +
            $";SnapshotTypeAvailable={result.SnapshotTypeAvailable}" +
            $";RuntimeSnapshotAvailable={result.RuntimeSnapshotAvailable}" +
            $";RefitterSnapshotAvailable={result.RefitterSnapshotAvailable}" +
            $";PresenceControlsAvailable={result.PresenceControlsAvailable}" +
            $";ClearControlsAvailable={result.ClearControlsAvailable}" +
            $";CopiedSnapshotObserved={result.CopiedSnapshotObserved}" +
            $";HasRecorder={result.HasRecorder}" +
            $";ErrorCount={result.ErrorCount}" +
            $";HasOverflowed={result.HasOverflowed}" +
            $";CopiedRecordCount={result.CopiedRecordCount}" +
            $";CopiedDiagnosticsReady={result.CopiedDiagnosticsReady}" +
            $";SnapshotRecordCopyReady={result.SnapshotRecordCopyReady}" +
            $";RecorderPointerExposed={result.RecorderPointerExposed}" +
            $";RecorderPointerProduced={result.RecorderPointerProduced}" +
            $";BorrowedRecorderPointerEscaped={result.BorrowedRecorderPointerEscaped}" +
            $";RefCountPublicOwnershipControl={result.RefCountPublicOwnershipControl}" +
            $";InterfaceInfoPublicOwnershipControl={result.InterfaceInfoPublicOwnershipControl}" +
            $";DirectRecorderOwnershipDeferred={result.DirectRecorderOwnershipDeferred}" +
            $";PointerFreeSurfaceReady={result.PointerFreeSurfaceReady}" +
            $";DesignGateReady={result.DesignGateReady}" +
            $";CanPromoteWithoutDesignGate={result.CanPromoteWithoutDesignGate}" +
            $";CanPromoteWithoutRuntimeProof={result.CanPromoteWithoutRuntimeProof}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";CanPromoteRuntimeProof={result.CanPromoteRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDimensionExpressionSnapshotDesignGate(TensorRtDimensionExpressionSnapshotDesignGateResult result)
    {
        return "dimension-expression-snapshot-design-gate" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";IsRuntimeExecutionEvidence={result.IsRuntimeExecutionEvidence}" +
            $";IsRuntimeExecutionProof={result.IsRuntimeExecutionProof}" +
            $";DiagnosticsKind={result.DiagnosticsKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";LineSupportsDimensionExpression={result.LineSupportsDimensionExpression}" +
            $";LineSupportsSizeTensor={result.LineSupportsSizeTensor}" +
            $";SnapshotTypeReady={result.SnapshotTypeReady}" +
            $";ConstantSnapshotCopyReady={result.ConstantSnapshotCopyReady}" +
            $";SizeTensorMetadataCopyReady={result.SizeTensorMetadataCopyReady}" +
            $";OwnerLifetimeKnown={result.OwnerLifetimeKnown}" +
            $";ExprBuilderOwnershipModeled={result.ExprBuilderOwnershipModeled}" +
            $";PluginShapeCallbackLifetimeModeled={result.PluginShapeCallbackLifetimeModeled}" +
            $";ExpressionPointerExposed={result.ExpressionPointerExposed}" +
            $";ExpressionPointerProduced={result.ExpressionPointerProduced}" +
            $";BorrowedExpressionPointerEscaped={result.BorrowedExpressionPointerEscaped}" +
            $";ExprBuilderPointerExposed={result.ExprBuilderPointerExposed}" +
            $";ExprBuilderCreationEnabled={result.ExprBuilderCreationEnabled}" +
            $";ExpressionNodePublicOwnershipControl={result.ExpressionNodePublicOwnershipControl}" +
            $";DirectDimensionExpressionRowsDeferred={result.DirectDimensionExpressionRowsDeferred}" +
            $";DirectExpressionBuilderRowsDeferred={result.DirectExpressionBuilderRowsDeferred}" +
            $";PointerFreeSurfaceReady={result.PointerFreeSurfaceReady}" +
            $";CopiedMetadataShapeReady={result.CopiedMetadataShapeReady}" +
            $";DesignGateReady={result.DesignGateReady}" +
            $";CanPromoteWithoutDesignGate={result.CanPromoteWithoutDesignGate}" +
            $";CanPromoteWithoutRuntimeProof={result.CanPromoteWithoutRuntimeProof}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";CanPromoteRuntimeProof={result.CanPromoteRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatCalibratorMetadataDesignGate(TensorRtCalibratorMetadataDesignGateResult result)
    {
        return "calibrator-metadata-design-gate" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";IsRuntimeExecutionEvidence={result.IsRuntimeExecutionEvidence}" +
            $";IsRuntimeExecutionProof={result.IsRuntimeExecutionProof}" +
            $";DiagnosticsKind={result.DiagnosticsKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";LineSupportsCalibrator={result.LineSupportsCalibrator}" +
            $";PresenceProbeAvailable={result.PresenceProbeAvailable}" +
            $";CopiedAlgorithmMetadataReady={result.CopiedAlgorithmMetadataReady}" +
            $";CopiedInterfaceInfoMetadataReady={result.CopiedInterfaceInfoMetadataReady}" +
            $";BatchCallbackOwnershipModeled={result.BatchCallbackOwnershipModeled}" +
            $";CacheBufferOwnershipModeled={result.CacheBufferOwnershipModeled}" +
            $";CalibratorPointerExposed={result.CalibratorPointerExposed}" +
            $";CalibratorPointerProduced={result.CalibratorPointerProduced}" +
            $";BorrowedCalibratorPointerEscaped={result.BorrowedCalibratorPointerEscaped}" +
            $";CallbackInvocationEnabled={result.CallbackInvocationEnabled}" +
            $";BatchBufferAccessEnabled={result.BatchBufferAccessEnabled}" +
            $";CacheBufferAccessEnabled={result.CacheBufferAccessEnabled}" +
            $";DirectCalibratorCallbackRowsDeferred={result.DirectCalibratorCallbackRowsDeferred}" +
            $";DirectCalibratorCacheRowsDeferred={result.DirectCalibratorCacheRowsDeferred}" +
            $";PointerFreeSurfaceReady={result.PointerFreeSurfaceReady}" +
            $";CopiedMetadataShapeReady={result.CopiedMetadataShapeReady}" +
            $";DesignGateReady={result.DesignGateReady}" +
            $";CanPromoteWithoutDesignGate={result.CanPromoteWithoutDesignGate}" +
            $";CanPromoteWithoutRuntimeProof={result.CanPromoteWithoutRuntimeProof}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";CanPromoteRuntimeProof={result.CanPromoteRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatRuntimeDeserializationBoundaryPrecheck(TensorRtRuntimeDeserializationBoundaryPrecheckResult result)
    {
        return "runtime-deserialization-boundary-precheck" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";IsRuntimeExecutionEvidence={result.IsRuntimeExecutionEvidence}" +
            $";IsRuntimeExecutionProof={result.IsRuntimeExecutionProof}" +
            $";DiagnosticsKind={result.DiagnosticsKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";LineSupportsRuntimeDeserialization={result.LineSupportsRuntimeDeserialization}" +
            $";LineSupportsDeserializeCudaEngineV2={result.LineSupportsDeserializeCudaEngineV2}" +
            $";ManagedByteArrayDeserializeReady={result.ManagedByteArrayDeserializeReady}" +
            $";ManagedArraySegmentDeserializeReady={result.ManagedArraySegmentDeserializeReady}" +
            $";ManagedReadOnlySpanDeserializeReady={result.ManagedReadOnlySpanDeserializeReady}" +
            $";ManagedStreamDeserializeReady={result.ManagedStreamDeserializeReady}" +
            $";ManagedFileDeserializeReady={result.ManagedFileDeserializeReady}" +
            $";HostMemoryDeserializeReady={result.HostMemoryDeserializeReady}" +
            $";SerializedBufferCopiedBeforeInterop={result.SerializedBufferCopiedBeforeInterop}" +
            $";PinnedBufferScopedToInteropCall={result.PinnedBufferScopedToInteropCall}" +
            $";BorrowedSerializedBufferEscaped={result.BorrowedSerializedBufferEscaped}" +
            $";HostMemoryHandleOwnedByWrapper={result.HostMemoryHandleOwnedByWrapper}" +
            $";EngineHandleOwnedByWrapper={result.EngineHandleOwnedByWrapper}" +
            $";EnginePointerExposed={result.EnginePointerExposed}" +
            $";EnginePointerProduced={result.EnginePointerProduced}" +
            $";DirectDeserializeCudaEngineRowsDeferred={result.DirectDeserializeCudaEngineRowsDeferred}" +
            $";DirectDeserializeCudaEngineV2RowsDeferred={result.DirectDeserializeCudaEngineV2RowsDeferred}" +
            $";LoadRuntimeDeferred={result.LoadRuntimeDeferred}" +
            $";PluginLibraryDependencyDiagnosticsReady={result.PluginLibraryDependencyDiagnosticsReady}" +
            $";PointerFreeSurfaceReady={result.PointerFreeSurfaceReady}" +
            $";ManagedDeserializeSurfaceReady={result.ManagedDeserializeSurfaceReady}" +
            $";SafeDeserializeBridgeReady={result.SafeDeserializeBridgeReady}" +
            $";PrecheckReady={result.PrecheckReady}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";CanPromoteWithoutRuntimeProof={result.CanPromoteWithoutRuntimeProof}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";CanPromoteRuntimeProof={result.CanPromoteRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatRuntimeDeserializationDependencyDiagnostics(TensorRtRuntimeDeserializationDependencyDiagnosticsResult result)
    {
        return "runtime-deserialization-dependency-diagnostics" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";IsRuntimeExecutionEvidence={result.IsRuntimeExecutionEvidence}" +
            $";IsRuntimeExecutionProof={result.IsRuntimeExecutionProof}" +
            $";DiagnosticsKind={result.DiagnosticsKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";PrecheckReady={result.PrecheckReady}" +
            $";ManagedDeserializeSurfaceReady={result.ManagedDeserializeSurfaceReady}" +
            $";SafeDeserializeBridgeReady={result.SafeDeserializeBridgeReady}" +
            $";FullPackageConsumerReportPresent={result.FullPackageConsumerReportPresent}" +
            $";FullPackageConsumerSmokeRequested={result.FullPackageConsumerSmokeRequested}" +
            $";FullPackageConsumerSmokeResult={result.FullPackageConsumerSmokeResult}" +
            $";DependencyProbeOnly={result.DependencyProbeOnly}" +
            $";BlockedByCudaDriver={result.BlockedByCudaDriver}" +
            $";DriverRuntimeMismatchClassified={result.DriverRuntimeMismatchClassified}" +
            $";PackageConsumerEvidenceClassification={result.PackageConsumerEvidenceClassification}" +
            $";PluginLibraryDependencyDiagnosticsComplete={result.PluginLibraryDependencyDiagnosticsComplete}" +
            $";LoadRuntimeOwnershipModeled={result.LoadRuntimeOwnershipModeled}" +
            $";RuntimeProofBlockerCategory={result.RuntimeProofBlockerCategory}" +
            $";RuntimeProofOwnerActionRequired={result.RuntimeProofOwnerActionRequired}" +
            $";ExternalRuntimeProofRequired={result.ExternalRuntimeProofRequired}" +
            $";PackageConsumerRuntimeProofPresent={result.PackageConsumerRuntimeProofPresent}" +
            $";DirectDeserializeCudaEngineRowsDeferred={result.DirectDeserializeCudaEngineRowsDeferred}" +
            $";DirectDeserializeCudaEngineV2RowsDeferred={result.DirectDeserializeCudaEngineV2RowsDeferred}" +
            $";LoadRuntimeDeferred={result.LoadRuntimeDeferred}" +
            $";PointerFreeSurfaceReady={result.PointerFreeSurfaceReady}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";CanPromoteRuntimeProof={result.CanPromoteRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";WhyNotRuntimeProof={SanitizeSmokeValue(result.WhyNotRuntimeProof)}" +
            $";NextOwnerAction={SanitizeSmokeValue(result.NextOwnerAction)}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static void PrintAllocatorOwnerInternalRuntimePrototypeDiagnostic()
    {
        using TensorRtAllocatorCallbackOwner owner = new TensorRtAllocatorCallbackOwner(static request =>
            TensorRtAllocatorDryRunResult.Success($"prototype:{request.Reason}:{request.Size}:{request.Alignment}"));

        TensorRtAllocatorCallbackOwnerSnapshot invoke =
            owner.RunLifecycleDiagnostic(new TensorRtAllocatorDryRunRequest(4096, 256, "smoke-prototype"));
        Console.WriteLine("AllocatorOwnerInternalRuntimePrototype=" + FormatInternalRuntimePrototype(invoke));

        TensorRtAllocatorCallbackOwnerSnapshot beforeDispose = owner.GetSnapshot("pre-dispose");
        Console.WriteLine("AllocatorOwnerInternalRuntimePrototypePreDispose=" + FormatInternalRuntimePrototype(beforeDispose));

        owner.Dispose();
        TensorRtAllocatorCallbackOwnerSnapshot afterDispose = owner.GetSnapshot("post-dispose");
        Console.WriteLine("AllocatorOwnerInternalRuntimePrototypeDispose=" + FormatInternalRuntimePrototype(afterDispose));

        using TensorRtAllocatorCallbackOwner throwingOwner = new TensorRtAllocatorCallbackOwner(static _ =>
            throw new InvalidOperationException("smoke prototype failure"));
        TensorRtAllocatorCallbackOwnerSnapshot failure =
            throwingOwner.RunLifecycleDiagnostic(new TensorRtAllocatorDryRunRequest(8192, 512, "smoke-prototype-throw"));
        Console.WriteLine("AllocatorOwnerInternalRuntimePrototypeException=" + FormatInternalRuntimePrototype(failure));
    }

    private static void PrintAllocatorOwnerLedgerSafetyGateDiagnostic(TensorRtApiLine line)
    {
        using TensorRtAllocatorCallbackOwner owner = new TensorRtAllocatorCallbackOwner(static request =>
            TensorRtAllocatorDryRunResult.Success($"ledger-safety:{request.Reason}:{request.Size}:{request.Alignment}"));

        TensorRtAllocatorLedgerSafetyGateResult gate = TensorRtAllocatorLedgerSafetyGate.Evaluate(
            owner,
            line,
            new TensorRtAllocatorDryRunRequest(16384, 1024, "smoke-ledger-safety"),
            "IGpuAllocator",
            0UL);
        Console.WriteLine("AllocatorOwnerLedgerSafetyGate=" + FormatAllocatorOwnerLedgerSafetyGate(gate));

        owner.Dispose();
        TensorRtAllocatorLedgerSafetyGateResult dispose = TensorRtAllocatorLedgerSafetyGate.GetSnapshot(owner, line, "post-dispose");
        Console.WriteLine("AllocatorOwnerLedgerSafetyGateDispose=" + FormatAllocatorOwnerLedgerSafetyGate(dispose));
    }

    private static string FormatAllocatorOwnerLedgerSafetyGate(TensorRtAllocatorLedgerSafetyGateResult result)
    {
        return "allocator-owner-ledger-safety-gate" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";OwnerId={result.OwnerId}" +
            $";Operation={SanitizeSmokeValue(result.Operation)}" +
            $";LastStatus={result.LastStatus}" +
            $";InternalPrototypeStatus={result.InternalPrototypeStatus}" +
            $";NativeLedgerStatus={result.NativeLedgerStatus}" +
            $";NativeLedgerAvailable={result.NativeLedgerAvailable}" +
            $";InvocationCount={result.InvocationCount}" +
            $";FailureCount={result.FailureCount}" +
            $";InFlightCallbackCount={result.InFlightCallbackCount}" +
            $";MaxInFlightCallbackCount={result.MaxInFlightCallbackCount}" +
            $";ReleaseHookCount={result.ReleaseHookCount}" +
            $";CallbackStatePinned={result.CallbackStatePinned}" +
            $";DelegatePinned={result.DelegatePinned}" +
            $";DisposeRequested={result.DisposeRequested}" +
            $";ManagedKeepAliveReady={result.ManagedKeepAliveReady}" +
            $";DisposeReleaseReady={result.DisposeReleaseReady}" +
            $";NativeLedgerDesignReady={result.NativeLedgerDesignReady}" +
            $";StateTransitionCount={result.StateTransitionCount}" +
            $";LedgerAllocationCount={result.LedgerAllocationCount}" +
            $";LedgerReleaseCount={result.LedgerReleaseCount}" +
            $";LedgerFailureCount={result.LedgerFailureCount}" +
            $";HasLiveAllocation={result.HasLiveAllocation}" +
            $";PointerFreeSurfaceReady={result.PointerFreeSurfaceReady}" +
            $";LineSpecificAttachDetachReady={result.LineSpecificAttachDetachReady}" +
            $";DevicePointerLedgerRuntimeReady={result.DevicePointerLedgerRuntimeReady}" +
            $";StreamLifetimeReady={result.StreamLifetimeReady}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatInternalRuntimePrototype(TensorRtAllocatorCallbackOwnerSnapshot result)
    {
        return "allocator-owner-internal-runtime-prototype" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";OwnerId={result.OwnerId}" +
            $";Operation={result.Operation}" +
            $";LastStatus={result.LastStatus}" +
            $";InvocationCount={result.InvocationCount}" +
            $";FailureCount={result.FailureCount}" +
            $";InFlightCallbackCount={result.InFlightCallbackCount}" +
            $";MaxInFlightCallbackCount={result.MaxInFlightCallbackCount}" +
            $";ActivePrototypeCallCount={result.ActivePrototypeCallCount}" +
            $";ReleaseHookCount={result.ReleaseHookCount}" +
            $";CallbackStatePinned={result.CallbackStatePinned}" +
            $";DelegatePinned={result.DelegatePinned}" +
            $";DisposeRequested={result.DisposeRequested}" +
            $";IsAttached={result.IsAttached}" +
            $";DevicePointerExposed={result.DevicePointerExposed}" +
            $";DevicePointerProduced={result.DevicePointerProduced}" +
            $";BorrowedPointerEscaped={result.BorrowedPointerEscaped}" +
            $";ManagedKeepAliveReady={result.ManagedKeepAliveReady}" +
            $";DisposeReleaseReady={result.DisposeReleaseReady}" +
            $";PointerFreeSurfaceReady={result.PointerFreeSurfaceReady}" +
            $";Succeeded={result.Succeeded}" +
            $";LastDiagnostic={SanitizeSmokeValue(result.LastDiagnostic)}" +
            $";ReleaseDiagnostic={SanitizeSmokeValue(result.ReleaseDiagnostic)}";
    }

    private static void PrintOutputAllocatorInternalRuntimeGateDiagnostic()
    {
        object gate = CreateOutputAllocatorRuntimeGate();
        try
        {
            object notifyRequest = CreateOutputAllocatorRuntimeGateRequest(
                "smoke_output",
                0UL,
                1UL,
                new long[] { 1, 3, 224, 224 },
                "smoke-notify-shape",
                false);
            object notify = InvokeOutputAllocatorRuntimeGate(gate, "RunInternalNotifyShapeRuntimeGate", notifyRequest);
            Console.WriteLine("OutputAllocatorInternalRuntimeGateNotifyShape=" + FormatOutputAllocatorRuntimeGate(notify));

            object reallocateRequest = CreateOutputAllocatorRuntimeGateRequest(
                "smoke_output",
                4096UL,
                256UL,
                new long[] { 1, 1000 },
                "smoke-reallocate-output",
                true);
            object reallocate = InvokeOutputAllocatorRuntimeGate(gate, "RunInternalReallocateOutputRuntimeGate", reallocateRequest);
            Console.WriteLine("OutputAllocatorInternalRuntimeGateReallocateOutput=" + FormatOutputAllocatorRuntimeGate(reallocate));

            object preDispose = GetOutputAllocatorRuntimeGateSnapshot(gate, "pre-dispose");
            Console.WriteLine("OutputAllocatorInternalRuntimeGatePreDispose=" + FormatOutputAllocatorRuntimeGate(preDispose));

            ((IDisposable)gate).Dispose();
            object postDispose = GetOutputAllocatorRuntimeGateSnapshot(gate, "post-dispose");
            Console.WriteLine("OutputAllocatorInternalRuntimeGateDispose=" + FormatOutputAllocatorRuntimeGate(postDispose));
        }
        finally
        {
            ((IDisposable)gate).Dispose();
        }

        object throwingGate = CreateOutputAllocatorRuntimeGate();
        try
        {
            object throwingRequest = CreateOutputAllocatorRuntimeGateRequest(
                "smoke_output",
                8192UL,
                512UL,
                new long[] { 1, 64 },
                "throw",
                false);
            object failure = InvokeOutputAllocatorRuntimeGate(throwingGate, "RunInternalReallocateOutputRuntimeGate", throwingRequest);
            Console.WriteLine("OutputAllocatorInternalRuntimeGateException=" + FormatOutputAllocatorRuntimeGate(failure));
        }
        finally
        {
            ((IDisposable)throwingGate).Dispose();
        }
    }

    private static object CreateOutputAllocatorRuntimeGate()
    {
        Type gateType = typeof(TensorRtAllocatorCallbackOwner).Assembly.GetType("JYPPX.TensorRtSharp.TensorRtOutputAllocatorRuntimeGate")
            ?? throw new MissingMemberException("TensorRtOutputAllocatorRuntimeGate");
        return Activator.CreateInstance(gateType, nonPublic: true)
            ?? throw new InvalidOperationException("Output allocator runtime gate returned null.");
    }

    private static object CreateOutputAllocatorRuntimeGateRequest(
        string tensorName,
        ulong requestedSize,
        ulong alignment,
        long[] shapeDimensions,
        string reason,
        bool hasCurrentMemory)
    {
        Type requestType = typeof(TensorRtAllocatorCallbackOwner).Assembly.GetType("JYPPX.TensorRtSharp.TensorRtOutputAllocatorRuntimeGateRequest")
            ?? throw new MissingMemberException("TensorRtOutputAllocatorRuntimeGateRequest");
        object[] arguments = new object[] { tensorName, requestedSize, alignment, shapeDimensions, reason, hasCurrentMemory };
        return Activator.CreateInstance(
            requestType,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
            binder: null,
            args: arguments,
            culture: CultureInfo.InvariantCulture)
            ?? throw new InvalidOperationException("Output allocator runtime gate request returned null.");
    }

    private static object InvokeOutputAllocatorRuntimeGate(object gate, string methodName, object request)
    {
        MethodInfo? method = gate.GetType().GetMethod(methodName, BindingFlags.Instance | BindingFlags.NonPublic);
        if (method == null)
        {
            throw new MissingMethodException(gate.GetType().Name, methodName);
        }

        return method.Invoke(gate, new[] { request }) ?? throw new InvalidOperationException("Output allocator runtime gate returned null.");
    }

    private static object GetOutputAllocatorRuntimeGateSnapshot(object gate, string operation)
    {
        MethodInfo? method = gate.GetType().GetMethod("GetInternalRuntimeGateSnapshot", BindingFlags.Instance | BindingFlags.NonPublic);
        if (method == null)
        {
            throw new MissingMethodException(gate.GetType().Name, "GetInternalRuntimeGateSnapshot");
        }

        return method.Invoke(gate, new object[] { operation }) ?? throw new InvalidOperationException("Output allocator runtime gate snapshot returned null.");
    }

    private static string FormatOutputAllocatorRuntimeGate(object result)
    {
        return "output-allocator-internal-runtime-gate" +
            $";EvidenceKind={ReadPrototypeProperty(result, "EvidenceKind")}" +
            $";RealCallbackRuntime={ReadPrototypeProperty(result, "RealCallbackRuntime")}" +
            $";CallbackKind={ReadPrototypeProperty(result, "CallbackKind")}" +
            $";OwnerId={ReadPrototypeProperty(result, "OwnerId")}" +
            $";Operation={ReadPrototypeProperty(result, "Operation")}" +
            $";TensorName={SanitizeSmokeValue(ReadPrototypeProperty(result, "TensorName"))}" +
            $";RequestedSize={ReadPrototypeProperty(result, "RequestedSize")}" +
            $";Alignment={ReadPrototypeProperty(result, "Alignment")}" +
            $";ShapeRank={ReadPrototypeProperty(result, "ShapeRank")}" +
            $";ShapeSummary={SanitizeSmokeValue(ReadPrototypeProperty(result, "ShapeSummary"))}" +
            $";HasCurrentMemory={ReadPrototypeProperty(result, "HasCurrentMemory")}" +
            $";LastStatus={ReadPrototypeProperty(result, "LastStatus")}" +
            $";InvocationCount={ReadPrototypeProperty(result, "InvocationCount")}" +
            $";NotifyShapeCount={ReadPrototypeProperty(result, "NotifyShapeCount")}" +
            $";ReallocateOutputCount={ReadPrototypeProperty(result, "ReallocateOutputCount")}" +
            $";FailureCount={ReadPrototypeProperty(result, "FailureCount")}" +
            $";InFlightCallbackCount={ReadPrototypeProperty(result, "InFlightCallbackCount")}" +
            $";MaxInFlightCallbackCount={ReadPrototypeProperty(result, "MaxInFlightCallbackCount")}" +
            $";ActiveGateCallCount={ReadPrototypeProperty(result, "ActiveGateCallCount")}" +
            $";ReleaseHookCount={ReadPrototypeProperty(result, "ReleaseHookCount")}" +
            $";CallbackStatePinned={ReadPrototypeProperty(result, "CallbackStatePinned")}" +
            $";DelegatePinned={ReadPrototypeProperty(result, "DelegatePinned")}" +
            $";DisposeRequested={ReadPrototypeProperty(result, "DisposeRequested")}" +
            $";IsAttached={ReadPrototypeProperty(result, "IsAttached")}" +
            $";OutputBufferPointerExposed={ReadPrototypeProperty(result, "OutputBufferPointerExposed")}" +
            $";OutputBufferPointerProduced={ReadPrototypeProperty(result, "OutputBufferPointerProduced")}" +
            $";Succeeded={ReadPrototypeProperty(result, "Succeeded")}" +
            $";LastDiagnostic={SanitizeSmokeValue(ReadPrototypeProperty(result, "LastDiagnostic"))}" +
            $";ReleaseDiagnostic={SanitizeSmokeValue(ReadPrototypeProperty(result, "ReleaseDiagnostic"))}";
    }

    private static void PrintOutputAllocatorCallbackOwnerDesignDiagnostic(TensorRtApiLine line)
    {
        using TensorRtOutputAllocatorCallbackOwner owner = new TensorRtOutputAllocatorCallbackOwner();
        TensorRtOutputAllocatorCallbackRequest request = new TensorRtOutputAllocatorCallbackRequest(
            "smoke_output",
            4096UL,
            256UL,
            new long[] { 1, 1000 },
            "smoke-output-allocator-owner-design",
            true);

        TensorRtOutputAllocatorCallbackOwnerSnapshot diagnostic = owner.RunDesignDiagnostic(line, request, 0UL);
        Console.WriteLine("OutputAllocatorCallbackOwnerDesign=" + FormatOutputAllocatorCallbackOwnerDesign(diagnostic));

        TensorRtOutputAllocatorCallbackOwnerSnapshot preDispose = owner.GetSnapshot("pre-dispose");
        Console.WriteLine("OutputAllocatorCallbackOwnerDesignPreDispose=" + FormatOutputAllocatorCallbackOwnerDesign(preDispose));

        owner.Dispose();
        TensorRtOutputAllocatorCallbackOwnerSnapshot postDispose = owner.GetSnapshot("post-dispose");
        Console.WriteLine("OutputAllocatorCallbackOwnerDesignDispose=" + FormatOutputAllocatorCallbackOwnerDesign(postDispose));

        TensorRtOutputAllocatorAttachDetachDesignGateResult attachDetachGate = TensorRtOutputAllocatorAttachDetachDesignGate.Evaluate(postDispose);
        Console.WriteLine("OutputAllocatorAttachDetachDesignGate=" + FormatOutputAllocatorAttachDetachDesignGate(attachDetachGate));

        TensorRtOutputBufferOwnershipSafetyGateResult ownershipGate = TensorRtOutputBufferOwnershipSafetyGate.Evaluate(postDispose, attachDetachGate);
        Console.WriteLine("OutputBufferOwnershipSafetyGate=" + FormatOutputBufferOwnershipSafetyGate(ownershipGate));

        TensorRtOutputAllocatorRuntimeProofPrecheckResult precheck = TensorRtOutputAllocatorRuntimeProofPrecheck.Evaluate(postDispose, attachDetachGate, ownershipGate);
        Console.WriteLine("OutputAllocatorRuntimeProofPrecheck=" + FormatOutputAllocatorRuntimeProofPrecheck(precheck));
    }

    private static string FormatOutputAllocatorCallbackOwnerDesign(TensorRtOutputAllocatorCallbackOwnerSnapshot result)
    {
        return "output-allocator-callback-owner-design" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";OwnerId={result.OwnerId}" +
            $";NativeOwnerId={result.NativeOwnerId}" +
            $";Operation={result.Operation}" +
            $";TensorName={SanitizeSmokeValue(result.TensorName)}" +
            $";RequestedSize={result.RequestedSize}" +
            $";Alignment={result.Alignment}" +
            $";ShapeRank={result.ShapeRank}" +
            $";ShapeSummary={SanitizeSmokeValue(result.ShapeSummary)}" +
            $";HasCurrentMemory={result.HasCurrentMemory}" +
            $";LastStatus={result.LastStatus}" +
            $";RuntimeGateStatus={result.RuntimeGateStatus}" +
            $";NativeLedgerStatus={result.NativeLedgerStatus}" +
            $";NativeLedgerAvailable={result.NativeLedgerAvailable}" +
            $";InvocationCount={result.InvocationCount}" +
            $";NotifyShapeCount={result.NotifyShapeCount}" +
            $";ReallocateOutputCount={result.ReallocateOutputCount}" +
            $";FailureCount={result.FailureCount}" +
            $";InFlightCallbackCount={result.InFlightCallbackCount}" +
            $";MaxInFlightCallbackCount={result.MaxInFlightCallbackCount}" +
            $";ReleaseHookCount={result.ReleaseHookCount}" +
            $";CallbackStatePinned={result.CallbackStatePinned}" +
            $";DelegatePinned={result.DelegatePinned}" +
            $";DisposeRequested={result.DisposeRequested}" +
            $";IsAttached={result.IsAttached}" +
            $";StateTransitionCount={result.StateTransitionCount}" +
            $";LedgerAllocationCount={result.LedgerAllocationCount}" +
            $";LedgerReleaseCount={result.LedgerReleaseCount}" +
            $";LedgerFailureCount={result.LedgerFailureCount}" +
            $";HasLiveAllocation={result.HasLiveAllocation}" +
            $";OutputBufferPointerExposed={result.OutputBufferPointerExposed}" +
            $";OutputBufferPointerProduced={result.OutputBufferPointerProduced}" +
            $";Succeeded={result.Succeeded}" +
            $";NativeLastOperation={SanitizeSmokeValue(result.NativeLastOperation)}" +
            $";LastDiagnostic={SanitizeSmokeValue(result.LastDiagnostic)}" +
            $";NativeLedgerDiagnostic={SanitizeSmokeValue(result.NativeLedgerDiagnostic)}" +
            $";ReleaseDiagnostic={SanitizeSmokeValue(result.ReleaseDiagnostic)}";
    }

    private static string FormatOutputAllocatorAttachDetachDesignGate(TensorRtOutputAllocatorAttachDetachDesignGateResult result)
    {
        return "output-allocator-attach-detach-design-gate" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";LineSupportsOutputAllocator={result.LineSupportsOutputAllocator}" +
            $";OwnerDesignReady={result.OwnerDesignReady}" +
            $";ManagedOwnerStateMachineReady={result.ManagedOwnerStateMachineReady}" +
            $";PointerFreeSurfaceReady={result.PointerFreeSurfaceReady}" +
            $";AttachControlAvailable={result.AttachControlAvailable}" +
            $";DetachClearControlAvailable={result.DetachClearControlAvailable}" +
            $";LineSpecificAttachDetachReady={result.LineSpecificAttachDetachReady}" +
            $";StableNativeOwnerAddressReady={result.StableNativeOwnerAddressReady}" +
            $";NoThrowNativeVTableReady={result.NoThrowNativeVTableReady}" +
            $";NativeVTableReady={result.NativeVTableReady}" +
            $";OutputBufferOwnershipRuntimeReady={result.OutputBufferOwnershipRuntimeReady}" +
            $";DesignGateReady={result.DesignGateReady}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatOutputBufferOwnershipSafetyGate(TensorRtOutputBufferOwnershipSafetyGateResult result)
    {
        return "output-buffer-ownership-safety-gate" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";TensorName={SanitizeSmokeValue(result.TensorName)}" +
            $";RequestedSize={result.RequestedSize}" +
            $";Alignment={result.Alignment}" +
            $";ShapeRank={result.ShapeRank}" +
            $";HasCurrentMemory={result.HasCurrentMemory}" +
            $";NotifyShapeCount={result.NotifyShapeCount}" +
            $";ReallocateOutputCount={result.ReallocateOutputCount}" +
            $";AttachDetachDesignGateReady={result.AttachDetachDesignGateReady}" +
            $";OwnerDesignReady={result.OwnerDesignReady}" +
            $";PointerFreeSurfaceReady={result.PointerFreeSurfaceReady}" +
            $";CopiedCurrentMemoryMetadataReady={result.CopiedCurrentMemoryMetadataReady}" +
            $";CopiedShapeMetadataReady={result.CopiedShapeMetadataReady}" +
            $";CopiedRequestMetadataReady={result.CopiedRequestMetadataReady}" +
            $";SafetyGateReady={result.SafetyGateReady}" +
            $";OutputBufferOwnershipRuntimeReady={result.OutputBufferOwnershipRuntimeReady}" +
            $";CurrentMemoryReusePolicyReady={result.CurrentMemoryReusePolicyReady}" +
            $";BorrowedPointerEscapeBlocked={result.BorrowedPointerEscapeBlocked}" +
            $";OwnedDevicePointerReleasePolicyReady={result.OwnedDevicePointerReleasePolicyReady}" +
            $";ShapeNotificationOrderingReady={result.ShapeNotificationOrderingReady}" +
            $";ReallocateOutputRuntimeReady={result.ReallocateOutputRuntimeReady}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatOutputAllocatorRuntimeProofPrecheck(TensorRtOutputAllocatorRuntimeProofPrecheckResult result)
    {
        return "output-allocator-runtime-proof-precheck" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";LineSupportsOutputAllocator={result.LineSupportsOutputAllocator}" +
            $";OwnerDesignReady={result.OwnerDesignReady}" +
            $";NativeLedgerDesignReady={result.NativeLedgerDesignReady}" +
            $";DisposeReleaseReady={result.DisposeReleaseReady}" +
            $";PointerFreeSurfaceReady={result.PointerFreeSurfaceReady}" +
            $";AttachDetachDesignGateReady={result.AttachDetachDesignGateReady}" +
            $";AttachControlAvailable={result.AttachControlAvailable}" +
            $";DetachClearControlAvailable={result.DetachClearControlAvailable}" +
            $";ManagedOwnerStateMachineReady={result.ManagedOwnerStateMachineReady}" +
            $";LineSpecificAttachDetachReady={result.LineSpecificAttachDetachReady}" +
            $";StableNativeOwnerAddressReady={result.StableNativeOwnerAddressReady}" +
            $";NoThrowNativeVTableReady={result.NoThrowNativeVTableReady}" +
            $";NativeVTableReady={result.NativeVTableReady}" +
            $";DevicePointerLedgerRuntimeReady={result.DevicePointerLedgerRuntimeReady}" +
            $";StreamLifetimeReady={result.StreamLifetimeReady}" +
            $";OutputBufferOwnershipSafetyGateReady={result.OutputBufferOwnershipSafetyGateReady}" +
            $";OutputBufferOwnershipRuntimeReady={result.OutputBufferOwnershipRuntimeReady}" +
            $";CurrentMemoryReusePolicyReady={result.CurrentMemoryReusePolicyReady}" +
            $";BorrowedPointerEscapeBlocked={result.BorrowedPointerEscapeBlocked}" +
            $";OwnedDevicePointerReleasePolicyReady={result.OwnedDevicePointerReleasePolicyReady}" +
            $";ShapeNotificationOrderingReady={result.ShapeNotificationOrderingReady}" +
            $";ReallocateOutputRuntimeReady={result.ReallocateOutputRuntimeReady}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static void PrintCallbackAllocatorReadinessSnapshot(TensorRtApiLine line)
    {
        using TensorRtAllocatorCallbackOwner allocatorOwner = new TensorRtAllocatorCallbackOwner(static request =>
            TensorRtAllocatorDryRunResult.Success($"readiness:{request.Reason}:{request.Size}:{request.Alignment}"));
        TensorRtAllocatorLedgerSafetyGateResult allocatorGate = TensorRtAllocatorLedgerSafetyGate.Evaluate(
            allocatorOwner,
            line,
            new TensorRtAllocatorDryRunRequest(32768, 256, "smoke-readiness-allocator"),
            "IGpuAllocator",
            0UL);

        using TensorRtOutputAllocatorCallbackOwner outputOwner = new TensorRtOutputAllocatorCallbackOwner();
        TensorRtOutputAllocatorCallbackRequest outputRequest = new TensorRtOutputAllocatorCallbackRequest(
            "smoke_readiness_output",
            4096UL,
            256UL,
            new long[] { 1, 1000 },
            "smoke-readiness-output",
            hasCurrentMemory: true);
        _ = outputOwner.RunDesignDiagnostic(line, outputRequest, 0UL);
        outputOwner.Dispose();
        TensorRtOutputAllocatorRuntimeProofPrecheckResult outputPrecheck =
            TensorRtOutputAllocatorRuntimeProofPrecheck.Evaluate(outputOwner.GetSnapshot("post-dispose-readiness"));

        using TensorRtDebugListenerCallbackOwner debugOwner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest debugRequest = new TensorRtDebugListenerCallbackRequest(
            "smoke_readiness_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "smoke-readiness-debug-listener",
            isInput: true,
            isExecutionTensor: true);
        _ = debugOwner.RunDesignDiagnostic(line, debugRequest);
        debugOwner.Dispose();
        TensorRtDebugListenerRuntimeProofPrecheckResult debugPrecheck =
            TensorRtDebugListenerRuntimeProofPrecheck.Evaluate(debugOwner.GetSnapshot("post-dispose-readiness"));

        TensorRtCallbackAllocatorReadinessSnapshot readiness = TensorRtCallbackAllocatorReadiness.Evaluate(
            allocatorGate,
            outputPrecheck,
            debugPrecheck);
        Console.WriteLine("CallbackAllocatorReadinessSnapshot=" + FormatCallbackAllocatorReadinessSnapshot(readiness));

        TensorRtStreamIoInterfaceInfoDesignGateResult streamGate =
            TensorRtStreamIoInterfaceInfoDesignGate.EvaluateKnownSurface(line);
        TensorRtCallbackOwnerClosureMatrixResult ownerClosureMatrix =
            TensorRtCallbackOwnerClosureMatrix.Evaluate(
                allocatorGate,
                outputPrecheck,
                debugPrecheck,
                streamGate);
        Console.WriteLine("CallbackOwnerClosureMatrix=" + FormatCallbackOwnerClosureMatrix(ownerClosureMatrix));
    }

    private static string FormatCallbackAllocatorReadinessSnapshot(TensorRtCallbackAllocatorReadinessSnapshot result)
    {
        return "callback-allocator-readiness-snapshot" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";LoggerCallbackReady={result.LoggerCallbackReady}" +
            $";ProfilerCallbackReady={result.ProfilerCallbackReady}" +
            $";ProgressMonitorCallbackReady={result.ProgressMonitorCallbackReady}" +
            $";AllocatorOwnerDryRunReady={result.AllocatorOwnerDryRunReady}" +
            $";AllocatorLedgerSafetyGateReady={result.AllocatorLedgerSafetyGateReady}" +
            $";OutputAllocatorOwnerDesignReady={result.OutputAllocatorOwnerDesignReady}" +
            $";OutputAllocatorRuntimeGateReady={result.OutputAllocatorRuntimeGateReady}" +
            $";DebugListenerOwnerDesignReady={result.DebugListenerOwnerDesignReady}" +
            $";DebugListenerNoThrowVTableGateReady={result.DebugListenerNoThrowVTableGateReady}" +
            $";DebugListenerRuntimeProofPrecheckReady={result.DebugListenerRuntimeProofPrecheckReady}" +
            $";RealCallbackInvocationProofReady={result.RealCallbackInvocationProofReady}" +
            $";IsPublishSafeForManagedCallbacks={result.IsPublishSafeForManagedCallbacks}" +
            $";IsRuntimeInvocationProofComplete={result.IsRuntimeInvocationProofComplete}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";BlockedReasonCount={result.BlockedReasonCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Summary={SanitizeSmokeValue(result.Summary)}";
    }

    private static string FormatCallbackOwnerClosureMatrix(TensorRtCallbackOwnerClosureMatrixResult result)
    {
        List<string> rowSummaries = new List<string>();
        foreach (TensorRtCallbackOwnerClosureMatrixRow row in result.Rows)
        {
            rowSummaries.Add(
                row.OwnerFamily +
                ":" +
                row.ReadyClosureColumnCount +
                "/" +
                row.TotalClosureColumnCount +
                ":closure=" +
                row.ClosureReady +
                ":attempt=" +
                row.CanAttemptRuntimeProof +
                ":proof=" +
                row.PackageConsumerRuntimeProofReady);
        }

        return "callback-owner-closure-matrix" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";FamilyCount={result.FamilyCount}" +
            $";DesignGateReadyFamilyCount={result.DesignGateReadyFamilyCount}" +
            $";ClosureReadyFamilyCount={result.ClosureReadyFamilyCount}" +
            $";RuntimeProofAttemptReadyFamilyCount={result.RuntimeProofAttemptReadyFamilyCount}" +
            $";PackageConsumerRuntimeProofReadyFamilyCount={result.PackageConsumerRuntimeProofReadyFamilyCount}" +
            $";PointerFreeSurfaceReady={result.PointerFreeSurfaceReady}" +
            $";AllFamiliesClosureReady={result.AllFamiliesClosureReady}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";Rows={SanitizeSmokeValue(string.Join("|", rowSummaries))}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Summary={SanitizeSmokeValue(result.Summary)}";
    }

    private static void PrintDebugListenerCallbackOwnerDesignDiagnostic()
    {
        PrintDebugListenerCallbackOwnerDesignDiagnostic(
            TensorRtApiLine.TensorRt11,
            enableRuntimeSmoke: false,
            runtimePackageKey: string.Empty);
    }

    private static void PrintDebugListenerCallbackOwnerDesignDiagnostic(
        TensorRtApiLine line,
        bool enableRuntimeSmoke,
        string runtimePackageKey)
    {
        using TensorRtDebugListenerCallbackOwner owner = new TensorRtDebugListenerCallbackOwner();
        TensorRtDebugListenerCallbackRequest request = new TensorRtDebugListenerCallbackRequest(
            "smoke_debug_tensor",
            TensorRtDataType.Float,
            TensorRtTensorLocation.Device,
            new long[] { 1, 3, 224, 224 },
            "smoke-debug-listener-owner-design",
            isInput: true,
            isExecutionTensor: true);

        TensorRtDebugListenerCallbackOwnerSnapshot diagnostic = owner.RunDesignDiagnostic(line, request);
        Console.WriteLine("DebugListenerCallbackOwnerDesign=" + FormatDebugListenerCallbackOwnerDesign(diagnostic));

        TensorRtDebugListenerCallbackOwnerSnapshot preDispose = owner.GetSnapshot("pre-dispose");
        Console.WriteLine("DebugListenerCallbackOwnerDesignPreDispose=" + FormatDebugListenerCallbackOwnerDesign(preDispose));

        owner.Dispose();
        TensorRtDebugListenerCallbackOwnerSnapshot postDispose = owner.GetSnapshot("post-dispose");
        Console.WriteLine("DebugListenerCallbackOwnerDesignDispose=" + FormatDebugListenerCallbackOwnerDesign(postDispose));

        TensorRtDebugListenerAttachDetachDesignGateResult attachDetachGate = TensorRtDebugListenerAttachDetachDesignGate.Evaluate(postDispose);
        Console.WriteLine("DebugListenerAttachDetachDesignGate=" + FormatDebugListenerAttachDetachDesignGate(attachDetachGate));

        TensorRtDebugListenerBorrowedTensorSafetyGateResult borrowedTensorGate =
            TensorRtDebugListenerBorrowedTensorSafetyGate.Evaluate(postDispose, attachDetachGate);
        Console.WriteLine("DebugListenerBorrowedTensorSafetyGate=" + FormatDebugListenerBorrowedTensorSafetyGate(borrowedTensorGate));

        TensorRtDebugListenerAttachVTableSafetyGateResult attachVTableGate =
            TensorRtDebugListenerAttachVTableSafetyGate.Evaluate(postDispose, attachDetachGate, borrowedTensorGate);
        Console.WriteLine("DebugListenerAttachVTableSafetyGate=" + FormatDebugListenerAttachVTableSafetyGate(attachVTableGate));

        TensorRtDebugListenerNativeAttachNoThrowPreflightResult nativeAttachNoThrowPreflight =
            TensorRtDebugListenerNativeAttachNoThrowPreflight.Evaluate(postDispose, attachDetachGate, borrowedTensorGate, attachVTableGate);
        Console.WriteLine("DebugListenerNativeAttachNoThrowPreflight=" + FormatDebugListenerNativeAttachNoThrowPreflight(nativeAttachNoThrowPreflight));

        TensorRtDebugListenerNativeOwnerAddressDesignGateResult nativeOwnerAddressDesignGate =
            TensorRtDebugListenerNativeOwnerAddressDesignGate.Evaluate(postDispose, attachDetachGate, borrowedTensorGate, attachVTableGate, nativeAttachNoThrowPreflight);
        Console.WriteLine("DebugListenerNativeOwnerAddressDesignGate=" + FormatDebugListenerNativeOwnerAddressDesignGate(nativeOwnerAddressDesignGate));

        TensorRtDebugListenerNativeNoThrowVTableDesignGateResult nativeNoThrowVTableDesignGate =
            TensorRtDebugListenerNativeNoThrowVTableDesignGate.Evaluate(postDispose, attachDetachGate, borrowedTensorGate, attachVTableGate, nativeAttachNoThrowPreflight, nativeOwnerAddressDesignGate);
        Console.WriteLine("DebugListenerNativeNoThrowVTableDesignGate=" + FormatDebugListenerNativeNoThrowVTableDesignGate(nativeNoThrowVTableDesignGate));

        TensorRtDebugListenerNativeAttachEntryDesignGateResult nativeAttachEntryDesignGate =
            TensorRtDebugListenerNativeAttachEntryDesignGate.Evaluate(postDispose, attachDetachGate, borrowedTensorGate, attachVTableGate, nativeAttachNoThrowPreflight, nativeOwnerAddressDesignGate, nativeNoThrowVTableDesignGate);
        Console.WriteLine("DebugListenerNativeAttachEntryDesignGate=" + FormatDebugListenerNativeAttachEntryDesignGate(nativeAttachEntryDesignGate));

        TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult nativeDetachBeforeReleaseDesignGate =
            TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate.Evaluate(postDispose, attachDetachGate, borrowedTensorGate, attachVTableGate, nativeAttachNoThrowPreflight, nativeOwnerAddressDesignGate, nativeNoThrowVTableDesignGate, nativeAttachEntryDesignGate);
        Console.WriteLine("DebugListenerNativeDetachBeforeReleaseDesignGate=" + FormatDebugListenerNativeDetachBeforeReleaseDesignGate(nativeDetachBeforeReleaseDesignGate));

        TensorRtDebugListenerNativeOwnerLifecycleDryRunResult nativeOwnerLifecycleDryRun =
            TensorRtDebugListenerNativeOwnerLifecycleDryRun.Evaluate(postDispose, attachDetachGate, borrowedTensorGate, attachVTableGate, nativeAttachNoThrowPreflight, nativeOwnerAddressDesignGate, nativeNoThrowVTableDesignGate, nativeAttachEntryDesignGate, nativeDetachBeforeReleaseDesignGate);
        Console.WriteLine("DebugListenerNativeOwnerLifecycleDryRun=" + FormatDebugListenerNativeOwnerLifecycleDryRun(nativeOwnerLifecycleDryRun));

        TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult nativeAttachEntryRuntimeScaffold =
            TensorRtDebugListenerNativeAttachEntryRuntimeScaffold.Evaluate(postDispose, nativeOwnerLifecycleDryRun);
        Console.WriteLine("DebugListenerNativeAttachEntryRuntimeScaffold=" + FormatDebugListenerNativeAttachEntryRuntimeScaffold(nativeAttachEntryRuntimeScaffold));

        TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult nativeAttachEntryMinimalSafety =
            TensorRtDebugListenerNativeAttachEntryMinimalSafety.Evaluate(postDispose, nativeAttachEntryRuntimeScaffold);
        Console.WriteLine("DebugListenerNativeAttachEntryMinimalSafety=" + FormatDebugListenerNativeAttachEntryMinimalSafety(nativeAttachEntryMinimalSafety));

        TensorRtDebugListenerNativeOwnerStableIdentityResult nativeOwnerStableIdentity =
            TensorRtDebugListenerNativeOwnerStableIdentity.Evaluate(postDispose, nativeAttachEntryRuntimeScaffold);
        Console.WriteLine("DebugListenerNativeOwnerStableIdentity=" + FormatDebugListenerNativeOwnerStableIdentity(nativeOwnerStableIdentity));

        TensorRtDebugListenerNativeOwnerNonCopyableStorageResult nativeOwnerNonCopyableStorage =
            TensorRtDebugListenerNativeOwnerNonCopyableStorage.Evaluate(postDispose, nativeOwnerStableIdentity);
        Console.WriteLine("DebugListenerNativeOwnerNonCopyableStorage=" + FormatDebugListenerNativeOwnerNonCopyableStorage(nativeOwnerNonCopyableStorage));

        TensorRtDebugListenerNativeNoThrowDestructorResult nativeNoThrowDestructor =
            TensorRtDebugListenerNativeNoThrowDestructor.Evaluate(postDispose, nativeOwnerNonCopyableStorage);
        Console.WriteLine("DebugListenerNativeNoThrowDestructor=" + FormatDebugListenerNativeNoThrowDestructor(nativeNoThrowDestructor));

        TensorRtDebugListenerNativeOwnerLifecycleGateResult nativeOwnerLifecycleGate =
            TensorRtDebugListenerNativeOwnerLifecycleGate.Evaluate(postDispose, nativeNoThrowDestructor);
        Console.WriteLine("DebugListenerNativeOwnerLifecycleGate=" + FormatDebugListenerNativeOwnerLifecycleGate(nativeOwnerLifecycleGate));

        TensorRtDebugListenerNativeAttachBridgeShapeGateResult nativeAttachBridgeShapeGate =
            TensorRtDebugListenerNativeAttachBridgeShapeGate.Evaluate(postDispose, nativeOwnerLifecycleGate);
        Console.WriteLine("DebugListenerNativeAttachBridgeShapeGate=" + FormatDebugListenerNativeAttachBridgeShapeGate(nativeAttachBridgeShapeGate));

        TensorRtDebugListenerExceptionStatusMappingGateResult exceptionStatusMappingGate =
            TensorRtDebugListenerExceptionStatusMappingGate.Evaluate(postDispose, nativeAttachBridgeShapeGate);
        Console.WriteLine("DebugListenerExceptionStatusMappingGate=" + FormatDebugListenerExceptionStatusMappingGate(exceptionStatusMappingGate));

        TensorRtDebugListenerInFlightAccountingGateResult inFlightAccountingGate =
            TensorRtDebugListenerInFlightAccountingGate.Evaluate(postDispose, exceptionStatusMappingGate);
        Console.WriteLine("DebugListenerInFlightAccountingGate=" + FormatDebugListenerInFlightAccountingGate(inFlightAccountingGate));

        TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult nativeNoThrowVTableScaffoldGate =
            TensorRtDebugListenerNativeNoThrowVTableScaffoldGate.Evaluate(postDispose, nativeAttachBridgeShapeGate, exceptionStatusMappingGate, inFlightAccountingGate);
        Console.WriteLine("DebugListenerNativeNoThrowVTableScaffoldGate=" + FormatDebugListenerNativeNoThrowVTableScaffoldGate(nativeNoThrowVTableScaffoldGate));

        TensorRtDebugListenerNoThrowVTableCallbackStubResult noThrowVTableCallbackStub =
            TensorRtDebugListenerNoThrowVTableCallbackStub.Evaluate(postDispose, nativeAttachEntryMinimalSafety, nativeNoThrowVTableScaffoldGate);
        Console.WriteLine("DebugListenerNoThrowVTableCallbackStub=" + FormatDebugListenerNoThrowVTableCallbackStub(noThrowVTableCallbackStub));

        TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult borrowedDebugTensorMetadataRuntimeGate =
            TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate.Evaluate(postDispose, borrowedTensorGate, noThrowVTableCallbackStub);
        Console.WriteLine("DebugListenerBorrowedDebugTensorMetadataRuntimeGate=" + FormatDebugListenerBorrowedDebugTensorMetadataRuntimeGate(borrowedDebugTensorMetadataRuntimeGate));

        TensorRtDebugListenerNativeVTableInstallPreflightResult nativeVTableInstallPreflight =
            TensorRtDebugListenerNativeVTableInstallPreflight.Evaluate(
                postDispose,
                nativeOwnerLifecycleGate,
                nativeAttachBridgeShapeGate,
                nativeNoThrowVTableScaffoldGate,
                borrowedDebugTensorMetadataRuntimeGate);
        Console.WriteLine("DebugListenerNativeVTableInstallPreflight=" + FormatDebugListenerNativeVTableInstallPreflight(nativeVTableInstallPreflight));

        TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult nativeOwnerVTableInstallExperiment =
            TensorRtDebugListenerNativeOwnerVTableInstallExperiment.Evaluate(
                postDispose,
                nativeOwnerLifecycleGate,
                nativeAttachBridgeShapeGate,
                nativeNoThrowVTableScaffoldGate,
                borrowedDebugTensorMetadataRuntimeGate,
                nativeVTableInstallPreflight);
        Console.WriteLine("DebugListenerNativeOwnerVTableInstallExperiment=" + FormatDebugListenerNativeOwnerVTableInstallExperiment(nativeOwnerVTableInstallExperiment));

        TensorRtDebugListenerRuntimeProofPrecheckResult precheck =
            TensorRtDebugListenerRuntimeProofPrecheck.Evaluate(postDispose, attachDetachGate, borrowedTensorGate, attachVTableGate, nativeAttachNoThrowPreflight, nativeOwnerAddressDesignGate, nativeNoThrowVTableDesignGate, nativeAttachEntryDesignGate, nativeDetachBeforeReleaseDesignGate, nativeOwnerLifecycleDryRun, nativeAttachEntryRuntimeScaffold, nativeOwnerStableIdentity, nativeOwnerNonCopyableStorage, nativeNoThrowDestructor, nativeOwnerLifecycleGate, nativeAttachBridgeShapeGate, exceptionStatusMappingGate, inFlightAccountingGate, nativeNoThrowVTableScaffoldGate);
        Console.WriteLine("DebugListenerRuntimeProofPrecheck=" + FormatDebugListenerRuntimeProofPrecheck(precheck));

        TensorRtDebugListenerRuntimeProofAttemptPreflightResult attemptPreflight =
            TensorRtDebugListenerRuntimeProofAttemptPreflight.Evaluate(precheck);
        Console.WriteLine("DebugListenerRuntimeProofAttemptPreflight=" + FormatDebugListenerRuntimeProofAttemptPreflight(attemptPreflight));

        TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult runtimeSmoke =
            TensorRtDebugListenerRealNonNullAttachRuntimeSmoke.Evaluate(
                attemptPreflight,
                runtimePackageKey,
                enableRuntimeSmoke,
                fullPackageConsumerReport: false);
        Console.WriteLine("DebugListenerRealNonNullAttachRuntimeSmoke=" + FormatDebugListenerRealNonNullAttachRuntimeSmoke(runtimeSmoke));

        TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult callbackTrampoline =
            TensorRtDebugListenerProcessDebugTensorCallbackTrampoline.Evaluate(
                noThrowVTableCallbackStub,
                borrowedDebugTensorMetadataRuntimeGate,
                runtimeSmoke);
        Console.WriteLine("DebugListenerProcessDebugTensorCallbackTrampoline=" + FormatDebugListenerProcessDebugTensorCallbackTrampoline(callbackTrampoline));

        TensorRtDebugListenerRealCallbackRuntimeProofResult realCallbackRuntimeProof =
            TensorRtDebugListenerRealCallbackRuntimeProof.Evaluate(
                runtimeSmoke,
                callbackTrampoline);
        Console.WriteLine("DebugListenerRealCallbackRuntimeProof=" + FormatDebugListenerRealCallbackRuntimeProof(realCallbackRuntimeProof));

        TensorRtDebugListenerCallbackProofGapReportResult callbackProofGapReport =
            TensorRtDebugListenerCallbackProofGapReport.Evaluate(
                attemptPreflight,
                runtimeSmoke,
                callbackTrampoline,
                realCallbackRuntimeProof);
        Console.WriteLine("DebugListenerCallbackProofGapReport=" + FormatDebugListenerCallbackProofGapReport(callbackProofGapReport));
    }

    private static string FormatDebugListenerCallbackOwnerDesign(TensorRtDebugListenerCallbackOwnerSnapshot result)
    {
        return "debug-listener-callback-owner-design" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";OwnerId={result.OwnerId}" +
            $";Operation={result.Operation}" +
            $";TensorName={SanitizeSmokeValue(result.TensorName)}" +
            $";DataType={result.DataType}" +
            $";Location={result.Location}" +
            $";ShapeRank={result.ShapeRank}" +
            $";ShapeSummary={SanitizeSmokeValue(result.ShapeSummary)}" +
            $";IsInput={result.IsInput}" +
            $";IsOutput={result.IsOutput}" +
            $";IsShapeTensor={result.IsShapeTensor}" +
            $";IsExecutionTensor={result.IsExecutionTensor}" +
            $";LastStatus={result.LastStatus}" +
            $";InvocationCount={result.InvocationCount}" +
            $";ProcessDebugTensorCount={result.ProcessDebugTensorCount}" +
            $";FailureCount={result.FailureCount}" +
            $";InFlightCallbackCount={result.InFlightCallbackCount}" +
            $";MaxInFlightCallbackCount={result.MaxInFlightCallbackCount}" +
            $";ActiveGateCallCount={result.ActiveGateCallCount}" +
            $";ReleaseHookCount={result.ReleaseHookCount}" +
            $";CallbackStatePinned={result.CallbackStatePinned}" +
            $";DelegatePinned={result.DelegatePinned}" +
            $";DisposeRequested={result.DisposeRequested}" +
            $";IsAttached={result.IsAttached}" +
            $";DebugTensorMetadataCopied={result.DebugTensorMetadataCopied}" +
            $";DebugTensorPointerExposed={result.DebugTensorPointerExposed}" +
            $";DebugTensorPointerProduced={result.DebugTensorPointerProduced}" +
            $";BorrowedDebugTensorPointerEscaped={result.BorrowedDebugTensorPointerEscaped}" +
            $";Succeeded={result.Succeeded}" +
            $";LastDiagnostic={SanitizeSmokeValue(result.LastDiagnostic)}" +
            $";ReleaseDiagnostic={SanitizeSmokeValue(result.ReleaseDiagnostic)}";
    }

    private static string FormatDebugListenerAttachDetachDesignGate(TensorRtDebugListenerAttachDetachDesignGateResult result)
    {
        return "debug-listener-attach-detach-design-gate" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";LineSupportsDebugListener={result.LineSupportsDebugListener}" +
            $";OwnerDesignReady={result.OwnerDesignReady}" +
            $";ManagedOwnerStateMachineReady={result.ManagedOwnerStateMachineReady}" +
            $";DebugTensorMetadataCopied={result.DebugTensorMetadataCopied}" +
            $";PointerFreeSurfaceReady={result.PointerFreeSurfaceReady}" +
            $";AttachControlAvailable={result.AttachControlAvailable}" +
            $";DetachClearControlAvailable={result.DetachClearControlAvailable}" +
            $";LineSpecificAttachDetachReady={result.LineSpecificAttachDetachReady}" +
            $";StableNativeOwnerAddressReady={result.StableNativeOwnerAddressReady}" +
            $";NoThrowNativeVTableReady={result.NoThrowNativeVTableReady}" +
            $";NativeVTableReady={result.NativeVTableReady}" +
            $";BorrowedDebugTensorLifetimeReady={result.BorrowedDebugTensorLifetimeReady}" +
            $";DesignGateReady={result.DesignGateReady}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerBorrowedTensorSafetyGate(TensorRtDebugListenerBorrowedTensorSafetyGateResult result)
    {
        return "debug-listener-borrowed-tensor-safety-gate" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";TensorName={SanitizeSmokeValue(result.TensorName)}" +
            $";DataType={result.DataType}" +
            $";Location={result.Location}" +
            $";ShapeRank={result.ShapeRank}" +
            $";ShapeSummary={SanitizeSmokeValue(result.ShapeSummary)}" +
            $";IsInput={result.IsInput}" +
            $";IsOutput={result.IsOutput}" +
            $";IsShapeTensor={result.IsShapeTensor}" +
            $";IsExecutionTensor={result.IsExecutionTensor}" +
            $";ProcessDebugTensorCount={result.ProcessDebugTensorCount}" +
            $";AttachDetachDesignGateReady={result.AttachDetachDesignGateReady}" +
            $";OwnerDesignReady={result.OwnerDesignReady}" +
            $";DebugTensorMetadataCopied={result.DebugTensorMetadataCopied}" +
            $";PointerFreeSurfaceReady={result.PointerFreeSurfaceReady}" +
            $";BorrowedDebugTensorPointerEscapeBlocked={result.BorrowedDebugTensorPointerEscapeBlocked}" +
            $";BorrowedDebugTensorLifetimeReady={result.BorrowedDebugTensorLifetimeReady}" +
            $";BorrowedDebugTensorDataLifetimeReady={result.BorrowedDebugTensorDataLifetimeReady}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerAttachVTableSafetyGate(TensorRtDebugListenerAttachVTableSafetyGateResult result)
    {
        return "debug-listener-attach-vtable-safety-gate" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";LineSupportsDebugListener={result.LineSupportsDebugListener}" +
            $";OwnerDesignReady={result.OwnerDesignReady}" +
            $";AttachDetachDesignGateReady={result.AttachDetachDesignGateReady}" +
            $";BorrowedTensorSafetyGateReady={result.BorrowedTensorSafetyGateReady}" +
            $";ManagedOwnerStateMachineReady={result.ManagedOwnerStateMachineReady}" +
            $";DebugTensorMetadataCopied={result.DebugTensorMetadataCopied}" +
            $";PointerFreeSurfaceReady={result.PointerFreeSurfaceReady}" +
            $";BorrowedDebugTensorPointerEscapeBlocked={result.BorrowedDebugTensorPointerEscapeBlocked}" +
            $";AttachControlAvailable={result.AttachControlAvailable}" +
            $";DetachClearControlAvailable={result.DetachClearControlAvailable}" +
            $";LineSpecificAttachDetachReady={result.LineSpecificAttachDetachReady}" +
            $";StableNativeOwnerAddressReady={result.StableNativeOwnerAddressReady}" +
            $";NoThrowNativeVTableReady={result.NoThrowNativeVTableReady}" +
            $";NativeVTableReady={result.NativeVTableReady}" +
            $";ExceptionToStatusMappingReady={result.ExceptionToStatusMappingReady}" +
            $";BorrowedDebugTensorLifetimeReady={result.BorrowedDebugTensorLifetimeReady}" +
            $";BorrowedDebugTensorDataLifetimeReady={result.BorrowedDebugTensorDataLifetimeReady}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerNativeAttachNoThrowPreflight(TensorRtDebugListenerNativeAttachNoThrowPreflightResult result)
    {
        return "debug-listener-native-attach-nothrow-preflight" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";AttachVTableSafetyGateReady={result.AttachVTableSafetyGateReady}" +
            $";NativeAttachEntryLocated={result.NativeAttachEntryLocated}" +
            $";NativeDetachEntryLocated={result.NativeDetachEntryLocated}" +
            $";StableNativeOwnerAddressDesignReady={result.StableNativeOwnerAddressDesignReady}" +
            $";ManagedCallbackKeepAliveDesignReady={result.ManagedCallbackKeepAliveDesignReady}" +
            $";NoThrowVTableDesignReady={result.NoThrowVTableDesignReady}" +
            $";ExceptionToStatusMappingDesignReady={result.ExceptionToStatusMappingDesignReady}" +
            $";BorrowedDebugTensorMetadataCopyDesignReady={result.BorrowedDebugTensorMetadataCopyDesignReady}" +
            $";BorrowedDebugTensorPointerEscapeBlocked={result.BorrowedDebugTensorPointerEscapeBlocked}" +
            $";NativeVTableDesignReady={result.NativeVTableDesignReady}" +
            $";BorrowedDebugTensorLifetimeRuntimeReady={result.BorrowedDebugTensorLifetimeRuntimeReady}" +
            $";BorrowedDebugTensorDataLifetimeRuntimeReady={result.BorrowedDebugTensorDataLifetimeRuntimeReady}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";PreflightReady={result.PreflightReady}" +
            $";CanImplementNativeAttach={result.CanImplementNativeAttach}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerNativeOwnerAddressDesignGate(TensorRtDebugListenerNativeOwnerAddressDesignGateResult result)
    {
        return "debug-listener-native-owner-address-design-gate" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";NativeAttachNoThrowPreflightReady={result.NativeAttachNoThrowPreflightReady}" +
            $";NativeAttachEntryLocated={result.NativeAttachEntryLocated}" +
            $";NativeDetachEntryLocated={result.NativeDetachEntryLocated}" +
            $";StableNativeOwnerAddressReady={result.StableNativeOwnerAddressReady}" +
            $";StableNativeOwnerAddressDesignReady={result.StableNativeOwnerAddressDesignReady}" +
            $";ManagedCallbackKeepAliveDesignReady={result.ManagedCallbackKeepAliveDesignReady}" +
            $";NativeOwnerNonCopyableReady={result.NativeOwnerNonCopyableReady}" +
            $";NativeOwnerDisposeOrderReady={result.NativeOwnerDisposeOrderReady}" +
            $";NativeOwnerReleaseHookReady={result.NativeOwnerReleaseHookReady}" +
            $";NativeOwnerInFlightDrainReady={result.NativeOwnerInFlightDrainReady}" +
            $";NoThrowNativeDestructorReady={result.NoThrowNativeDestructorReady}" +
            $";NoThrowVTableDesignReady={result.NoThrowVTableDesignReady}" +
            $";ExceptionToStatusMappingDesignReady={result.ExceptionToStatusMappingDesignReady}" +
            $";BorrowedDebugTensorMetadataCopyDesignReady={result.BorrowedDebugTensorMetadataCopyDesignReady}" +
            $";BorrowedDebugTensorPointerEscapeBlocked={result.BorrowedDebugTensorPointerEscapeBlocked}" +
            $";NativeOwnerLifecycleReady={result.NativeOwnerLifecycleReady}" +
            $";NativeVTableDesignReady={result.NativeVTableDesignReady}" +
            $";BorrowedDebugTensorLifetimeRuntimeReady={result.BorrowedDebugTensorLifetimeRuntimeReady}" +
            $";BorrowedDebugTensorDataLifetimeRuntimeReady={result.BorrowedDebugTensorDataLifetimeRuntimeReady}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";DesignGateReady={result.DesignGateReady}" +
            $";CanImplementNativeAttach={result.CanImplementNativeAttach}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerNativeNoThrowVTableDesignGate(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult result)
    {
        return "debug-listener-native-nothrow-vtable-design-gate" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";NativeOwnerAddressDesignGateReady={result.NativeOwnerAddressDesignGateReady}" +
            $";NativeAttachNoThrowPreflightReady={result.NativeAttachNoThrowPreflightReady}" +
            $";NativeAttachEntryLocated={result.NativeAttachEntryLocated}" +
            $";NativeOwnerLifecycleReady={result.NativeOwnerLifecycleReady}" +
            $";ManagedCallbackKeepAliveDesignReady={result.ManagedCallbackKeepAliveDesignReady}" +
            $";NoThrowNativeDestructorReady={result.NoThrowNativeDestructorReady}" +
            $";NoThrowVTableDesignReady={result.NoThrowVTableDesignReady}" +
            $";ExceptionToStatusMappingDesignReady={result.ExceptionToStatusMappingDesignReady}" +
            $";BorrowedDebugTensorMetadataCopyDesignReady={result.BorrowedDebugTensorMetadataCopyDesignReady}" +
            $";BorrowedDebugTensorPointerEscapeBlocked={result.BorrowedDebugTensorPointerEscapeBlocked}" +
            $";NativeVTableTrampolineReady={result.NativeVTableTrampolineReady}" +
            $";CallbackExceptionCaptureReady={result.CallbackExceptionCaptureReady}" +
            $";CallbackStatusMappingReady={result.CallbackStatusMappingReady}" +
            $";CallbackInFlightAccountingReady={result.CallbackInFlightAccountingReady}" +
            $";NativeVTableDesignReady={result.NativeVTableDesignReady}" +
            $";BorrowedDebugTensorLifetimeRuntimeReady={result.BorrowedDebugTensorLifetimeRuntimeReady}" +
            $";BorrowedDebugTensorDataLifetimeRuntimeReady={result.BorrowedDebugTensorDataLifetimeRuntimeReady}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";DesignGateReady={result.DesignGateReady}" +
            $";CanImplementNativeAttach={result.CanImplementNativeAttach}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerNativeAttachEntryDesignGate(TensorRtDebugListenerNativeAttachEntryDesignGateResult result)
    {
        return "debug-listener-native-attach-entry-design-gate" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";NativeNoThrowVTableDesignGateReady={result.NativeNoThrowVTableDesignGateReady}" +
            $";NativeOwnerAddressDesignGateReady={result.NativeOwnerAddressDesignGateReady}" +
            $";NativeAttachNoThrowPreflightReady={result.NativeAttachNoThrowPreflightReady}" +
            $";NativeDetachEntryLocated={result.NativeDetachEntryLocated}" +
            $";NativeAttachEntryLocated={result.NativeAttachEntryLocated}" +
            $";LineSpecificAttachEntryDesignReady={result.LineSpecificAttachEntryDesignReady}" +
            $";AttachEntryNoThrowReady={result.AttachEntryNoThrowReady}" +
            $";AttachEntryVersionGuardReady={result.AttachEntryVersionGuardReady}" +
            $";AttachEntryOwnershipReady={result.AttachEntryOwnershipReady}" +
            $";DetachBeforeReleaseReady={result.DetachBeforeReleaseReady}" +
            $";NativeOwnerLifecycleReady={result.NativeOwnerLifecycleReady}" +
            $";NativeVTableDesignReady={result.NativeVTableDesignReady}" +
            $";ManagedCallbackKeepAliveDesignReady={result.ManagedCallbackKeepAliveDesignReady}" +
            $";BorrowedDebugTensorMetadataCopyDesignReady={result.BorrowedDebugTensorMetadataCopyDesignReady}" +
            $";BorrowedDebugTensorPointerEscapeBlocked={result.BorrowedDebugTensorPointerEscapeBlocked}" +
            $";BorrowedDebugTensorLifetimeRuntimeReady={result.BorrowedDebugTensorLifetimeRuntimeReady}" +
            $";BorrowedDebugTensorDataLifetimeRuntimeReady={result.BorrowedDebugTensorDataLifetimeRuntimeReady}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";DesignGateReady={result.DesignGateReady}" +
            $";CanImplementNativeAttach={result.CanImplementNativeAttach}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerNativeDetachBeforeReleaseDesignGate(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult result)
    {
        return "debug-listener-native-detach-before-release-design-gate" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";NativeAttachEntryDesignGateReady={result.NativeAttachEntryDesignGateReady}" +
            $";NativeNoThrowVTableDesignGateReady={result.NativeNoThrowVTableDesignGateReady}" +
            $";NativeOwnerAddressDesignGateReady={result.NativeOwnerAddressDesignGateReady}" +
            $";NativeDetachEntryLocated={result.NativeDetachEntryLocated}" +
            $";NativeAttachEntryLocated={result.NativeAttachEntryLocated}" +
            $";LineSpecificAttachEntryDesignReady={result.LineSpecificAttachEntryDesignReady}" +
            $";AttachEntryNoThrowReady={result.AttachEntryNoThrowReady}" +
            $";AttachEntryVersionGuardReady={result.AttachEntryVersionGuardReady}" +
            $";AttachEntryOwnershipReady={result.AttachEntryOwnershipReady}" +
            $";DetachBeforeReleaseReady={result.DetachBeforeReleaseReady}" +
            $";ReleaseHookOrderingReady={result.ReleaseHookOrderingReady}" +
            $";DisposeIdempotencyReady={result.DisposeIdempotencyReady}" +
            $";InFlightDrainBeforeReleaseReady={result.InFlightDrainBeforeReleaseReady}" +
            $";CallbackStateUnpinAfterDetachReady={result.CallbackStateUnpinAfterDetachReady}" +
            $";DelegateUnpinAfterDetachReady={result.DelegateUnpinAfterDetachReady}" +
            $";NativeOwnerLifecycleReady={result.NativeOwnerLifecycleReady}" +
            $";NativeVTableDesignReady={result.NativeVTableDesignReady}" +
            $";ManagedCallbackKeepAliveDesignReady={result.ManagedCallbackKeepAliveDesignReady}" +
            $";BorrowedDebugTensorMetadataCopyDesignReady={result.BorrowedDebugTensorMetadataCopyDesignReady}" +
            $";BorrowedDebugTensorPointerEscapeBlocked={result.BorrowedDebugTensorPointerEscapeBlocked}" +
            $";BorrowedDebugTensorLifetimeRuntimeReady={result.BorrowedDebugTensorLifetimeRuntimeReady}" +
            $";BorrowedDebugTensorDataLifetimeRuntimeReady={result.BorrowedDebugTensorDataLifetimeRuntimeReady}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";DesignGateReady={result.DesignGateReady}" +
            $";CanImplementNativeAttach={result.CanImplementNativeAttach}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerNativeOwnerLifecycleDryRun(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult result)
    {
        return "debug-listener-native-owner-lifecycle-dry-run" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";OwnerId={result.OwnerId}" +
            $";LastStatus={result.LastStatus}" +
            $";ReleaseHookCount={result.ReleaseHookCount}" +
            $";InFlightCallbackCount={result.InFlightCallbackCount}" +
            $";CallbackStatePinned={result.CallbackStatePinned}" +
            $";DelegatePinned={result.DelegatePinned}" +
            $";DisposeRequested={result.DisposeRequested}" +
            $";NativeDetachBeforeReleaseDesignGateReady={result.NativeDetachBeforeReleaseDesignGateReady}" +
            $";NativeAttachEntryDesignGateReady={result.NativeAttachEntryDesignGateReady}" +
            $";NativeNoThrowVTableDesignGateReady={result.NativeNoThrowVTableDesignGateReady}" +
            $";NativeOwnerAddressDesignGateReady={result.NativeOwnerAddressDesignGateReady}" +
            $";NativeDetachEntryLocated={result.NativeDetachEntryLocated}" +
            $";NativeAttachEntryLocated={result.NativeAttachEntryLocated}" +
            $";StableNativeOwnerIdentityReady={result.StableNativeOwnerIdentityReady}" +
            $";NativeOwnerNonCopyableReady={result.NativeOwnerNonCopyableReady}" +
            $";NativeOwnerDisposeOrderReady={result.NativeOwnerDisposeOrderReady}" +
            $";NativeOwnerReleaseHookReady={result.NativeOwnerReleaseHookReady}" +
            $";NativeOwnerInFlightDrainReady={result.NativeOwnerInFlightDrainReady}" +
            $";DetachBeforeReleaseReady={result.DetachBeforeReleaseReady}" +
            $";ReleaseHookOrderingReady={result.ReleaseHookOrderingReady}" +
            $";DisposeIdempotencyReady={result.DisposeIdempotencyReady}" +
            $";InFlightDrainBeforeReleaseReady={result.InFlightDrainBeforeReleaseReady}" +
            $";CallbackStateUnpinAfterDetachReady={result.CallbackStateUnpinAfterDetachReady}" +
            $";DelegateUnpinAfterDetachReady={result.DelegateUnpinAfterDetachReady}" +
            $";NoThrowNativeDestructorReady={result.NoThrowNativeDestructorReady}" +
            $";NativeOwnerLifecycleReady={result.NativeOwnerLifecycleReady}" +
            $";NativeVTableDesignReady={result.NativeVTableDesignReady}" +
            $";ManagedCallbackKeepAliveDesignReady={result.ManagedCallbackKeepAliveDesignReady}" +
            $";BorrowedDebugTensorMetadataCopyDesignReady={result.BorrowedDebugTensorMetadataCopyDesignReady}" +
            $";BorrowedDebugTensorPointerEscapeBlocked={result.BorrowedDebugTensorPointerEscapeBlocked}" +
            $";DryRunReady={result.DryRunReady}" +
            $";BorrowedDebugTensorLifetimeRuntimeReady={result.BorrowedDebugTensorLifetimeRuntimeReady}" +
            $";BorrowedDebugTensorDataLifetimeRuntimeReady={result.BorrowedDebugTensorDataLifetimeRuntimeReady}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";CanImplementNativeAttach={result.CanImplementNativeAttach}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";LastDiagnostic={SanitizeSmokeValue(result.LastDiagnostic)}" +
            $";ReleaseDiagnostic={SanitizeSmokeValue(result.ReleaseDiagnostic)}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerNativeAttachEntryRuntimeScaffold(TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult result)
    {
        return "debug-listener-native-attach-entry-runtime-scaffold" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";OwnerId={result.OwnerId}" +
            $";LastStatus={result.LastStatus}" +
            $";NativeOwnerLifecycleDryRunReady={result.NativeOwnerLifecycleDryRunReady}" +
            $";NativeAttachEntryLocated={result.NativeAttachEntryLocated}" +
            $";NativeDetachEntryLocated={result.NativeDetachEntryLocated}" +
            $";AttachEntryParameterShapeReady={result.AttachEntryParameterShapeReady}" +
            $";AttachEntryVersionGuardReady={result.AttachEntryVersionGuardReady}" +
            $";AttachEntryNoThrowBoundaryReady={result.AttachEntryNoThrowBoundaryReady}" +
            $";AttachEntryOwnershipDiagnosticsReady={result.AttachEntryOwnershipDiagnosticsReady}" +
            $";StableNativeOwnerIdentityReady={result.StableNativeOwnerIdentityReady}" +
            $";NativeOwnerNonCopyableReady={result.NativeOwnerNonCopyableReady}" +
            $";NoThrowNativeDestructorReady={result.NoThrowNativeDestructorReady}" +
            $";NativeOwnerLifecycleReady={result.NativeOwnerLifecycleReady}" +
            $";RuntimeScaffoldReady={result.RuntimeScaffoldReady}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";CanImplementNativeAttach={result.CanImplementNativeAttach}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerNativeAttachEntryMinimalSafety(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult result)
    {
        return "debug-listener-native-attach-entry-minimal-safety" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";OwnerId={result.OwnerId}" +
            $";LastStatus={result.LastStatus}" +
            $";RuntimeScaffoldReady={result.RuntimeScaffoldReady}" +
            $";LifecycleGateReady={result.LifecycleGateReady}" +
            $";LifecyclePointerFree={result.LifecyclePointerFree}" +
            $";NativeAttachEntryLocated={result.NativeAttachEntryLocated}" +
            $";NativeDetachEntryLocated={result.NativeDetachEntryLocated}" +
            $";AttachEntryParameterShapeReady={result.AttachEntryParameterShapeReady}" +
            $";AttachEntryNoThrowReady={result.AttachEntryNoThrowReady}" +
            $";AttachEntryVersionGuardReady={result.AttachEntryVersionGuardReady}" +
            $";AttachEntryOwnershipDiagnosticsReady={result.AttachEntryOwnershipDiagnosticsReady}" +
            $";SetDebugListenerNonNullEnabled={result.SetDebugListenerNonNullEnabled}" +
            $";NonNullAttachStillDisabled={result.NonNullAttachStillDisabled}" +
            $";NativeAttachWouldBeBlocked={result.NativeAttachWouldBeBlocked}" +
            $";MinimalSafetyReady={result.MinimalSafetyReady}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";CanImplementNativeAttach={result.CanImplementNativeAttach}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";ReasonNativeAttachStillBlocked={SanitizeSmokeValue(result.ReasonNativeAttachStillBlocked)}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerNativeOwnerStableIdentity(TensorRtDebugListenerNativeOwnerStableIdentityResult result)
    {
        return "debug-listener-native-owner-stable-identity" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";OwnerId={result.OwnerId}" +
            $";LastStatus={result.LastStatus}" +
            $";NativeAttachEntryRuntimeScaffoldReady={result.NativeAttachEntryRuntimeScaffoldReady}" +
            $";StableNativeOwnerIdentityReady={result.StableNativeOwnerIdentityReady}" +
            $";OwnerIdentityDiagnosticsReady={result.OwnerIdentityDiagnosticsReady}" +
            $";OwnerIdentityPointerFree={result.OwnerIdentityPointerFree}" +
            $";NativeAttachEntryLocated={result.NativeAttachEntryLocated}" +
            $";NativeDetachEntryLocated={result.NativeDetachEntryLocated}" +
            $";NativeOwnerNonCopyableReady={result.NativeOwnerNonCopyableReady}" +
            $";NoThrowNativeDestructorReady={result.NoThrowNativeDestructorReady}" +
            $";NativeOwnerLifecycleReady={result.NativeOwnerLifecycleReady}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";CanImplementNativeAttach={result.CanImplementNativeAttach}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";LastDiagnostic={SanitizeSmokeValue(result.LastDiagnostic)}" +
            $";ReleaseDiagnostic={SanitizeSmokeValue(result.ReleaseDiagnostic)}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerNativeOwnerNonCopyableStorage(TensorRtDebugListenerNativeOwnerNonCopyableStorageResult result)
    {
        return "debug-listener-native-owner-noncopyable-storage" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";OwnerId={result.OwnerId}" +
            $";LastStatus={result.LastStatus}" +
            $";NativeOwnerStableIdentityReady={result.NativeOwnerStableIdentityReady}" +
            $";OwnerIdentityDiagnosticsReady={result.OwnerIdentityDiagnosticsReady}" +
            $";OwnerIdentityPointerFree={result.OwnerIdentityPointerFree}" +
            $";NativeOwnerNonCopyableReady={result.NativeOwnerNonCopyableReady}" +
            $";NativeOwnerCopyBlocked={result.NativeOwnerCopyBlocked}" +
            $";NativeOwnerMoveBlocked={result.NativeOwnerMoveBlocked}" +
            $";NativeOwnerAddressExposed={result.NativeOwnerAddressExposed}" +
            $";NativeOwnerPointerProduced={result.NativeOwnerPointerProduced}" +
            $";NativeAttachEntryLocated={result.NativeAttachEntryLocated}" +
            $";NativeDetachEntryLocated={result.NativeDetachEntryLocated}" +
            $";NoThrowNativeDestructorReady={result.NoThrowNativeDestructorReady}" +
            $";NativeOwnerLifecycleReady={result.NativeOwnerLifecycleReady}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";CanImplementNativeAttach={result.CanImplementNativeAttach}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";LastDiagnostic={SanitizeSmokeValue(result.LastDiagnostic)}" +
            $";ReleaseDiagnostic={SanitizeSmokeValue(result.ReleaseDiagnostic)}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerNativeNoThrowDestructor(TensorRtDebugListenerNativeNoThrowDestructorResult result)
    {
        return "debug-listener-native-nothrow-destructor" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";OwnerId={result.OwnerId}" +
            $";LastStatus={result.LastStatus}" +
            $";NativeOwnerNonCopyableStorageReady={result.NativeOwnerNonCopyableStorageReady}" +
            $";NativeOwnerNonCopyableReady={result.NativeOwnerNonCopyableReady}" +
            $";NativeOwnerCopyBlocked={result.NativeOwnerCopyBlocked}" +
            $";NativeOwnerMoveBlocked={result.NativeOwnerMoveBlocked}" +
            $";NativeOwnerAddressExposed={result.NativeOwnerAddressExposed}" +
            $";NativeOwnerPointerProduced={result.NativeOwnerPointerProduced}" +
            $";DestructorNoThrowScaffoldReady={result.DestructorNoThrowScaffoldReady}" +
            $";DestructorExceptionEscapeBlocked={result.DestructorExceptionEscapeBlocked}" +
            $";DestructorAddressExposed={result.DestructorAddressExposed}" +
            $";DestructorPointerProduced={result.DestructorPointerProduced}" +
            $";NativeAttachEntryLocated={result.NativeAttachEntryLocated}" +
            $";NativeDetachEntryLocated={result.NativeDetachEntryLocated}" +
            $";NoThrowNativeDestructorReady={result.NoThrowNativeDestructorReady}" +
            $";NativeOwnerLifecycleReady={result.NativeOwnerLifecycleReady}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";CanImplementNativeAttach={result.CanImplementNativeAttach}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";LastDiagnostic={SanitizeSmokeValue(result.LastDiagnostic)}" +
            $";ReleaseDiagnostic={SanitizeSmokeValue(result.ReleaseDiagnostic)}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerNativeOwnerLifecycleGate(TensorRtDebugListenerNativeOwnerLifecycleGateResult result)
    {
        return "debug-listener-native-owner-lifecycle-gate" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";OwnerId={result.OwnerId}" +
            $";LastStatus={result.LastStatus}" +
            $";ReleaseHookCount={result.ReleaseHookCount}" +
            $";InFlightCallbackCount={result.InFlightCallbackCount}" +
            $";CallbackStatePinned={result.CallbackStatePinned}" +
            $";DelegatePinned={result.DelegatePinned}" +
            $";DisposeRequested={result.DisposeRequested}" +
            $";NativeNoThrowDestructorGateReady={result.NativeNoThrowDestructorGateReady}" +
            $";NativeOwnerNonCopyableReady={result.NativeOwnerNonCopyableReady}" +
            $";NativeOwnerCopyBlocked={result.NativeOwnerCopyBlocked}" +
            $";NativeOwnerMoveBlocked={result.NativeOwnerMoveBlocked}" +
            $";NativeOwnerAddressExposed={result.NativeOwnerAddressExposed}" +
            $";NativeOwnerPointerProduced={result.NativeOwnerPointerProduced}" +
            $";DestructorNoThrowScaffoldReady={result.DestructorNoThrowScaffoldReady}" +
            $";DestructorExceptionEscapeBlocked={result.DestructorExceptionEscapeBlocked}" +
            $";DestructorAddressExposed={result.DestructorAddressExposed}" +
            $";DestructorPointerProduced={result.DestructorPointerProduced}" +
            $";ManagedDisposeSnapshotReady={result.ManagedDisposeSnapshotReady}" +
            $";LifecycleScaffoldReady={result.LifecycleScaffoldReady}" +
            $";ReleaseHookOrderingGateReady={result.ReleaseHookOrderingGateReady}" +
            $";DisposeIdempotencyGateReady={result.DisposeIdempotencyGateReady}" +
            $";InFlightDrainGateReady={result.InFlightDrainGateReady}" +
            $";CallbackStateUnpinAfterDetachGateReady={result.CallbackStateUnpinAfterDetachGateReady}" +
            $";DelegateUnpinAfterDetachGateReady={result.DelegateUnpinAfterDetachGateReady}" +
            $";LifecycleAddressExposed={result.LifecycleAddressExposed}" +
            $";LifecyclePointerProduced={result.LifecyclePointerProduced}" +
            $";NativeAttachEntryLocated={result.NativeAttachEntryLocated}" +
            $";NativeDetachEntryLocated={result.NativeDetachEntryLocated}" +
            $";NoThrowNativeDestructorReady={result.NoThrowNativeDestructorReady}" +
            $";ReleaseHookOrderingReady={result.ReleaseHookOrderingReady}" +
            $";DisposeIdempotencyReady={result.DisposeIdempotencyReady}" +
            $";InFlightDrainBeforeReleaseReady={result.InFlightDrainBeforeReleaseReady}" +
            $";CallbackStateUnpinAfterDetachReady={result.CallbackStateUnpinAfterDetachReady}" +
            $";DelegateUnpinAfterDetachReady={result.DelegateUnpinAfterDetachReady}" +
            $";NativeOwnerLifecycleReady={result.NativeOwnerLifecycleReady}" +
            $";NativeVTableDesignReady={result.NativeVTableDesignReady}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";LifecycleGateReady={result.LifecycleGateReady}" +
            $";CanImplementNativeAttach={result.CanImplementNativeAttach}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";LastDiagnostic={SanitizeSmokeValue(result.LastDiagnostic)}" +
            $";ReleaseDiagnostic={SanitizeSmokeValue(result.ReleaseDiagnostic)}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerNativeAttachBridgeShapeGate(TensorRtDebugListenerNativeAttachBridgeShapeGateResult result)
    {
        return "debug-listener-native-attach-bridge-shape-gate" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";OwnerId={result.OwnerId}" +
            $";LastStatus={result.LastStatus}" +
            $";NativeOwnerLifecycleGateReady={result.NativeOwnerLifecycleGateReady}" +
            $";AttachBridgeShapeReady={result.AttachBridgeShapeReady}" +
            $";AttachBridgeNoThrowBoundaryReady={result.AttachBridgeNoThrowBoundaryReady}" +
            $";AttachBridgeVersionGuardReady={result.AttachBridgeVersionGuardReady}" +
            $";AttachBridgeOwnershipDiagnosticsReady={result.AttachBridgeOwnershipDiagnosticsReady}" +
            $";AttachBridgePointerFree={result.AttachBridgePointerFree}" +
            $";SetDebugListenerNonNullEnabled={result.SetDebugListenerNonNullEnabled}" +
            $";NonNullAttachStillDisabled={result.NonNullAttachStillDisabled}" +
            $";NativeAttachEntryLocated={result.NativeAttachEntryLocated}" +
            $";NativeDetachEntryLocated={result.NativeDetachEntryLocated}" +
            $";NativeOwnerLifecycleReady={result.NativeOwnerLifecycleReady}" +
            $";NativeVTableDesignReady={result.NativeVTableDesignReady}" +
            $";AttachBridgeShapeGateReady={result.AttachBridgeShapeGateReady}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";CanImplementNativeAttach={result.CanImplementNativeAttach}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerExceptionStatusMappingGate(TensorRtDebugListenerExceptionStatusMappingGateResult result)
    {
        return "debug-listener-exception-status-mapping-gate" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";OwnerId={result.OwnerId}" +
            $";LastStatus={result.LastStatus}" +
            $";AttachBridgeShapeGateReady={result.AttachBridgeShapeGateReady}" +
            $";ManagedCallbackExceptionCaptureReady={result.ManagedCallbackExceptionCaptureReady}" +
            $";NativeCallbackExceptionCaptureReady={result.NativeCallbackExceptionCaptureReady}" +
            $";CallbackStatusMappingGateReady={result.CallbackStatusMappingGateReady}" +
            $";ExceptionEscapeBlocked={result.ExceptionEscapeBlocked}" +
            $";DiagnosticCopyReady={result.DiagnosticCopyReady}" +
            $";MappingAddressExposed={result.MappingAddressExposed}" +
            $";MappingPointerProduced={result.MappingPointerProduced}" +
            $";NativeAttachEntryLocated={result.NativeAttachEntryLocated}" +
            $";ExceptionStatusMappingGateReady={result.ExceptionStatusMappingGateReady}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerInFlightAccountingGate(TensorRtDebugListenerInFlightAccountingGateResult result)
    {
        return "debug-listener-inflight-accounting-gate" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";OwnerId={result.OwnerId}" +
            $";LastStatus={result.LastStatus}" +
            $";ProcessDebugTensorCount={result.ProcessDebugTensorCount}" +
            $";InFlightCallbackCount={result.InFlightCallbackCount}" +
            $";MaxInFlightCallbackCount={result.MaxInFlightCallbackCount}" +
            $";ReleaseHookCount={result.ReleaseHookCount}" +
            $";CallbackStatePinned={result.CallbackStatePinned}" +
            $";DelegatePinned={result.DelegatePinned}" +
            $";DisposeRequested={result.DisposeRequested}" +
            $";ExceptionStatusMappingGateReady={result.ExceptionStatusMappingGateReady}" +
            $";CallbackEnterAccountingGateReady={result.CallbackEnterAccountingGateReady}" +
            $";CallbackLeaveAccountingGateReady={result.CallbackLeaveAccountingGateReady}" +
            $";CallbackInFlightNeverNegativeReady={result.CallbackInFlightNeverNegativeReady}" +
            $";ReleaseAfterDrainGateReady={result.ReleaseAfterDrainGateReady}" +
            $";CallbackStateUnpinAfterDrainGateReady={result.CallbackStateUnpinAfterDrainGateReady}" +
            $";AccountingAddressExposed={result.AccountingAddressExposed}" +
            $";AccountingPointerProduced={result.AccountingPointerProduced}" +
            $";NativeAttachEntryLocated={result.NativeAttachEntryLocated}" +
            $";InFlightAccountingGateReady={result.InFlightAccountingGateReady}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerNativeNoThrowVTableScaffoldGate(TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult result)
    {
        return "debug-listener-native-nothrow-vtable-scaffold-gate" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";OwnerId={result.OwnerId}" +
            $";LastStatus={result.LastStatus}" +
            $";NativeAttachBridgeShapeGateReady={result.NativeAttachBridgeShapeGateReady}" +
            $";ExceptionStatusMappingGateReady={result.ExceptionStatusMappingGateReady}" +
            $";InFlightAccountingGateReady={result.InFlightAccountingGateReady}" +
            $";NoThrowVTableScaffoldReady={result.NoThrowVTableScaffoldReady}" +
            $";VTableDestructorNoThrowReady={result.VTableDestructorNoThrowReady}" +
            $";ProcessDebugTensorCallbackStubNoThrowReady={result.ProcessDebugTensorCallbackStubNoThrowReady}" +
            $";ExceptionEscapeBlocked={result.ExceptionEscapeBlocked}" +
            $";CallbackExceptionCaptureGateReady={result.CallbackExceptionCaptureGateReady}" +
            $";CallbackStatusMappingGateReady={result.CallbackStatusMappingGateReady}" +
            $";CallbackInFlightAccountingGateReady={result.CallbackInFlightAccountingGateReady}" +
            $";BorrowedDebugTensorPointerEscapeBlocked={result.BorrowedDebugTensorPointerEscapeBlocked}" +
            $";VTableAddressExposed={result.VTableAddressExposed}" +
            $";VTablePointerProduced={result.VTablePointerProduced}" +
            $";NativeAttachEntryLocated={result.NativeAttachEntryLocated}" +
            $";NativeVTableDesignReady={result.NativeVTableDesignReady}" +
            $";VTableScaffoldGateReady={result.VTableScaffoldGateReady}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";CanImplementNativeAttach={result.CanImplementNativeAttach}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerNoThrowVTableCallbackStub(TensorRtDebugListenerNoThrowVTableCallbackStubResult result)
    {
        return "debug-listener-nothrow-vtable-callback-stub" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";OwnerId={result.OwnerId}" +
            $";LastStatus={result.LastStatus}" +
            $";TensorName={SanitizeSmokeValue(result.TensorName)}" +
            $";DataType={result.DataType}" +
            $";Location={result.Location}" +
            $";ShapeRank={result.ShapeRank}" +
            $";ShapeSummary={SanitizeSmokeValue(result.ShapeSummary)}" +
            $";IsInput={result.IsInput}" +
            $";IsExecutionTensor={result.IsExecutionTensor}" +
            $";CallbackEntryCount={result.CallbackEntryCount}" +
            $";CallbackLeaveCount={result.CallbackLeaveCount}" +
            $";FailureCount={result.FailureCount}" +
            $";MinimalSafetyReady={result.MinimalSafetyReady}" +
            $";NoThrowVTableScaffoldGateReady={result.NoThrowVTableScaffoldGateReady}" +
            $";NoThrowVTableScaffoldReady={result.NoThrowVTableScaffoldReady}" +
            $";CallbackStubShapeReady={result.CallbackStubShapeReady}" +
            $";CallbackStubNoThrowReady={result.CallbackStubNoThrowReady}" +
            $";CallbackMetadataCopyReady={result.CallbackMetadataCopyReady}" +
            $";CallbackExceptionCaptureReady={result.CallbackExceptionCaptureReady}" +
            $";CallbackStatusMappingReady={result.CallbackStatusMappingReady}" +
            $";CallbackInFlightEnterReady={result.CallbackInFlightEnterReady}" +
            $";CallbackInFlightLeaveReady={result.CallbackInFlightLeaveReady}" +
            $";CallbackInFlightPairingReady={result.CallbackInFlightPairingReady}" +
            $";CallbackInFlightNeverNegativeReady={result.CallbackInFlightNeverNegativeReady}" +
            $";BorrowedDebugTensorMetadataCopyReady={result.BorrowedDebugTensorMetadataCopyReady}" +
            $";BorrowedDebugTensorPointerEscapeBlocked={result.BorrowedDebugTensorPointerEscapeBlocked}" +
            $";DebugTensorPointerExposed={result.DebugTensorPointerExposed}" +
            $";DebugTensorDataPointerExposed={result.DebugTensorDataPointerExposed}" +
            $";SetDebugListenerNonNullEnabled={result.SetDebugListenerNonNullEnabled}" +
            $";NativeAttachWouldBeBlocked={result.NativeAttachWouldBeBlocked}" +
            $";NativeVTableInstalled={result.NativeVTableInstalled}" +
            $";CallbackStubGateReady={result.CallbackStubGateReady}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";CanInstallNativeVTable={result.CanInstallNativeVTable}" +
            $";CanCallProcessDebugTensorRuntime={result.CanCallProcessDebugTensorRuntime}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";ReasonCallbackRuntimeStillBlocked={SanitizeSmokeValue(result.ReasonCallbackRuntimeStillBlocked)}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerBorrowedDebugTensorMetadataRuntimeGate(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult result)
    {
        return "debug-listener-borrowed-debug-tensor-metadata-runtime-gate" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";OwnerId={result.OwnerId}" +
            $";LastStatus={result.LastStatus}" +
            $";TensorName={SanitizeSmokeValue(result.TensorName)}" +
            $";TensorNameLength={result.TensorNameLength}" +
            $";DataType={result.DataType}" +
            $";Location={result.Location}" +
            $";TensorShapeRank={result.TensorShapeRank}" +
            $";ShapeSummary={SanitizeSmokeValue(result.ShapeSummary)}" +
            $";IsInput={result.IsInput}" +
            $";IsOutput={result.IsOutput}" +
            $";IsShapeTensor={result.IsShapeTensor}" +
            $";IsExecutionTensor={result.IsExecutionTensor}" +
            $";BorrowedTensorSafetyGateReady={result.BorrowedTensorSafetyGateReady}" +
            $";CallbackStubGateReady={result.CallbackStubGateReady}" +
            $";MetadataGateReady={result.MetadataGateReady}" +
            $";TensorNameCopied={result.TensorNameCopied}" +
            $";TensorTypeCopied={result.TensorTypeCopied}" +
            $";TensorLocationCopied={result.TensorLocationCopied}" +
            $";TensorShapeCopied={result.TensorShapeCopied}" +
            $";TensorFlagsCopied={result.TensorFlagsCopied}" +
            $";BorrowedDebugTensorMetadataCopyReady={result.BorrowedDebugTensorMetadataCopyReady}" +
            $";BorrowedDebugTensorPointerEscapeBlocked={result.BorrowedDebugTensorPointerEscapeBlocked}" +
            $";BorrowedDebugTensorDataPointerEscapeBlocked={result.BorrowedDebugTensorDataPointerEscapeBlocked}" +
            $";DebugTensorPointerExposed={result.DebugTensorPointerExposed}" +
            $";DebugTensorDataPointerExposed={result.DebugTensorDataPointerExposed}" +
            $";BorrowedDebugTensorLifetimeReady={result.BorrowedDebugTensorLifetimeReady}" +
            $";BorrowedDebugTensorDataLifetimeReady={result.BorrowedDebugTensorDataLifetimeReady}" +
            $";SetDebugListenerNonNullEnabled={result.SetDebugListenerNonNullEnabled}" +
            $";NativeVTableInstalled={result.NativeVTableInstalled}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";CanCallProcessDebugTensorRuntime={result.CanCallProcessDebugTensorRuntime}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";ReasonMetadataRuntimeStillBlocked={SanitizeSmokeValue(result.ReasonMetadataRuntimeStillBlocked)}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerNativeVTableInstallPreflight(TensorRtDebugListenerNativeVTableInstallPreflightResult result)
    {
        return "debug-listener-native-vtable-install-preflight" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";OwnerId={result.OwnerId}" +
            $";LastStatus={result.LastStatus}" +
            $";NativeOwnerLifecycleGateReady={result.NativeOwnerLifecycleGateReady}" +
            $";NativeAttachBridgeShapeGateReady={result.NativeAttachBridgeShapeGateReady}" +
            $";NativeNoThrowVTableScaffoldGateReady={result.NativeNoThrowVTableScaffoldGateReady}" +
            $";BorrowedDebugTensorMetadataGateReady={result.BorrowedDebugTensorMetadataGateReady}" +
            $";VTableInstallShapeReady={result.VTableInstallShapeReady}" +
            $";VTableInstallVersionGuardReady={result.VTableInstallVersionGuardReady}" +
            $";VTableInstallNoThrowBoundaryReady={result.VTableInstallNoThrowBoundaryReady}" +
            $";VTableInstallOwnershipDiagnosticsReady={result.VTableInstallOwnershipDiagnosticsReady}" +
            $";VTableInstallPointerFree={result.VTableInstallPointerFree}" +
            $";AttachBridgeSetDebugListenerNonNullEnabled={result.AttachBridgeSetDebugListenerNonNullEnabled}" +
            $";SetDebugListenerNonNullEnabled={result.SetDebugListenerNonNullEnabled}" +
            $";NonNullAttachStillDisabled={result.NonNullAttachStillDisabled}" +
            $";VTableAddressExposed={result.VTableAddressExposed}" +
            $";VTablePointerProduced={result.VTablePointerProduced}" +
            $";DebugTensorPointerExposed={result.DebugTensorPointerExposed}" +
            $";DebugTensorDataPointerExposed={result.DebugTensorDataPointerExposed}" +
            $";BorrowedDebugTensorPointerEscapeBlocked={result.BorrowedDebugTensorPointerEscapeBlocked}" +
            $";BorrowedDebugTensorDataPointerEscapeBlocked={result.BorrowedDebugTensorDataPointerEscapeBlocked}" +
            $";BorrowedDebugTensorLifetimeReady={result.BorrowedDebugTensorLifetimeReady}" +
            $";BorrowedDebugTensorDataLifetimeReady={result.BorrowedDebugTensorDataLifetimeReady}" +
            $";NativeVTableInstallPreflightReady={result.NativeVTableInstallPreflightReady}" +
            $";NativeVTableInstalled={result.NativeVTableInstalled}" +
            $";NativeVTableInstallRuntimeReady={result.NativeVTableInstallRuntimeReady}" +
            $";CanEnableSetDebugListenerNonNull={result.CanEnableSetDebugListenerNonNull}" +
            $";CanInstallNativeVTable={result.CanInstallNativeVTable}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";CanCallProcessDebugTensorRuntime={result.CanCallProcessDebugTensorRuntime}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";ReasonNativeVTableInstallStillBlocked={SanitizeSmokeValue(result.ReasonNativeVTableInstallStillBlocked)}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerNativeOwnerVTableInstallExperiment(TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult result)
    {
        return "debug-listener-native-owner-vtable-install-experiment" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";OwnerId={result.OwnerId}" +
            $";LastStatus={result.LastStatus}" +
            $";NativeOwnerLifecycleGateReady={result.NativeOwnerLifecycleGateReady}" +
            $";NativeAttachBridgeShapeGateReady={result.NativeAttachBridgeShapeGateReady}" +
            $";NativeNoThrowVTableScaffoldGateReady={result.NativeNoThrowVTableScaffoldGateReady}" +
            $";BorrowedDebugTensorMetadataGateReady={result.BorrowedDebugTensorMetadataGateReady}" +
            $";NativeVTableInstallPreflightReady={result.NativeVTableInstallPreflightReady}" +
            $";ExperimentShapeReady={result.ExperimentShapeReady}" +
            $";InstallAttemptGuardReady={result.InstallAttemptGuardReady}" +
            $";NonNullAttachEnabled={result.NonNullAttachEnabled}" +
            $";RuntimeProofEnabled={result.RuntimeProofEnabled}" +
            $";NativeVTableInstallAttempted={result.NativeVTableInstallAttempted}" +
            $";NativeVTableInstalled={result.NativeVTableInstalled}" +
            $";RollbackReady={result.RollbackReady}" +
            $";DetachBeforeReleaseReady={result.DetachBeforeReleaseReady}" +
            $";FailureStatusMappingReady={result.FailureStatusMappingReady}" +
            $";PointerFree={result.PointerFree}" +
            $";VTableAddressExposed={result.VTableAddressExposed}" +
            $";VTablePointerProduced={result.VTablePointerProduced}" +
            $";DebugTensorPointerExposed={result.DebugTensorPointerExposed}" +
            $";DebugTensorDataPointerExposed={result.DebugTensorDataPointerExposed}" +
            $";CanEnableSetDebugListenerNonNull={result.CanEnableSetDebugListenerNonNull}" +
            $";CanInstallNativeVTable={result.CanInstallNativeVTable}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";CanCallProcessDebugTensorRuntime={result.CanCallProcessDebugTensorRuntime}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";ReasonNativeOwnerVTableInstallStillBlocked={SanitizeSmokeValue(result.ReasonNativeOwnerVTableInstallStillBlocked)}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerRuntimeProofPrecheck(TensorRtDebugListenerRuntimeProofPrecheckResult result)
    {
        return "debug-listener-runtime-proof-precheck" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";OwnerDesignReady={result.OwnerDesignReady}" +
            $";DebugTensorMetadataCopied={result.DebugTensorMetadataCopied}" +
            $";DisposeReleaseReady={result.DisposeReleaseReady}" +
            $";PointerFreeSurfaceReady={result.PointerFreeSurfaceReady}" +
            $";AttachDetachDesignGateReady={result.AttachDetachDesignGateReady}" +
            $";AttachControlAvailable={result.AttachControlAvailable}" +
            $";DetachClearControlAvailable={result.DetachClearControlAvailable}" +
            $";ManagedOwnerStateMachineReady={result.ManagedOwnerStateMachineReady}" +
            $";LineSpecificAttachDetachReady={result.LineSpecificAttachDetachReady}" +
            $";StableNativeOwnerAddressReady={result.StableNativeOwnerAddressReady}" +
            $";NoThrowNativeVTableReady={result.NoThrowNativeVTableReady}" +
            $";NativeVTableReady={result.NativeVTableReady}" +
            $";ExceptionToStatusMappingReady={result.ExceptionToStatusMappingReady}" +
            $";BorrowedTensorSafetyGateReady={result.BorrowedTensorSafetyGateReady}" +
            $";AttachVTableSafetyGateReady={result.AttachVTableSafetyGateReady}" +
            $";NativeAttachNoThrowPreflightReady={result.NativeAttachNoThrowPreflightReady}" +
            $";NativeOwnerAddressDesignGateReady={result.NativeOwnerAddressDesignGateReady}" +
            $";NativeNoThrowVTableDesignGateReady={result.NativeNoThrowVTableDesignGateReady}" +
            $";NativeAttachEntryDesignGateReady={result.NativeAttachEntryDesignGateReady}" +
            $";NativeDetachBeforeReleaseDesignGateReady={result.NativeDetachBeforeReleaseDesignGateReady}" +
            $";NativeOwnerLifecycleDryRunReady={result.NativeOwnerLifecycleDryRunReady}" +
            $";NativeAttachEntryRuntimeScaffoldReady={result.NativeAttachEntryRuntimeScaffoldReady}" +
            $";NativeOwnerStableIdentityReady={result.NativeOwnerStableIdentityReady}" +
            $";OwnerIdentityDiagnosticsReady={result.OwnerIdentityDiagnosticsReady}" +
            $";OwnerIdentityPointerFree={result.OwnerIdentityPointerFree}" +
            $";NativeOwnerNonCopyableStorageReady={result.NativeOwnerNonCopyableStorageReady}" +
            $";NativeOwnerCopyBlocked={result.NativeOwnerCopyBlocked}" +
            $";NativeOwnerMoveBlocked={result.NativeOwnerMoveBlocked}" +
            $";NativeOwnerAddressExposed={result.NativeOwnerAddressExposed}" +
            $";NativeOwnerPointerProduced={result.NativeOwnerPointerProduced}" +
            $";NativeNoThrowDestructorGateReady={result.NativeNoThrowDestructorGateReady}" +
            $";DestructorNoThrowScaffoldReady={result.DestructorNoThrowScaffoldReady}" +
            $";DestructorExceptionEscapeBlocked={result.DestructorExceptionEscapeBlocked}" +
            $";DestructorAddressExposed={result.DestructorAddressExposed}" +
            $";DestructorPointerProduced={result.DestructorPointerProduced}" +
            $";NativeOwnerLifecycleGateReady={result.NativeOwnerLifecycleGateReady}" +
            $";ManagedDisposeSnapshotReady={result.ManagedDisposeSnapshotReady}" +
            $";LifecycleScaffoldReady={result.LifecycleScaffoldReady}" +
            $";ReleaseHookOrderingGateReady={result.ReleaseHookOrderingGateReady}" +
            $";DisposeIdempotencyGateReady={result.DisposeIdempotencyGateReady}" +
            $";InFlightDrainGateReady={result.InFlightDrainGateReady}" +
            $";CallbackStateUnpinAfterDetachGateReady={result.CallbackStateUnpinAfterDetachGateReady}" +
            $";DelegateUnpinAfterDetachGateReady={result.DelegateUnpinAfterDetachGateReady}" +
            $";LifecycleAddressExposed={result.LifecycleAddressExposed}" +
            $";LifecyclePointerProduced={result.LifecyclePointerProduced}" +
            $";NativeAttachBridgeShapeGateReady={result.NativeAttachBridgeShapeGateReady}" +
            $";AttachBridgeShapeReady={result.AttachBridgeShapeReady}" +
            $";AttachBridgeNoThrowBoundaryReady={result.AttachBridgeNoThrowBoundaryReady}" +
            $";AttachBridgeVersionGuardReady={result.AttachBridgeVersionGuardReady}" +
            $";AttachBridgeOwnershipDiagnosticsReady={result.AttachBridgeOwnershipDiagnosticsReady}" +
            $";AttachBridgePointerFree={result.AttachBridgePointerFree}" +
            $";NonNullAttachStillDisabled={result.NonNullAttachStillDisabled}" +
            $";ExceptionStatusMappingGateReady={result.ExceptionStatusMappingGateReady}" +
            $";NativeCallbackExceptionCaptureReady={result.NativeCallbackExceptionCaptureReady}" +
            $";CallbackStatusMappingGateReady={result.CallbackStatusMappingGateReady}" +
            $";ExceptionEscapeBlocked={result.ExceptionEscapeBlocked}" +
            $";DiagnosticCopyReady={result.DiagnosticCopyReady}" +
            $";InFlightAccountingGateReady={result.InFlightAccountingGateReady}" +
            $";CallbackEnterAccountingGateReady={result.CallbackEnterAccountingGateReady}" +
            $";CallbackLeaveAccountingGateReady={result.CallbackLeaveAccountingGateReady}" +
            $";CallbackInFlightNeverNegativeReady={result.CallbackInFlightNeverNegativeReady}" +
            $";ReleaseAfterDrainGateReady={result.ReleaseAfterDrainGateReady}" +
            $";CallbackStateUnpinAfterDrainGateReady={result.CallbackStateUnpinAfterDrainGateReady}" +
            $";NativeNoThrowVTableScaffoldGateReady={result.NativeNoThrowVTableScaffoldGateReady}" +
            $";NoThrowVTableScaffoldReady={result.NoThrowVTableScaffoldReady}" +
            $";VTableDestructorNoThrowReady={result.VTableDestructorNoThrowReady}" +
            $";ProcessDebugTensorCallbackStubNoThrowReady={result.ProcessDebugTensorCallbackStubNoThrowReady}" +
            $";VTableAddressExposed={result.VTableAddressExposed}" +
            $";VTablePointerProduced={result.VTablePointerProduced}" +
            $";NativeAttachEntryLocated={result.NativeAttachEntryLocated}" +
            $";NativeDetachEntryLocated={result.NativeDetachEntryLocated}" +
            $";AttachEntryParameterShapeReady={result.AttachEntryParameterShapeReady}" +
            $";LineSpecificAttachEntryDesignReady={result.LineSpecificAttachEntryDesignReady}" +
            $";AttachEntryNoThrowReady={result.AttachEntryNoThrowReady}" +
            $";AttachEntryNoThrowBoundaryReady={result.AttachEntryNoThrowBoundaryReady}" +
            $";AttachEntryVersionGuardReady={result.AttachEntryVersionGuardReady}" +
            $";AttachEntryOwnershipReady={result.AttachEntryOwnershipReady}" +
            $";AttachEntryOwnershipDiagnosticsReady={result.AttachEntryOwnershipDiagnosticsReady}" +
            $";DetachBeforeReleaseReady={result.DetachBeforeReleaseReady}" +
            $";ReleaseHookOrderingReady={result.ReleaseHookOrderingReady}" +
            $";DisposeIdempotencyReady={result.DisposeIdempotencyReady}" +
            $";InFlightDrainBeforeReleaseReady={result.InFlightDrainBeforeReleaseReady}" +
            $";CallbackStateUnpinAfterDetachReady={result.CallbackStateUnpinAfterDetachReady}" +
            $";DelegateUnpinAfterDetachReady={result.DelegateUnpinAfterDetachReady}" +
            $";StableNativeOwnerAddressDesignReady={result.StableNativeOwnerAddressDesignReady}" +
            $";ManagedCallbackKeepAliveDesignReady={result.ManagedCallbackKeepAliveDesignReady}" +
            $";NativeOwnerNonCopyableReady={result.NativeOwnerNonCopyableReady}" +
            $";NativeOwnerDisposeOrderReady={result.NativeOwnerDisposeOrderReady}" +
            $";NativeOwnerReleaseHookReady={result.NativeOwnerReleaseHookReady}" +
            $";NativeOwnerInFlightDrainReady={result.NativeOwnerInFlightDrainReady}" +
            $";NoThrowNativeDestructorReady={result.NoThrowNativeDestructorReady}" +
            $";NativeOwnerLifecycleReady={result.NativeOwnerLifecycleReady}" +
            $";NoThrowVTableDesignReady={result.NoThrowVTableDesignReady}" +
            $";ExceptionToStatusMappingDesignReady={result.ExceptionToStatusMappingDesignReady}" +
            $";NativeVTableTrampolineReady={result.NativeVTableTrampolineReady}" +
            $";CallbackExceptionCaptureReady={result.CallbackExceptionCaptureReady}" +
            $";CallbackStatusMappingReady={result.CallbackStatusMappingReady}" +
            $";CallbackInFlightAccountingReady={result.CallbackInFlightAccountingReady}" +
            $";CanImplementNativeAttach={result.CanImplementNativeAttach}" +
            $";BorrowedDebugTensorPointerEscapeBlocked={result.BorrowedDebugTensorPointerEscapeBlocked}" +
            $";BorrowedDebugTensorLifetimeReady={result.BorrowedDebugTensorLifetimeReady}" +
            $";BorrowedDebugTensorDataLifetimeReady={result.BorrowedDebugTensorDataLifetimeReady}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerRuntimeProofAttemptPreflight(TensorRtDebugListenerRuntimeProofAttemptPreflightResult result)
    {
        return "debug-listener-runtime-proof-attempt-preflight" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";Line={(int)result.Line}" +
            $";Status={result.Status}" +
            $";NativeAttachEntryLocated={result.NativeAttachEntryLocated}" +
            $";NonNullAttachStillDisabled={result.NonNullAttachStillDisabled}" +
            $";NativeOwnerLifecycleReady={result.NativeOwnerLifecycleReady}" +
            $";CanImplementNativeAttach={result.CanImplementNativeAttach}" +
            $";NativeVTableReady={result.NativeVTableReady}" +
            $";NoThrowVTableDesignReady={result.NoThrowVTableDesignReady}" +
            $";NativeVTableTrampolineReady={result.NativeVTableTrampolineReady}" +
            $";CallbackExceptionCaptureReady={result.CallbackExceptionCaptureReady}" +
            $";CallbackStatusMappingReady={result.CallbackStatusMappingReady}" +
            $";CallbackInFlightAccountingReady={result.CallbackInFlightAccountingReady}" +
            $";VTableAddressExposed={result.VTableAddressExposed}" +
            $";VTablePointerProduced={result.VTablePointerProduced}" +
            $";BorrowedDebugTensorPointerEscapeBlocked={result.BorrowedDebugTensorPointerEscapeBlocked}" +
            $";BorrowedDebugTensorLifetimeReady={result.BorrowedDebugTensorLifetimeReady}" +
            $";BorrowedDebugTensorDataLifetimeReady={result.BorrowedDebugTensorDataLifetimeReady}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";FullPackageConsumerRuntimeEvidenceReady={result.FullPackageConsumerRuntimeEvidenceReady}" +
            $";PrecheckCanAttemptRuntimeProof={result.PrecheckCanAttemptRuntimeProof}" +
            $";CanEnableSetDebugListenerNonNull={result.CanEnableSetDebugListenerNonNull}" +
            $";CanInstallNativeVTable={result.CanInstallNativeVTable}" +
            $";CanCallProcessDebugTensorRuntime={result.CanCallProcessDebugTensorRuntime}" +
            $";CanPromoteRealCallbackRuntime={result.CanPromoteRealCallbackRuntime}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";ReasonNonNullAttachStillBlocked={SanitizeSmokeValue(result.ReasonNonNullAttachStillBlocked)}" +
            $";ReasonNativeVTableStillBlocked={SanitizeSmokeValue(result.ReasonNativeVTableStillBlocked)}" +
            $";ReasonRuntimeProofStillBlocked={SanitizeSmokeValue(result.ReasonRuntimeProofStillBlocked)}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerRealNonNullAttachRuntimeSmoke(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult result)
    {
        return "debug-listener-real-non-null-attach-runtime-smoke" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";TensorRtLine={result.TensorRtLine}" +
            $";RuntimePackageKey={SanitizeSmokeValue(result.RuntimePackageKey)}" +
            $";Status={result.Status}" +
            $";OptInEnabled={result.OptInEnabled}" +
            $";FullPackageConsumerReport={result.FullPackageConsumerReport}" +
            $";AttachGuardReady={result.AttachGuardReady}" +
            $";NativeVTableReady={result.NativeVTableReady}" +
            $";BorrowedDebugTensorRuntimeReady={result.BorrowedDebugTensorRuntimeReady}" +
            $";CallbackInvocationReady={result.CallbackInvocationReady}" +
            $";AttachAttempted={result.AttachAttempted}" +
            $";AttachSucceeded={result.AttachSucceeded}" +
            $";DetachAttempted={result.DetachAttempted}" +
            $";DetachSucceeded={result.DetachSucceeded}" +
            $";RollbackAttempted={result.RollbackAttempted}" +
            $";RollbackSucceeded={result.RollbackSucceeded}" +
            $";NativeVTableInstalled={result.NativeVTableInstalled}" +
            $";ProcessDebugTensorInvoked={result.ProcessDebugTensorInvoked}" +
            $";InvocationCount={result.InvocationCount}" +
            $";AllocationCount={result.AllocationCount}" +
            $";ReleaseCount={result.ReleaseCount}" +
            $";FailureCount={result.FailureCount}" +
            $";InFlightCallbackCount={result.InFlightCallbackCount}" +
            $";LastStatus={result.LastStatus}" +
            $";LastDiagnostic={SanitizeSmokeValue(result.LastDiagnostic)}" +
            $";ReportPointerFree={result.ReportPointerFree}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";CanPromoteRealCallbackRuntime={result.CanPromoteRealCallbackRuntime}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";ReasonRuntimeProofStillBlocked={SanitizeSmokeValue(result.ReasonRuntimeProofStillBlocked)}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerProcessDebugTensorCallbackTrampoline(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult result)
    {
        return "debug-listener-process-debug-tensor-callback-trampoline" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";TensorRtLine={result.TensorRtLine}" +
            $";RuntimePackageKey={SanitizeSmokeValue(result.RuntimePackageKey)}" +
            $";Status={result.Status}" +
            $";TrampolineShapeReady={result.TrampolineShapeReady}" +
            $";NativeCallbackEntryLocated={result.NativeCallbackEntryLocated}" +
            $";NoThrowCallbackEntryReady={result.NoThrowCallbackEntryReady}" +
            $";ExceptionCaptureReady={result.ExceptionCaptureReady}" +
            $";CallbackStatusMappingReady={result.CallbackStatusMappingReady}" +
            $";InFlightAccountingReady={result.InFlightAccountingReady}" +
            $";DetachBeforeReleaseReady={result.DetachBeforeReleaseReady}" +
            $";BorrowedDebugTensorMetadataCopyReady={result.BorrowedDebugTensorMetadataCopyReady}" +
            $";BorrowedDebugTensorPointerExposed={result.BorrowedDebugTensorPointerExposed}" +
            $";BorrowedDebugTensorDataPointerExposed={result.BorrowedDebugTensorDataPointerExposed}" +
            $";PointerFreeSurfaceReady={result.PointerFreeSurfaceReady}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";OptInEnabled={result.OptInEnabled}" +
            $";FullPackageConsumerReport={result.FullPackageConsumerReport}" +
            $";AttachAttempted={result.AttachAttempted}" +
            $";AttachSucceeded={result.AttachSucceeded}" +
            $";NativeVTableInstalled={result.NativeVTableInstalled}" +
            $";ProcessDebugTensorInvoked={result.ProcessDebugTensorInvoked}" +
            $";InvocationCount={result.InvocationCount}" +
            $";CallbackStubEntryCount={result.CallbackStubEntryCount}" +
            $";CallbackStubLeaveCount={result.CallbackStubLeaveCount}" +
            $";FailureCount={result.FailureCount}" +
            $";InFlightCallbackCount={result.InFlightCallbackCount}" +
            $";LastStatus={result.LastStatus}" +
            $";LastDiagnostic={SanitizeSmokeValue(result.LastDiagnostic)}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";CanPromoteRealCallbackRuntime={result.CanPromoteRealCallbackRuntime}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";TensorName={SanitizeSmokeValue(result.Metadata.TensorName)}" +
            $";TensorNameLength={result.Metadata.TensorNameLength}" +
            $";DataType={result.Metadata.DataType}" +
            $";Location={result.Metadata.Location}" +
            $";TensorShapeRank={result.Metadata.TensorShapeRank}" +
            $";ShapeSummary={SanitizeSmokeValue(result.Metadata.ShapeSummary)}" +
            $";MetadataCopied={result.Metadata.MetadataCopied}" +
            $";ReasonRuntimeProofStillBlocked={SanitizeSmokeValue(result.ReasonRuntimeProofStillBlocked)}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerRealCallbackRuntimeProof(TensorRtDebugListenerRealCallbackRuntimeProofResult result)
    {
        return "debug-listener-real-callback-runtime-proof" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";TensorRtLine={result.TensorRtLine}" +
            $";RuntimePackageKey={SanitizeSmokeValue(result.RuntimePackageKey)}" +
            $";Status={result.Status}" +
            $";OptInEnabled={result.OptInEnabled}" +
            $";FullPackageConsumerReport={result.FullPackageConsumerReport}" +
            $";RuntimeSmokeReady={result.RuntimeSmokeReady}" +
            $";TrampolineShapeReady={result.TrampolineShapeReady}" +
            $";AttachAttempted={result.AttachAttempted}" +
            $";AttachSucceeded={result.AttachSucceeded}" +
            $";DetachAttempted={result.DetachAttempted}" +
            $";DetachSucceeded={result.DetachSucceeded}" +
            $";RollbackAttempted={result.RollbackAttempted}" +
            $";RollbackSucceeded={result.RollbackSucceeded}" +
            $";NativeVTableInstalled={result.NativeVTableInstalled}" +
            $";ProcessDebugTensorInvoked={result.ProcessDebugTensorInvoked}" +
            $";InvocationCount={result.InvocationCount}" +
            $";FailureCount={result.FailureCount}" +
            $";InFlightCallbackCount={result.InFlightCallbackCount}" +
            $";BorrowedDebugTensorMetadataCopied={result.BorrowedDebugTensorMetadataCopied}" +
            $";PointerFreeSurfaceReady={result.PointerFreeSurfaceReady}" +
            $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
            $";AttemptedNoInvocation={result.AttemptedNoInvocation}" +
            $";LastStatus={result.LastStatus}" +
            $";LastDiagnostic={SanitizeSmokeValue(result.LastDiagnostic)}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";CanPromoteRealCallbackRuntime={result.CanPromoteRealCallbackRuntime}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
            $";ReasonRuntimeProofStillBlocked={SanitizeSmokeValue(result.ReasonRuntimeProofStillBlocked)}" +
            $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string FormatDebugListenerCallbackProofGapReport(TensorRtDebugListenerCallbackProofGapReportResult result)
    {
        return "debug-listener-callback-proof-gap-report" +
            $";EvidenceKind={result.EvidenceKind}" +
            $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
            $";RealCallbackRuntime={result.RealCallbackRuntime}" +
            $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
            $";CallbackKind={result.CallbackKind}" +
            $";TensorRtLine={result.TensorRtLine}" +
            $";RuntimePackageKey={SanitizeSmokeValue(result.RuntimePackageKey)}" +
            $";Status={result.Status}" +
            $";NonNullAttachStillDisabled={result.NonNullAttachStillDisabled}" +
            $";NativeAttachEntryReady={result.NativeAttachEntryReady}" +
            $";NativeVTableInstallBlocked={result.NativeVTableInstallBlocked}" +
            $";NoThrowCallbackEntryReady={result.NoThrowCallbackEntryReady}" +
            $";ExceptionStatusMappingReady={result.ExceptionStatusMappingReady}" +
            $";InFlightAccountingReady={result.InFlightAccountingReady}" +
            $";BorrowedDebugTensorMetadataCopied={result.BorrowedDebugTensorMetadataCopied}" +
            $";DetachRollbackReady={result.DetachRollbackReady}" +
            $";ProcessDebugTensorRuntimeInvoked={result.ProcessDebugTensorRuntimeInvoked}" +
            $";FullPackageConsumerRuntimeProofReady={result.FullPackageConsumerRuntimeProofReady}" +
            $";PointerFreeSurfaceReady={result.PointerFreeSurfaceReady}" +
            $";AttemptedNoInvocation={result.AttemptedNoInvocation}" +
            $";InvocationCount={result.InvocationCount}" +
            $";FailureCount={result.FailureCount}" +
            $";InFlightCallbackCount={result.InFlightCallbackCount}" +
            $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
            $";CanPromoteRealCallbackRuntime={result.CanPromoteRealCallbackRuntime}" +
            $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
            $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
            $";GapReasonCount={result.GapReasonCount}" +
            $";PrimaryGapReason={SanitizeSmokeValue(result.PrimaryGapReason)}" +
            $";RuntimeProofBlockerCategory={SanitizeSmokeValue(result.RuntimeProofBlockerCategory)}" +
            $";PackageConsumerRuntimeProofRequired={result.PackageConsumerRuntimeProofRequired}" +
            $";RuntimeInvocationRequired={result.RuntimeInvocationRequired}" +
            $";EvidenceSource={SanitizeSmokeValue(result.EvidenceSource)}" +
            $";NextOwnerAction={SanitizeSmokeValue(result.NextOwnerAction)}" +
            $";GapReasons={SanitizeSmokeValue(string.Join(",", result.GapReasons))}" +
            $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
    }

    private static string ReadPrototypeProperty(object result, string propertyName)
    {
        PropertyInfo? property = result.GetType().GetProperty(propertyName, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        object? value = property?.GetValue(result);
        return Convert.ToString(value, CultureInfo.InvariantCulture) ?? string.Empty;
    }

    private static string SanitizeSmokeValue(string value)
    {
        return value.Replace(Environment.NewLine, " ").Replace(';', ',');
    }

    private static TensorRtApiLine ResolveProbeLine(string requestedLine)
    {
        if (string.Equals(requestedLine, "auto", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtApiLine.TensorRt11;
        }

        TensorRtApiLine? line = ResolveTensorRtLineWithoutSnapshot(requestedLine);
        if (line.HasValue)
        {
            return line.Value;
        }

        throw new ArgumentException("TensorRT line must be auto, 8, 10, or 11.", nameof(requestedLine));
    }

    private static TensorRtApiLine? ResolveTensorRtLine(TensorRtEnvironmentSnapshot snapshot, string requestedLine)
    {
        if (string.Equals(requestedLine, "auto", StringComparison.OrdinalIgnoreCase))
        {
            if (snapshot.TensorRt11.RuntimeCreationSupported && snapshot.TensorRt11.BuilderCreationSupported)
            {
                return TensorRtApiLine.TensorRt11;
            }

            if (snapshot.TensorRt10.RuntimeCreationSupported && snapshot.TensorRt10.BuilderCreationSupported)
            {
                return TensorRtApiLine.TensorRt10;
            }

            if (snapshot.TensorRt8.RuntimeCreationSupported && snapshot.TensorRt8.BuilderCreationSupported)
            {
                return TensorRtApiLine.TensorRt8;
            }

            return null;
        }

        return ResolveTensorRtLineWithoutSnapshot(requestedLine);
    }

    private static TensorRtApiLine? ResolveTensorRtLineWithoutSnapshot(string requestedLine)
    {
        if (string.Equals(requestedLine, "8", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(requestedLine, "trt8", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtApiLine.TensorRt8;
        }

        if (string.Equals(requestedLine, "10", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(requestedLine, "trt10", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtApiLine.TensorRt10;
        }

        if (string.Equals(requestedLine, "11", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(requestedLine, "trt11", StringComparison.OrdinalIgnoreCase))
        {
            return TensorRtApiLine.TensorRt11;
        }

        throw new ArgumentException("TensorRT line must be auto, 8, 10, or 11.", nameof(requestedLine));
    }

    private static TensorRtAdapterInfo GetAdapter(TensorRtEnvironmentSnapshot snapshot, TensorRtApiLine line)
    {
        return line switch
        {
            TensorRtApiLine.TensorRt8 => snapshot.TensorRt8,
            TensorRtApiLine.TensorRt10 => snapshot.TensorRt10,
            TensorRtApiLine.TensorRt11 => snapshot.TensorRt11,
            _ => snapshot.TensorRt11
        };
    }

    private static bool IsSkippableEnvironmentException(Exception exception)
    {
        if (exception is DllNotFoundException || exception is BadImageFormatException)
        {
            return true;
        }

        if (exception is BridgeProbeException bridgeProbe)
        {
            return bridgeProbe.StatusCode == BridgeStatusCode.DependencyMissing ||
                bridgeProbe.StatusCode == BridgeStatusCode.NotSupported ||
                bridgeProbe.StatusCode == BridgeStatusCode.InvalidState ||
                (bridgeProbe.StatusCode == BridgeStatusCode.RuntimeError &&
                    bridgeProbe.Message.Contains("structured exception", StringComparison.OrdinalIgnoreCase));
        }

        return false;
    }
}

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

internal static class Program
{
    private static void Main(string[] args)
    {
        string requestedLine = JYPPX.SampleSupport.SampleCommandLine.GetStringArgument(args, "--tensor-rt-line", "auto");
        bool dependencyProbeOnly = JYPPX.SampleSupport.SampleCommandLine.HasSwitch(args, "--dependency-probe-only");

        Console.WriteLine($"PluginRegistryInventorySmokeRunner TensorRtLineRequest={requestedLine} DependencyProbeOnly={dependencyProbeOnly}");

        if (dependencyProbeOnly)
        {
            TensorRtApiLine probeLine = ResolveProbeLine(requestedLine);
            PrintDependencyProbe(probeLine);
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

        try
        {
            RunPluginRegistryInventorySmoke(line.Value);
        }
        catch (Exception exception) when (IsSkippableEnvironmentException(exception))
        {
            Console.WriteLine($"Skipped=True Reason={exception.GetType().Name}:{exception.Message}");
            return;
        }

        Console.WriteLine("PluginRegistryInventorySmokeRunner Passed=True");
    }

    private static void RunPluginRegistryInventorySmoke(TensorRtApiLine line)
    {
        if (line == TensorRtApiLine.TensorRt8)
        {
            Console.WriteLine("TensorRt8GlobalAndCapabilityPluginRegistriesSkipped=True Reason=GlobalRuntimeAndCapabilityRegistriesRemainDeferred");
            RunRuntimeLocalPluginRegistrySmoke(line);
            RunBuilderOwnedPluginRegistrySmoke(line);
            return;
        }

        if (!TensorRtEnvironmentProbe.TryIsGlobalPluginRegistryAvailable(line, out bool globalExists, out string globalExistsDiagnostic))
        {
            Console.WriteLine($"GlobalPluginRegistry Skipped=True Reason={globalExistsDiagnostic}");
        }
        else
        {
            Console.WriteLine($"GlobalPluginRegistry Exists={globalExists}");
        }

        TensorRtPluginRegistryInventory? globalInventory = null;
        string globalDiagnostic = string.Empty;
        bool globalInventoryRead = globalExists &&
            TensorRtEnvironmentProbe.TryGetGlobalPluginRegistryInventory(line, out globalInventory, out globalDiagnostic);

        if (globalInventoryRead)
        {
            ValidateInventory(globalInventory!, "GlobalPluginRegistry");
            Console.WriteLine(FormatInventory("GlobalPluginRegistry", globalInventory!));
            ValidateLookup(line, globalInventory!, TensorRtEngineCapability.Standard, useBuilderCapability: false);
        }
        else if (!globalExists)
        {
            Console.WriteLine("GlobalPluginRegistry Skipped=True Reason=RegistryUnavailable");
        }
        else
        {
            Console.WriteLine($"GlobalPluginRegistry Skipped=True Reason={globalDiagnostic}");
        }

        if (!TensorRtEnvironmentProbe.TryIsBuilderCapabilityPluginRegistryAvailable(line, TensorRtEngineCapability.Standard, out bool exists, out string existsDiagnostic))
        {
            Console.WriteLine($"BuilderCapabilityPluginRegistry Skipped=True Reason={existsDiagnostic}");
        }
        else
        {
            Console.WriteLine($"BuilderCapabilityPluginRegistry Exists={exists} Capability={TensorRtEngineCapability.Standard}");
            if (!exists)
            {
                Console.WriteLine("BuilderCapabilityPluginRegistry Skipped=True Reason=RegistryUnavailable");
            }
            else
            {
                if (TensorRtEnvironmentProbe.TryIsBuilderSafePluginRegistryAvailable(line, TensorRtEngineCapability.Standard, out bool safeExists, out string safeDiagnostic))
                {
                    Console.WriteLine($"BuilderSafePluginRegistry Exists={safeExists} Capability={TensorRtEngineCapability.Standard}");
                }
                else
                {
                    Console.WriteLine($"BuilderSafePluginRegistry Skipped=True Reason={safeDiagnostic}");
                }

                if (!TensorRtEnvironmentProbe.TryGetBuilderCapabilityPluginRegistryInventory(line, TensorRtEngineCapability.Standard, out TensorRtPluginRegistryInventory? builderInventory, out string builderDiagnostic))
                {
                    Console.WriteLine($"BuilderCapabilityPluginRegistry Skipped=True Reason={builderDiagnostic}");
                }
                else
                {
                    ValidateInventory(builderInventory!, "BuilderCapabilityPluginRegistry");
                    Console.WriteLine(FormatInventory("BuilderCapabilityPluginRegistry", builderInventory!));
                    ValidateLookup(line, builderInventory!, TensorRtEngineCapability.Standard, useBuilderCapability: true);
                }
            }
        }

        RunRuntimeLocalPluginRegistrySmoke(line);
        RunBuilderOwnedPluginRegistrySmoke(line);
    }

    private static void RunRuntimeLocalPluginRegistrySmoke(TensorRtApiLine line)
    {
        try
        {
            using TensorRtLogger logger = new TensorRtLogger(line);
            using TensorRtRuntime runtime = new TensorRtRuntime(logger);

            if (!runtime.TryIsPluginRegistryAvailable(out bool exists, out string existsDiagnostic))
            {
                Console.WriteLine($"RuntimePluginRegistry Skipped=True Reason={existsDiagnostic}");
                return;
            }

            Console.WriteLine($"RuntimePluginRegistry Exists={exists}");
            if (!exists)
            {
                Console.WriteLine("RuntimePluginRegistry Skipped=True Reason=RegistryUnavailable");
                return;
            }

            if (!runtime.TryGetPluginRegistryInventory(out TensorRtPluginRegistryInventory runtimeInventory, out string diagnostic))
            {
                Console.WriteLine($"RuntimePluginRegistry Skipped=True Reason={diagnostic}");
                return;
            }

            ValidateInventory(runtimeInventory, "RuntimePluginRegistry");
            Console.WriteLine(FormatInventory("RuntimePluginRegistry", runtimeInventory));
            ValidateRuntimeLocalLookup(runtime, runtimeInventory);
        }
        catch (Exception exception) when (IsSkippableEnvironmentException(exception))
        {
            Console.WriteLine($"RuntimePluginRegistry Skipped=True Reason={exception.GetType().Name}:{exception.Message}");
        }
    }

    private static void RunBuilderOwnedPluginRegistrySmoke(TensorRtApiLine line)
    {
        try
        {
            using TensorRtLogger logger = new TensorRtLogger(line);
            using TensorRtBuilder builder = new TensorRtBuilder(logger);

            if (!builder.TryIsPluginRegistryAvailable(out bool exists, out string existsDiagnostic))
            {
                Console.WriteLine($"BuilderPluginRegistry Skipped=True Reason={existsDiagnostic}");
                return;
            }

            Console.WriteLine($"BuilderPluginRegistry Exists={exists}");
            if (!exists)
            {
                Console.WriteLine("BuilderPluginRegistry Skipped=True Reason=RegistryUnavailable");
                return;
            }

            if (!builder.TryGetPluginRegistryInventory(out TensorRtPluginRegistryInventory builderOwnedInventory, out string diagnostic))
            {
                Console.WriteLine($"BuilderPluginRegistry Skipped=True Reason={diagnostic}");
                return;
            }

            ValidateInventory(builderOwnedInventory, "BuilderPluginRegistry");
            Console.WriteLine(FormatInventory("BuilderPluginRegistry", builderOwnedInventory));
            ValidateBuilderOwnedLookup(builder, builderOwnedInventory);
        }
        catch (Exception exception) when (IsSkippableEnvironmentException(exception))
        {
            Console.WriteLine($"BuilderPluginRegistry Skipped=True Reason={exception.GetType().Name}:{exception.Message}");
        }
    }

    private static void ValidateInventory(TensorRtPluginRegistryInventory inventory, string label)
    {
        TensorRtPluginRegistryInventoryDiagnostics diagnostics = inventory.GetDiagnostics();
        Console.WriteLine($"{label} PluginRegistryInventoryDiagnostics IsConsistent={diagnostics.IsConsistent} CreatorCount={diagnostics.CreatorCount} SummaryCount={diagnostics.SummaryCount} TotalFieldCount={diagnostics.TotalFieldCount} EmptyNameCount={diagnostics.EmptyNameCount} EmptyVersionCount={diagnostics.EmptyVersionCount} EmptyFieldNameCount={diagnostics.EmptyFieldNameCount} NegativeFieldLengthCount={diagnostics.NegativeFieldLengthCount}");

        if (!diagnostics.IsConsistent)
        {
            throw new InvalidOperationException($"{label} copied metadata diagnostics failed: {diagnostics.DiagnosticSummary}");
        }

        if (inventory.CreatorCount < 0)
        {
            throw new InvalidOperationException($"{label} reported a negative creator count.");
        }

        if (inventory.CreatorCount != inventory.Creators.Count)
        {
            throw new InvalidOperationException($"{label} creator count does not match copied creator list.");
        }

        if (diagnostics.CreatorCount != inventory.CreatorCount)
        {
            throw new InvalidOperationException($"{label} diagnostic creator count does not match copied creator list.");
        }

        if (inventory.RecursiveCreatorCount.HasValue && inventory.RecursiveCreatorCount.Value < inventory.CreatorCount)
        {
            throw new InvalidOperationException($"{label} recursive creator count is smaller than creator count.");
        }

        if (inventory.CreatorCount == 0)
        {
            IReadOnlyList<TensorRtPluginCreatorSummary> emptySummaries = inventory.GetCreatorSummaries();
            if (emptySummaries.Count != 0)
            {
                throw new InvalidOperationException($"{label} returned creator summaries for an empty inventory.");
            }

            IReadOnlyList<TensorRtPluginFieldSummary> emptyFieldSummaries = inventory.GetFieldSummaries();
            if (emptyFieldSummaries.Count != 0)
            {
                throw new InvalidOperationException($"{label} returned field summaries for an empty inventory.");
            }

            return;
        }

        IReadOnlyList<TensorRtPluginCreatorSummary> summaries = inventory.GetCreatorSummaries(maxCreators: 1);
        if (summaries.Count != 1)
        {
            throw new InvalidOperationException($"{label} did not return the first creator summary.");
        }

        TensorRtPluginCreatorInfo first = inventory.Creators[0];
        TensorRtPluginCreatorSummary firstSummary = summaries[0];
        if (string.IsNullOrWhiteSpace(first.Name) || string.IsNullOrWhiteSpace(first.Version))
        {
            throw new InvalidOperationException($"{label} first creator did not include name and version.");
        }

        if (!string.Equals(first.Name, firstSummary.Name, StringComparison.Ordinal) ||
            !string.Equals(first.Version, firstSummary.Version, StringComparison.Ordinal) ||
            !string.Equals(first.Namespace, firstSummary.Namespace, StringComparison.Ordinal) ||
            first.ApiLanguage != firstSummary.ApiLanguage ||
            first.Fields.Count != firstSummary.FieldCount)
        {
            throw new InvalidOperationException($"{label} first creator summary does not match copied creator metadata.");
        }

        if (first.Fields.Count < 0)
        {
            throw new InvalidOperationException($"{label} first creator reported a negative field count.");
        }

        IReadOnlyList<TensorRtPluginFieldSummary> fieldSummaries = inventory.GetFieldSummaries(maxCreators: 1, maxFieldsPerCreator: 1);
        if (fieldSummaries.Count > 0)
        {
            TensorRtPluginFieldSummary firstFieldSummary = fieldSummaries[0];
            TensorRtPluginFieldInfo firstField = first.Fields[0];
            if (firstFieldSummary.CreatorIndex != first.Index ||
                !string.Equals(firstFieldSummary.CreatorName, first.Name, StringComparison.Ordinal) ||
                !string.Equals(firstFieldSummary.CreatorVersion, first.Version, StringComparison.Ordinal) ||
                !string.Equals(firstFieldSummary.CreatorNamespace, first.Namespace, StringComparison.Ordinal) ||
                firstFieldSummary.FieldIndex != 0 ||
                !string.Equals(firstFieldSummary.FieldName, firstField.Name, StringComparison.Ordinal) ||
                firstFieldSummary.FieldType != firstField.FieldType ||
                firstFieldSummary.Length != firstField.Length ||
                firstFieldSummary.HasData != firstField.HasData)
            {
                throw new InvalidOperationException($"{label} first field summary does not match copied creator field metadata.");
            }

            Console.WriteLine($"{label} FieldSummary Creator={firstFieldSummary.CreatorName}/{firstFieldSummary.CreatorVersion}/{firstFieldSummary.CreatorNamespace} Field={firstFieldSummary.FieldName} Type={firstFieldSummary.FieldType} Length={firstFieldSummary.Length} HasData={firstFieldSummary.HasData}");
        }

        Console.WriteLine($"{label} CreatorSummary Name={firstSummary.Name} Version={firstSummary.Version} Namespace={firstSummary.Namespace} Interface={firstSummary.InterfaceKind}/{firstSummary.InterfaceMajor}.{firstSummary.InterfaceMinor} ApiLanguage={firstSummary.ApiLanguage} Fields={firstSummary.FieldCount}");
    }

    private static void ValidateLookup(TensorRtApiLine line, TensorRtPluginRegistryInventory inventory, TensorRtEngineCapability capability, bool useBuilderCapability)
    {
        if (inventory.CreatorCount == 0)
        {
            Console.WriteLine($"{(useBuilderCapability ? "BuilderCapability" : "Global")}PluginCreatorLookup Skipped=True Reason=NoCreators");
            return;
        }

        foreach (TensorRtPluginCreatorInfo candidate in inventory.Creators)
        {
            bool found;
            TensorRtPluginCreatorInfo? copiedCreator = null;
            string diagnostic = "OK";

            if (useBuilderCapability)
            {
                if (!TensorRtEnvironmentProbe.TryIsBuilderCapabilityPluginCreatorRegistered(
                    line,
                    capability,
                    candidate.Name,
                    candidate.Version,
                    candidate.Namespace,
                    out found,
                    out diagnostic))
                {
                    Console.WriteLine($"BuilderCapabilityPluginCreatorLookup CandidateSkipped=True Name={candidate.Name} Version={candidate.Version} Namespace={candidate.Namespace} Reason={diagnostic}");
                    continue;
                }

                if (!found)
                {
                    Console.WriteLine($"BuilderCapabilityPluginCreatorLookup CandidateFound=False Name={candidate.Name} Version={candidate.Version} Namespace={candidate.Namespace} Diagnostic={diagnostic}");
                    continue;
                }

                found = TensorRtEnvironmentProbe.TryGetBuilderCapabilityPluginCreator(
                    line,
                    capability,
                    candidate.Name,
                    candidate.Version,
                    candidate.Namespace,
                    out copiedCreator,
                    out diagnostic);
            }
            else
            {
                if (!TensorRtEnvironmentProbe.TryIsGlobalPluginCreatorRegistered(line, candidate.Name, candidate.Version, candidate.Namespace, out found, out diagnostic))
                {
                    Console.WriteLine($"GlobalPluginCreatorLookup CandidateSkipped=True Name={candidate.Name} Version={candidate.Version} Namespace={candidate.Namespace} Reason={diagnostic}");
                    continue;
                }

                if (!found)
                {
                    Console.WriteLine($"GlobalPluginCreatorLookup CandidateFound=False Name={candidate.Name} Version={candidate.Version} Namespace={candidate.Namespace} Diagnostic={diagnostic}");
                    continue;
                }

                found = TensorRtEnvironmentProbe.TryGetGlobalPluginCreator(
                    line,
                    candidate.Name,
                    candidate.Version,
                    candidate.Namespace,
                    out copiedCreator,
                    out diagnostic);
            }

            if (!found)
            {
                Console.WriteLine($"{(useBuilderCapability ? "BuilderCapability" : "Global")}PluginCreatorLookup CandidateMetadataSkipped=True Name={candidate.Name} Version={candidate.Version} Namespace={candidate.Namespace} Reason={diagnostic}");
                continue;
            }

            Console.WriteLine($"{(useBuilderCapability ? "BuilderCapability" : "Global")}PluginCreatorLookup Found=True Name={candidate.Name} Version={candidate.Version} Namespace={candidate.Namespace} Diagnostic={diagnostic}");
            if (copiedCreator != null)
            {
                if (!string.Equals(candidate.Name, copiedCreator.Name, StringComparison.Ordinal) ||
                    !string.Equals(candidate.Version, copiedCreator.Version, StringComparison.Ordinal) ||
                    !string.Equals(candidate.Namespace, copiedCreator.Namespace, StringComparison.Ordinal))
                {
                    throw new InvalidOperationException("Plugin creator lookup metadata does not match inventory metadata.");
                }
            }
            else if (useBuilderCapability)
            {
                throw new InvalidOperationException("Builder capability plugin creator lookup did not return copied metadata.");
            }

            TensorRtPluginCreatorInfo? snapshotCreator = inventory.FindCreator(candidate.Name, candidate.Version, candidate.Namespace);
            bool snapshotFound = inventory.TryFindCreator(candidate.Name, candidate.Version, candidate.Namespace, out TensorRtPluginCreatorInfo? trySnapshotCreator);
            Console.WriteLine($"{(useBuilderCapability ? "BuilderCapability" : "Global")}PluginCreatorSnapshot Found={snapshotFound} Name={candidate.Name} Version={candidate.Version} Namespace={candidate.Namespace}");
            if (!snapshotFound || snapshotCreator == null || trySnapshotCreator == null)
            {
                throw new InvalidOperationException("Managed plugin creator snapshot lookup failed.");
            }

            if (!string.Equals(candidate.Name, snapshotCreator.Name, StringComparison.Ordinal) ||
                !string.Equals(candidate.Version, snapshotCreator.Version, StringComparison.Ordinal) ||
                !string.Equals(candidate.Namespace, snapshotCreator.Namespace, StringComparison.Ordinal) ||
                !string.Equals(snapshotCreator.Name, trySnapshotCreator.Name, StringComparison.Ordinal) ||
                !string.Equals(snapshotCreator.Version, trySnapshotCreator.Version, StringComparison.Ordinal) ||
                !string.Equals(snapshotCreator.Namespace, trySnapshotCreator.Namespace, StringComparison.Ordinal))
            {
                throw new InvalidOperationException("Managed snapshot creator metadata does not match the lookup candidate.");
            }

            return;
        }

        Console.WriteLine($"{(useBuilderCapability ? "BuilderCapability" : "Global")}PluginCreatorLookup Skipped=True Reason=NoLookupableCreator");
    }

    private static void ValidateBuilderOwnedLookup(TensorRtBuilder builder, TensorRtPluginRegistryInventory inventory)
    {
        if (inventory.CreatorCount == 0)
        {
            Console.WriteLine("BuilderPluginCreatorLookup Skipped=True Reason=NoCreators");
            return;
        }

        foreach (TensorRtPluginCreatorInfo candidate in inventory.Creators)
        {
            if (!builder.TryIsPluginCreatorRegistered(candidate.Name, candidate.Version, candidate.Namespace, out bool found, out string foundDiagnostic))
            {
                Console.WriteLine($"BuilderPluginCreatorLookup CandidateSkipped=True Name={candidate.Name} Version={candidate.Version} Namespace={candidate.Namespace} Reason={foundDiagnostic}");
                continue;
            }

            if (!found)
            {
                Console.WriteLine($"BuilderPluginCreatorLookup CandidateFound=False Name={candidate.Name} Version={candidate.Version} Namespace={candidate.Namespace} Diagnostic={foundDiagnostic}");
                continue;
            }

            bool copied = builder.TryGetPluginCreator(candidate.Name, candidate.Version, candidate.Namespace, out TensorRtPluginCreatorInfo? copiedCreator, out string copiedDiagnostic);
            if (!copied || copiedCreator == null)
            {
                Console.WriteLine($"BuilderPluginCreatorLookup CandidateMetadataSkipped=True Name={candidate.Name} Version={candidate.Version} Namespace={candidate.Namespace} Reason={copiedDiagnostic}");
                continue;
            }

            Console.WriteLine($"BuilderPluginCreatorLookup Found=True Name={candidate.Name} Version={candidate.Version} Namespace={candidate.Namespace} Diagnostic={copiedDiagnostic}");
            if (!string.Equals(candidate.Name, copiedCreator.Name, StringComparison.Ordinal) ||
                !string.Equals(candidate.Version, copiedCreator.Version, StringComparison.Ordinal) ||
                !string.Equals(candidate.Namespace, copiedCreator.Namespace, StringComparison.Ordinal))
            {
                throw new InvalidOperationException("Builder plugin creator lookup metadata does not match inventory metadata.");
            }

            TensorRtPluginCreatorInfo? snapshotCreator = inventory.FindCreator(candidate.Name, candidate.Version, candidate.Namespace);
            bool snapshotFound = inventory.TryFindCreator(candidate.Name, candidate.Version, candidate.Namespace, out TensorRtPluginCreatorInfo? trySnapshotCreator);
            Console.WriteLine($"BuilderPluginCreatorSnapshot Found={snapshotFound} Name={candidate.Name} Version={candidate.Version} Namespace={candidate.Namespace}");
            if (!snapshotFound || snapshotCreator == null || trySnapshotCreator == null)
            {
                throw new InvalidOperationException("Builder-managed plugin creator snapshot lookup failed.");
            }

            if (!string.Equals(candidate.Name, snapshotCreator.Name, StringComparison.Ordinal) ||
                !string.Equals(candidate.Version, snapshotCreator.Version, StringComparison.Ordinal) ||
                !string.Equals(candidate.Namespace, snapshotCreator.Namespace, StringComparison.Ordinal) ||
                !string.Equals(snapshotCreator.Name, trySnapshotCreator.Name, StringComparison.Ordinal) ||
                !string.Equals(snapshotCreator.Version, trySnapshotCreator.Version, StringComparison.Ordinal) ||
                !string.Equals(snapshotCreator.Namespace, trySnapshotCreator.Namespace, StringComparison.Ordinal))
            {
                throw new InvalidOperationException("Builder snapshot creator metadata does not match the lookup candidate.");
            }

            return;
        }

        Console.WriteLine("BuilderPluginCreatorLookup Skipped=True Reason=NoLookupableCreator");
    }

    private static void ValidateRuntimeLocalLookup(TensorRtRuntime runtime, TensorRtPluginRegistryInventory inventory)
    {
        if (inventory.CreatorCount == 0)
        {
            Console.WriteLine("RuntimePluginCreatorLookup Skipped=True Reason=NoCreators");
            return;
        }

        foreach (TensorRtPluginCreatorInfo candidate in inventory.Creators)
        {
            if (!runtime.TryIsPluginCreatorRegistered(candidate.Name, candidate.Version, candidate.Namespace, out bool found, out string foundDiagnostic))
            {
                Console.WriteLine($"RuntimePluginCreatorLookup CandidateSkipped=True Name={candidate.Name} Version={candidate.Version} Namespace={candidate.Namespace} Reason={foundDiagnostic}");
                continue;
            }

            if (!found)
            {
                Console.WriteLine($"RuntimePluginCreatorLookup CandidateFound=False Name={candidate.Name} Version={candidate.Version} Namespace={candidate.Namespace} Diagnostic={foundDiagnostic}");
                continue;
            }

            bool copied = runtime.TryGetPluginCreator(candidate.Name, candidate.Version, candidate.Namespace, out TensorRtPluginCreatorInfo? copiedCreator, out string copiedDiagnostic);
            if (!copied || copiedCreator == null)
            {
                Console.WriteLine($"RuntimePluginCreatorLookup CandidateMetadataSkipped=True Name={candidate.Name} Version={candidate.Version} Namespace={candidate.Namespace} Reason={copiedDiagnostic}");
                continue;
            }

            Console.WriteLine($"RuntimePluginCreatorLookup Found=True Name={candidate.Name} Version={candidate.Version} Namespace={candidate.Namespace} Diagnostic={copiedDiagnostic}");
            if (!string.Equals(candidate.Name, copiedCreator.Name, StringComparison.Ordinal) ||
                !string.Equals(candidate.Version, copiedCreator.Version, StringComparison.Ordinal) ||
                !string.Equals(candidate.Namespace, copiedCreator.Namespace, StringComparison.Ordinal))
            {
                throw new InvalidOperationException("Runtime plugin creator lookup metadata does not match inventory metadata.");
            }

            TensorRtPluginCreatorInfo? snapshotCreator = inventory.FindCreator(candidate.Name, candidate.Version, candidate.Namespace);
            bool snapshotFound = inventory.TryFindCreator(candidate.Name, candidate.Version, candidate.Namespace, out TensorRtPluginCreatorInfo? trySnapshotCreator);
            Console.WriteLine($"RuntimePluginCreatorSnapshot Found={snapshotFound} Name={candidate.Name} Version={candidate.Version} Namespace={candidate.Namespace}");
            if (!snapshotFound || snapshotCreator == null || trySnapshotCreator == null)
            {
                throw new InvalidOperationException("Runtime-managed plugin creator snapshot lookup failed.");
            }

            if (!string.Equals(candidate.Name, snapshotCreator.Name, StringComparison.Ordinal) ||
                !string.Equals(candidate.Version, snapshotCreator.Version, StringComparison.Ordinal) ||
                !string.Equals(candidate.Namespace, snapshotCreator.Namespace, StringComparison.Ordinal) ||
                !string.Equals(snapshotCreator.Name, trySnapshotCreator.Name, StringComparison.Ordinal) ||
                !string.Equals(snapshotCreator.Version, trySnapshotCreator.Version, StringComparison.Ordinal) ||
                !string.Equals(snapshotCreator.Namespace, trySnapshotCreator.Namespace, StringComparison.Ordinal))
            {
                throw new InvalidOperationException("Runtime snapshot creator metadata does not match the lookup candidate.");
            }

            return;
        }

        Console.WriteLine("RuntimePluginCreatorLookup Skipped=True Reason=NoLookupableCreator");
    }

    private static string FormatInventory(string label, TensorRtPluginRegistryInventory inventory)
    {
        string first = inventory.CreatorCount > 0
            ? $"{inventory.Creators[0].Name}/{inventory.Creators[0].Version}/{inventory.Creators[0].Namespace}/api={inventory.Creators[0].ApiLanguage}/fields={inventory.Creators[0].Fields.Count}"
            : "n/a";
        IReadOnlyList<TensorRtPluginCreatorSummary> summaries = inventory.GetCreatorSummaries(maxCreators: 1);
        string firstSummary = summaries.Count > 0
            ? summaries[0].ToString()
            : "n/a";
        return $"{label} Source={inventory.Source} Creators={inventory.CreatorCount} Recursive={inventory.RecursiveCreatorCount?.ToString() ?? "n/a"} ParentSearch={inventory.ParentSearchEnabled} ErrorRecorder={inventory.HasErrorRecorder} First={first} FirstSummary={firstSummary}";
    }

    private static void PrintDependencyProbe(TensorRtApiLine line)
    {
        TensorRtDependencyProbeReport dependencyProbe = TensorRtEnvironmentProbe.ProbeNativeDependencies(line);
        Console.WriteLine($"DependencyProbe Line={(int)line} BridgeInitialized={dependencyProbe.BridgeInitialized} Candidates={dependencyProbe.NativeBridgeCandidates.Count} Loaded={dependencyProbe.LoadedModuleCount} SearchPathCandidates={dependencyProbe.SearchPathCandidateCount} Diagnostics={dependencyProbe.Diagnostics.Count} Message={dependencyProbe.BridgeDiagnostic}");
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
            if (snapshot.TensorRt11.VendorDependencyAvailable)
            {
                return TensorRtApiLine.TensorRt11;
            }

            if (snapshot.TensorRt10.VendorDependencyAvailable)
            {
                return TensorRtApiLine.TensorRt10;
            }

            if (snapshot.TensorRt8.VendorDependencyAvailable)
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
        if (exception is DllNotFoundException || exception is BadImageFormatException || exception is SEHException || exception is AccessViolationException)
        {
            return true;
        }

        if (exception is BridgeProbeException bridgeProbe)
        {
            return bridgeProbe.StatusCode == BridgeStatusCode.DependencyMissing ||
                bridgeProbe.StatusCode == BridgeStatusCode.NotSupported ||
                bridgeProbe.StatusCode == BridgeStatusCode.InvalidState ||
                IsDelayLoadDependencyException(bridgeProbe);
        }

        return false;
    }

    private static bool IsDelayLoadDependencyException(BridgeProbeException exception)
    {
        if (exception.StatusCode != BridgeStatusCode.RuntimeError)
        {
            return false;
        }

        return exception.Message.Contains("structured exception with code 3228369022", StringComparison.Ordinal);
    }
}

using System.Text;

namespace JYPPX.ProjectQuality.Tests;

internal static class RepositorySourceReader
{
    private static readonly IReadOnlyDictionary<string, string[]> SourceSets =
        new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase)
        {
            ["TensorRtDebugListenerRuntimeProofPrecheck.cs"] =
            [
                "TensorRtDebugListenerRuntimeProofPrecheck.cs",
                "TensorRtDebugListenerRuntimeProofPrecheck.DesignPrerequisites.cs",
                "TensorRtDebugListenerRuntimeProofPrecheck.NativeAttachDesign.cs",
                "TensorRtDebugListenerRuntimeProofPrecheck.OwnerLifecycle.cs",
                "TensorRtDebugListenerRuntimeProofPrecheck.RuntimeScaffold.cs",
                "TensorRtDebugListenerRuntimeProofPrecheck.FinalRuntimeGates.cs",
                "TensorRtDebugListenerRuntimeProofPrecheckResult.cs"
            ],
            ["TensorRtAllocatorCallbackOwner.cs"] =
            [
                "TensorRtAllocatorCallbackOwner.cs",
                "TensorRtAllocatorCallbackOwner.Lifecycle.cs",
                "TensorRtAllocatorCallbackOwner.ManagedDryRun.cs",
                "TensorRtAllocatorCallbackOwner.NativeDryRun.cs",
                "TensorRtAllocatorCallbackOwner.StateLedger.cs",
                "TensorRtAllocatorCallbackOwner.InternalPrototype.cs",
                "TensorRtAllocatorCallbackOwner.ResultMapping.cs",
                "TensorRtAllocatorDryRunRequest.cs",
                "TensorRtAllocatorDryRunResult.cs",
                "TensorRtAllocatorDryRunHandler.cs",
                "TensorRtAllocatorNativeDryRunResult.cs",
                "TensorRtAllocatorOwnerStateDryRunResult.cs",
                "TensorRtAllocatorCallbackOwnerSnapshot.cs",
                "TensorRtAllocatorInternalRuntimePrototypeResult.cs"
            ]
        };

    public static string Read(string path)
    {
        string fileName = Path.GetFileName(path);
        if (!SourceSets.TryGetValue(fileName, out string[]? sourceSet))
        {
            return File.ReadAllText(path);
        }

        string directory = Path.GetDirectoryName(path)
            ?? throw new InvalidOperationException($"Source path has no parent directory: {path}");
        StringBuilder source = new();
        foreach (string sourceFileName in sourceSet)
        {
            source.AppendLine(File.ReadAllText(Path.Combine(directory, sourceFileName)));
        }

        return source.ToString();
    }
}

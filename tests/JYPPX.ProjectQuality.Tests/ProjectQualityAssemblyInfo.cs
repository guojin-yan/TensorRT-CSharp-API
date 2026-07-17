using Xunit;

// Most ProjectQuality tests execute scripts against the same canonical release-evidence graph.
[assembly: CollectionBehavior(DisableTestParallelization = true)]

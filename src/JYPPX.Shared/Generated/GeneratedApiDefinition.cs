namespace JYPPX.Shared.Generated;

/// <summary>
/// Describes one generated API manifest entry.
/// </summary>
public sealed class GeneratedApiDefinition
{
    public GeneratedApiDefinition(
        string id,
        string module,
        string versionLine,
        string entryPoint,
        string ownership,
        bool manualOverride,
        string versionGuard,
        string parameterSummary)
    {
        Id = id;
        Module = module;
        VersionLine = versionLine;
        EntryPoint = entryPoint;
        Ownership = ownership;
        ManualOverride = manualOverride;
        VersionGuard = versionGuard;
        ParameterSummary = parameterSummary;
    }

    public string Id { get; }
    public string Module { get; }
    public string VersionLine { get; }
    public string EntryPoint { get; }
    public string Ownership { get; }
    public bool ManualOverride { get; }
    public string VersionGuard { get; }
    public string ParameterSummary { get; }
}


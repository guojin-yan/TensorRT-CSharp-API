namespace JYPPX.Shared.Generated;

/// <summary>
/// Describes one generated API manifest entry.
/// 描述一个生成的 API manifest 条目。
/// </summary>
internal sealed class GeneratedApiDefinition
{
    /// <summary>
    /// Creates one generated API manifest entry description.
    /// 创建单个生成 API manifest 条目描述。
    /// </summary>
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

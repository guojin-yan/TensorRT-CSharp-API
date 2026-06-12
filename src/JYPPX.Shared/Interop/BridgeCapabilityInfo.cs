namespace JYPPX.Shared.Interop;

/// <summary>
/// Capability flags surfaced by the native bridge.
/// </summary>
public sealed class BridgeCapabilityInfo
{
    public BridgeCapabilityInfo(
        bool supportsTrt8Adapter,
        bool supportsTrt10Adapter,
        bool supportsTrt11Adapter,
        bool supportsTrt8RuntimeCreation,
        bool supportsTrt10RuntimeCreation,
        bool supportsTrt11RuntimeCreation,
        bool supportsTrt8BuilderCreation,
        bool supportsTrt10BuilderCreation,
        bool supportsTrt11BuilderCreation,
        bool supportsLastErrorQuery,
        bool supportsBuildInfoQuery,
        bool supportsRuntimeInfoQuery)
    {
        SupportsTrt8Adapter = supportsTrt8Adapter;
        SupportsTrt10Adapter = supportsTrt10Adapter;
        SupportsTrt11Adapter = supportsTrt11Adapter;
        SupportsTrt8RuntimeCreation = supportsTrt8RuntimeCreation;
        SupportsTrt10RuntimeCreation = supportsTrt10RuntimeCreation;
        SupportsTrt11RuntimeCreation = supportsTrt11RuntimeCreation;
        SupportsTrt8BuilderCreation = supportsTrt8BuilderCreation;
        SupportsTrt10BuilderCreation = supportsTrt10BuilderCreation;
        SupportsTrt11BuilderCreation = supportsTrt11BuilderCreation;
        SupportsLastErrorQuery = supportsLastErrorQuery;
        SupportsBuildInfoQuery = supportsBuildInfoQuery;
        SupportsRuntimeInfoQuery = supportsRuntimeInfoQuery;
    }

    public bool SupportsTrt8Adapter { get; }
    public bool SupportsTrt10Adapter { get; }
    public bool SupportsTrt11Adapter { get; }
    public bool SupportsTrt8RuntimeCreation { get; }
    public bool SupportsTrt10RuntimeCreation { get; }
    public bool SupportsTrt11RuntimeCreation { get; }
    public bool SupportsTrt8BuilderCreation { get; }
    public bool SupportsTrt10BuilderCreation { get; }
    public bool SupportsTrt11BuilderCreation { get; }
    public bool SupportsLastErrorQuery { get; }
    public bool SupportsBuildInfoQuery { get; }
    public bool SupportsRuntimeInfoQuery { get; }
}

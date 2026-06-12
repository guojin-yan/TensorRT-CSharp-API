using System;
using System.Linq;
using JYPPX.CudaSharp;

Console.WriteLine("CudaGraphSmokeRunner");

const int ByteCount = 64;
byte[] input = Enumerable.Range(0, ByteCount).Select(static value => unchecked((byte)(value + 1))).ToArray();

using CudaPinnedMemory source = new CudaPinnedMemory(ByteCount);
using CudaPinnedMemory destination = new CudaPinnedMemory(ByteCount);
using CudaMemory device = new CudaMemory(ByteCount);
using CudaStream stream = new CudaStream(CudaStreamCreationFlags.NonBlocking);
using CudaEvent startEvent = new CudaEvent();
using CudaEvent endEvent = new CudaEvent();

source.CopyFrom(input);
destination.CopyFrom(new byte[ByteCount]);

CudaStreamCaptureInfo captureBefore = stream.GetCaptureInfo();
stream.BeginCapture(CudaStreamCaptureMode.Relaxed);
CudaStreamCaptureInfo captureDuring = stream.GetCaptureInfo();
device.CopyFromAsync(source, ByteCount, stream);
device.CopyToAsync(destination, ByteCount, stream);
using CudaGraph graph = stream.EndCapture();
CudaStreamCaptureInfo captureAfter = stream.GetCaptureInfo();
using CudaGraphExec graphExec = graph.Instantiate();
using CudaGraph graphClone = graph.Clone();
CudaGraphNode capturedNode = graph.GetNode(0);
CudaGraphNode capturedRootNode = graph.GetRootNode(0);
CudaGraphNode capturedCloneNode = graphClone.FindNodeInClone(capturedNode);
CudaGraphNodeType capturedNodeType = CudaGraph.GetNodeType(capturedNode);
ulong capturedNodeDependencies = CudaGraph.GetDependencyCount(capturedNode);
ulong capturedNodeDependents = CudaGraph.GetDependentCount(capturedNode);
CudaGraphEdge? capturedEdge = graph.EdgeCount > 0 ? graph.GetEdge(0) : null;
ulong graphExecFlags = 0;
string graphExecFlagsState = "Unsupported";
try
{
    graphExecFlags = graphExec.Flags;
    graphExecFlagsState = graphExecFlags.ToString();
}
catch (CudaException exception)
{
    graphExecFlagsState = $"Skipped:{exception.Message}";
}

using CudaGraph topologyGraph = CudaGraph.Create();
CudaGraphNode topologyRoot = topologyGraph.AddEmptyNode();
CudaGraphNode topologyChild = topologyGraph.AddEmptyNodeAfter(topologyRoot);
CudaGraphEdge topologyEdge = topologyGraph.GetEdge(0);
ulong topologyChildDependencyCount = CudaGraph.GetDependencyCount(topologyChild);
CudaGraphNode topologyChildDependency = CudaGraph.GetDependency(topologyChild, 0);
ulong topologyRootDependentCount = CudaGraph.GetDependentCount(topologyRoot);
CudaGraphNode topologyRootDependent = CudaGraph.GetDependent(topologyRoot, 0);
if (topologyEdge.From != topologyRoot || topologyEdge.To != topologyChild || topologyChildDependency != topologyRoot || topologyRootDependent != topologyChild)
{
    throw new InvalidOperationException("CUDA graph topology token round trip failed.");
}

topologyGraph.RemoveDependency(topologyRoot, topologyChild);
topologyGraph.AddDependency(topologyRoot, topologyChild);
using CudaGraphExec topologyExec = topologyGraph.Instantiate();
string topologyNodeEnabledState = "Unsupported";
try
{
    bool topologyNodeEnabled = topologyExec.GetNodeEnabled(topologyRoot);
    topologyExec.SetNodeEnabled(topologyRoot, topologyNodeEnabled);
    topologyNodeEnabledState = topologyNodeEnabled.ToString();
}
catch (CudaException exception)
{
    topologyNodeEnabledState = $"Skipped:{exception.Message}";
}

string topologyGraphIdState = "Unsupported";
try
{
    topologyGraphIdState = topologyGraph.Id.ToString();
}
catch (CudaException exception)
{
    topologyGraphIdState = $"Skipped:{exception.Message}";
}

string topologyExecIdState = "Unsupported";
try
{
    topologyExecIdState = topologyExec.Id.ToString();
}
catch (CudaException exception)
{
    topologyExecIdState = $"Skipped:{exception.Message}";
}

string topologyNodeIdentityState = ProbeGraphNodeIdentity(topologyGraph, topologyRoot);

destination.CopyFrom(new byte[ByteCount]);
startEvent.Record(stream);
graphExec.Upload(stream);
graphExec.Launch(stream);
endEvent.Record(stream);
stream.Synchronize();
float elapsedMilliseconds = endEvent.ElapsedTimeSince(startEvent);
string graphMemoryState = ProbeGraphMemory(CudaDevice.Current);

byte[] output = destination.ToArray(ByteCount);
bool match = input.SequenceEqual(output);
if (!match)
{
    throw new InvalidOperationException($"CUDA graph copy round trip failed. Expected=[{string.Join(", ", input)}] Actual=[{string.Join(", ", output)}]");
}

Console.WriteLine($"CudaGraphCaptureRoundTrip=True Bytes={ByteCount} Capture={captureBefore.Status}->{captureDuring.Status}->{captureAfter.Status} CaptureId={captureDuring.CaptureId} Nodes={graph.NodeCount} Roots={graph.RootNodeCount} Edges={graph.EdgeCount} CloneNodes={graphClone.NodeCount} ExecFlags={graphExecFlagsState} EventElapsedMilliseconds={elapsedMilliseconds:0.###}");
Console.WriteLine($"CudaGraphCapturedTopology Node0={capturedNode} Root0={capturedRootNode} CloneNode0={capturedCloneNode} NodeType={capturedNodeType} NodeDeps={capturedNodeDependencies} NodeDependents={capturedNodeDependents} Edge0={capturedEdge?.ToString() ?? "None"}");
Console.WriteLine($"CudaGraphManualTopology Nodes={topologyGraph.NodeCount} Roots={topologyGraph.RootNodeCount} Edges={topologyGraph.EdgeCount} ChildDeps={topologyChildDependencyCount} RootDependents={topologyRootDependentCount} NodeEnabled={topologyNodeEnabledState} GraphId={topologyGraphIdState} ExecId={topologyExecIdState} NodeIdentity={topologyNodeIdentityState}");
Console.WriteLine($"CudaGraphMemory {graphMemoryState}");

static string ProbeGraphNodeIdentity(CudaGraph graph, CudaGraphNode node)
{
    try
    {
        bool containsNode = graph.ContainsNode(node);
        uint localId = CudaGraph.GetNodeLocalId(node);
        ulong toolsId = CudaGraph.GetNodeToolsId(node);
        return $"Contains={containsNode} LocalId={localId} ToolsId={toolsId}";
    }
    catch (CudaException exception)
    {
        return $"Skipped:{exception.Message}";
    }
}

static string ProbeGraphMemory(int deviceOrdinal)
{
    try
    {
        CudaDeviceGraphMemoryInfo before = CudaDevice.GetGraphMemoryInfo(deviceOrdinal);
        CudaDevice.ResetGraphMemoryHighWatermarks(deviceOrdinal);
        CudaDevice.TrimGraphMemory(deviceOrdinal);
        CudaDeviceGraphMemoryInfo after = CudaDevice.GetGraphMemoryInfo(deviceOrdinal);
        return $"Before=[{before}] AfterResetTrim=[{after}]";
    }
    catch (CudaException exception)
    {
        return $"Skipped:{exception.Message}";
    }
}

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using JYPPX.CudaSharp;

internal static class Program
{
    private static void Main(string[] args)
    {
        Console.WriteLine("CudaGraphSmokeRunner");

        try
        {

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
        device.FillAsync(0, ByteCount, stream);
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
        string capturedMemsetParamsState = ProbeGraphMemsetNodeParameters(graph, device, ByteCount);
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
        CudaGraphTopologySnapshot topologySnapshot = topologyGraph.GetTopologySnapshot();
        CudaGraphNodeTopologySnapshot topologyRootSnapshot = CudaGraph.GetNodeTopologySnapshot(topologyRoot);
        CudaGraphNodeTopologySnapshot topologyChildSnapshot = CudaGraph.GetNodeTopologySnapshot(topologyChild);
        string topologySnapshotListState = ProbeGraphSnapshotLists(topologyGraph, topologyRoot, topologyChild);
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
        string topologyEdgeDataState = ProbeGraphEdgeData(topologyGraph, topologyRoot, topologyChild);
        string topologyDebugDotState = ProbeGraphDebugDot(topologyGraph);
        string eventNodeState = ProbeGraphEventNodes();
        string memcpy1DNodeState = ProbeGraphMemcpy1D(source, destination, device, ByteCount);
        string typedDescriptorState = ProbeGraphNodeParamsTypedDescriptors(source, destination, device, ByteCount);
        string childGraphUpdateState = ProbeChildGraphUpdate(stream);
        using CudaGraphExec topologyExec = topologyGraph.Instantiate();
        string topologyNodeEnabledState = "Unsupported";
        string topologyExecNodeSnapshotState = "Unsupported";
        try
        {
            bool topologyNodeEnabled = topologyExec.GetNodeEnabled(topologyRoot);
            topologyExec.SetNodeEnabled(topologyRoot, topologyNodeEnabled);
            CudaGraphExecNodeStateSnapshot execNodeSnapshot = topologyExec.GetNodeStateSnapshot(topologyRoot);
            CudaGraphExecDiagnosticSnapshot execDiagnosticSnapshot = topologyExec.GetDiagnosticSnapshot(new[] { topologyRoot, topologyChild });
            CudaGraphExecDiagnosticSummary execDiagnosticSummary = execDiagnosticSnapshot.ToSummary();
            topologyNodeEnabledState = topologyNodeEnabled.ToString();
            topologyExecNodeSnapshotState = $"{execNodeSnapshot}; {execDiagnosticSnapshot}; GraphExecDiagnosticSummary={execDiagnosticSummary}";
        }
        catch (CudaException exception)
        {
            topologyNodeEnabledState = $"Skipped:{exception.Message}";
            topologyExecNodeSnapshotState = $"Skipped:{exception.Message}";
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
        Console.WriteLine($"CudaGraphCapturedTopology Node0={capturedNode} Root0={capturedRootNode} CloneNode0={capturedCloneNode} NodeType={capturedNodeType} NodeDeps={capturedNodeDependencies} NodeDependents={capturedNodeDependents} MemsetParams={capturedMemsetParamsState} Edge0={capturedEdge?.ToString() ?? "None"}");
        Console.WriteLine($"CudaGraphManualTopology Nodes={topologyGraph.NodeCount} Roots={topologyGraph.RootNodeCount} Edges={topologyGraph.EdgeCount} ChildDeps={topologyChildDependencyCount} RootDependents={topologyRootDependentCount} EdgeData={topologyEdgeDataState} DebugDot={topologyDebugDotState} EventNodes={eventNodeState} Memcpy1D={memcpy1DNodeState} NodeParamsDescriptor={typedDescriptorState} ChildGraphUpdate={childGraphUpdateState} NodeEnabled={topologyNodeEnabledState} GraphId={topologyGraphIdState} ExecId={topologyExecIdState} NodeIdentity={topologyNodeIdentityState}");
        Console.WriteLine($"CudaGraphSnapshots GraphSnapshot=[{topologySnapshot}] RootNodeSnapshot=[{topologyRootSnapshot}] ChildNodeSnapshot=[{topologyChildSnapshot}] ExecNodeSnapshot=[{topologyExecNodeSnapshotState}]");
        Console.WriteLine($"CudaGraphSnapshotLists {topologySnapshotListState}");
        Console.WriteLine($"CudaGraphMemory {graphMemoryState}");
        }
        catch (CudaException exception)
        {
            Console.WriteLine($"Skipped=True Reason=CudaException:{exception.Message}");
        }
        catch (DllNotFoundException exception)
        {
            Console.WriteLine($"Skipped=True Reason=DllNotFoundException:{exception.Message}");
        }
        catch (BadImageFormatException exception)
        {
            Console.WriteLine($"Skipped=True Reason=BadImageFormatException:{exception.Message}");
        }
    }

    static string ProbeGraphEdgeData(CudaGraph graph, CudaGraphNode root, CudaGraphNode child)
    {
        bool dependencyRemoved = false;
        try
        {
            ulong dependencyCount = CudaGraph.GetDependencyWithEdgeDataCount(child);
            CudaGraphNodeDependency dependency = CudaGraph.GetDependencyWithEdgeData(child, 0);
            ulong dependentCount = CudaGraph.GetDependentWithEdgeDataCount(root);
            CudaGraphNodeDependency dependent = CudaGraph.GetDependentWithEdgeData(root, 0);
            ulong graphEdgeCount = graph.GetEdgeWithEdgeDataCount();
            CudaGraphEdgeWithData graphEdge = graph.GetEdgeWithEdgeData(0);
            if (dependencyCount != 1 || dependentCount != 1 || graphEdgeCount != 1 ||
                dependency.Node != root || dependent.Node != child ||
                graphEdge.From != root || graphEdge.To != child ||
                dependency.EdgeData != CudaGraphEdgeData.Default ||
                dependent.EdgeData != CudaGraphEdgeData.Default ||
                graphEdge.EdgeData != CudaGraphEdgeData.Default)
            {
                throw new InvalidOperationException("CUDA graph edge data topology query returned unexpected values.");
            }

            graph.RemoveDependency(root, child, dependency.EdgeData);
            dependencyRemoved = true;
            if (graph.EdgeCount != 0 || graph.GetEdgeWithEdgeDataCount() != 0)
            {
                throw new InvalidOperationException("CUDA graph v2 dependency remove did not remove the expected edge.");
            }

            graph.AddDependency(root, child, dependency.EdgeData);
            dependencyRemoved = false;
            CudaGraphNodeDependency restoredDependency = CudaGraph.GetDependencyWithEdgeData(child, 0);
            if (restoredDependency.Node != root || restoredDependency.EdgeData != dependency.EdgeData)
            {
                throw new InvalidOperationException("CUDA graph v2 dependency add did not restore the expected edge data.");
            }

            return $"GraphEdges={graphEdgeCount} Deps={dependencyCount} Dependents={dependentCount} Data={dependency.EdgeData} Restored={restoredDependency.EdgeData}";
        }
        catch (CudaException exception)
        {
            if (dependencyRemoved)
            {
                try
                {
                    graph.AddDependency(root, child);
                }
                catch (CudaException)
                {
                }
            }

            return $"Skipped:{exception.Message}";
        }
    }

    static string ProbeGraphSnapshotLists(CudaGraph graph, CudaGraphNode root, CudaGraphNode child)
    {
        try
        {
            CudaGraphDiagnosticSnapshot diagnostic = graph.GetDiagnosticSnapshot(maxNodes: 8, maxRootNodes: 8, maxEdges: 8);
            CudaGraphDiagnosticSummary diagnosticSummary = diagnostic.ToSummary();
            IReadOnlyList<CudaGraphNodeSnapshot> nodes = graph.GetNodeSnapshotList(maxNodes: 8);
            IReadOnlyList<CudaGraphNodeSnapshot> roots = graph.GetRootNodeSnapshotList(maxNodes: 8);
            IReadOnlyList<CudaGraphEdgeSnapshot> edgeDataEdges = graph.GetEdgeSnapshotList(includeEdgeData: true, maxEdges: 8);
            IReadOnlyList<CudaGraphAdjacentNodeSnapshot> childDependencies = CudaGraph.GetNodeDependencySnapshotList(child, includeEdgeData: true, maxDependencies: 8);
            IReadOnlyList<CudaGraphAdjacentNodeSnapshot> rootDependents = CudaGraph.GetNodeDependentSnapshotList(root, includeEdgeData: true, maxDependents: 8);

            if (nodes.Count != 2 ||
                roots.Count != 1 ||
                edgeDataEdges.Count != 1 ||
                childDependencies.Count != 1 ||
                rootDependents.Count != 1 ||
                diagnostic.Nodes.Count != 2 ||
                diagnostic.RootNodes.Count != 1 ||
                diagnostic.Edges.Count != 1 ||
                nodes[0].Topology.NodeType != CudaGraphNodeType.Empty ||
                childDependencies[0].Node != root ||
                rootDependents[0].Node != child ||
                !edgeDataEdges[0].HasEdgeData ||
                !childDependencies[0].HasEdgeData ||
                !rootDependents[0].HasEdgeData)
            {
                throw new InvalidOperationException("CUDA graph snapshot list diagnostics returned unexpected values.");
            }

            IReadOnlyList<CudaGraphEdgeSnapshot> legacyEdges = graph.GetEdgeSnapshotList(includeEdgeData: false, maxEdges: 8);
            IReadOnlyList<CudaGraphAdjacentNodeSnapshot> legacyDependencies = CudaGraph.GetNodeDependencySnapshotList(child, includeEdgeData: false, maxDependencies: 8);
            IReadOnlyList<CudaGraphAdjacentNodeSnapshot> legacyDependents = CudaGraph.GetNodeDependentSnapshotList(root, includeEdgeData: false, maxDependents: 8);
            if (legacyEdges.Count != 1 ||
                legacyDependencies.Count != 1 ||
                legacyDependents.Count != 1 ||
                legacyEdges[0].HasEdgeData ||
                legacyDependencies[0].HasEdgeData ||
                legacyDependents[0].HasEdgeData)
            {
                throw new InvalidOperationException("CUDA graph legacy snapshot list fallback returned unexpected values.");
            }

            return $"Diagnostic=[{diagnostic}] GraphDiagnosticSummary=[{diagnosticSummary}] Nodes={nodes.Count} Roots={roots.Count} Edges={edgeDataEdges.Count} LegacyEdges={legacyEdges.Count} Dependencies={childDependencies.Count} Dependents={rootDependents.Count} EdgeDataState={edgeDataEdges[0].EdgeDataState}";
        }
        catch (CudaException exception)
        {
            return $"Skipped:{exception.Message}";
        }
    }

    static string ProbeGraphMemsetNodeParameters(CudaGraph graph, CudaMemory destination, int byteCount)
    {
        try
        {
            for (ulong index = 0; index < graph.NodeCount; index++)
            {
                CudaGraphNode node = graph.GetNode(index);
                if (CudaGraph.GetNodeType(node) != CudaGraphNodeType.Memset)
                {
                    continue;
                }

                CudaGraphMemsetNodeParameters parameters = CudaGraph.GetMemsetNodeParameters(node);
                if (parameters.DestinationAddress == 0 || parameters.ElementSize == 0 || parameters.Width == 0 || parameters.Height == 0)
                {
                    throw new InvalidOperationException("CUDA graph memset node parameters were unexpectedly empty.");
                }

                CudaGraph.SetMemsetNodeParameters(node, destination, 0, byteCount);
                CudaGraphMemsetNodeParameters updated = CudaGraph.GetMemsetNodeParameters(node);
                if (updated.Value != 0 || updated.Width != (ulong)byteCount || updated.Height != 1)
                {
                    throw new InvalidOperationException("CUDA graph memset node parameter update returned unexpected values.");
                }

                return $"{parameters} Updated={updated.Value}/{updated.Width}x{updated.Height}";
            }

            return "Skipped:NoMemsetNode";
        }
        catch (CudaException exception)
        {
            return $"Skipped:{exception.Message}";
        }
    }

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

    static string ProbeGraphDebugDot(CudaGraph graph)
    {
        try
        {
            string directory = Path.Combine(Path.GetTempPath(), "jyppx-cuda-graph-smoke", Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(directory);
            string path = Path.Combine(directory, "graph.dot");
            graph.ExportDebugDot(path);
            FileInfo file = new FileInfo(path);
            if (!file.Exists || file.Length == 0)
            {
                throw new InvalidOperationException("CUDA graph debug DOT export did not create a non-empty file.");
            }

            string text = File.ReadAllText(path);
            bool containsDigraph = text.IndexOf("digraph", StringComparison.OrdinalIgnoreCase) >= 0;
            if (!containsDigraph)
            {
                throw new InvalidOperationException("CUDA graph debug DOT export did not contain a digraph marker.");
            }

            return $"Path={path} Bytes={file.Length} ContainsDigraph={containsDigraph}";
        }
        catch (CudaException exception)
        {
            return $"Skipped:{exception.Message}";
        }
    }

    static string ProbeGraphEventNodes()
    {
        try
        {
            using CudaGraph graph = CudaGraph.Create();
            using CudaEvent recordEvent = new CudaEvent(CudaEventCreationFlags.DisableTiming);
            using CudaEvent waitEvent = new CudaEvent(CudaEventCreationFlags.DisableTiming);
            using CudaEvent replacementRecordEvent = new CudaEvent(CudaEventCreationFlags.DisableTiming);
            using CudaEvent replacementWaitEvent = new CudaEvent(CudaEventCreationFlags.DisableTiming);

            CudaGraphNode recordNode = graph.AddEventRecordNode(recordEvent);
            CudaGraphNode waitNode = graph.AddEventWaitNodeAfter(recordNode, waitEvent);
            CudaGraph.SetEventRecordNodeEvent(recordNode, replacementRecordEvent);
            CudaGraph.SetEventWaitNodeEvent(waitNode, replacementWaitEvent);
            bool recordHasEvent = CudaGraph.EventRecordNodeHasEvent(recordNode);
            bool waitHasEvent = CudaGraph.EventWaitNodeHasEvent(waitNode);

            if (CudaGraph.GetNodeType(recordNode) != CudaGraphNodeType.EventRecord ||
                CudaGraph.GetNodeType(waitNode) != CudaGraphNodeType.WaitEvent ||
                CudaGraph.GetDependencyCount(waitNode) != 1 ||
                !recordHasEvent ||
                !waitHasEvent)
            {
                throw new InvalidOperationException("CUDA graph event node topology returned unexpected values.");
            }

            using CudaGraphExec graphExec = graph.Instantiate();
            graphExec.SetEventRecordNodeEvent(recordNode, recordEvent);
            graphExec.SetEventWaitNodeEvent(waitNode, waitEvent);

            return $"Record={recordNode} Wait={waitNode} RecordHasEvent={recordHasEvent} WaitHasEvent={waitHasEvent} Nodes={graph.NodeCount} Edges={graph.EdgeCount}";
        }
        catch (CudaException exception)
        {
            return $"Skipped:{exception.Message}";
        }
    }

    static string ProbeGraphNodeParamsTypedDescriptors(CudaPinnedMemory source, CudaPinnedMemory destination, CudaMemory device, int byteCount)
    {
        try
        {
            using CudaMemory staging = new CudaMemory(byteCount);
            using CudaGraph graph = CudaGraph.Create();
            using CudaEvent recordEvent = new CudaEvent(CudaEventCreationFlags.DisableTiming);
            using CudaEvent waitEvent = new CudaEvent(CudaEventCreationFlags.DisableTiming);

            CudaGraphNode empty = graph.AddEmptyNode();
            CudaGraphNode eventRecord = graph.AddEventRecordNodeAfter(empty, recordEvent);
            CudaGraphNode eventWait = graph.AddEventWaitNodeAfter(eventRecord, waitEvent);
            CudaGraphNode memcpy = graph.AddHostToDeviceMemcpyNodeAfter(eventWait, device, source, byteCount);
            CudaGraphNode memcpyDevice = graph.AddDeviceToDeviceMemcpyNodeAfter(memcpy, staging, device, byteCount);

            CudaGraphNodeParamsDescriptor emptyDescriptor = CudaGraph.GetNodeParamsDescriptor(empty);
            CudaGraphNodeParamsDescriptor eventRecordDescriptor = CudaGraph.GetNodeParamsDescriptor(eventRecord);
            CudaGraphNodeParamsDescriptor eventWaitDescriptor = CudaGraph.GetNodeParamsDescriptor(eventWait);
            CudaGraphNodeParamsDescriptor memcpyDescriptor = CudaGraph.GetNodeParamsDescriptor(memcpy);
            CudaGraphNodeParamsDescriptor memcpyDeviceDescriptor = CudaGraph.GetNodeParamsDescriptor(memcpyDevice);

            if (emptyDescriptor.DescriptorKind != CudaGraphNodeParamsDescriptorKind.Empty ||
                eventRecordDescriptor.DescriptorKind != CudaGraphNodeParamsDescriptorKind.EventRecord ||
                eventWaitDescriptor.DescriptorKind != CudaGraphNodeParamsDescriptorKind.EventWait ||
                memcpyDescriptor.DescriptorKind != CudaGraphNodeParamsDescriptorKind.Memcpy ||
                memcpyDeviceDescriptor.DescriptorKind != CudaGraphNodeParamsDescriptorKind.Memcpy ||
                !eventRecordDescriptor.HasEvent ||
                !eventWaitDescriptor.HasEvent ||
                eventRecordDescriptor.HasBorrowedHandleExposure ||
                eventWaitDescriptor.HasBorrowedHandleExposure ||
                memcpyDescriptor.HasBorrowedHandleExposure ||
                memcpyDescriptor.ByteCount != (ulong)byteCount ||
                memcpyDeviceDescriptor.MemcpyKind != CudaMemcpyKind.DeviceToDevice)
            {
                throw new InvalidOperationException("CUDA graph typed node parameter descriptors returned unexpected values.");
            }

            return $"Kinds={emptyDescriptor.DescriptorKind}/{eventRecordDescriptor.DescriptorKind}/{eventWaitDescriptor.DescriptorKind}/{memcpyDescriptor.DescriptorKind}/{memcpyDeviceDescriptor.MemcpyKind} Bytes={memcpyDescriptor.ByteCount} Borrowed={memcpyDescriptor.HasBorrowedHandleExposure}";
        }
        catch (CudaException exception)
        {
            return $"Skipped:{exception.Message}";
        }
    }

    static string ProbeChildGraphUpdate(CudaStream uploadStream)
    {
        try
        {
            using CudaGraph childGraph = CudaGraph.Create();
            CudaGraphNode childRoot = childGraph.AddEmptyNode();
            childGraph.AddEmptyNodeAfter(childRoot);

            using CudaGraph parentGraph = CudaGraph.Create();
            CudaGraphNode parentRoot = parentGraph.AddEmptyNode();
            CudaGraphNode childNode = parentGraph.AddChildGraphNodeAfter(parentRoot, childGraph);
            CudaGraphChildSnapshot snapshot = parentGraph.GetChildGraphSnapshot(childNode);
            if (!snapshot.HasEmbeddedGraph || snapshot.NodeCount != 2 || snapshot.RootNodeCount != 1 || snapshot.EdgeCount != 1)
            {
                throw new InvalidOperationException("CUDA embedded child graph snapshot returned unexpected topology.");
            }

            using CudaGraphExec graphExec = parentGraph.Instantiate();
            using CudaGraph replacementChild = CudaGraph.Create();
            CudaGraphNode replacementRoot = replacementChild.AddEmptyNode();
            replacementChild.AddEmptyNodeAfter(replacementRoot);
            graphExec.SetChildGraphNodeParameters(childNode, replacementChild);
            CudaGraphExecUpdateSnapshot updateSnapshot = graphExec.Update(parentGraph);
            if (!updateSnapshot.Succeeded)
            {
                throw new InvalidOperationException($"CUDA executable graph update failed: {updateSnapshot}");
            }

            string parameterizedState;
            try
            {
                using CudaGraphExec parameterized = parentGraph.InstantiateWithParameters();
                using CudaGraphExec uploaded = parentGraph.InstantiateWithParameters(uploadStream, flags: 2);
                parameterizedState = $"Default=True Upload=True Flags={uploaded.Flags}";
            }
            catch (CudaException exception)
            {
                parameterizedState = $"Skipped:{exception.Message}";
            }

            return $"Snapshot=[{snapshot}] Update=[{updateSnapshot}] Parameterized={parameterizedState}";
        }
        catch (CudaException exception)
        {
            return $"Skipped:{exception.Message}";
        }
    }

    static string ProbeGraphMemcpy1D(CudaPinnedMemory source, CudaPinnedMemory destination, CudaMemory device, int byteCount)
    {
        try
        {
            using CudaMemory staging = new CudaMemory(byteCount);
            using CudaGraph graph = CudaGraph.Create();

            CudaGraphNode hostToDevice = graph.AddHostToDeviceMemcpyNode(device, source, byteCount);
            CudaGraphNode deviceToDevice = graph.AddDeviceToDeviceMemcpyNodeAfter(hostToDevice, staging, device, byteCount);
            CudaGraphNode deviceToHost = graph.AddDeviceToHostMemcpyNodeAfter(deviceToDevice, destination, staging, byteCount);

            CudaGraphMemcpyNodeParameters hostToDeviceParameters = CudaGraph.GetMemcpyNodeParameters(hostToDevice);
            CudaGraphMemcpyNodeParameters deviceToDeviceParameters = CudaGraph.GetMemcpyNodeParameters(deviceToDevice);
            CudaGraphMemcpyNodeParameters deviceToHostParameters = CudaGraph.GetMemcpyNodeParameters(deviceToHost);
            if (hostToDeviceParameters.Kind != CudaMemcpyKind.HostToDevice ||
                deviceToDeviceParameters.Kind != CudaMemcpyKind.DeviceToDevice ||
                deviceToHostParameters.Kind != CudaMemcpyKind.DeviceToHost ||
                hostToDeviceParameters.Width != (ulong)byteCount ||
                deviceToDeviceParameters.Width != (ulong)byteCount ||
                deviceToHostParameters.Width != (ulong)byteCount)
            {
                throw new InvalidOperationException("CUDA graph memcpy 1D diagnostics returned unexpected values.");
            }

            CudaGraph.SetHostToDeviceMemcpyNodeParameters(hostToDevice, device, source, byteCount);
            CudaGraph.SetDeviceToDeviceMemcpyNodeParameters(deviceToDevice, staging, device, byteCount);
            CudaGraph.SetDeviceToHostMemcpyNodeParameters(deviceToHost, destination, staging, byteCount);

            using CudaGraphExec graphExec = graph.Instantiate();
            graphExec.SetHostToDeviceMemcpyNodeParameters(hostToDevice, device, source, byteCount);
            graphExec.SetDeviceToDeviceMemcpyNodeParameters(deviceToDevice, staging, device, byteCount);
            graphExec.SetDeviceToHostMemcpyNodeParameters(deviceToHost, destination, staging, byteCount);

            return $"Nodes={graph.NodeCount} H2D={hostToDeviceParameters.Kind}/{hostToDeviceParameters.Width} D2D={deviceToDeviceParameters.Kind}/{deviceToDeviceParameters.Width} D2H={deviceToHostParameters.Kind}/{deviceToHostParameters.Width}";
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
            CudaDeviceGraphMemorySummary beforeSummary = before.ToSummary();
            CudaDevice.ResetGraphMemoryHighWatermarks(deviceOrdinal);
            CudaDevice.TrimGraphMemory(deviceOrdinal);
            CudaDeviceGraphMemoryInfo after = CudaDevice.GetGraphMemoryInfo(deviceOrdinal);
            CudaDeviceGraphMemorySummary afterSummary = after.ToSummary();
            return $"Before=[{before}] Summary=[{beforeSummary}] AfterResetTrim=[{after}] AfterSummary=[{afterSummary}]";
        }
        catch (CudaException exception)
        {
            return $"Skipped:{exception.Message}";
        }
    }
}

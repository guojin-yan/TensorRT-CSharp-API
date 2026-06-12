using System;
using System.Collections.Generic;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtBuilder
{
    /// <summary>
    /// Gets a read-only snapshot of TensorRT plugin creators registered for this builder.
    /// 获取当前 builder 可见的 TensorRT plugin creator 只读快照。
    /// </summary>
    /// <remarks>
    /// This method does not create plugins, deserialize plugins, load plugin libraries, or take ownership of creator objects.
    /// 该方法不会创建 plugin、反序列化 plugin、加载 plugin library，也不会接管 creator 对象所有权。
    /// </remarks>
    /// <returns>A plugin registry inventory snapshot. Plugin registry inventory 快照。</returns>
    public TensorRtPluginRegistryInventory GetPluginRegistryInventory()
    {
        int creatorCount = NativeBridgeApi.GetBuilderPluginRegistryCreatorCount(Line, _handle);
        bool hasErrorRecorder = NativeBridgeApi.HasBuilderPluginRegistryErrorRecorder(Line, _handle);
        bool parentSearchEnabled = NativeBridgeApi.IsBuilderPluginRegistryParentSearchEnabled(Line, _handle);
        List<TensorRtPluginCreatorInfo> creators = new List<TensorRtPluginCreatorInfo>(creatorCount);

        for (int creatorIndex = 0; creatorIndex < creatorCount; creatorIndex++)
        {
            string name = NativeBridgeApi.GetBuilderPluginCreatorName(Line, _handle, creatorIndex);
            string version = NativeBridgeApi.GetBuilderPluginCreatorVersion(Line, _handle, creatorIndex);
            string pluginNamespace = NativeBridgeApi.GetBuilderPluginCreatorNamespace(Line, _handle, creatorIndex);
            string interfaceKind = NativeBridgeApi.GetBuilderPluginCreatorInterfaceKind(Line, _handle, creatorIndex, out int interfaceMajor, out int interfaceMinor);
            int fieldCount = NativeBridgeApi.GetBuilderPluginCreatorFieldCount(Line, _handle, creatorIndex);
            List<TensorRtPluginFieldInfo> fields = new List<TensorRtPluginFieldInfo>(fieldCount);

            for (int fieldIndex = 0; fieldIndex < fieldCount; fieldIndex++)
            {
                string fieldName = NativeBridgeApi.GetBuilderPluginCreatorFieldName(Line, _handle, creatorIndex, fieldIndex);
                NativeBridgeApi.GetBuilderPluginCreatorFieldMetadata(Line, _handle, creatorIndex, fieldIndex, out TensorRtPluginFieldType fieldType, out int length, out bool hasData);
                fields.Add(new TensorRtPluginFieldInfo(fieldName, fieldType, length, hasData));
            }

            creators.Add(new TensorRtPluginCreatorInfo(
                creatorIndex,
                name,
                version,
                pluginNamespace,
                interfaceKind,
                interfaceMajor,
                interfaceMinor,
                fields));
        }

        return new TensorRtPluginRegistryInventory(Line, TensorRtPluginRegistrySource.Builder, hasErrorRecorder, parentSearchEnabled, recursiveCreatorCount: null, creators);
    }

    /// <summary>
    /// Tries to get a plugin registry inventory without throwing for unsupported TensorRT lines.
    /// 尝试获取 plugin registry inventory；当 TensorRT line 不支持时不抛出异常。
    /// </summary>
    /// <param name="inventory">The inventory snapshot when the query succeeds. 查询成功时的 inventory 快照。</param>
    /// <param name="diagnostic">A diagnostic string describing success or failure. 描述成功或失败原因的诊断字符串。</param>
    /// <returns><see langword="true"/> when the inventory was collected successfully. 成功收集 inventory 时返回 <see langword="true"/>。</returns>
    public bool TryGetPluginRegistryInventory(out TensorRtPluginRegistryInventory inventory, out string diagnostic)
    {
        try
        {
            inventory = GetPluginRegistryInventory();
            diagnostic = "OK";
            return true;
        }
        catch (Exception exception) when (exception is BridgeProbeException || exception is NotSupportedException || exception is InvalidOperationException)
        {
            inventory = new TensorRtPluginRegistryInventory(Line, TensorRtPluginRegistrySource.Builder, hasErrorRecorder: false, parentSearchEnabled: false, recursiveCreatorCount: null, Array.Empty<TensorRtPluginCreatorInfo>());
            diagnostic = exception.Message;
            return false;
        }
    }
}

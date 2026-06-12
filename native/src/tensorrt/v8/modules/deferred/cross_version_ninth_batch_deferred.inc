#define JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(function_name, interface_name, reason) \
    JYPPX_StatusCode function_name(void) \
    { \
        return jyppx::tensorrt::report_not_implemented(kLine, interface_name " deferred: " reason); \
    }

JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_blob_name_to_tensor_find_deferred, "IBlobNameToTensor::find", "borrowed tensor-map lookup needs a network-owned tensor reference policy before exposing handles")
JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_network_add_plugin_v2_deferred, "INetworkDefinition::addPluginV2", "plugin object ownership and pointer-array lifetimes require a dedicated safe plugin bridge")
JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_network_add_rnn_v2_deferred, "INetworkDefinition::addRNNv2", "deprecated RNNv2 setup mixes layer state, tensor handles, and weight lifetimes; it needs a dedicated safe RNN model")
JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_network_get_error_recorder_deferred, "INetworkDefinition::getErrorRecorder", "borrowed callback ownership is not safe across the C ABI")
JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_network_set_error_recorder_deferred, "INetworkDefinition::setErrorRecorder", "callback reference counting and lifetime require a managed recorder bridge")
JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_plugin_v2_layer_get_plugin_deferred, "IPluginV2Layer::getPlugin", "borrowed plugin pointer ownership is not safe across the C ABI")
JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_rnn_v2_layer_get_bias_for_gate_deferred, "IRNNv2Layer::getBiasForGate", "RNNv2 gate weights need dedicated ownership and indexing validation")
JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_rnn_v2_layer_get_cell_state_deferred, "IRNNv2Layer::getCellState", "borrowed tensor state needs a network-owned tensor reference policy")
JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_rnn_v2_layer_get_data_length_deferred, "IRNNv2Layer::getDataLength", "deprecated RNNv2 tensor access is grouped with the dedicated safe RNN model")
JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_rnn_v2_layer_get_direction_deferred, "IRNNv2Layer::getDirection", "deprecated RNNv2 accessors are deferred as a single safe RNN surface")
JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_rnn_v2_layer_get_hidden_size_deferred, "IRNNv2Layer::getHiddenSize", "deprecated RNNv2 accessors are deferred as a single safe RNN surface")
JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_rnn_v2_layer_get_hidden_state_deferred, "IRNNv2Layer::getHiddenState", "borrowed tensor state needs a network-owned tensor reference policy")
JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_rnn_v2_layer_get_input_mode_deferred, "IRNNv2Layer::getInputMode", "deprecated RNNv2 accessors are deferred as a single safe RNN surface")
JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_rnn_v2_layer_get_layer_count_deferred, "IRNNv2Layer::getLayerCount", "deprecated RNNv2 accessors are deferred as a single safe RNN surface")
JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_rnn_v2_layer_get_max_seq_length_deferred, "IRNNv2Layer::getMaxSeqLength", "deprecated RNNv2 accessors are deferred as a single safe RNN surface")
JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_rnn_v2_layer_get_operation_deferred, "IRNNv2Layer::getOperation", "deprecated RNNv2 accessors are deferred as a single safe RNN surface")
JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_rnn_v2_layer_get_sequence_lengths_deferred, "IRNNv2Layer::getSequenceLengths", "borrowed tensor state needs a network-owned tensor reference policy")
JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_rnn_v2_layer_get_weights_for_gate_deferred, "IRNNv2Layer::getWeightsForGate", "RNNv2 gate weights need dedicated ownership and indexing validation")
JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_rnn_v2_layer_set_bias_for_gate_deferred, "IRNNv2Layer::setBiasForGate", "RNNv2 gate weights need dedicated bridge-owned storage")
JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_rnn_v2_layer_set_cell_state_deferred, "IRNNv2Layer::setCellState", "borrowed tensor state needs a network-owned tensor reference policy")
JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_rnn_v2_layer_set_direction_deferred, "IRNNv2Layer::setDirection", "deprecated RNNv2 setters are deferred as a single safe RNN surface")
JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_rnn_v2_layer_set_hidden_state_deferred, "IRNNv2Layer::setHiddenState", "borrowed tensor state needs a network-owned tensor reference policy")
JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_rnn_v2_layer_set_input_mode_deferred, "IRNNv2Layer::setInputMode", "deprecated RNNv2 setters are deferred as a single safe RNN surface")
JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_rnn_v2_layer_set_operation_deferred, "IRNNv2Layer::setOperation", "deprecated RNNv2 setters are deferred as a single safe RNN surface")
JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_rnn_v2_layer_set_sequence_lengths_deferred, "IRNNv2Layer::setSequenceLengths", "borrowed tensor state needs a network-owned tensor reference policy")
JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB(jyppx_trt8_rnn_v2_layer_set_weights_for_gate_deferred, "IRNNv2Layer::setWeightsForGate", "RNNv2 gate weights need dedicated bridge-owned storage")

#undef JYPPX_TRT8_NINTH_BATCH_DEFERRED_STUB

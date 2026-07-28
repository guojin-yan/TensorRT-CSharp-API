#pragma once

#include <stddef.h>
#include <stdint.h>

#include "jyppx/common/bridge_exports.h"
#include "jyppx/common/status.h"

typedef struct JYPPX_CudaStream JYPPX_CudaStream;
typedef struct JYPPX_CudaEvent JYPPX_CudaEvent;
typedef struct JYPPX_CudaMemory JYPPX_CudaMemory;
typedef struct JYPPX_CudaPinnedMemory JYPPX_CudaPinnedMemory;
typedef struct JYPPX_CudaPitchedMemory JYPPX_CudaPitchedMemory;
typedef struct JYPPX_CudaGraph JYPPX_CudaGraph;
typedef struct JYPPX_CudaGraphExec JYPPX_CudaGraphExec;
typedef struct JYPPX_CudaGraphMemoryAllocation JYPPX_CudaGraphMemoryAllocation;
typedef struct JYPPX_CudaGraphConditionalHandle JYPPX_CudaGraphConditionalHandle;
typedef struct JYPPX_CudaGraphConditionalNode JYPPX_CudaGraphConditionalNode;
typedef struct JYPPX_CudaArray JYPPX_CudaArray;
typedef struct JYPPX_CudaMipmappedArray JYPPX_CudaMipmappedArray;
typedef struct JYPPX_CudaTextureObject JYPPX_CudaTextureObject;
typedef struct JYPPX_CudaSurfaceObject JYPPX_CudaSurfaceObject;
typedef struct JYPPX_CudaKernelLibrary JYPPX_CudaKernelLibrary;
typedef struct JYPPX_CudaKernelLaunch JYPPX_CudaKernelLaunch;
typedef struct JYPPX_CudaDriverModule JYPPX_CudaDriverModule;
typedef struct JYPPX_CudaDriverKernelLaunch JYPPX_CudaDriverKernelLaunch;
typedef struct JYPPX_CudaExecutionContext JYPPX_CudaExecutionContext;
typedef struct JYPPX_CudaRtcProgram JYPPX_CudaRtcProgram;

typedef enum JYPPX_CudaKernelArgumentKind
{
    JYPPX_CUDA_KERNEL_ARGUMENT_SCALAR = 1,
    JYPPX_CUDA_KERNEL_ARGUMENT_DEVICE_MEMORY = 2
} JYPPX_CudaKernelArgumentKind;

typedef struct JYPPX_CudaKernelArgumentDescriptor
{
    int32_t kind;
    uint32_t reserved;
    size_t scalar_offset;
    size_t scalar_size;
    JYPPX_CudaMemory* memory;
    size_t memory_offset;
} JYPPX_CudaKernelArgumentDescriptor;

typedef enum JYPPX_CudaRtcArtifactKind
{
    JYPPX_CUDA_RTC_ARTIFACT_PTX = 1,
    JYPPX_CUDA_RTC_ARTIFACT_CUBIN = 2,
    JYPPX_CUDA_RTC_ARTIFACT_LTO_IR = 3
} JYPPX_CudaRtcArtifactKind;

typedef struct JYPPX_CudaRtcCapabilityInfo
{
    JYPPX_Boolean dependency_available;
    int32_t version_major;
    int32_t version_minor;
    JYPPX_Boolean supports_ptx;
    JYPPX_Boolean supports_cubin;
    JYPPX_Boolean supports_lto_ir;
    JYPPX_Boolean supports_deprecated_nvvm;
    JYPPX_Boolean supports_name_expressions;
} JYPPX_CudaRtcCapabilityInfo;

typedef struct JYPPX_CudaDriverCapabilityInfo
{
    JYPPX_Boolean dependency_available;
    int32_t driver_version;
    JYPPX_Boolean supports_module_load;
    JYPPX_Boolean supports_function_lookup;
    JYPPX_Boolean supports_typed_launch;
    JYPPX_Boolean supports_context_interop;
    JYPPX_Boolean supports_completion_events;
} JYPPX_CudaDriverCapabilityInfo;

typedef enum JYPPX_CudaMemcpyKind
{
    JYPPX_CUDA_MEMCPY_HOST_TO_DEVICE = 1,
    JYPPX_CUDA_MEMCPY_DEVICE_TO_HOST = 2,
    JYPPX_CUDA_MEMCPY_DEVICE_TO_DEVICE = 3
} JYPPX_CudaMemcpyKind;

typedef struct JYPPX_CudaRuntimeInfo
{
    JYPPX_Boolean vendor_dependency_available;
    JYPPX_Boolean supports_streams;
    JYPPX_Boolean supports_events;
    JYPPX_Boolean supports_memory;
    int32_t runtime_version;
    int32_t driver_version;
    int32_t device_count;
    const char* status_message;
} JYPPX_CudaRuntimeInfo;

typedef struct JYPPX_CudaDeviceInfo
{
    int32_t ordinal;
    char name[256];
    int32_t major;
    int32_t minor;
    int32_t multi_processor_count;
    int32_t warp_size;
    int32_t max_threads_per_block;
    int32_t can_map_host_memory;
    int32_t integrated;
    uint64_t total_global_memory;
} JYPPX_CudaDeviceInfo;

typedef struct JYPPX_CudaDeviceSelectionRequirements
{
    int32_t major;
    int32_t minor;
    int32_t multi_processor_count;
    int32_t warp_size;
    int32_t max_threads_per_block;
    int32_t can_map_host_memory;
    int32_t integrated;
    uint64_t total_global_memory;
} JYPPX_CudaDeviceSelectionRequirements;

typedef struct JYPPX_CudaMemoryInfo
{
    uint64_t free_bytes;
    uint64_t total_bytes;
} JYPPX_CudaMemoryInfo;

typedef struct JYPPX_CudaKernelLibraryInventory
{
    uint32_t reported_kernel_count;
    uint32_t enumerated_kernel_count;
    uint32_t null_kernel_count;
    JYPPX_Boolean is_complete;
} JYPPX_CudaKernelLibraryInventory;

typedef struct JYPPX_CudaPointerAttributes
{
    int32_t memory_type;
    int32_t device;
    uint64_t device_pointer;
    uint64_t host_pointer;
} JYPPX_CudaPointerAttributes;

typedef struct JYPPX_CudaMemRangeAttributeValue
{
    int32_t attribute;
    int32_t value;
} JYPPX_CudaMemRangeAttributeValue;

typedef struct JYPPX_CudaManagedMemoryBatchRange
{
    JYPPX_CudaMemory* memory;
    size_t offset;
    size_t size;
    int32_t destination_device;
} JYPPX_CudaManagedMemoryBatchRange;

typedef struct JYPPX_CudaPitchedMemoryInfo
{
    uint64_t pitch_bytes;
    uint64_t width_bytes;
    uint64_t height;
} JYPPX_CudaPitchedMemoryInfo;

typedef struct JYPPX_CudaChannelFormatDesc
{
    int32_t x;
    int32_t y;
    int32_t z;
    int32_t w;
    int32_t format_kind;
} JYPPX_CudaChannelFormatDesc;

typedef struct JYPPX_CudaArrayExtent
{
    uint64_t width;
    uint64_t height;
    uint64_t depth;
} JYPPX_CudaArrayExtent;

typedef struct JYPPX_CudaArrayInfo
{
    JYPPX_CudaChannelFormatDesc channel;
    JYPPX_CudaArrayExtent extent;
    uint32_t flags;
} JYPPX_CudaArrayInfo;

typedef struct JYPPX_CudaArrayMemoryRequirements
{
    uint64_t size;
    uint64_t alignment;
} JYPPX_CudaArrayMemoryRequirements;

typedef struct JYPPX_CudaArraySparseProperties
{
    uint32_t tile_width;
    uint32_t tile_height;
    uint32_t tile_depth;
    uint32_t mip_tail_first_level;
    uint64_t mip_tail_size;
    uint32_t flags;
} JYPPX_CudaArraySparseProperties;

typedef struct JYPPX_CudaResourceDescriptorSnapshot
{
    int32_t resource_type;
    JYPPX_Boolean has_array;
    JYPPX_Boolean has_mipmapped_array;
    JYPPX_Boolean has_device_pointer;
    uint64_t size_in_bytes;
    uint64_t width;
    uint64_t height;
    uint64_t pitch_in_bytes;
} JYPPX_CudaResourceDescriptorSnapshot;

typedef struct JYPPX_CudaTextureDescriptor
{
    int32_t address_mode_x;
    int32_t address_mode_y;
    int32_t address_mode_z;
    int32_t filter_mode;
    int32_t read_mode;
    JYPPX_Boolean srgb;
    float border_color_r;
    float border_color_g;
    float border_color_b;
    float border_color_a;
    JYPPX_Boolean normalized_coordinates;
    uint32_t max_anisotropy;
    int32_t mipmap_filter_mode;
    float mipmap_level_bias;
    float min_mipmap_level_clamp;
    float max_mipmap_level_clamp;
    JYPPX_Boolean disable_trilinear_optimization;
    JYPPX_Boolean seamless_cubemap;
} JYPPX_CudaTextureDescriptor;

typedef struct JYPPX_CudaTextureResourceViewSnapshot
{
    JYPPX_Boolean is_specified;
    int32_t format;
    uint64_t width;
    uint64_t height;
    uint64_t depth;
    uint32_t first_mipmap_level;
    uint32_t last_mipmap_level;
    uint32_t first_layer;
    uint32_t last_layer;
} JYPPX_CudaTextureResourceViewSnapshot;

typedef struct JYPPX_CudaMemLocation
{
    int32_t type;
    int32_t id;
} JYPPX_CudaMemLocation;

typedef struct JYPPX_CudaMemcpyAttributes
{
    int32_t src_access_order;
    JYPPX_CudaMemLocation src_location_hint;
    JYPPX_CudaMemLocation dst_location_hint;
    uint32_t flags;
} JYPPX_CudaMemcpyAttributes;

typedef struct JYPPX_CudaPitchedPtr
{
    void* pointer;
    size_t pitch;
    size_t x_size;
    size_t y_size;
} JYPPX_CudaPitchedPtr;

typedef struct JYPPX_CudaPos
{
    size_t x;
    size_t y;
    size_t z;
} JYPPX_CudaPos;

typedef struct JYPPX_CudaMemcpy3DPeerParams
{
    JYPPX_CudaArray* src_array;
    JYPPX_CudaPos src_pos;
    JYPPX_CudaPitchedPtr src_ptr;
    int32_t src_device;
    JYPPX_CudaArray* dst_array;
    JYPPX_CudaPos dst_pos;
    JYPPX_CudaPitchedPtr dst_ptr;
    int32_t dst_device;
    JYPPX_CudaArrayExtent extent;
} JYPPX_CudaMemcpy3DPeerParams;

typedef struct JYPPX_CudaOffset3D
{
    size_t x;
    size_t y;
    size_t z;
} JYPPX_CudaOffset3D;

typedef struct JYPPX_CudaMemcpy3DOperandPointer
{
    void* pointer;
    size_t row_length;
    size_t layer_height;
    JYPPX_CudaMemLocation location_hint;
} JYPPX_CudaMemcpy3DOperandPointer;

typedef struct JYPPX_CudaMemcpy3DOperandArray
{
    JYPPX_CudaArray* array;
    JYPPX_CudaOffset3D offset;
} JYPPX_CudaMemcpy3DOperandArray;

typedef struct JYPPX_CudaMemcpy3DOperand
{
    int32_t type;
    JYPPX_CudaMemcpy3DOperandPointer pointer;
    JYPPX_CudaMemcpy3DOperandArray array;
} JYPPX_CudaMemcpy3DOperand;

typedef struct JYPPX_CudaMemcpy3DBatchOp
{
    JYPPX_CudaMemcpy3DOperand source;
    JYPPX_CudaMemcpy3DOperand destination;
    JYPPX_CudaArrayExtent extent;
    int32_t source_access_order;
    uint32_t flags;
} JYPPX_CudaMemcpy3DBatchOp;

typedef struct JYPPX_CudaMemPoolPtrExportData
{
    uint8_t reserved[64];
} JYPPX_CudaMemPoolPtrExportData;

typedef struct JYPPX_CudaGraphEdgeData
{
    uint8_t from_port;
    uint8_t to_port;
    uint8_t type;
    uint8_t reserved0;
    uint8_t reserved1;
    uint8_t reserved2;
    uint8_t reserved3;
    uint8_t reserved4;
} JYPPX_CudaGraphEdgeData;

typedef struct JYPPX_CudaDevResourceSnapshot
{
    int32_t type;
    int32_t is_valid;
    int32_t device_ordinal;
    uint32_t sm_count;
    uint32_t min_sm_partition_size;
    uint32_t sm_coscheduled_alignment;
    uint32_t sm_flags;
    uint32_t workqueue_concurrency_limit;
    int32_t workqueue_sharing_scope;
    JYPPX_Boolean has_opaque_workqueue;
    JYPPX_Boolean has_next_resource;
} JYPPX_CudaDevResourceSnapshot;

typedef struct JYPPX_CudaGraphMemsetNodeParams
{
    uint64_t destination_address;
    uint64_t pitch;
    uint32_t value;
    uint32_t element_size;
    uint64_t width;
    uint64_t height;
} JYPPX_CudaGraphMemsetNodeParams;

typedef struct JYPPX_CudaGraphMemcpyNodeParams
{
    uint64_t source_address;
    uint64_t destination_address;
    uint64_t source_pitch;
    uint64_t destination_pitch;
    uint64_t source_x_size;
    uint64_t source_y_size;
    uint64_t destination_x_size;
    uint64_t destination_y_size;
    uint64_t source_position_x;
    uint64_t source_position_y;
    uint64_t source_position_z;
    uint64_t destination_position_x;
    uint64_t destination_position_y;
    uint64_t destination_position_z;
    uint64_t width;
    uint64_t height;
    uint64_t depth;
    int32_t kind;
    uint32_t source_is_array;
    uint32_t destination_is_array;
} JYPPX_CudaGraphMemcpyNodeParams;

typedef struct JYPPX_CudaGraphKernelNodeAttributeValue
{
    int32_t attribute;
    int32_t int_value;
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} JYPPX_CudaGraphKernelNodeAttributeValue;

typedef struct JYPPX_CudaGraphKernelNodeParamsSnapshot
{
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t grid_z;
    uint32_t block_x;
    uint32_t block_y;
    uint32_t block_z;
    uint32_t shared_memory_bytes;
    JYPPX_Boolean has_function;
    JYPPX_Boolean has_kernel_params;
    JYPPX_Boolean has_extra;
} JYPPX_CudaGraphKernelNodeParamsSnapshot;

typedef struct JYPPX_CudaGraphHostNodeParamsSnapshot
{
    JYPPX_Boolean has_callback;
    JYPPX_Boolean has_user_data;
} JYPPX_CudaGraphHostNodeParamsSnapshot;

typedef struct JYPPX_CudaGraphMemAllocNodeParamsSnapshot
{
    uint64_t byte_count;
    uint64_t access_descriptor_count;
    int32_t allocation_type;
    uint64_t handle_types;
    int32_t location_type;
    int32_t location_id;
    JYPPX_Boolean has_access_descriptors;
    JYPPX_Boolean has_device_pointer;
    JYPPX_Boolean has_security_attributes;
} JYPPX_CudaGraphMemAllocNodeParamsSnapshot;

typedef struct JYPPX_CudaGraphMemFreeNodeParamsSnapshot
{
    JYPPX_Boolean has_device_pointer;
} JYPPX_CudaGraphMemFreeNodeParamsSnapshot;

typedef struct JYPPX_CudaGraphExternalSemaphoreNodeParamsSnapshot
{
    uint32_t semaphore_count;
    JYPPX_Boolean has_semaphore_array;
    JYPPX_Boolean has_parameter_array;
} JYPPX_CudaGraphExternalSemaphoreNodeParamsSnapshot;

typedef struct JYPPX_CudaDim3
{
    uint32_t x;
    uint32_t y;
    uint32_t z;
} JYPPX_CudaDim3;

typedef struct JYPPX_CudaFuncAttributes
{
    size_t shared_size_bytes;
    size_t const_size_bytes;
    size_t local_size_bytes;
    int32_t max_threads_per_block;
    int32_t num_registers;
    int32_t ptx_version;
    int32_t binary_version;
    int32_t cache_mode_ca;
    int32_t max_dynamic_shared_size_bytes;
    int32_t preferred_shared_memory_carveout;
    int32_t cluster_dim_must_be_set;
    int32_t required_cluster_width;
    int32_t required_cluster_height;
    int32_t required_cluster_depth;
    int32_t cluster_scheduling_policy_preference;
    int32_t non_portable_cluster_size_allowed;
} JYPPX_CudaFuncAttributes;

typedef struct JYPPX_CudaLaunchConfig
{
    JYPPX_CudaDim3 grid_dim;
    JYPPX_CudaDim3 block_dim;
    size_t dynamic_shared_memory_bytes;
    JYPPX_CudaStream* stream;
    void* attributes;
    uint32_t attribute_count;
} JYPPX_CudaLaunchConfig;

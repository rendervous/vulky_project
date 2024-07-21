from enum import IntEnum


class PresenterMode(IntEnum):
    """
    Presentation mode of a device. Decide if the render target resources are optimized for window
    presentation or not.
    """
    NONE = 0
    """
    The presentation mode is unknown
    """

    OFFLINE = 1
    """
    Render target is not optimized for window presentation.
    """

    WINDOW = 2
    """
    Render target is optimized for presenting on a window.
    """



class QueueType(IntEnum):
    """
    Type of the graphics engine queue used for submitting tasks.
    """
    NONE = 0
    """
    The queue is unknown
    """
    COPY = 1
    """
    The queue supports only transfer commands.
    """
    COMPUTE = 2
    """
    The queue supports transfer and compute commands.
    """
    GRAPHICS = 3
    """
    The queue supports transfer, compute and rasterization commands.
    """
    RAYTRACING = 4
    """
    The queue supports transfer, compute, graphics and raytracing commands.
    """


class BufferUsage(IntEnum):
    """
    Determines the usage for a buffer object.
    """
    NONE = 0
    """
    The usage is unknown.
    """
    STAGING = 1
    """
    The buffer is used for staging.
    """
    VERTEX = 2
    """
    The buffer is used as vertex buffer input in a rasterization pipeline.
    """
    INDEX = 3
    """
    The buffer is used as index buffer input in a rasterization pipeline.
    """
    UNIFORM = 4
    """
    The buffer is used as uniform object.
    """
    STORAGE = 5
    """
    The buffer is used as a storage buffer.
    """
    RAYTRACING_ADS = 6
    """
    The buffer is used to store a BVH.
    """
    RAYTRACING_RESOURCE = 7
    """
    The buffer is used as vertices or indices for building a BVH
    """
    SHADER_TABLE = 8
    """
    The buffer is used to store shader handles.
    """


class ImageUsage(IntEnum):
    """
    Determines the usage for an image object.
    """
    NONE = 0
    """
    The usage is unknown.
    """
    TRANSFER = 1
    """
    The image can be copied to/from the cpu
    """
    SAMPLED = 2
    """
    The image can be sampled from a shader.
    """
    RENDER_TARGET = 3
    """
    The image can be used only as render target.
    """
    DEPTH_STENCIL = 4
    """
    The image can be used only as a depth-stencil buffer
    """
    STORAGE = 5
    """
    The image can be used only as storage
    """
    ANY = 6
    """
    The image can be used for sampling, storage and as render target.
    """


class Format(IntEnum):
    NONE = 0
    UINT_RGBA = 1
    UINT_RGB = 2
    UINT_BGRA_STD = 3
    UINT_RGBA_STD = 4
    UINT_RGBA_UNORM = 5
    UINT_BGRA_UNORM = 6
    FLOAT = 7
    INT = 8
    UINT = 9
    VEC2 = 10
    VEC3 = 11
    VEC4 = 12
    IVEC2 = 13
    IVEC3 = 14
    IVEC4 = 15
    UVEC2 = 16
    UVEC3 = 17
    UVEC4 = 18
    PRESENTER = 19
    DEPTH_STENCIL = 20


class ImageType(IntEnum):
    NONE = 0
    TEXTURE_1D = 1
    TEXTURE_2D = 2
    TEXTURE_3D = 3


class MemoryLocation(IntEnum):
    """
    Memory configurations.
    """
    NONE = 0
    """
    Memory location is unknown
    """
    GPU = 1
    """
    Efficient memory for reading and writing on the GPU.
    """
    CPU = 2
    """
    Memory can be read and write directly from the CPU
    """


class ShaderStage(IntEnum):
    """
    Determines the shader stage
    """
    NONE = 0
    """
    Shader stage is unknown.
    """
    VERTEX = 1
    """
    Shader stage for per-vertex process in a rasterization pipeline.
    """
    FRAGMENT = 2
    """
    Shader stage for per-fragment process in a rasterization pipeline.
    """
    COMPUTE = 3
    """
    Shader stage for a compute process in a compute pipeline.
    """
    RT_GENERATION = 4
    """
    Shader stage for ray generation in a ray-tracing pipeline.
    """
    RT_CLOSEST_HIT = 5
    """
    Shader stage for closest hit process in a ray-tracing pipeline.
    """
    RT_MISS = 6
    """
    Shader stage for a ray miss process in a ray-tracing pipeline.
    """
    RT_ANY_HIT = 7
    """
    Shader stage for any ray hit process in a ray-tracing pipeline.
    """
    RT_INTERSECTION_HIT = 8
    """
    Shader stage for final intersection hit process in a ray-tracing pipeline.
    """
    RT_CALLABLE = 9
    """
    Shader stage for a callable shader in a ray-tracing pipeline.  
    """
    GEOMETRY = 10,
    """
    Shader stage for geometry process in a rasterization pipeline.
    """



class Filter(IntEnum):
    """
    Filter used during sampling a texel.
    """
    NONE = 0
    """
    The filter is unknown.
    """
    POINT = 1
    """
    Nearest sample.
    """
    LINEAR = 2
    """
    Linear (bilinear) interpolation is used.
    """


class MipMapMode(IntEnum):
    """
    Filter used during sampling between mip maps.
    """
    NONE = 0
    """
    The filter is unknown.
    """
    POINT = 1
    """
    Nearest sample.
    """
    LINEAR = 2
    """
    Linear (bilinear, trilinear) interpolation is used.
    """


class AddressMode(IntEnum):
    """
    Determines how the sampler behaves at edges.
    """
    NONE = 0
    """
    The behaviour is unknown.
    """
    REPEAT = 1
    """
    Sampler repeat coordinates.
    """
    CLAMP_EDGE = 2
    """
    Sampler repeat color at edge
    """
    BORDER = 3
    """
    Sampler uses a constant value for coordinates out of range.
    """


class CompareOp(IntEnum):
    """
    Determines different comparison operations for depth, blending, stencil and other operations.
    """
    NONE = 0
    """
    The operation is unknown.
    """
    NEVER = 1
    """
    Test always fails.
    """
    LESS = 2
    """
    First element is less than second.
    """
    EQUAL = 3
    """
    Elements are equals.
    """
    LESS_OR_EQUAL = 4
    """
    First element is less or equals than second.
    """
    GREATER = 5
    """
    First element is greater than second.
    """
    NOT_EQUAL = 6
    """
    Elements are different.
    """
    GREATER_OR_EQUAL = 7
    """
    First element is greater or equals than second.
    """
    ALWAYS = 8
    """
    Test always succeed.
    """


class BorderColor(IntEnum):
    """
    Determines possible values for the border sampling.
    """
    NONE = 0
    """
    The value is unknown.
    """
    TRANSPARENT_BLACK_FLOAT = 1
    """
    All values are 0.0f.
    """
    TRANSPARENT_BLACK_INT = 2
    """
    All values are 0
    """
    OPAQUE_BLACK_FLOAT = 3
    """
    All components are 0.0 but the alpha is full (1.0f)
    """
    OPAQUE_BLACK_INT = 4
    """
    All components are 0 but the alpha is full
    """
    OPAQUE_WHITE_FLOAT = 5
    """
    All components are 1.0f
    """
    OPAQUE_WHITE_INT = 6
    """
    All components are full
    """


class ADSNodeType(IntEnum):
    """
    Determines possible types of an ADS buffer.
    """
    NONE = 0
    """
    The element type is unknown.
    """
    TRIANGLES = 1
    """
    Elements of the (bottom) ADS buffer are triangles. 
    """
    AABB = 2
    """
    Elements of the (bottom) ADS buffer are boxes. 
    """
    INSTANCE = 3
    """
    Elements of the (top) ADS buffer are instances.
    """


class PipelineType(IntEnum):
    """
    Determines the pipeline type.
    """
    NONE = 0
    """
    The pipeline is unknown.
    """
    COMPUTE = 1
    """
    The pipeline is for a compute process.
    """
    GRAPHICS = 2
    """
    The pipeline is for a rasterization process.
    """
    RAYTRACING = 3
    """
    The pipeline is for a ray-tracing process.
    """


class DescriptorType(IntEnum):
    """
    Determines the type of descriptor of a resource binding.
    """
    NONE = 0
    """
    The descriptor is unknown.
    """
    SAMPLER = 1
    """
    The descriptor is for a sampler object.
    """
    UNIFORM_BUFFER = 2
    """
    The descriptor is for a uniform buffer object.
    """
    STORAGE_BUFFER = 3
    """
    The descriptor is for a storage buffer object.
    """
    STORAGE_IMAGE = 4
    """
    The descriptor is for a storage image object.
    """
    SAMPLED_IMAGE = 5
    """
    The descriptor is for an image that can be sampled.
    """
    COMBINED_IMAGE = 6
    """
    The descriptor is for an image and a sampler object.
    """
    SCENE_ADS = 7
    """
    The descriptor is for an (top) ADS buffer.
    """



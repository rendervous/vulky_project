try:
    import imgui
    __GUI_AVAILABLE__ = True
except:
    __GUI_AVAILABLE__ = False

from ._common import *
from ._vulkan_memory_allocator import VulkanMemory
from . import _vulkan_internal as _internal
from ._vulkan_internal import (
    ShaderHandlerWrapper as ShaderHandler,
    FrameBufferWrapper as FrameBuffer,
    SamplerWrapper as Sampler
)
import torch as _torch
import typing as _typing
import numpy as _np


__TRACE_WRAP__ = False

def compile_shader_source(code, stage, include_dirs):
    import subprocess
    import os
    idirs = " ".join(" -I\"" + d + "\" " for d in include_dirs)
    if os.name == 'nt':  # Windows
        p = subprocess.Popen(
            os.path.expandvars(
                '%VULKAN_SDK%/Bin/glslangValidator.exe --stdin -r -Os -V --target-env vulkan1.2 ').replace("\\", "/")
            + f'-S {stage} {idirs}', stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE
        )
        outs, errs = p.communicate(code.encode('utf-8'))
    else:  # Assuming Linux
        # Quick monkeyhack for linux based distribution
        import shlex
        p = subprocess.Popen(
            shlex.split(
                os.path.expandvars('/usr/bin/glslangValidator --stdin -r -V --target-env vulkan1.3 ').replace("\\",
                                                                                                      "/")
                + f'-S {stage} {idirs}'), stdin=subprocess.PIPE
        )
        outs, errs = p.communicate(code.encode('utf-8'))
    perr = p.wait()
    if perr != 0:
        numbered_code = '\n'.join(f"{i+1}\t\t{l}" for i,l in enumerate(code.split('\n')))
        print("[ERROR] Compilation failed")
        print(outs)
        raise RuntimeError(f"Cannot compile {numbered_code}")
    with open(f'{stage}.spv', 'rb') as f:
        binary_output = f.read(-1)
    # print(code)
    # print(f'[INFO] Compiled code for {stage}')
    return binary_output


def compile_shader_source_file(filename, stage, binary_file_name, include_dirs=[]):
    import subprocess
    import os
    idirs = "".join("-I\"" + d + "\"" for d in include_dirs)
    if os.name == 'nt':  # Windows
        p = subprocess.Popen(
            os.path.expandvars(
                '%VULKAN_SDK%/Bin/glslangValidator.exe -r -V --target-env vulkan1.2 ').replace("\\", "/")
            + f'-S {stage} {idirs} \"{filename}\" -o \"{binary_file_name}\"'
        )
    else:  # Assuming Linux
        # Quick monkeyhack for linux based distribution
        import shlex
        p = subprocess.Popen(
            shlex.split(
                os.path.expandvars('/usr/bin/glslangValidator -r -V --target-env vulkan1.2 ').replace("\\",
                                                                                                      "/")
                + f'-S {stage} {idirs} \"{filename}\" -o \"{binary_file_name}\"')
        )
    if p.wait() != 0:
        raise RuntimeError(f"Cannot compile {filename}")
    # print(f'[INFO] Compiled... {filename}')


__HEADER_LAST_MODIFIED_TIME__ = dict()


def _get_last_modified_time_for_headers_at(d: str):
    import os
    if d in __HEADER_LAST_MODIFIED_TIME__:
        return __HEADER_LAST_MODIFIED_TIME__[d]

    last_modified_time = 0.0
    for f in os.listdir(d):
        filename = d + '/' + f
        if os.path.isfile(filename):
            if f.endswith('.h'): # header
                last_modified_time = max(last_modified_time, os.path.getmtime(filename))
        else:
            last_modified_time = max(last_modified_time, _get_last_modified_time_for_headers_at(filename))
    __HEADER_LAST_MODIFIED_TIME__[d] = last_modified_time
    return last_modified_time


def _get_last_modified_time_for_headers_included(include_dirs: _typing.List):
    return max(_get_last_modified_time_for_headers_at(d) for d in include_dirs)


def compile_shader_file(filename: str, include_dirs: _typing.List):
    import os
    filename_without_extension, extension = os.path.splitext(filename)
    binary_file = filename_without_extension + ".spv"
    if os.path.isfile(binary_file):
        # Check if it is valid
        if os.path.getmtime(binary_file) > max(os.path.getmtime(filename), _get_last_modified_time_for_headers_included(include_dirs)):
            return binary_file  # compiled and uptodate
    assert extension == '.glsl'
    stage = os.path.splitext(filename_without_extension)[1][1:]  # [1:] for removing the dot .
    compile_shader_source_file(filename, stage, binary_file, include_dirs)
    return binary_file


def compile_shader_sources(directory='.', force_all: bool = False):
    import os
    def needs_to_update(source, binary):
        return not os.path.exists(binary) or os.path.getmtime(source) > os.path.getmtime(binary)
    directory = directory.replace('\\', '/')
    for filename in os.listdir(directory):
        filename = directory + "/" + filename
        filename_without_extension, extension = os.path.splitext(filename)
        if extension == '.glsl':
            stage = os.path.splitext(filename_without_extension)[1][1:]  # [1:] for removing the dot .
            binary_file = filename_without_extension + ".spv"
            if needs_to_update(filename, binary_file) or force_all:
                compile_shader_source_file(filename, stage, binary_file)


class Resource(object):
    def __init__(self, device, w_resource: _internal.ResourceWrapper):
        self.w_resource = w_resource
        self.device = device

    @lazy_constant
    def is_buffer(self):
        return self.w_resource.resource_data.is_buffer

    @lazy_constant
    def is_ads(self):
        return self.w_resource.resource_data.is_ads

    @lazy_constant
    def is_image(self):
        return not self.w_resource.resource_data.is_buffer

    @lazy_constant
    def is_on_cpu(self):
        return self.w_resource.resource_data.is_cpu

    @lazy_constant
    def is_on_gpu(self):
        return self.w_resource.resource_data.is_gpu

    def clear(self) -> 'Resource':
        self.w_resource.clear(self.device.w_device)
        return self

    def load(self, src_data):
        if src_data is self:
            return self
        if isinstance(src_data, Resource):
            src_data = src_data.w_resource
        self.w_resource.load(self.device.w_device, src_data)
        return self

    def save(self, dst_data):
        if dst_data is self:
            return self
        if isinstance(dst_data, Resource):
            dst_data = dst_data.w_resource
        self.w_resource.save(self.device.w_device, dst_data)
        return self

    @lazy_constant
    def cuda_ptr(self):
        return self.w_resource.cuda_ptr

    @lazy_constant
    def device_ptr(self):
        return self.w_resource.device_ptr

    @lazy_constant
    def cpu_ptr(self):
        return self.w_resource.cpu_ptr

    @lazy_constant
    def size(self):
        raise Exception('Not implemented')


class ObjectBufferAccessor:
    _rdv_memory: memoryview = None      # memory of the whole object
    _rdv_layout: Layout = None      # layout for the whole object
    _rdv_fields: dict = None      # gets precomputed accessors, tensors or references

    def __init__(self, memory: memoryview, layout: Layout):
        object.__setattr__(self, '_rdv_memory', memory)
        object.__setattr__(self, '_rdv_layout', layout)
        object.__setattr__(self, '_rdv_fields', dict())
        self._build()

    def _collect_references(self, s: set):
        for k, v in self._rdv_fields.items():
            if self._rdv_layout.is_structure:
                _, layout = self._rdv_layout.fields_layout[k]
            else:  # is_array
                layout = self._rdv_layout.element_layout
            if layout.scalar_format == 'Q':
                if not isinstance(v, int): # Reference without object
                    s.add(v)
            if layout.is_structure or layout.is_array:
                v: ObjectBufferAccessor
                v._collect_references(s)

    def references(self) -> set:
        s = set()
        self._collect_references(s)
        return s

    def _build(self):
        if self._rdv_layout.is_structure:
            for item, (offset, layout) in self._rdv_layout.fields_layout.items():
                self._build_element(item, offset, layout)
        else:  #array
            for i in range(self._rdv_layout.declaration[0]):
                field_layout = self._rdv_layout.element_layout
                offset = i * field_layout.aligned_size
                self._build_element(i, offset, field_layout)

    def _build_element(self, item, offset, field_layout):
        field_memory = self._rdv_memory[offset: offset + field_layout.aligned_size]
        if field_layout.is_structure or field_layout.is_array:
            s = ObjectBufferAccessor(field_memory, field_layout)
            self._rdv_fields[item] = s
            return s
        if field_layout.is_scalar:
            value = field_memory.cast(field_layout.scalar_format)[0]
            if field_layout.scalar_format == 'Q':
                assert value == 0, 'Can not build a reference from non-null memory info'
                value = None  # Special None case for reference types
            self._rdv_fields[item] = value
            return value
        t = _torch.frombuffer(field_memory, dtype=field_layout.element_layout.declaration)
        tensor_type = field_layout.declaration
        if field_layout.is_vector:  # possible vec4 to vec3
            t = t[0:tensor_type.tensor_shape[0]]
        if not field_layout.is_vector:
            t = t.view(tensor_type.tensor_shape[0], -1)[:, 0:tensor_type.tensor_shape[1]]
        value = tensor_type(t)
        self._rdv_fields[item] = value
        return value

    @staticmethod
    def _equal_values(field_layout: Layout, v1: _typing.Any, v2: _typing.Any):
        if field_layout.is_scalar:
            if field_layout.scalar_format == 'Q':
                return v1 is v2
            return v1 == v2
        return v1 is v2 # or _torch.all(v1 == v2).item()

    def _set_element(self, key, offset, field_layout, value):
        assert not field_layout.is_structure and not field_layout.is_array
        current_value = self._rdv_fields[key]
        if ObjectBufferAccessor._equal_values(field_layout, current_value, value):
            return
        if field_layout.is_scalar:
            field_memory = self._rdv_memory[offset: offset + field_layout.aligned_size]
            if field_layout.scalar_format == 'Q':  #reference type
                if value is None:
                    field_memory.cast('Q')[0] = 0
                    self._rdv_fields[key] = None  # save reference as a cached value
                    return
                if isinstance(value, int):
                    field_memory.cast('Q')[0] = value
                    self._rdv_fields[key] = value  # save reference as a cached value
                    return
                assert isinstance(value, GPUPtr), 'Invalid type, bind a GPUPtr object'
                field_memory.cast('Q')[0] = value.device_ptr
                self._rdv_fields[key] = value
                return
            field_memory.cast(field_layout.scalar_format)[0] = value
            self._rdv_fields[key] = value  # update cached value
            return
        # else update tensor value
        current_value[:] = value

    def __getattr__(self, item):
        assert self._rdv_layout.is_structure
        if item in self._rdv_fields:
            return self._rdv_fields[item]
        raise AttributeError(item)

    def __setattr__(self, key, value):
        assert self._rdv_layout.is_structure
        offset, field_layout = self._rdv_layout.fields_layout[key]
        self._set_element(key, offset, field_layout, value)

    def __getitem__(self, item):
        assert self._rdv_layout.is_array
        return self._rdv_fields[item]

    def __len__(self):
        assert self._rdv_layout.is_array
        return self._rdv_layout.aligned_size // self._rdv_layout.array_stride

    def __setitem__(self, key, value):
        assert self._rdv_layout.is_array
        field_layout = self._rdv_layout.element_layout
        offset = key * field_layout.aligned_size
        return self._set_element(key, offset, field_layout, value)


class Buffer(Resource):
    """
    Represents a continuous memory on the device
    """
    def __init__(self, device: 'DeviceManager', w_buffer: _internal.ResourceWrapper):
        super().__init__(device, w_buffer)

    # def tensor(self, *shape: int, dtype: _torch.dtype = _torch.uint8, offset: int = 0):
    #     if len(shape) == 0:
    #         size = self.size() - offset
    #     else:
    #         type_size = Layout.scalar_size(dtype)
    #         size = math.prod(shape) * type_size
    #     return self.w_resource.slice_buffer(offset, size).as_tensor(dtype)

    def slice(self, offset: int, size: int):
        return Buffer(self.device, self.w_resource.slice_buffer(offset, size))

    @lazy_constant
    def size(self):
        return self.w_resource.size

    @lazy_constant
    def memory(self):
        assert self.w_resource.resource_data.is_cpu
        return self.w_resource.bytes

    def __repr__(self):
        if self.w_resource.resource_data.support_direct_tensor_map:
            return repr(self.w_resource.as_tensor(_torch.uint8))
        stag = self.device.create_buffer(min(4*32, self.size), BufferUsage.STAGING, MemoryLocation.CPU)
        self.slice(0, stag.size).save(stag)
        return repr(_np.asarray(stag.memory.cast('f')))


class ObjectBuffer(Buffer):
    def __init__(self, device: 'DeviceManager', w_buffer: _internal.ResourceWrapper, layout: Layout):
        super(ObjectBuffer, self).__init__(device, w_buffer)
        assert layout.aligned_size == w_buffer.size
        if not w_buffer.resource_data.is_cpu:
            self._staging = device.create_buffer(self.size, BufferUsage.STAGING, MemoryLocation.CPU)
        else:
            self._staging = self
        self._staging.clear()
        self.accessor = ObjectBufferAccessor(self._staging.memory, layout)

    def update_gpu(self):
        self.load(self._staging)

    def __enter__(self):
        return self.accessor

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.load(self._staging)


class StructuredBufferAccess:
    def __init__(self, memory: memoryview, structure_stride: int, offset: int, structure_layout: Layout):
        object.__setattr__(self, '_rdv_memory', memory)  # All buffer memory
        object.__setattr__(self, '_rdv_layout', structure_layout)  # the layout of a single element structure
        object.__setattr__(self, '_rdv_structure_stride', structure_stride)  # the structure stride
        object.__setattr__(self, '_rdv_offset', offset)  # the offset for a specific element in the whole struct
        object.__setattr__(self, '_rdv_fields', dict())  #cached fields converted into tensors

    @staticmethod
    def build_accessor(memory: memoryview, stride: int, offset: int, layout: Layout):
        if layout.is_structure or layout.is_array:
            return StructuredBufferAccess(memory, stride, offset, layout)
        scalar_format = layout.scalar_format if layout.is_scalar else layout.element_layout.scalar_format
        torch_dtype = layout.declaration if layout.is_scalar else layout.element_layout.declaration
        scalar_size = Layout.scalar_size(torch_dtype)
        element_offset = offset // scalar_size
        element_count = layout.aligned_size // scalar_size
        t = _torch.frombuffer(memory, dtype=torch_dtype).view(-1, stride // scalar_size)[:, element_offset:element_offset + element_count]
        if layout.is_scalar:
            return t
        tensor_type = layout.declaration
        if layout.is_vector:
            return tensor_type(t)
        t = t.view(len(t), tensor_type.tensor_shape[0], -1)[:, :, 0:tensor_type.tensor_shape[1]]
        return tensor_type(t)
        # dtype = layout.get_type() if layout.is_scalar else layout.element_layout.get_type()
        # t = w_buffer.as_tensor(dtype)
        # element_size = t.element_size()
        # start_element = offset // element_size
        # stride_element = stride // element_size
        # count_element = layout.aligned_size // element_size
        # t = t.view(-1, stride_element)[:, start_element:start_element + count_element]
        # if layout.is_scalar:
        #     return t
        # tensor_type = layout.get_type()
        # if layout.is_matrix:
        #     t = t.view(t.shape[0], tensor_type.tensor_shape[0], -1)[...,0:tensor_type.tensor_shape[1]]
        # return tensor_type(t)

    def _get_subelement(self, offset, layout):
        if offset not in self._rdv_fields:
            self._rdv_fields[offset] = StructuredBufferAccess.build_accessor(self._rdv_memory, self._rdv_structure_stride, self._rdv_offset + offset, layout)
        return self._rdv_fields[offset]

    def _set_subelement(self, offset, layout, value):
        assert not layout.is_array and not layout.is_structure
        t = self._get_subelement(offset, layout)
        # if isinstance(value, _torch.Tensor):  # and value.shape == t.shape:
        #    assert value.device.type == 'cpu'
        if _np.isscalar(value):
            t[:] = value
        else:
            t[:] = value
        # else:
        #     t[:] = value
        #     # _torch.fill_(t, value)

    def __getattr__(self, item):
        assert self._rdv_layout.is_structure, f'Trying to get {item} to a non-structured layout'
        offset, layout = self._rdv_layout.fields_layout[item]
        return self._get_subelement(offset, layout)

    def __setattr__(self, key, value):
        assert self._rdv_layout.is_structure
        offset, layout = self._rdv_layout.fields_layout[key]
        self._set_subelement(offset, layout, value)

    def __getitem__(self, item):
        assert self._rdv_layout.is_array
        assert isinstance(item, int), 'Can not index with slice or anything else.'
        size = self._rdv_layout.element_layout.aligned_size
        offset = size * item
        return self._get_subelement(offset, self._rdv_layout.element_layout)

    def __setitem__(self, key, value):
        assert isinstance(key, int), 'Can not index with slice or anything else'
        assert self._rdv_layout.is_array
        size = self._rdv_layout.element_layout.aligned_size
        offset = size * key
        return self._set_subelement(offset, self._rdv_layout.element_layout, value)


class StructuredBuffer(Buffer):
    def __init__(self, device: 'DeviceManager', w_buffer: _internal.ResourceWrapper, layout: Layout):
        assert w_buffer.size % layout.aligned_size == 0
        super(StructuredBuffer, self).__init__(device, w_buffer)
        self.layout = layout

    def map(self, mode: _typing.Literal['in', 'out', 'inout'], clear: bool = False):
        assert not clear or mode == 'in', 'Can only clear when map in'
        _self: StructuredBuffer = self
        if self.w_resource.resource_data.is_gpu:
            staging = buffer_like(self, MemoryLocation.CPU)
        else:
            staging = self
        if clear:
            staging.clear()
        class Context:
            def __enter__(self):
                if mode == 'out' or mode == 'inout':
                    _self.save(staging)
                return StructuredBufferAccess(staging.memory, _self.layout.aligned_size, 0, _self.layout)
            def __exit__(self, exc_type, exc_val, exc_tb):
                if mode == 'in' or mode == 'inout':
                    _self.load(staging)
        return Context()


# class ResourceAccess:
#     def __init__(self, w_buffer: ResourceWrapper, offset: int, stride: int, layout: Layout, is_structured_buffer: bool = False):
#         assert w_buffer.support_direct_tensor_map()
#         object.__setattr__(self, 'w_buffer', w_buffer)
#         object.__setattr__(self, 'layout', layout)
#         object.__setattr__(self, 'offset', offset)
#         object.__setattr__(self, 'stride', stride)
#         object.__setattr__(self, 'is_structured_buffer', is_structured_buffer)
#         object.__setattr__(self, '_cached', dict())
#
#     @staticmethod
#     def build_accessor(w_buffer: ResourceWrapper, offset: int, stride: int, layout: Layout, is_structured_buffer: bool = False):
#         if layout.is_structure or layout.is_array:
#             return ResourceAccess(w_buffer, offset, stride, layout, is_structured_buffer)
#         return w_buffer.as_bytes()[offset: offset + layout.aligned_size]
#         # dtype = layout.get_type() if layout.is_scalar else layout.element_layout.get_type()
#         # t = w_buffer.as_tensor(dtype)
#         # element_size = t.element_size()
#         # start_element = offset // element_size
#         # stride_element = stride // element_size
#         # count_element = layout.aligned_size // element_size
#         # t = t.view(-1, stride_element)[:, start_element:start_element + count_element]
#         # if layout.is_scalar:
#         #     return t
#         # tensor_type = layout.get_type()
#         # if layout.is_matrix:
#         #     t = t.view(t.shape[0], tensor_type.tensor_shape[0], -1)[...,0:tensor_type.tensor_shape[1]]
#         # return tensor_type(t)
#
#     def _get_subelement(self, offset, layout):
#         if offset not in self._cached:
#             self._cached[offset] = ResourceAccess.build_accessor(self.w_buffer, self.offset + offset, self.stride, layout)
#         return self._cached[offset]
#
#     def _set_subelement(self, offset, layout, value):
#         assert not layout.is_array and not layout.is_structure
#         t = self._get_subelement(offset, layout)
#         # if isinstance(value, _torch.Tensor):  # and value.shape == t.shape:
#         #    assert value.device.type == 'cpu'
#         if _np.isscalar(value):
#             t[:] = struct.pack(layout.scalar_format, value)
#         else:
#             t[:] = memoryview(_np.asarray(value)).cast('B')
#         # else:
#         #     t[:] = value
#         #     # _torch.fill_(t, value)
#
#     def __getattr__(self, item):
#         assert self.layout.is_structure
#         offset, layout = self.layout.fields_layout(item)
#         return self._get_subelement(offset, layout)
#
#     def __setattr__(self, key, value):
#         assert self.layout.is_structure
#         offset, layout = self.layout.fields_layout(key)
#         self._set_subelement(offset, layout, value)
#
#     def __getitem__(self, item):
#         assert isinstance(item, int), 'Can not index with slice or anything else.'
#         if self.is_structured_buffer:
#             return ResourceAccess(self.w_buffer.buffer_slice(item*self.stride, (item+1)*self.stride), 0, self.stride, self.layout)
#         size = self.layout.element_layout.aligned_size
#         offset = size * item
#         return self._get_subelement(offset, self.layout.element_layout)
#
#     def __setitem__(self, key, value):
#         assert isinstance(key, int), 'Can not index with slice or anything else'
#         assert not self.is_structured_buffer
#         assert not self.layout.is_structure
#         size = self.layout.element_layout.aligned_size
#         offset = size * key
#         return self._set_subelement(offset, self.layout.element_layout, value)
#
#
# class _LayoutBufferBase(Buffer):
#
#     def __init__(self, device: 'DeviceManager', w_buffer: ResourceWrapper, layout: Layout, is_structured_buffer: bool):
#         super(_LayoutBufferBase, self).__init__(device, w_buffer)
#         self.layout = layout
#         self.is_structured_buffer = is_structured_buffer
#         self.__map_staging = None
#         self.__cache_accessor = None
#         self.__maps = 0
#         self.__is_dynamic = True
#
#     def make_dynamic(self, value=True):
#         self.__is_dynamic = value
#         return self
#
#     def __enter__(self):
#         self.__maps += 1
#         if self.__cache_accessor is None:
#             if self.w_resource.resource_data.is_cpu_visible: #support_direct_tensor_map():
#                 self.__map_staging = self.w_resource
#             else:
#                 self.__map_staging = self.device.w_device.create_buffer(self.w_resource.get_size(), BufferUsage.STAGING, MemoryLocation.CPU)
#             self.__cache_accessor = ResourceAccess.build_accessor(self.__map_staging, 0, self.layout.aligned_size, self.layout, self.is_structured_buffer)
#         if self.__maps == 1:
#             self.__map_staging.load(self.device.w_device, self.w_resource)
#         return self.__cache_accessor
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.__maps -= 1
#         if not self.w_resource.resource_data.is_cpu_visible: #support_direct_tensor_map():
#             if self.__maps == 0:
#                 self.__map_staging.save(self.device.w_device, self.w_resource)
#         if not self.__is_dynamic:
#             self.__map_staging = None
#             self.__cache_accessor = None
#
#     def get_access(self):
#         assert self.w_resource.resource_data.is_cpu_visible
#         if self.__cache_accessor is None:
#             self.__cache_accessor = ResourceAccess.build_accessor(self.w_resource, 0, self.layout.aligned_size, self.layout, self.is_structured_buffer)
#         return self.__cache_accessor
#
#
# class UniformBuffer(_LayoutBufferBase):
#     def __init__(self, device: 'DeviceManager', w_buffer: ResourceWrapper, layout: Layout):
#         super(UniformBuffer, self).__init__(device, w_buffer, layout, False)
#
#
# class StructuredBuffer(_LayoutBufferBase):
#     def __init__(self, device: 'DeviceManager', w_buffer: ResourceWrapper, layout: Layout):
#         super(StructuredBuffer, self).__init__(device, w_buffer, layout, True)


# class ObjectBuffer(Buffer):
#     def __init__(self, device: 'DeviceManager', w_buffer: ResourceWrapper, layout: Layout):
#         super(ObjectBuffer, self).__init__(device, w_buffer)
#         self.layout = layout
#         self.__map_staging = None
#         self.__cache_accessor = None
#         self.__maps = 0
#         self.__is_dynamic = True


class Image(Resource):
    """
    TODO: NOT IDEAL DESIGN
    """
    @staticmethod
    def compute_dimension(width: int, height: int, depth: int, mip_level: int):
        return max(1, width // (1 << mip_level)), max(1, height // (1 << mip_level)), max(1, depth // (1 << mip_level))

    def __init__(self, device, w_image: _internal.ResourceWrapper, texel_layout: Layout):
        super().__init__(device, w_image)
        self.width, self.height, self.depth = Image.compute_dimension(
            w_image.resource_data.vk_description.extent.width,
            w_image.resource_data.vk_description.extent.height,
            w_image.resource_data.vk_description.extent.depth,
            w_image.current_slice["mip_start"]
        )
        self.layout = texel_layout
        self._cache_size = None

    def get_image_dimension(self) -> int:
        type = self.w_resource.resource_data.vk_description.imageType
        return type + 1

    def get_mip_count(self) -> int:
        return self.w_resource.current_slice["mip_count"]

    def get_array_count(self) -> int:
        return self.w_resource.current_slice["array_count"]

    def slice_mips(self, mip_start, mip_count):
        return Image(self.device, self.w_resource.slice_mips(mip_start, mip_count), self.layout)

    def slice_array(self, array_start, array_count):
        return Image(self.device, self.w_resource.slice_array(array_start, array_count), self.layout)

    def subresource(self, mip: int = 0, layer: int = 0):
        return Image(self.device, self.w_resource.subresource(mip, layer), self.layout)

    def as_readonly(self):
        return Image(self.device, self.w_resource.as_readonly(), self.layout)

    def size(self):
        return self.w_resource.size

    def element_size(self):
        return self.layout.aligned_size

    def numel(self):
        return self.size() // self.element_size()

    def subresource_footprint(self, mip: int = 0, layer: int = 0):
        return self.w_resource.get_subresource_footprint(mip, layer)


class GeometryCollection:

    def __init__(self, device: _internal.DeviceWrapper):
        self.w_device = device
        self.descriptions = []

    def __del__(self):
        self.w_device = None
        self.descriptions = []

    def get_collection_type(self) -> ADSNodeType:
        pass


class TriangleCollection(GeometryCollection):

    def __init__(self, device: _internal.DeviceWrapper):
        super().__init__(device)

    def append(self, vertices: Buffer,
               indices: Buffer = None,
               transform: Buffer = None):
        self.descriptions.append((vertices, indices, transform))

    def get_collection_type(self) -> ADSNodeType:
        return ADSNodeType.TRIANGLES  # Triangles


class AABBCollection(GeometryCollection):
    def __init__(self, device: _internal.DeviceWrapper):
        super().__init__(device)

    def append(self, aabb: Buffer):
        self.descriptions.append(aabb)

    def get_collection_type(self) -> ADSNodeType:
        return ADSNodeType.AABB  # AABB


class ADS(Resource):
    def __init__(self, device, w_resource: _internal.ResourceWrapper, handle, scratch_size,
                 info: _typing.Any, ranges, instance_buffer=None):
        super().__init__(device, w_resource)
        self.ads = w_resource.resource_data.ads
        self.ads_info = info
        self.handle = handle
        self.scratch_size = scratch_size
        self.ranges = ranges
        self.instance_buffer = instance_buffer


class RTProgram:

    def __init__(self, pipeline: 'Pipeline',
                 w_shader_table: _internal.ResourceWrapper,
                 miss_offset,
                 hit_offset,
                 callable_offset):
        self.pipeline = pipeline
        prop = self.pipeline.w_pipeline.w_device.raytracing_properties
        self.shader_handle_stride = prop.shaderGroupHandleSize
        self.w_table = w_shader_table
        self.__map_w_table = None
        self.__map_w_table_tensor = None
        # self.raygen_slice = w_shader_table.slice_buffer(0, self.shader_handle_stride)
        self.miss_slice = w_shader_table.slice_buffer(miss_offset, hit_offset - miss_offset)
        self.hitgroup_slice = w_shader_table.slice_buffer(hit_offset, callable_offset - hit_offset)
        self.callable_slice = w_shader_table.slice_buffer(callable_offset, w_shader_table.size - callable_offset)
        self.miss_offset = miss_offset
        self.hit_offset = hit_offset
        self.callable_offset = callable_offset
        self.main_program = 0

    def select_generation(self, main_index: int):
        self.main_program = main_index

    def active_raygen_slice(self):
        offset = self.main_program * self.shader_handle_stride
        return self.w_table.slice_buffer(offset, self.shader_handle_stride)

    def __enter__(self):
        if self.w_table.support_direct_tensor_map:
            self.__map_w_table = self.w_table
            self.__map_w_table_tensor = self.__map_w_table.as_tensor(_torch.uint8)
        else:
            self.__map_w_table = self.pipeline.w_pipeline.w_device.create_buffer(self.w_table.size, BufferUsage.STAGING, MemoryLocation.CPU)
            self.__map_w_table.load(self.pipeline.w_pipeline.w_device, self.w_table)
            self.__map_w_table_tensor = self.__map_w_table.as_tensor(_torch.uint8)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.w_table.support_direct_tensor_map:
            self.__map_w_table.save(self.pipeline.w_pipeline.w_device, self.w_table)
        self.__map_w_table = None
        self.__map_w_table_tensor = None

    def __del__(self):
        self.pipeline = None
        self.w_table = None
        self.raygen_slice = None
        self.miss_slice = None
        self.hitgroup_slice = None
        self.callable_slice = None

    def set_generation(self, gen_index: int, shader_group: ShaderHandler):
        assert self.__map_w_table is not None, 'Program need to updated inside a context, use with program...'
        offset = gen_index * self.shader_handle_stride
        self.__map_w_table_tensor[offset:offset + self.shader_handle_stride] = _torch.frombuffer(shader_group.handle, dtype=_torch.uint8)

    def set_miss(self, miss_index: int, shader_group: ShaderHandler):
        assert self.__map_w_table is not None, 'Program need to updated inside a context, use with program...'
        offset = self.miss_offset + miss_index * self.shader_handle_stride
        self.__map_w_table_tensor[offset:offset + self.shader_handle_stride] = _torch.frombuffer(shader_group.handle, dtype=_torch.uint8)

    def set_hit_group(self, hit_group_index: int, shader_group: ShaderHandler):
        assert self.__map_w_table is not None, 'Program need to updated inside a context, use with program...'
        offset = self.hit_offset + hit_group_index * self.shader_handle_stride
        self.__map_w_table_tensor[offset:offset + self.shader_handle_stride] = _torch.frombuffer(shader_group.handle, dtype=_torch.uint8)

    def set_callable(self, callable_index: int, shader_group: ShaderHandler):
        assert self.__map_w_table is not None, 'Program need to updated inside a context, use with program...'
        offset = self.callable_offset + callable_index * self.shader_handle_stride
        self.__map_w_table_tensor[offset:offset + self.shader_handle_stride] = _torch.frombuffer(shader_group.handle, dtype=_torch.uint8)


class DescriptorSet:
    def __init__(self, w_ds: 'DescriptorSetWrapper', layout_reference_names: _typing.Dict[str, _typing.Tuple[int, int]]):
        self.w_ds = w_ds
        self.layout_reference_name = layout_reference_names

    def update(self, **bindings: _typing.Union['Resource', _typing.List['Resource'], _typing.Tuple['Resource', 'Sampler']]):
        """
        Updates the descriptors for bindings in the descriptor set.
        Notice that grouping bindings updates in a single call might lead to better performance.
        >>> pipeline : Pipeline = ...
        >>> pipeline.layout(set=0, binding=0, transforms=DescriptorType.UNIFORM_BUFFER)
        >>> pipeline.layout(set=0, binding=1, environment=DescriptorType.SAMPLED_IMAGE)
        >>> ...
        >>> pipeline.close()
        >>> dss = pipeline.create_descriptor_set_collection(set=0, count=1)
        >>> ds = dss[0]  # peek the only descriptor set in the collection
        >>> ds.update(  # write resources descriptors to the slots named within pipeline.layout
        >>>     transforms=my_transforms_buffer,
        >>>     environment=my_environment_image
        >>> )
        """
        def convert_to_wrapped(r: _typing.Union['Resource', _typing.List['Resource'], _typing.Tuple['Resource', 'Sampler']]):
            if r is None:
                return r
            if isinstance(r, tuple):
                return (convert_to_wrapped(r[0]), r[1])  # sampler element in the tuple is already the wrapper version
            if isinstance(r, list):
                return [convert_to_wrapped(a) for a in r]
            return r.w_resource
        to_update = {}
        for k, v in bindings.items():
            assert k in self.layout_reference_name
            set, binding = self.layout_reference_name[k]
            assert set == self.w_ds.set
            to_update[binding] = convert_to_wrapped(v)
        self.w_ds.update(to_update)


class DescriptorSetCollection:
    def __init__(self, w_dsc: _internal.DescriptorSetCollectionWrapper, layout_reference_names: _typing.Dict[str, _typing.Tuple[int, int]]):
        self.w_dsc = w_dsc
        self.layout_reference_names = layout_reference_names

    def __len__(self):
        return len(self.w_dsc.desc_sets_wrappers)

    def __getitem__(self, item):
        return DescriptorSet(self.w_dsc.desc_sets_wrappers[item], self.layout_reference_names)



class Pipeline:
    def __init__(self, w_pipeline: _internal.PipelineWrapper):
        self.w_pipeline = w_pipeline
        self.layout_reference_names = {}  # maps name to layout (set, binding)

    def is_closed(self):
        return self.w_pipeline.initialized

    def close(self):
        self.w_pipeline._build()

    class _ShaderStageContext:

        def __init__(self, pipeline, *new_stages):
            self.pipeline = pipeline
            self.new_stages = new_stages

        def __enter__(self):
            self.pipeline.w_pipeline.push_active_stages(*self.new_stages)
            return self.pipeline

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.pipeline.w_pipeline.pop_active_stages()

    def shader_stages(self, *shader_stages: ShaderStage):
        """
        Activates temporarily a set of shader stages.
        To be used in a context scope. e.g.,
        >>> pipeline: Pipeline = ...
        >>> # here pipeline has all stages active
        >>> with pipeline.shader_stages(ShaderStage.VERTEX, ShaderStage.FRAGMENT):
        >>> ... # settings requiring only VS and FS stages active
        >>> # here pipeline has all stages active again
        """
        assert len(shader_stages) > 0, 'At least one shader stage should be active'
        return Pipeline._ShaderStageContext(self, *shader_stages)

    def layout(self,
               set: int,
               binding: int,
               array_size: _typing.Optional[int] = None,
               is_variable: bool = False,
               **bind_declaration: DescriptorType):
        """
        Declares a part of the pipeline layout.
        set: specify the set number this bind belongs to.
        binding: specify the binding slot.
        array_size: Number of resources bound as an array.
        is_variable: If true, the bound array is not fixed although count is used as upper bound.
        bind_declaration: A single key-value pair indicating the reference name and the descriptor type,
        Example:
        >>> pipeline: Pipeline = ...
        >>> pipeline.layout(set=0, binding=0, camera_transforms=DescriptorType.UNIFORM_BUFFER)
        >>> pipeline.layout(set=1, binding=0, model_transform=DescriptorType.UNIFORM_BUFFER)
        >>> pipeline.layout(set=1, binding=1, array_size=10, model_textures=DescriptorType.SAMPLED_IMAGE)
        """
        assert len(bind_declaration) == 1, 'One and only one bind declaration must be provided.'
        k, v = next(iter(bind_declaration.items()))
        if is_variable:
            assert array_size is not None, 'Only arrays can be unbound size. In that case, array_size must be specified as upper bound.'
        else:
            if array_size is None:
                array_size = 1  # used as count for single descriptors
        assert isinstance(v, DescriptorType)
        assert k not in self.layout_reference_names, f'The name {k} is already used in the pipeline layout'
        self.layout_reference_names[k] = (set, binding)
        self.w_pipeline.layout(set=set, binding=binding, descriptor_type=v, count=array_size, is_variable=is_variable)

    def load_shader(self, path, *specialization, main_function = 'main'):
        return self.w_pipeline.load_shader(_internal.ShaderStageWrapper.from_file(
            device=self.w_pipeline.w_device,
            main_function=main_function,
            path=path,
            specialization=specialization
        ))

    def load_shader_from_source(self, code, *specialization, main_function = 'main', include_dirs = []):
        stage_prefix = {
            ShaderStage.VERTEX: 'vert',
            ShaderStage.FRAGMENT: 'frag',
            ShaderStage.COMPUTE: 'comp',
            ShaderStage.RT_GENERATION: 'rgen',
            ShaderStage.RT_MISS: 'rmiss',
            ShaderStage.RT_ANY_HIT: 'rahit',
            ShaderStage.RT_INTERSECTION_HIT: 'rint',
            ShaderStage.RT_CLOSEST_HIT: 'rchit',
            ShaderStage.RT_CALLABLE: 'rcall',
        }[self.w_pipeline.get_single_active_shader_stage()]
        shader_id  = -1
        # try:
        binary_code = compile_shader_source(code, stage_prefix, include_dirs)
        shader_id = self.w_pipeline.load_shader(_internal.ShaderStageWrapper.from_binary(
            device=self.w_pipeline.w_device,
            main_function=main_function,
            bytecode=binary_code,
            specialization=specialization
        ))
        # except:
        #     pass

        if shader_id < 0:
            raise Exception('Error compiling code')

        return shader_id

    def create_descriptor_set_collection(self, set: int, count: int):
        return DescriptorSetCollection(
            self.w_pipeline.create_descriptor_set_collection(set, count, variable_size=count),
            self.layout_reference_names
        )


class GraphicsPipeline(Pipeline):
    def __init__(self, w_pipeline: _internal.PipelineWrapper):
        super().__init__(w_pipeline)
        self.attach_reference_names = {}  # maps name to attach slot
        self.vertex_reference_names = {}  # maps name to vertex attributes
        self.vertex_bindings = {}  # maps bindings to dict[attribute_name, offset]

    def attach(self, slot: int, **attach_declaration: Format):
        assert len(attach_declaration) == 1
        k, v = next(iter(attach_declaration.items()))
        assert k not in self.attach_reference_names
        assert isinstance(v, Format), 'Attachment declaration must be a Format'
        self.w_pipeline.attach(slot, format=v)
        self.attach_reference_names[k] = slot

    def vertex(self, location: int, **attribute_declaration):
        assert len(attribute_declaration) == 1
        k, v = next(iter(attribute_declaration.items()))
        assert k not in self.vertex_reference_names
        assert isinstance(v, Format), 'Vertex attribute declaration must be a Format'
        self.w_pipeline.vertex(location, v)
        self.vertex_reference_names[k] = (location, v)

    def vertex_binding(self, binding: int, stride: int, **attribute_offset_map):
        assert binding not in self.vertex_bindings, 'Vertex-buffer binding slot already set'
        assert all(k in self.vertex_reference_names for k in attribute_offset_map), 'Vertex-buffer binding refers to a vertex attribute has not declared.'
        self.vertex_bindings[binding] = attribute_offset_map
        self.w_pipeline.vertex_binding(binding, stride, {
            self.vertex_reference_names[k][0]: offset for k, offset in attribute_offset_map.items()
        })

    def create_framebuffer(self, width: int, height: int, layers: int = 1, **bindings: _typing.Union['Image', None]) -> FrameBuffer:
        max_slot = -1 if len(self.attach_reference_names)==0 else max(self.attach_reference_names.values())
        attachments = [None]*(max_slot + 1)
        for k,v in bindings.items():
            attachments[self.attach_reference_names[k]] = None if v is None else v.w_resource
        return self.w_pipeline.create_framebuffer(width, height, layers, attachments)


class RaytracingPipeline(Pipeline):
    def __init__(self, w_pipeline: _internal.PipelineWrapper):
        super().__init__(w_pipeline)

    def create_rt_hit_group(self, closest_hit: int = None, any_hit: int = None, intersection: int = None):
        return self.w_pipeline.create_hit_group(closest_hit, any_hit, intersection)

    def create_rt_gen_group(self, generation_shader_index: int):
        return self.w_pipeline.create_general_group(generation_shader_index)

    def create_rt_miss_group(self, miss_shader_index: int):
        return self.w_pipeline.create_general_group(miss_shader_index)

    def create_rt_callable_group(self, callable_index: int):
        return self.w_pipeline.create_general_group(callable_index)

    def _get_aligned_size(self, size, align):
        return (size + align - 1) & (~(align - 1))

    def create_rt_program(self, max_raygen_shader=1, max_miss_shader=10, max_hit_groups=1000, max_callable=1000) -> RTProgram:
        assert self.is_closed(), 'Can not create programs from a pipeline is still open. Please close the pipeline first.'

        shaderHandlerSize = self.w_pipeline.w_device.raytracing_properties.shaderGroupHandleSize
        groupAlignment = self.w_pipeline.w_device.raytracing_properties.shaderGroupBaseAlignment

        raygen_size = self._get_aligned_size(shaderHandlerSize * max_raygen_shader, groupAlignment)
        raymiss_size = self._get_aligned_size(shaderHandlerSize * max_miss_shader, groupAlignment)
        rayhit_size = self._get_aligned_size(shaderHandlerSize * max_hit_groups, groupAlignment)
        raycall_size = self._get_aligned_size(shaderHandlerSize * max_callable, groupAlignment)

        w_buffer = self.w_pipeline.w_device.create_buffer(
            raygen_size + raymiss_size + rayhit_size + raycall_size,
            usage=BufferUsage.SHADER_TABLE,
            memory=MemoryLocation.CPU)# .clear(self.w_pipeline.w_device)
        return RTProgram(self, w_buffer, raygen_size, raygen_size + raymiss_size, raygen_size + raymiss_size + rayhit_size)

    def stack_size(self, depth: int):
        self.w_pipeline.set_max_recursion(depth)


class CommandManager:

    def __init__(self, device, w_cmdList: _internal.CommandBufferWrapper):
        self.w_cmdList = w_cmdList
        self.device = device

    def __del__(self):
        self.w_cmdList = None
        self.device = None

    @classmethod
    def get_queue_required(cls) -> int:
        pass

    def freeze(self):
        self.w_cmdList.freeze()

    def is_frozen(self):
        return self.w_cmdList.is_frozen()

    def is_closed(self):
        return self.w_cmdList.is_closed()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.w_cmdList.end()
        self.w_cmdList.flush_and_wait()
        # self.device.safe_dispatch_function(lambda: self.w_cmdList.flush_and_wait())

    def use(self, image: Image, image_usage: ImageUsage):
        self.w_cmdList.use_image_as(image.w_resource, image_usage)

    def image_barrier(self, image: Image, image_usage: ImageUsage):
        self.w_cmdList.transition_image_layout(image.w_resource, image_usage)


class CopyManager(CommandManager):
    def __init__(self, device, w_cmdList: _internal.CommandBufferWrapper):
        super().__init__(device, w_cmdList)

    @classmethod
    def get_queue_required(cls) -> int:
        return QueueType.COPY

    def copy(self, src_resource, dst_resource):
        if src_resource is None or dst_resource is None:
            return
        self.w_cmdList.copy(src_resource.w_resource, dst_resource.w_resource)

    # def copy_image(self, src_image: Image, dst_image: Image):
    #     self.w_cmdList.copy_image(src_image.w_resource, dst_image.w_resource)
    #
    # def copy_buffer_to_image(self, src_buffer: Buffer, dst_image: Image):
    #     self.w_cmdList.copy_buffer_to_image(src_buffer.w_resource, dst_image.w_resource)


class ComputeManager(CopyManager):
    def __init__(self, device, w_cmdList: _internal.CommandBufferWrapper):
        super().__init__(device, w_cmdList)

    @classmethod
    def get_queue_required(cls) -> int:
        return QueueType.COMPUTE

    def clear_color(self, image: Image, color):
        self.w_cmdList.clear_color(image.w_resource, color)

    def clear_depth_stencil(self, image: Image, depth: float = 1.0, stencil: int = 0):
        self.w_cmdList.clear_depth_stencil(image.w_resource, depth, stencil)

    def clear_buffer(self, buffer: Buffer, value: int = 0):
        self.w_cmdList.clear_buffer(buffer.w_resource, value)

    def set_pipeline(self, pipeline: Pipeline):
        if not pipeline.is_closed():
            raise Exception("Error, can not set a pipeline has not been closed.")
        self.w_cmdList.set_pipeline(pipeline=pipeline.w_pipeline)

    def bind(self, ds: DescriptorSet):
        self.w_cmdList.bind(w_ds = ds.w_ds)

    def update_sets(self, *sets):
        for s in sets:
            self.w_cmdList.update_bindings_level(s)

    def update_constants(self, **fields):
        self.w_cmdList.update_constants(**fields)

    def dispatch_groups(self, groups_x: int, groups_y: int = 1, groups_z: int = 1):
        self.w_cmdList.dispatch_groups(groups_x, groups_y, groups_z)

    def dispatch_threads(self, dim_x: int, dim_y: int, dim_z: int, group_size_x: int, group_size_y: int, group_size_z: int):
        self.dispatch_groups((dim_x + group_size_x - 1)//group_size_x, (dim_y + group_size_y - 1)//group_size_y, (dim_z + group_size_z - 1)//group_size_z)

    def dispatch_threads_1D(self, dim_x: int, group_size_x: int = 1024):
        import math
        self.dispatch_groups(math.ceil(dim_x / group_size_x))

    def dispatch_threads_2D(self, dim_x: int, dim_y: int, group_size_x: int = 32, group_size_y: int = 32):
        import math
        self.dispatch_groups(math.ceil(dim_x / group_size_x), math.ceil(dim_y / group_size_y))


class GraphicsManager(ComputeManager):
    def __init__(self, device, w_cmdList: _internal.CommandBufferWrapper):
        super().__init__(device, w_cmdList)
        self.current_framebuffer = None

    def set_framebuffer(self, framebuffer: FrameBuffer):
        self.w_cmdList.set_framebuffer(framebuffer)

    @classmethod
    def get_queue_required(cls) -> int:
        return QueueType.GRAPHICS

    def blit_image(self, src_image: Image, dst_image: Image, filter: Filter = Filter.POINT):
        self.w_cmdList.blit_image(src_image.w_resource, dst_image.w_resource, filter)

    def dispatch_primitives(self, vertices: int, instances: int = 1, vertex_start: int = 0, instance_start: int = 0):
        self.w_cmdList.dispatch_primitives(
            vertices, instances, vertex_start, instance_start
        )

    def dispatch_indexed_primitives(self, indices: int, instances: int = 1, first_index: int = 0, vertex_offset: int = 0, first_instance: int = 0):
        self.w_cmdList.dispatch_indexed_primitives(
            indices, instances, first_index, vertex_offset, first_instance
        )
    def bind_vertex_buffer(self, binding: int, vertex_buffer: Buffer):
        self.w_cmdList.bind_vertex_buffer(binding, vertex_buffer.w_resource)

    def bind_index_buffer(self, index_buffer: Buffer):
        self.w_cmdList.bind_index_buffer(index_buffer.w_resource)


class RaytracingManager(GraphicsManager):
    def __init__(self, device, w_cmdList: _internal.CommandBufferWrapper):
        super().__init__(device, w_cmdList)

    @classmethod
    def get_queue_required(cls) -> int:
        return QueueType.RAYTRACING

    def build_ads(self, ads: ADS, scratch_buffer: Buffer):
        self.w_cmdList.build_ads(
            ads.w_resource,
            ads.ads_info,
            ads.ranges,
            scratch_buffer.w_resource)

    def update_ads(self, ads: ADS, scratch_buffer: Buffer):
        self.w_cmdList.update_ads(
            ads.w_resource,
            ads.ads_info,
            ads.ranges,
            scratch_buffer.w_resource)

    def dispatch_rays(self, program: RTProgram, dim_x: int, dim_y: int, dim_z: int = 1):
        self.w_cmdList.dispatch_rays(
            program.active_raygen_slice(), program.miss_slice, program.hitgroup_slice, program.callable_slice,
            dim_x, dim_y, dim_z
        )


class Caps:
    ray_tracing = False
    zero_copy_buffer_map = False
    zero_copy_torch_map = False
    cooperative_matrices = False


# class GPUWrapping:
#     def __init__(self, device: 'DeviceManager'):
#         self.device = device
#         self.wrapped_objs = { }  # maps obj with [gpu_ptr, mode, backend_resource, owner]
#
#     def resolve_ptr(self, obj, mode: str) -> int:
#         if obj is None:
#             return 0
#         if obj in self.wrapped_objs:
#             ptr, current_mode, buffer, data = self.wrapped_objs[obj]
#             if current_mode == mode:
#                 return ptr
#             if mode == 'in':
#                 buffer.load(data)
#             self.wrapped_objs[obj][1] = 'inout'
#             return ptr
#         if isinstance(obj, Buffer):  # buffer
#             return obj.device_ptr()
#         if isinstance(obj, ViewTensor):  # tensor with memory in vulkan memory
#             if isinstance(obj.memory_owner, VulkanMemory):
#                 return self.device.torch_ptr_to_device_ptr(obj)
#                 # memory = obj.memory_owner
#                 # return memory.cuda_to_device_ptr(obj.data_ptr())
#         if isinstance(obj, _torch.Tensor):
#             if obj.device == __TORCH_DEVICE__ and support().zero_copy_torch_map and obj.is_contiguous():  # No need to vulkanize
#                 return obj.data_ptr()
#             else:
#                 print(f'Backing {obj.is_contiguous()}')
#                 backend_tensor = tensor_like(obj)
#                 memory = backend_tensor.memory_owner
#                 if mode == 'in' or mode == 'inout':
#                     if mode == 'inout':
#                         assert obj.is_contiguous(), 'Can not copy out non-contiguous tensors'
#                     backend_tensor.copy_(obj.contiguous())
#                 # self.wrapped_objs[obj] = [memory.cuda_to_device_ptr(obj.data_ptr()), mode, backend_tensor, obj]
#                 self.wrapped_objs[obj] = [backend_tensor.data_ptr(), mode, backend_tensor, obj]
#                 return self.wrapped_objs[obj][0]
#         raise Exception(f'Invalid object type {type(obj)} for wrap')
#
#     def release(self):
#         if self.wrapped_objs is not None:
#             for ptr, mode, backup, obj in self.wrapped_objs.values():
#                 if mode == 'out' or mode == 'inout':
#                     obj.copy_(backup)
#         self.wrapped_objs = None
#
#     # def _wrap_objs(self):
#     #     l = list(self._wrapping_objs.values())
#     #     l.sort(key=lambda t: t[1])  #sort by ptr
#     #     i = 0
#     #     while i < len(l):
#     #         start_ptr = l[i][1]
#     #         end_ptr = start_ptr + l[i][2]
#     #         j = i + 1
#     #         while j < len(l) and l[j][1] <= end_ptr:
#     #             end_ptr = max(end_ptr, l[j][1]+l[j][2])
#     #             j+=1
#     #         full_buffer = self.device.create_buffer(end_ptr - start_ptr, usage=BufferUsage.STAGING, memory=MemoryLocation.GPU)
#     #         for x in range(i, j):
#     #             current_slice = full_buffer.slice(l[x][1] - start_ptr, l[x][2])
#     #             self.wrapped_objs[l[x][3]] = (current_slice.device_ptr(), l[x][0], current_slice, l[x][3])
#     #             if l[x][0] == 'in' or l[x][0] == 'inout':
#     #                 current_slice.load(l[x][3])  # update gpu resource with wrapped data
#     #         i = j
#     #     self.state = 1
#     #     self._wrapping_objs = None
#
#     def __del__(self):
#         self.release()
#
#     # def get_gpu_ptr(self, obj):
#     #     assert self.state != 2, 'Can not retrieve pointers from objects already unwrapped'
#     #     if self.state == 0:
#     #         self._wrap_objs()
#     #     if obj is None:
#     #         return 0
#     #     assert obj in self.wrapped_objs, 'Some of the objects were not previously wrapped.'
#     #     return self.wrapped_objs[obj][0]
#     #     # return _torch.tensor(ptrs.astype(_np.int64))  # using np to cast uint64 to int64, only int64 supported in _torch.


# class WrappedGPUPtr:
#     def __init__(self, device: 'DeviceManager', objs: Tuple[Union[_torch.Tensor, Buffer],...], mode: Literal['in', 'inout', 'out']='in'):
#         self.device = device
#         self.objs = objs
#         self.mode = mode
#         self.ptrs = _np.zeros((len(objs), 1), dtype=_np.uint64)
#         self.owners = [None]*len(objs)
#
#     def _wrap(self, obj) -> Tuple[int, Any]:
#         if obj is None:
#             return 0, None
#         if isinstance(obj, Buffer):
#             return obj.device_ptr(), obj
#         if isinstance(obj, ViewTensor):
#             if isinstance(obj.memory_owner, VulkanMemory):
#                 memory = obj.memory_owner
#                 return memory.cuda_to_device_ptr(obj.data_ptr()), obj
#         if isinstance(obj, _torch.Tensor):
#             if obj.is_cuda and os.name == 'nt':  # No need to vulkanize
#                 return obj.data_ptr(), obj
#             back_buffer = self.device.create_buffer_like(obj, memory=MemoryLocation.GPU)
#             if self.mode == 'inout' or self.mode == 'in':
#                 self.device.copy(back_buffer, obj)
#             return back_buffer.device_ptr(), back_buffer
#         raise Exception(f'Invalid object type {type(obj)} for wrap')
#
#     def _unwrap(self, obj, owner):
#         if obj is owner:
#             return
#         if self.mode == 'inout' or self.mode == 'out':
#             owner.save(obj)
#
#     def __enter__(self):
#         for i, obj in enumerate(self.objs):
#             self.ptrs[i], self.owners[i] = self._wrap(obj)
#         return _torch.from_numpy(self.ptrs.astype(_np.int64))
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         for i, obj in enumerate(self.objs):
#             self._unwrap(obj, self.owners[i])
#             self.ptrs[i] = 0
#             self.owners[i] = None

class GPUPtr:

    def __init__(self, device_ptr: int, obj: _typing.Any, is_direct: bool = True):
        self.device_ptr = device_ptr
        self.obj = obj
        self.is_direct = is_direct

    def flush(self):
        raise NotImplementedError()

    def invalidate(self):
        raise NotImplementedError()

    def mark_as_dirty(self):
        raise NotImplementedError()

    def update_mode(self, additional_mode: _typing.Literal['in', 'out', 'inout']):
        raise NotImplementedError()


class DirectGPUPtr(GPUPtr):
    def __init__(self, device_ptr: int, obj: _typing.Any):
        super(DirectGPUPtr, self).__init__(device_ptr, obj)

    __NULL__ = None
    @staticmethod
    def null():
        if DirectGPUPtr.__NULL__ is None:
            DirectGPUPtr.__NULL__ = DirectGPUPtr(0, None)
        return DirectGPUPtr.__NULL__

    def flush(self):
        pass

    def invalidate(self):
        pass

    def mark_as_dirty(self):
        pass

    def update_mode(self, additional_mode: _typing.Literal['in', 'out', 'inout']):
        pass


class WrappedTensorPtr(GPUPtr):
    def __init__(self, obj: _typing.Any, t: _torch.Tensor, v: ViewTensor, mode: _typing.Literal['in', 'out', 'inout']):
        super(WrappedTensorPtr, self).__init__(v.memory_owner.device_ptr, obj, is_direct=False)
        self.cpu_tensor = t
        self.gpu_backend = v
        self.mode = mode
        if self.mode == 'in' or self.mode == 'inout':
            self.gpu_version = -1
            self.flush()
        else:
            self.gpu_version = t._version
        # self.mark_as_dirty()  # automatic assumed dirty after map since common use is to call invalidate after submit.

    def update_mode(self, additional_mode: _typing.Literal['in', 'out', 'inout']):
        if self.mode == additional_mode or self.mode == 'inout':
            return
        if additional_mode == 'in':
            self.flush()
        self.mode = 'inout'

    # def __del__(self):
    #     if self.mode == 'out' or self.mode == 'inout':
    #         self.mark_as_dirty()
    #         self.invalidate()

    def mark_as_dirty(self):
        if self.mode == 'out' or self.mode == 'inout':
            self.gpu_version = self.cpu_tensor._version + 1

    def flush(self):
        if self.mode == 'out':
            return
        if self.gpu_version < self.cpu_tensor._version:
            self.gpu_backend.copy_(self.cpu_tensor)
            self.gpu_version = self.cpu_tensor._version
            # print(f"[UPDATED] shape {self.cpu_tensor.shape} to {self.gpu_version}")

    def invalidate(self):
        if self.mode == 'in':
            return
        if self.gpu_version > self.cpu_tensor._version:
            self.cpu_tensor.copy_(self.gpu_backend)
            self.gpu_version = self.cpu_tensor._version


class _GPUWrappingManager:
    def __init__(self, device: 'DeviceManager'):
        import weakref
        self.device = weakref.ref(device)
        self.hashed_wraps : _typing.List[_typing.Optional[weakref.WeakSet]] = [None] * 10013

    def wrap_gpu(self, t: _typing.Any, mode: _typing.Literal['in', 'out', 'inout']) -> GPUPtr:
        import os
        import weakref
        import torch.cuda
        if t is None:
            return DirectGPUPtr.null()
        obj = t
        if hasattr(t, '__bindable__'):
            t = t.__bindable__
        if isinstance(t, Buffer):
            return DirectGPUPtr(t.device_ptr, obj)
        if isinstance(t, ViewTensor) and isinstance(t.memory_owner, VulkanMemory):
            return DirectGPUPtr(t.memory_owner.cuda_to_device_ptr(t.data_ptr()), obj)
        assert isinstance(t, _torch.Tensor), f'Type {type(obj)} can not be wrapped on the GPU'
        if os.name == 'nt' and _torch.cuda.is_available() and t.is_contiguous() and t.is_cuda:
           return DirectGPUPtr(t.data_ptr(), t)
        code = t.data_ptr() ^ t.numel() #id(t)
        entry = code % len(self.hashed_wraps)
        if self.hashed_wraps[entry] is not None:
            for w in self.hashed_wraps[entry]:
                if w.obj.data_ptr() == t.data_ptr():
                    if __TRACE_WRAP__:
                        print(f"[WARNING] Collision {t.shape} with mode {mode} from mode {w.mode}")
                    w.update_mode(mode)
                    # w.flush()
                    return w
        if self.hashed_wraps[entry] is None:
            self.hashed_wraps[entry] = weakref.WeakSet()
        import gc
        gc.collect()
        v = self.device().create_tensor(*t.shape, dtype=t.dtype)
        w = WrappedTensorPtr(obj, t, v, mode)
        if __TRACE_WRAP__:
            print(f"[WARNING] wrapped tensor of {t.shape} elements")
        self.hashed_wraps[entry].add(w)
        return w


# class GPUWrap:
#     @staticmethod
#     def _wrap_obj(obj: Any) -> (int, Union[Buffer, ViewTensor]):
#         if isinstance(obj, Buffer):
#             return obj.device_ptr, None
#         if isinstance(obj, ViewTensor):
#             return obj.memory_owner.device_ptr, None
#         if isinstance(obj, _torch.Tensor):
#             b = buffer_like(obj, MemoryLocation.GPU)
#             return b.device_ptr, b
#         raise Exception(f'Not supported wrapping {type(obj)}')
#
#     def __init__(self, obj: Any):
#         self.obj = obj
#         self.ptr, self._backing = GPUWrap._wrap_obj(obj)
#
#     def push(self):
#         if self._backing is None:
#             return
#         self._backing.load(self.obj)
#
#     def pull(self):
#         if self._backing is None:
#             return
#         self._backing.save(self.obj)


class Window:

    def __init__(self, device: 'DeviceManager', w_window: _internal.WindowWrapper, format: Format):
        self.w_window = w_window
        self.device = device
        self._stats_fps = 0
        self._stats_spf = 0
        self._last_time_enter  = 0
        self._rt_present = Image(self.device, w_window.render_target, Layout.from_format(Format.PRESENTER))
        self._image = device.create_image(image_type=ImageType.TEXTURE_2D, is_cube=False, image_format=format,
                            width=w_window.width, height=w_window.height, depth=1, mips=1, layers=1,
                            usage=ImageUsage.TRANSFER, memory=MemoryLocation.GPU)
        self._buffer = device.create_buffer(w_window.width * w_window.height * Layout.from_format(format).aligned_size,
                                            BufferUsage.STORAGE, MemoryLocation.GPU)
        self._staging = self._buffer if self._buffer.w_resource.support_direct_tensor_map else \
                            device.create_buffer(w_window.width * w_window.height * Layout.from_format(format).aligned_size,
                                            BufferUsage.STAGING, MemoryLocation.CPU)
        self._tensor = self._staging.w_resource.as_tensor(_torch.float32).view(w_window.height, w_window.width, -1)
        # create managers for Tensor->Present, Buffer->Present, Image->Present

        # image -> present
        man = device.get_graphics()
        man.blit_image(self._image, self._rt_present)
        man.freeze()
        self._present_image_man = man

        man = device.get_graphics()
        man.copy(self._buffer, self._image)
        man.blit_image(self._image, self._rt_present)
        man.freeze()
        self._present_buffer_man = man

        man = device.get_graphics()
        man.copy(self._staging, self._image)
        man.blit_image(self._image, self._rt_present)
        man.freeze()
        self._present_tensor_man = man

        self.backbuffer_images = { }  # map from name to (vulkan backbuffer image, same but as opengl image) used to draw tensors and textures in imgui through an opengl texture

    class ClientContext:
        def __init__(self, window: 'Window', map: _typing.Union[Image, Buffer, _torch.Tensor], man: _typing.Optional[GraphicsManager]):
            self.window = window
            self.w_window = window.w_window
            self.map = map
            self.man = man
            self._timer = 0

        def __enter__(self):
            import time
            self.w_window._begin_frame()
            self._timer = time.perf_counter()
            return self.map

        def __exit__(self, exc_type, exc_val, exc_tb):
            import time
            d = time.perf_counter() - self._timer
            if self.man is not None:
                submit(self.man)
            self.w_window._end_frame()
            fps = 1000 if d <= 0.001 else 1 / d
            if self.window._stats_fps > 0:
                self.window._stats_fps = self.window._stats_fps * 0.9 + fps * 0.1
                self.window._stats_spf = self.window._stats_spf * 0.9 + d * 0.1
            else:
                self.window._stats_fps = fps
                self.window._stats_spf = d


    def render_target(self):
        return Window.ClientContext(window=self, map=self._rt_present, man=None)

    def image(self):
        return Window.ClientContext(window=self, map=self._image, man=self._present_image_man)

    def buffer(self):
        return Window.ClientContext(window=self, map=self._buffer, man=self._present_buffer_man)

    def tensor(self):
        return Window.ClientContext(window=self, map=self._tensor, man=self._present_tensor_man)

    def show_tensor(self, name: str, t: _torch.Tensor, width: int, height: int, cmap: str = 'viridis'):
        if not name in self.backbuffer_images:
            backbuffer = self.device.create_image(image_type=ImageType.TEXTURE_2D, is_cube=False, image_format=Format.VEC4, width=width, height=height,
                                     depth=1, mips=1, layers=1, usage=ImageUsage.STORAGE)
            opengl_backbuffer = self.device.w_device.create_opengl_texture_from_vk(backbuffer.w_resource.resource_data)
            tensor_data_uniform = object_buffer(Layout.from_structure(mode='std430',
                                                                      tensor_data=_torch.int64,
                                                                      width=int,
                                                                      height=int,
                                                                      components=int,
                                                                      type=int
                                                                      ))
            color_map_data = object_buffer(Layout.from_structure(mode='std430',
                                                                 map_colors=[17, vec4]
                                                                 ))

            pipeline = pipeline_compute()
            pipeline.load_shader_from_source("""
#version 460
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference2: require
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_EXT_control_flow_attributes : require

layout (local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(set=0, binding=0, rgba8) uniform image2D _out;
layout(set=0, binding=1, std430) uniform TensorData{
    uint64_t tensor_data;
    int width;
    int height;
    int components; // 1 - 4  (if 1 then map is applied)
    int type; // 0 - int, 1 - float  (if int it is assumed to be 0..255)
};

layout(set=0, binding=2, std430) uniform ColorMapData{
    vec4 map_colors[17];
};

layout(buffer_reference, scalar, buffer_reference_align=4) buffer int_ptr { int data[]; };
layout(buffer_reference, scalar, buffer_reference_align=4) buffer float_ptr { float data[]; };

vec4 compute_color(vec2 c)
{
    int px = clamp(int(c.x * width), 0, width - 1);
    int py = clamp(int((1 - c.y) * height), 0, height - 1);
    int pixel_offset = py * width + px;
    int byte_offset = pixel_offset * components * 4;
    vec4 color = vec4(1.0);
    if (type == 0) // int
    {
        int_ptr pixel_buffer = int_ptr(tensor_data + byte_offset);
        [[unroll]]
        for (int i=0; i<components; i++)
            color[i] = pixel_buffer.data[i] / 255.0;
    }
    else {
        float_ptr pixel_buffer = float_ptr(tensor_data + byte_offset);
        [[unroll]]
        for (int i=0; i<components; i++)
            color[i] = pixel_buffer.data[i];
    }
    
    if (components == 1) // use color maps
    {
        float alpha = color.x * 16;
        int cm_index = clamp(int(alpha), 0, 15);
        alpha = fract(alpha);
        color = mix (map_colors[cm_index], map_colors[cm_index + 1], alpha);
    }
    
    return color;
}

void main() {
    ivec2 index = ivec2(gl_GlobalInvocationID.xy);
    ivec2 dim = imageSize(_out);
    if (any(greaterThanEqual(index, dim)))
    return;

    vec2 C = (index + vec2(0.5))/dim;

    imageStore(_out, index, compute_color(C));
}
            """)
            pipeline.bind_storage_image(0, backbuffer)
            pipeline.bind_uniform(1, tensor_data_uniform)
            pipeline.bind_uniform(2, color_map_data)
            pipeline.close()
            man = self.device.get_compute()
            man.set_pipeline(pipeline)
            man.dispatch_threads_2D(width, height)
            man.freeze()
            last_used_cmap = ''

            self.backbuffer_images[name] = dict(
                man=man,
                tensor_data_uniform=tensor_data_uniform,
                color_map_data=color_map_data,
                opengl_backbuffer=opengl_backbuffer,
                last_used_cmap=last_used_cmap
            )
        else:
            cached_data = self.backbuffer_images[name]
            man = cached_data['man']
            tensor_data_uniform = cached_data['tensor_data_uniform']
            color_map_data = cached_data['color_map_data']
            opengl_backbuffer = cached_data['opengl_backbuffer']
            last_used_cmap = cached_data['last_used_cmap']
        # update uniforms
        with tensor_data_uniform as b:
            b.tensor_data = wrap_gpu(t)
            b.width = t.shape[1]
            b.height = t.shape[0]
            b.components = t.shape[2]
            b.type = 0 if t.dtype == _torch.int32 else 1

        if t.shape[2] == 1 and last_used_cmap != cmap:  # update colormap
            import matplotlib
            cmap = matplotlib.cm.get_cmap(cmap)
            with color_map_data as c:
                for i in range(17):
                    alpha = i / 16.0
                    color_tuple = cmap(alpha)
                    c.map_colors[i] = vec4(*color_tuple)
            self.backbuffer_images[name]['last_used_cmap'] = cmap

        submit(man)  # draw tensor into backbuffer

        if __GUI_AVAILABLE__:
            imgui.image(opengl_backbuffer, width, height)







    # def __enter__(self):
    #     self._last_time_enter = time.perf_counter()
    #     w_resource = self.w_window._begin_frame()
    #     return Image(self.device, w_resource, self.texel_layout)
    #
    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     if rt is not rt_present:
    #         ResourceData.blit(_self, rt.resource_data, rt.current_slice, rt_present.resource_data,
    #                           rt_present.current_slice)
    #     self.w_window._end_frame()
    #     d = time.perf_counter() - self._last_time_enter
    #     fps = 1000 if d <= 0.001 else 1/d
    #     if self._stats_fps > 0:
    #         self._stats_fps = self._stats_fps * 0.9 + fps * 0.1
    #     else:
    #         self._stats_fps = fps

    @property
    def fps(self):
        return self._stats_fps

    @property
    def spf(self):
        return self._stats_spf

    @property
    def is_closed(self):
        return self.w_window.is_closed()

    def label_text(self, label: str, text: str):
        assert __GUI_AVAILABLE__, 'No GUI available. consider install imgui'
        return imgui.label_text(label, text)

    def __getattr__(self, item):
        assert __GUI_AVAILABLE__, 'No GUI available. consider install imgui'
        return imgui.__dict__[item]


def _thread_safe_call(f):
    def wrapper(self, *args, **kwargs):
        def eval():
            f(self, *args, **kwargs)
        self.safe_dispatch_function(eval)
    return wrapper


class DeviceManager:
    import threading

    def __init__(self):
        self.w_state = None
        self.width = 0
        self.height = 0
        self.__copying_on_the_gpu = None
        self.__queue = None
        self.__loop_process = None
        self.__allow_cross_thread_without_looping = False
        self.__caps = Caps()
        self.__wrapping = _GPUWrappingManager(self)


    def wrap_gpu(self, data: _typing.Any, mode: _typing.Literal['in', 'out', 'inout']) -> GPUPtr:
        return self.__wrapping.wrap_gpu(data, mode)

    def allow_cross_threading(self):
        self.__allow_cross_thread_without_looping = True
        print(
            "[WARNING] Allow access from other threads is dangerous. Think in wrapping the concurrent process in a loop")

    __event__ = threading.Event()
    __locker__ = threading.Lock()

    def safe_dispatch_function(self, function):
        import threading
        if threading.current_thread() == threading.main_thread() or self.__allow_cross_thread_without_looping:
            function()
            return
        if self.__queue is None:
            raise Exception("Can not dispatch cross-thread function without looping")
        with self.__locker__:
            self.__queue.append(function)
        DeviceManager.__event__.wait()
        # print(len(self.__queue))

    def execute_loop(self, main_process: _typing.Callable[[_typing.Optional[_typing.Any]], bool], *process: _typing.Callable[[_typing.Optional[_typing.Any]], None], context: _typing.Optional[_typing.Any] = None):
        import threading
        assert threading.current_thread() == threading.main_thread(), "execution of main process can only be triggered from main thread!"
        if self.__loop_process is not None:
            for p in self.__loop_process:
                p.join()  # wait for an unfinished loop first
        self.__queue = []  # start dispatch queue
        context_args = () if context is None else (context,)
        self.__loop_process = [threading.Thread(target=p, args=context_args) for p in process]
        for p in self.__loop_process:
            p.start()
        main_alive = True
        while main_alive or any(p.is_alive() for p in self.__loop_process):
            with self.__locker__:
                while len(self.__queue) > 0:  # consume all pending dispatched include
                    f = self.__queue.pop(0)
                    f()
            DeviceManager.__event__.set()
            DeviceManager.__event__.clear()
            main_alive = main_process(*context_args)
        self.__queue = None

    def release(self):
        self.__copying_on_the_gpu = None
        if self.w_device is not None:
            self.w_device.release()
        self.w_device = None

    def __del__(self):
        self.release()

    def support(self) -> Caps:
        return self.__caps

    def __bind__(self, w_device: _internal.DeviceWrapper):
        self.w_device = w_device
        self.__caps.cooperative_matrices = w_device.support_cooperative_matrices
        self.__caps.ray_tracing = w_device.support_raytracing
        self.__caps.zero_copy_buffer_map = w_device.support_buffer_map
        self.__caps.zero_copy_torch_map = w_device.support_torch_map

    def load_technique(self, technique):
        technique.__bind__(self.w_device)
        technique.__setup__()
        return technique

    def dispatch_technique(self, technique):
        assert technique.w_device, "Technique is not bound to a device, you must load the technique in some point before dispatching"
        technique.__dispatch__()

    def set_debug_name(self, name: str):
        self.w_device.set_debug_name(name)

    def torch_ptr_to_device_ptr(self, t: _torch.Tensor) -> int:
        return self.w_device.torch_ptr_to_device_ptr(t)

    def create_tensor(self, *shape: int, dtype: _torch.dtype, memory: MemoryLocation = MemoryLocation.GPU):
        # return _torch.zeros(*shape, dtype=dtype, device='cpu' if memory == MemoryLocation.CPU else 'cuda')
        return self.w_device.create_tensor(*shape, dtype=dtype, memory=memory)

    def create_tensor_like(self, t: _torch.Tensor) -> _torch.Tensor:
        shape = t.shape
        memory = MemoryLocation.CPU if t.device.type == 'cpu' else MemoryLocation.GPU
        return self.create_tensor(*shape, dtype=t.dtype, memory=memory)

    def create_buffer_like(self, t: _typing.Union[_torch.Tensor, Resource], memory: MemoryLocation):
        import math
        if isinstance(t, _torch.Tensor):
            return self.create_buffer(math.prod(t.shape) * t.element_size(), usage=BufferUsage.STAGING, memory=memory)
        return self.create_buffer(t.w_resource.size, usage=BufferUsage.STAGING, memory=memory)

    def create_buffer(self, size: int, usage: BufferUsage = BufferUsage.STORAGE, memory: MemoryLocation = MemoryLocation.GPU) -> Buffer:
        return Buffer(self, self.w_device.create_buffer(size, usage, memory))

    def create_structured_buffer(self, count: int, element_description: _typing.Union[type, _torch.dtype, Layout, _typing.List, _typing.Dict],
                      usage: BufferUsage = BufferUsage.STORAGE, memory: MemoryLocation = MemoryLocation.GPU) -> StructuredBuffer:
        if not isinstance(element_description, Layout):
            element_description = Layout.from_description(LayoutAlignment.SCALAR, element_description)
        size = element_description.aligned_size * count
        return StructuredBuffer(self, self.w_device.create_buffer(size, usage, memory), element_description)

    def create_object_buffer(self, element_description: _typing.Union[type, _torch.dtype, Layout, _typing.List, _typing.Dict],
                             usage: BufferUsage = BufferUsage.STORAGE, memory: MemoryLocation = MemoryLocation.GPU):
        if not isinstance(element_description, Layout):
            element_description = Layout.from_description(element_description)
        size = element_description.aligned_size
        return ObjectBuffer(self, self.w_device.create_buffer(size, usage, memory), element_description)

    def create_image(self, image_type: ImageType, is_cube: bool, image_format: Format,
                     width: int, height: int, depth: int,
                     mips: int, layers: int,
                     usage: ImageUsage = ImageUsage.SAMPLED, memory: MemoryLocation = MemoryLocation.GPU):
        texel_layout = Layout.from_format(image_format)
        return Image(self, self.w_device.create_image(
            image_type, image_format, is_cube, (width, height, depth), mips, layers, usage, memory
        ), texel_layout)

    def create_triangle_collection(self) -> TriangleCollection:
        return TriangleCollection(device=self.w_device)

    def create_aabb_collection(self) -> AABBCollection:
        return AABBCollection(device=self.w_device)

    def create_geometry_ads(self, collection: GeometryCollection) -> ADS:
        ads, info, ranges, handle, scratch_size = self.w_device.create_ads(
            geometry_type=collection.get_collection_type(),
            descriptions=[
                (v.w_resource, v.layout.aligned_size, None if i is None else i.w_resource, None if t is None else t.w_resource)
                for v, i, t in collection.descriptions
            ] if collection.get_collection_type() == ADSNodeType.TRIANGLES else [
                aabb.w_resource for aabb in collection.descriptions  # AABBs are described using the buffer only
            ]
        )
        return ADS(self, ads, handle, scratch_size, info, ranges)

    def create_scene_ads(self, instance_buffer: Buffer):
        ads, info, ranges, handle, scratch_size = self.w_device.create_ads(
            geometry_type=ADSNodeType.INSTANCE,
            descriptions=[
                instance_buffer.w_resource
            ]
        )
        return ADS(self, ads, handle, scratch_size, info, ranges, instance_buffer)

    def create_instance_buffer(self, instances: int, memory: MemoryLocation = MemoryLocation.GPU) -> StructuredBuffer:
        instance_layout = Layout.from_instance()
        b = self.create_structured_buffer(instances, element_description=instance_layout, usage=BufferUsage.RAYTRACING_RESOURCE, memory=memory).clear()
        # Load default values for b
        with b.map('in', clear=True) as t:
            t.transform[0][0] = 1.0
            t.transform[1][1] = 1.0
            t.transform[2][2] = 1.0
            t.mask8_idx24 = asint32(0xFF000000)
            # t.flags = 0
        return b

    def create_sampler(self,
                       mag_filter: Filter = Filter.POINT,
                       min_filter: Filter = Filter.POINT,
                       mipmap_mode: MipMapMode = MipMapMode.POINT,
                       address_U: AddressMode = AddressMode.REPEAT,
                       address_V: AddressMode = AddressMode.REPEAT,
                       address_W: AddressMode = AddressMode.REPEAT,
                       mip_LOD_bias: float = 0.0,
                       enable_anisotropy: bool = False,
                       max_anisotropy: float = 0.0,
                       enable_compare: bool = False,
                       compare_op: CompareOp = CompareOp.NEVER,
                       min_LOD: float = 0.0,
                       max_LOD: float = 0.0,
                       border_color: BorderColor = BorderColor.TRANSPARENT_BLACK_FLOAT,
                       use_unnormalized_coordinates: bool = False
                       ) -> Sampler:
        return self.w_device.create_sampler(mag_filter, min_filter, mipmap_mode,
                                            address_U, address_V, address_W,
                                            mip_LOD_bias, 1 if enable_anisotropy else 0,
                                            max_anisotropy, 1 if enable_compare else 0,
                                            compare_op, min_LOD, max_LOD,
                                            border_color,
                                            1 if use_unnormalized_coordinates else 0)

    def create_compute_pipeline(self):
        return Pipeline(self.w_device.create_pipeline(
            pipeline_type=PipelineType.COMPUTE))

    def create_graphics_pipeline(self):
        return GraphicsPipeline(self.w_device.create_pipeline(
            pipeline_type=PipelineType.GRAPHICS))

    def create_raytracing_pipeline(self):
        return RaytracingPipeline(self.w_device.create_pipeline(
            pipeline_type=PipelineType.RAYTRACING))

    def create_window(self, width: int, height: int, format: Format) -> Window:
        w_window = self.w_device.create_window(width, height)
        return Window(self, w_window, format)

    def _get_queue_manager(self, queue: QueueType):
        c = self.w_device.create_cmdList(queue)
        c.begin()
        return c

    def get_graphics(self) -> GraphicsManager:
        return GraphicsManager(self, self._get_queue_manager(QueueType.GRAPHICS))

    def get_compute(self) -> ComputeManager:
        return ComputeManager(self, self._get_queue_manager(QueueType.COMPUTE))

    def get_raytracing(self) -> RaytracingManager:
        return RaytracingManager(self, self._get_queue_manager(QueueType.RAYTRACING))

    def get_copy(self) -> CopyManager:
        return CopyManager(self, self._get_queue_manager(QueueType.COPY))

    def submit(self, man: CommandManager, wait: bool = True):
        assert man.is_frozen(), "Only frozen managers can be submitted"
        if wait:
            man.w_cmdList.flush_and_wait()
        else:
            man.w_cmdList.flush()

    def flush(self):
        self.w_device.flush_pending_and_wait()

    def copy(self, dst: _typing.Union[_torch.Tensor, Resource], src: _typing.Union[_torch.Tensor, Resource]):
        if isinstance(dst, Resource):
            return dst.load(src)
        if isinstance(src, Resource):
            return src.save(dst)
        dst = ViewTensor.reinterpret(dst, _torch.uint8)
        src = ViewTensor.reinterpret(src, _torch.uint8)
        if dst.shape == src.shape:
            dst.copy_(src)
        if dst.is_contiguous():
            dst.view(*src.shape).copy_(src)
        assert src.is_contiguous(), 'If shapes are not equal, then one of the two tensor has to be contiguous'
        dst.copy_(src.view(*dst.shape))

    # def map_gpu(self, *objs: Union[_torch.Tensor, Buffer], mode: Literal['in', 'inout', 'out']='in'):
    #     return DeviceManager.WrappedGPUPtr(self, objs, mode)

    def __enter__(self):
        self.__previous_device = __ACTIVE_DEVICE__
        return device_manager(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        device_manager(self.__previous_device)


class NoneDeviceManager(DeviceManager):
    def _raise_access_error(self) -> _typing.Any:
        raise Exception("No active device, use create_device")

    def submit(self, man: CommandManager, wait: bool = True):
        return self._raise_access_error()

    def torch_ptr_to_device_ptr(self, t: _torch.Tensor) -> int:
        return self._raise_access_error()

    def __bind__(self, w_device: _internal.DeviceWrapper):
        return self._raise_access_error()

    def __del__(self):
        pass

    def get_compute(self) -> ComputeManager:
        return self._raise_access_error()

    def get_copy(self) -> CopyManager:
        return self._raise_access_error()

    def get_graphics(self) -> GraphicsManager:
        return self._raise_access_error()

    def get_raytracing(self) -> RaytracingManager:
        return self._raise_access_error()

    def create_triangle_collection(self) -> TriangleCollection:
        return self._raise_access_error()

    def create_compute_pipeline(self):
        return self._raise_access_error()

    def create_graphics_pipeline(self):
        return self._raise_access_error()

    def create_scene_ads(self, instance_buffer: Buffer):
        return self._raise_access_error()

    def create_raytracing_pipeline(self):
        return self._raise_access_error()

    def create_aabb_collection(self) -> AABBCollection:
        return self._raise_access_error()

    def create_geometry_ads(self, collection: GeometryCollection) -> ADS:
        return self._raise_access_error()

    def create_tensor_like(self, t: _torch.Tensor) -> _torch.Tensor:
        return self._raise_access_error()

    def create_window(self, width: int, height: int, format: Format) -> Window:
        return self._raise_access_error()

    def create_buffer_like(self, t: _typing.Union[_torch.Tensor, Resource], memory: MemoryLocation):
        return self._raise_access_error()

    def create_instance_buffer(self, instances: int, memory: MemoryLocation = MemoryLocation.GPU) -> StructuredBuffer:
        return self._raise_access_error()

    def create_tensor(self, *shape: int, dtype: _torch.dtype, memory: MemoryLocation = MemoryLocation.GPU):
        return self._raise_access_error()

    def create_buffer(self, size: int, usage: BufferUsage = BufferUsage.STORAGE, memory: MemoryLocation = MemoryLocation.GPU) -> Buffer:
        return self._raise_access_error()

    def create_object_buffer(self, element_description: _typing.Union[type, _torch.dtype, Layout, _typing.List, _typing.Dict],
                             usage: BufferUsage = BufferUsage.STORAGE, memory: MemoryLocation = MemoryLocation.GPU):
        return self._raise_access_error()

    def create_structured_buffer(self, count: int, element_description: _typing.Union[type, _torch.dtype, Layout, _typing.List, _typing.Dict],
                      usage: BufferUsage = BufferUsage.STORAGE, memory: MemoryLocation = MemoryLocation.GPU) -> StructuredBuffer:
        return self._raise_access_error()

    def create_image(self, image_type: ImageType, is_cube: bool, image_format: Format,
                     width: int, height: int, depth: int,
                     mips: int, layers: int,
                     usage: ImageUsage = ImageUsage.SAMPLED, memory: MemoryLocation = MemoryLocation.GPU):
        return self._raise_access_error()

    def create_sampler(self,
                       mag_filter: Filter = Filter.POINT,
                       min_filter: Filter = Filter.POINT,
                       mipmap_mode: MipMapMode = MipMapMode.POINT,
                       address_U: AddressMode = AddressMode.REPEAT,
                       address_V: AddressMode = AddressMode.REPEAT,
                       address_W: AddressMode = AddressMode.REPEAT,
                       mip_LOD_bias: float = 0.0,
                       enable_anisotropy: bool = False,
                       max_anisotropy: float = 0.0,
                       enable_compare: bool = False,
                       compare_op: CompareOp = CompareOp.NEVER,
                       min_LOD: float = 0.0,
                       max_LOD: float = 0.0,
                       border_color: BorderColor = BorderColor.TRANSPARENT_BLACK_FLOAT,
                       use_unnormalized_coordinates: bool = False
                       ) -> Sampler:
        return self._raise_access_error()

    def wrap_gpu(self, data: _typing.Any, mode: _typing.Literal['in', 'out', 'inout']) -> GPUPtr:
        return self._raise_access_error()

    def support(self) -> Caps:
        return self._raise_access_error()

    def set_debug_name(self, name: str):
        return self._raise_access_error()

    def safe_dispatch_function(self, function):
        return self._raise_access_error()

    def release(self):
        return self._raise_access_error()

    def load_technique(self, technique):
        return self._raise_access_error()

    def flush(self):
        return self._raise_access_error()

    def execute_loop(self, main_process: _typing.Callable[[_typing.Optional[_typing.Any]], bool], *process: _typing.Callable[[_typing.Optional[_typing.Any]], None], context: _typing.Optional[_typing.Any] = None):
        return self._raise_access_error()

    def dispatch_technique(self, technique):
        return self._raise_access_error()

    def copy(self, dst: _typing.Union[_torch.Tensor, Resource], src: _typing.Union[_torch.Tensor, Resource]):
        return self._raise_access_error()

    def allow_cross_threading(self):
        return self._raise_access_error()


__ACTIVE_DEVICE__: DeviceManager = NoneDeviceManager()


def device_manager(new_device: _typing.Optional[DeviceManager] = None) -> DeviceManager:
    global __ACTIVE_DEVICE__
    if new_device is not None:
        __ACTIVE_DEVICE__ = new_device
    return __ACTIVE_DEVICE__


def create_device(*, device: int = 0, debug: bool = False, set_active: bool = True) -> DeviceManager:
    """
    Creates a device manager. This is the core of vulkan graphics call.
    This method automatically sets created device as active, further actions will use it.
    To change to other devices use device method. e.g: device_manager(other_device)
    """
    state = _internal.DeviceWrapper(
        device_index=device,
        enable_validation_layers=debug
    )
    dev = DeviceManager()
    dev.__bind__(state)
    return device_manager(dev) if set_active else dev


import atexit
@atexit.register
def quit():
    """
    Releases the active device from future usages.
    Use this function at the end of the program execution to avoid any hanging resources
    and properly shutdown vulkan objects
    """
    global __ACTIVE_DEVICE__
    if __ACTIVE_DEVICE__ is not None:
        __ACTIVE_DEVICE__.release()
    __ACTIVE_DEVICE__ = None
    import gc
    gc.collect()


class Technique(DeviceManager):
    def __setup__(self):
        pass

    def __dispatch__(self):
        pass


def asint32(v: int):
    """
    Converts a python integer to a valid signed int32 value
    """
    if v < 0:
        return v
    return _np.uint32(v).astype(_np.int32)

# def Extends(class_):
#     def wrapper(function):
#         setattr(class_, function.__name__, function)
#
#     return wrapper


# import functools
#
# __check_active_T = TypeVar('CheckActiveType')
#
# def _check_active_device(f: __check_active_T) -> __check_active_T:
#     @functools.wraps(f)
#     def wrap(*args, **kwargs):
#         assert __ACTIVE_DEVICE__ is not None, "Create a device, a presenter or set active a device."
#         return f(*args, **kwargs)
#
#     return wrap


# @_check_active_device
def window(width: int, height: int, format: Format = Format.VEC4) -> Window:
    return __ACTIVE_DEVICE__.create_window(width, height, format)

# @_check_active_device
def support() -> Caps:
    """
    Gets the set of features supported by the rendering system.
    """
    return __ACTIVE_DEVICE__.support()

# @_check_active_device
def tensor_like(t: _torch.Tensor) -> _torch.Tensor:
    """
    Creates a tensor in gpu vulkan memory using another tensor as reference.
    This tensor grants a zero_copy access from vulkan when used.
    """
    return __ACTIVE_DEVICE__.create_tensor_like(t)

# @_check_active_device
def tensor(*shape, dtype: dtype = _torch.float32) -> _torch.Tensor:
    """
    Creates a tensor using a specific shape and type with compatible memory with vulkan, cupy and numpy tensors.
    The underlying buffer is created with possible usage to transfer to/from, storage binding and gpu addressing.
    """
    return __ACTIVE_DEVICE__.create_tensor(*shape, dtype=dtype)
    # return __ACTIVE_DEVICE__.create_tensor_buffer(*shape, dtype=dtype, memory=memory, clear=clear).as_tensor()

# @_check_active_device
def tensor_copy(t: _torch.Tensor) -> ViewTensor:
    vk_t = tensor(*t.shape, dtype=t.dtype)
    vk_t.copy_(t)
    return vk_t

# @_check_active_device
def pipeline_raytracing() -> RaytracingPipeline:
    """
    Creates a Pipeline to manage the setup of a raytracing pipeline, resources, include and other attributes.
    """
    return __ACTIVE_DEVICE__.create_raytracing_pipeline()

# @_check_active_device
def pipeline_graphics() -> GraphicsPipeline:
    """
    Creates a Pipeline to manage the setup of a graphics pipeline, resources, include and other attributes.
    """
    return __ACTIVE_DEVICE__.create_graphics_pipeline()

# @_check_active_device
def pipeline_compute() -> Pipeline:
    """
    Creates a Pipeline to manage the setup of a compute pipeline, resources, include and other attributes.
    """
    return __ACTIVE_DEVICE__.create_compute_pipeline()

# @_check_active_device
def sampler(mag_filter: Filter = Filter.POINT,
            min_filter: Filter = Filter.POINT,
            mipmap_mode: MipMapMode = MipMapMode.POINT,
            address_U: AddressMode = AddressMode.REPEAT,
            address_V: AddressMode = AddressMode.REPEAT,
            address_W: AddressMode = AddressMode.REPEAT,
            mip_LOD_bias: float = 0.0,
            enable_anisotropy: bool = False,
            max_anisotropy: float = 0.0,
            enable_compare: bool = False,
            compare_op: CompareOp = CompareOp.NEVER,
            min_LOD: float = 0.0,
            max_LOD: float = 0.0,
            border_color: BorderColor = BorderColor.TRANSPARENT_BLACK_FLOAT,
            use_unnormalized_coordinates: bool = False
            ) -> Sampler:
    """
    Creates a sampler that can be used to bind texture objects.
    """
    return __ACTIVE_DEVICE__.create_sampler(
        mag_filter, min_filter, mipmap_mode, address_U, address_V, address_W, mip_LOD_bias, enable_anisotropy,
        max_anisotropy,
        enable_compare, compare_op, min_LOD, max_LOD, border_color, use_unnormalized_coordinates
    )

# @_check_active_device
def sampler_linear(address_U: AddressMode = AddressMode.REPEAT,
                  address_V: AddressMode = AddressMode.REPEAT,
                  address_W: AddressMode = AddressMode.REPEAT,
                  mip_LOD_bias: float = 0.0,
                  enable_anisotropy: bool = False,
                  max_anisotropy: float = 0.0,
                  enable_compare: bool = False,
                  compare_op: CompareOp = CompareOp.NEVER,
                  min_LOD: float = 0.0,
                  max_LOD: float = 1000.0,
                  border_color: BorderColor = BorderColor.TRANSPARENT_BLACK_FLOAT,
                  use_unnormalized_coordinates: bool = False
                  ) -> Sampler:
    """
    Creates a linear sampler that can be used to bind texture objects.
    """
    return __ACTIVE_DEVICE__.create_sampler(
        Filter.LINEAR, Filter.LINEAR, MipMapMode.LINEAR, address_U, address_V, address_W, mip_LOD_bias, enable_anisotropy,
        max_anisotropy,
        enable_compare, compare_op, min_LOD, max_LOD, border_color, use_unnormalized_coordinates
    )

# @_check_active_device
def image_1D(image_format: Format, width: int, mips=None, layers=1,
                     usage: ImageUsage = ImageUsage.SAMPLED, memory: MemoryLocation = MemoryLocation.GPU) -> Image:
    """
    Creates a one-dimensional image object on the GPU. If mips is None, then the maximum possible value is used.
    """
    import math
    if mips is None:
        mips = int(math.log(width, 2)) + 1
    return __ACTIVE_DEVICE__.create_image(ImageType.TEXTURE_1D, False, image_format,
                             width, 1, 1, mips, layers, usage, memory=memory)

# @_check_active_device
def image_2D(image_format: Format, width: int, height: int, mips=None, layers=1,
                     usage: ImageUsage = ImageUsage.SAMPLED, memory: MemoryLocation = MemoryLocation.GPU) -> Image:
    """
    Creates a two-dimensional image object on the GPU. If mips is None, then the maximum possible value is used.
    """
    import math
    if mips is None:
        mips = int(math.log(max(width, height), 2)) + 1
    return __ACTIVE_DEVICE__.create_image(ImageType.TEXTURE_2D, False, image_format,
                             width, height, 1, mips, layers, usage, memory=memory)

# @_check_active_device
def image_3D(image_format: Format, width: int, height: int, depth: int, mips : int = None, layers : int = 1,
                     usage: ImageUsage = ImageUsage.SAMPLED, memory: MemoryLocation = MemoryLocation.GPU) -> Image:
    """
    Creates a three-dimensional image object on the GPU. If mips is None, then the maximum possible value is used.
    """
    import math
    if mips is None:
        mips = int(math.log(max(width, height, depth), 2)) + 1
    return __ACTIVE_DEVICE__.create_image(ImageType.TEXTURE_3D, False, image_format,
                             width, height, depth, mips, layers, usage, memory=memory)


def external_sync():
    """
    Must be used if a tensor bound to a pipeline depends on some external computation, e.g. CUDA.
    """
    _internal.syncronize_external_computation()


# @_check_active_device
def image(image_type: ImageType, is_cube: bool, image_format: Format,
                 width: int, height: int, depth: int,
                 mips: int, layers: int,
                 usage: ImageUsage, memory: MemoryLocation) -> Image:
    """
    Creates an image object on the specified memory (HOST or DEVICE).
    """
    return __ACTIVE_DEVICE__.create_image(image_type, is_cube, image_format,
                              width, height, depth, mips, layers, usage, memory)

# @_check_active_device
def render_target(image_format: Format, width: int, height: int) -> Image:
    """
    Creates a two-dimensional image object on the GPU to be used as render target.
    """
    return __ACTIVE_DEVICE__.create_image(ImageType.TEXTURE_2D, False, image_format,
                             width, height, 1, 1, 1, ImageUsage.RENDER_TARGET, MemoryLocation.GPU)

# @_check_active_device
def depth_stencil(width: int, height: int) -> Image:
    """
    Creates a two-dimensional image object on the GPU to be used as depth stencil buffer.
    """
    return __ACTIVE_DEVICE__.create_image(ImageType.TEXTURE_2D, False, Format.DEPTH_STENCIL,
                             width, height, 1, 1, 1, ImageUsage.DEPTH_STENCIL, MemoryLocation.GPU)

# @_check_active_device
def scratch_buffer(*adss) -> Buffer:
    """
    Creates a buffer on the GPU to be used as scratch buffer for acceleration-datastructure creation.
    """
    size = max(a.scratch_size for a in adss)
    return __ACTIVE_DEVICE__.create_buffer(size, usage=BufferUsage.RAYTRACING_ADS, memory=MemoryLocation.GPU)

# @_check_active_device
def instance_buffer(instances: int, memory: MemoryLocation = MemoryLocation.GPU) -> StructuredBuffer:
    """
    Creates a buffer on the GPU to be used to store instances of a scene acceleration-datastructure.
    """
    return __ACTIVE_DEVICE__.create_instance_buffer(instances, memory)

# @_check_active_device
def ads_scene(instance_buffer: Buffer) -> ADS:
    """
    Creates an acceleration data structure for the scene elements (top-level ads).
    """
    return __ACTIVE_DEVICE__.create_scene_ads(instance_buffer)

# @_check_active_device
def ads_model(collection: GeometryCollection) -> ADS:
    """
    Creates an acceleration data structure for a model formed by a set of geometries (bottom-level ads).
    """
    return __ACTIVE_DEVICE__.create_geometry_ads(collection)

# @_check_active_device
def buffer(size: int, usage: BufferUsage, memory: MemoryLocation) -> Buffer:
    """
    Creates a buffer for a generic usage. Cuda-visible buffers exposes a cuda_ptr and
    can be wrap as tensors zero-copy operation.
    """
    return __ACTIVE_DEVICE__.create_buffer(size, usage=usage, memory=memory)

# @_check_active_device
def buffer_like(t: _typing.Union[_torch.Tensor, Resource], memory: MemoryLocation) -> Buffer:
    return __ACTIVE_DEVICE__.create_buffer_like(t, memory)

# @_check_active_device
def object_buffer(layout: Layout, usage: BufferUsage = BufferUsage.UNIFORM, memory: MemoryLocation = MemoryLocation.CPU) -> ObjectBuffer:
    """
    Creates a buffer for a uniform store. Uniform CPU data can be updated (cpu version) accessing to the fields.
    To finally update the resource (in case is allocated on the gpu) use flush_cpu().
    """
    return __ACTIVE_DEVICE__.create_object_buffer(element_description=layout, usage = usage, memory = memory)

# @_check_active_device
def structured_buffer(count: int, element_description: _typing.Union[type, _torch.dtype, Layout, _typing.Dict, _typing.List],
                             usage: BufferUsage = BufferUsage.STORAGE,
                             memory: MemoryLocation = MemoryLocation.GPU) -> StructuredBuffer:
    """
    Creates a buffer for a structured store. Each index is a Uniform that
    can be updated (cpu version) accessing to the fields.
    To finally update the resource (in case is allocated on the gpu) use flush_cpu().
    """
    return __ACTIVE_DEVICE__.create_structured_buffer(count, element_description=element_description, usage = usage, memory = memory)

# @_check_active_device
def index_buffer(count: int,
                          memory: MemoryLocation = MemoryLocation.GPU) -> StructuredBuffer:
    """
    Creates a buffer for an indices store. Each index is a int32 value that
    can be updated (cpu version).
    To finally update the resource (in case is allocated on the gpu) use flush_cpu().
    """
    return __ACTIVE_DEVICE__.create_structured_buffer(count, element_description=Layout.from_description(LayoutAlignment.SCALAR, _torch.int32), usage = BufferUsage.INDEX, memory = memory)

# @_check_active_device
def vertex_buffer(count: int, element_description: _typing.Union[type, _torch.dtype, Layout, _typing.Dict, _typing.List],
                             memory: MemoryLocation = MemoryLocation.GPU) -> StructuredBuffer:
    """
    Creates a buffer for a structured store. Each index is a Uniform that
    can be updated (cpu version) accessing to the fields.
    To finally update the resource (in case is allocated on the gpu) use flush_cpu().
    """
    return __ACTIVE_DEVICE__.create_structured_buffer(count, element_description=element_description, usage = BufferUsage.VERTEX, memory = memory)

# @_check_active_device
# def wrap_in(*objs: Any) -> DeviceManager.WrappedGPUPtr:
#     """
#     Creates an uniform buffer for an int64 ptr store.
#     Use wrap method to take the ptr from a valid vulkan-compatible object (vulkanized tensors, buffers) or
#     set directly the ptr field (in that case flush_cpu is necessary).
#     """
#     return __ACTIVE_DEVICE__.map_gpu(*objs, mode='in')
#
# @_check_active_device
# def wrap_out(*objs: Any) -> DeviceManager.WrappedGPUPtr:
#     """
#     Creates an uniform buffer for an int64 ptr store.
#     Use wrap method to take the ptr from a valid vulkan-compatible object (vulkanized tensors, buffers) or
#     set directly the ptr field (in that case flush_cpu is necessary).
#     """
#     return __ACTIVE_DEVICE__.map_gpu(*objs, mode='out')
#
# @_check_active_device
# def wrap_inout(*objs: Any) -> DeviceManager.WrappedGPUPtr:
#     """
#     Creates an uniform buffer for an int64 ptr store.
#     Use wrap method to take the ptr from a valid vulkan-compatible object (vulkanized tensors, buffers) or
#     set directly the ptr field (in that case flush_cpu is necessary).
#     """
#     return __ACTIVE_DEVICE__.map_gpu(*objs, mode='inout')

# @_check_active_device
def triangle_collection() -> TriangleCollection:
    """
    Creates a collection to store triangle meshes for ray-cast
    """
    return __ACTIVE_DEVICE__.create_triangle_collection()

# @_check_active_device
def aabb_collection() -> AABBCollection:
    """
    Creates a collection to store boxes for ray-cast
    """
    return __ACTIVE_DEVICE__.create_aabb_collection()

# @_check_active_device
def flush():
    """
    Flushes all pending work of vulkan submissions and wait for completion.
    """
    __ACTIVE_DEVICE__.flush()

# @_check_active_device
def torch_ptr_to_device_ptr(t: _torch.Tensor):
    return __ACTIVE_DEVICE__.torch_ptr_to_device_ptr(t)

# @_check_active_device
def compute_manager() -> ComputeManager:
    """
    Gets a compute manager object that can be used to populate with compute commands.
    Using this object as a context will flush automatically the command list at the end.
    Creating an object and freeze allows to submit several times after creation.
    """
    return __ACTIVE_DEVICE__.get_compute()

# @_check_active_device
def copy_manager() -> CopyManager:
    """
    Gets a copy manager object that can be used to populate with transfer commands.
    Using this object as a context will flush automatically the command list at the end.
    Creating an object and freeze allows to submit several times after creation.
    """
    return __ACTIVE_DEVICE__.get_copy()

# @_check_active_device
def graphics_manager() -> GraphicsManager:
    """
    Gets a graphics manager object that can be used to populate with graphics commands.
    Using this object as a context will flush automatically the command list at the end.
    Creating an object and freeze allows to submit several times after creation.
    """
    return __ACTIVE_DEVICE__.get_graphics()

# @_check_active_device
def raytracing_manager() -> RaytracingManager:
    """
    Gets a raytracing manager object that can be used to populate with raytracing commands.
    Using this object as a context will flush automatically the command list at the end.
    Creating an object and freeze allows to submit several times after creation.
    """
    return __ACTIVE_DEVICE__.get_raytracing()

# @_check_active_device
def submit(man: CommandManager, wait: bool = True):
    """
    Allows to submit the command list save in a manager using freeze.
    """
    __ACTIVE_DEVICE__.submit(man, wait)

# @_check_active_device
def load_technique(technique):
    return __ACTIVE_DEVICE__.load_technique(technique)

# @_check_active_device
def dispatch_technique(technique):
    return __ACTIVE_DEVICE__.dispatch_technique(technique)

# @_check_active_device
def execute_loop(main_process: _typing.Callable[[_typing.Optional[_typing.Any]], bool], *process: _typing.Callable[[_typing.Optional[_typing.Any]], None], context: _typing.Optional[_typing.Any] = None):
    """
    Creates a thread to execute the process safety dispatching vulkan calls in the main thread
    """
    __ACTIVE_DEVICE__.execute_loop(main_process, *process, context=context)

# @_check_active_device
def allow_cross_threading():
    """
    Allows dispatching cross-threading vulkan calls. The safest way is to include the cross-threading code
    inside a loop function
    """
    __ACTIVE_DEVICE__.allow_cross_threading()

# @_check_active_device
def set_debug_name(name: str):
    """
    Sets the name for the next resource to be created.
    """
    __ACTIVE_DEVICE__.set_debug_name(name)

# @_check_active_device
def wrap_gpu(t: _typing.Any, mode: _typing.Literal['in', 'out', 'inout'] = 'in') -> GPUPtr:
    """
    Wraps an object to be accessible from/to the GPU depending on the mode.
    Returned object can be assigned to fields of type int64_t and use as reference buffers.
    """
    return __ACTIVE_DEVICE__.wrap_gpu(t, mode)



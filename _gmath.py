import torch
import math
from typing import Any, Union, Optional


def tensor_to_vec(t: torch.Tensor) -> '_GTensorBase':
    assert len(t.shape) >= 1
    assert t.dtype == torch.float32 or t.dtype == torch.int32
    vec_shape = tuple(t.shape[-1:])
    try:
        if t.dtype == torch.float32:
            typ = {
                (2,): vec2,
                (3,): vec3,
                (4,): vec4,
            }[vec_shape]
        else:
            typ = {
                (2,): ivec2,
                (3,): ivec3,
                (4,): ivec4,
            }[vec_shape]
    except:
        raise Exception(f'Not supported vector type for shape {vec_shape}')
    return typ(t)


def tensor_to_mat(t: torch.Tensor) -> '_GTensorBase':
    assert len(t.shape) >= 2
    assert t.dtype == torch.float32
    mat_shape = tuple(t.shape[-2:])
    try:
        typ = {
            (2,2): mat2,
            (3,3): mat3,
            (3,4): mat3x4,
            (4,3): mat4x3,
            (4,4): mat4,
        }[mat_shape]
    except:
        raise Exception(f'Not supported vector type for shape {mat_shape}')
    return typ(t)


def tensor_to_gtensor_if_possible(t: torch.Tensor, dimension: int) -> Union[torch.Tensor, '_GTensorBase']:
    """
    Tool method to convert a torch tensor into a graphic tensor or scalar
    """
    assert dimension >= 0
    if dimension == 0 or dimension > 2:
        return t.as_subclass(torch.Tensor)  # Remove any attached gtensor class could be using
    if dimension == 1:
        return tensor_to_vec(t)
    if dimension == 2:
        return tensor_to_mat(t)
    raise Exception('?')


# def _tensify(t, dimension):
#     """
#     Tool method to convert a torch tensor into a graphic tensor or scalar
#     """
#     assert dimension >= 0, "Dimension can not be negative"
#     if dimension == 0 or dimension > 2:  #scalar or multidimensional array
#         return t
#     assert len(t.shape) >= dimension, "can not create a matrix from a vector tensor"
#     if tuple(t.shape[-dimension:]) not in __SHAPE_TO_TYPE__:
#         return t.as_subclass(torch.Tensor)
#     typ = __SHAPE_TO_TYPE__[t.shape[-dimension:]]
#     if t.dtype == torch.int32:
#         if typ == 'float':
#             typ = 'int'
#         else:
#             typ = 'i'+typ
#     typ = eval(typ)
#     return typ(t)


# def _tensify_batch(t, batch_shape):
#     if not isinstance(t, torch.Tensor):
#         return t
#     if len(t.shape) < len(batch_shape) or list(t.shape[:len(batch_shape)]) != list(batch_shape):
#         return t.as_subclass(torch.Tensor)  # is not batched
#     dimension = len(t.shape) - len(batch_shape)
#     return _tensify(t, dimension)


class GTensorMeta(torch._C._TensorMeta):
    def __new__(mcs, name, bases, dct):
        tensor_type = super().__new__(mcs, name, bases, dct)
        if name.startswith('i'):
            dtype = torch.int32
            name = name[1:]
        else:
            dtype = torch.float32
        dict = {
            'vec1': (1,),
            'vec2': (2,),
            'vec3': (3,),
            'vec4': (4,),
            'mat1': (1, 1),
            'mat2': (2, 2),
            'mat3': (3, 3),
            'mat3x4': (3, 4),
            'mat4x3': (4, 3),
            'mat4': (4, 4),
        }
        if name not in dict:
            return tensor_type
        tensor_type.tensor_shape = dict[name]
        tensor_type.tensor_dtype = dtype
        tensor_type.dimension = len(tensor_type.tensor_shape)
        return tensor_type

    def __promote_constant(self, value):
        if not isinstance(value, torch.Tensor):
            return torch.full(self.tensor_shape, fill_value=value, dtype=self.tensor_dtype)
        return value.repeat()

    @staticmethod
    def _check_componentwise_tensors(args):
        if not isinstance(args, list):
            return False
        if len(args) == 0:
            return False
        for a in args:
            if not isinstance(a, torch.Tensor):
                return False
            if a.shape != args[0].shape:
                return False
        return True

    def __call__(cls, *args, **kwargs):
        if len(args) == 0:
            args = [0.0]
        if len(args) == 1:  # promote
            if not isinstance(args[0], torch.Tensor):
                if isinstance(args[0], list):
                    args = torch.as_tensor(args[0], dtype=cls.tensor_dtype)
                    assert list(args.shape[-cls.dimension:]) == list(cls.tensor_shape), "Error with shapes in list elements"
                else:
                    # Assume is scalar
                    args = cls.__promote_constant(args[0])
            else:
                # Check if it is a scalar batched tensor
                args = args[0].type(cls.tensor_dtype)
                if args.shape[-1] == 1: # need to promote
                    args = args.repeat(*tuple([1]*len(args.shape[:-1])), math.prod(cls.tensor_shape))
                    args = args.view(*args.shape[:-1], *cls.tensor_shape)
                assert args.shape[-cls.dimension:] == cls.tensor_shape, f"Wrong shape for tensor argument, expected final dimension to be 1 or {cls.tensor_shape}"
        if not isinstance(args, torch.Tensor):
            if _GTensorBase._check_componentwise_tensors(args):
                args = torch.cat(args, dim=-1).type(cls.tensor_dtype)
            else:
                args = torch.as_tensor([*args], dtype=cls.tensor_dtype).view(cls.tensor_shape)
        assert args.shape[-cls.dimension:] == cls.tensor_shape, f'Wrong vector dimension, expected {cls.shape} provided {args.shape[-cls.dimension:]}'
        tensor_instance = super(GTensorMeta, cls).__call__(args)
        tensor_instance = tensor_instance.as_subclass(cls)
        batch_dimension = len(args.shape) - cls.dimension
        object.__setattr__(tensor_instance, 'batch_dimension', batch_dimension)
        object.__setattr__(tensor_instance, 'batch_shape', args.shape[:batch_dimension])
        object.__setattr__(tensor_instance, 'is_batch', batch_dimension > 0)
        return tensor_instance


# __SHAPE_TO_TYPE__ = {
#     # (1,): 'vec1',
#     (2,): 'vec2',
#     (3,): 'vec3',
#     (4,): 'vec4',
#     # (1, 1): 'mat1',
#     (2, 2): 'mat2',
#     (3, 3): 'mat3',
#     (3, 4): 'mat3x4',
#     (4, 3): 'mat4x3',
#     (4, 4): 'mat4'
# }


__FIELDS_INDEX__ = {
    'x': 0,
    'y': 1,
    'z': 2,
    'w': 3,
}


class _GTensorBase(torch.Tensor, metaclass=GTensorMeta):
    def __init__(self, *args):
        super(_GTensorBase, self).__init__()
        batch_dimension = len(self.shape) - self.dimension
        object.__setattr__(self, 'batch_dimension', batch_dimension)
        object.__setattr__(self, 'batch_shape', self.shape[:batch_dimension])
        object.__setattr__(self, 'is_batch', batch_dimension > 0)

    @classmethod
    def length(cls, x):
        """
        Computes a tensor with the lengths of vectors in x.
        """
        assert cls.dimension == 1
        return tensor_to_gtensor_if_possible(torch.sqrt((x**2).sum(dim=-1, keepdim=True)), dimension=0)

    @classmethod
    def dot(cls, a, b):
        """
        Computes a tensor with the dot product of vectors a and b.
        """
        assert cls.dimension == 1
        return tensor_to_gtensor_if_possible((a*b).sum(-1, keepdim=True), dimension=0)

    @classmethod
    def normalize(cls, v):
        """
        Computes the normalized version of v.
        """
        assert cls.dimension == 1
        return v / torch.sqrt(cls.dot(v, v))

    @classmethod
    def identity(cls):
        """
        Creates an identity matrix.
        """
        assert cls.dimension == 2, "Identity function only valid for matrices"
        assert cls.tensor_shape[0] == cls.tensor_shape[1], "Identity function only valid for squared matrices"
        return cls(torch.diag(torch.full(cls.tensor_shape[0:1], 1, dtype=cls.tensor_dtype)))

    @classmethod
    def zero(cls):
        """
        Creates a zero matrix.
        """
        return cls(torch.zeros(cls.tensor_shape, dtype=cls.tensor_dtype))

    @classmethod
    def one(cls):
        """
        Creates a matrix or vector full of ones.
        """
        return cls(torch.ones(cls.tensor_shape, dtype=cls.tensor_dtype))

    def __repr__(self):
        return super(_GTensorBase, self).__repr__().replace("tensor", self.__class__.__name__)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        with torch._C.DisableTorchFunction():
            if func == torch.Tensor.as_subclass:
                return func(*args, **kwargs)
            if func == torch.Tensor.requires_grad_:
                return func(*args, **kwargs)
            if kwargs is None:
                kwargs = {}
            if func.__name__ == '__get__':
                return func(*args, **kwargs)
            batch_dimension = None
            for tensor in args:
                if isinstance(tensor, _GTensorBase) and tensor.is_batch:
                    current_batch_dimension = len(tensor.shape[:tensor.batch_dimension])
                    if batch_shape is None:
                        batch_shape = current_batch_dimension
                    else:
                        assert batch_dimension == current_batch_dimension, "Can not operate two graphic tensors with different batch dimensions"
            if batch_dimension is None:
                batch_dimension = 0
            if kwargs is None:
                kwargs = {}
            args = tuple(a if not isinstance(a, _GTensorBase) else a.as_subclass(torch.Tensor) for a in args)  # retrieve all tensors from GTensors
            kwargs = {k: v if not isinstance(v, _GTensorBase) else v.as_subclass(torch.Tensor) for k, v in kwargs.items()}

            ret = func(*args, **kwargs)
            if isinstance(ret, torch.Tensor):
                return tensor_to_gtensor_if_possible(ret, len(ret.shape) - batch_dimension)
            return ret

    def __getitem__(self, item):
        last_indexes = [*item] if isinstance(item, tuple) else [item]
        full_index = [slice(None)]*self.batch_dimension + last_indexes
        t = super(_GTensorBase, self).__getitem__(full_index)
        return tensor_to_gtensor_if_possible(t, len(t.shape) - self.batch_dimension)

    def __setitem__(self, item, value):
        last_indexes = [*item] if isinstance(item, tuple) else [item]
        full_index = [slice(None)]*self.batch_dimension + last_indexes
        super(_GTensorBase, self).__setitem__(full_index, value)

    def __iter__(self):
        assert not self.is_batch, "Not supported yet iteration over tensor batch"
        return super(_GTensorBase, self).__iter__()

    def __getattr__(self, item):
        if item in __FIELDS_INDEX__:
            return self[__FIELDS_INDEX__[item]]
        if all(c in __FIELDS_INDEX__ for c in item):  # all are fields
            index = [__FIELDS_INDEX__[c] for c in item]
            return self[index]
        try:
            return super(_GTensorBase, self).__getattr__(item)
        except:
            pass
        return super(_GTensorBase, self).__getattribute__(item)
    
    def __setattr__(self, key, value):
        if key in __FIELDS_INDEX__:
            self[__FIELDS_INDEX__[key]] = value
            return
        if all(c in __FIELDS_INDEX__ for c in key):  # all are fields
            index = [__FIELDS_INDEX__[c] for c in key]
            self[index] = value
            return
        super(_GTensorBase, self).__setattr__(key, value)


# class vec1(_GTensorBase):
#     pass


class vec2(_GTensorBase):
    pass


class vec3(_GTensorBase):
    @classmethod
    def cross(cls, a: Union[torch.Tensor, 'vec3'], b: Union[torch.Tensor, 'vec3']) -> 'vec3':
        return vec3(torch.cross(a, b))


class vec4(_GTensorBase):
    pass


# class ivec1(_GTensorBase):
#     pass


class ivec2(_GTensorBase):
    pass


class ivec3(_GTensorBase):
    pass


class ivec4(_GTensorBase):
    pass


# class mat1(_GTensorBase):
#     pass


class mat2(_GTensorBase):
    pass


class mat3(_GTensorBase):
    @staticmethod
    def rotation(axis: Union[torch.Tensor, vec3], angle: torch.Tensor) -> torch.Tensor:
        axis, angle = broadcast_args_to_max_batch(
            (axis, (3,)),
            (angle, ()),
        )
        cos_theta = torch.cos(angle)
        sin_theta = torch.sin(angle)
        ux = axis[..., 0:1]
        uy = axis[..., 1:2]
        uz = axis[..., 2:3]
        m = torch.cat(
            [
                cos_theta + ux ** 2 * (1 - cos_theta),
                ux * uy * (1 - cos_theta) - uz * sin_theta,
                ux * uz * (1 - cos_theta) + uy * sin_theta,

                uy * ux * (1 - cos_theta) + uz * sin_theta,
                cos_theta + uy ** 2 * (1 - cos_theta),
                uy * uz * (1 - cos_theta) - ux * sin_theta,

                uz * ux * (1 - cos_theta) - uy * sin_theta,
                uz * uy * (1 - cos_theta) + ux * sin_theta,
                cos_theta + uz ** 2 * (1 - cos_theta)
            ], dim=-1
        )
        return mat3(m.view(*axis.shape[:-1], 3, 3))


class mat3x4(_GTensorBase):

    @staticmethod
    def composite(internal: Optional[Union['mat3x4', torch.Tensor]],
                  external: Optional[Union['mat3x4', torch.Tensor]]) -> 'mat3x4':
        """
        Gets the transform composition between two transforms (in transpose mode).
        x', 1 = M (x, 1)^T
        Internal transform is the more local, external the most global.
        """
        if internal is None:
            return external
        if external is None:
            return internal
        Ra = external[..., :3, :3]
        Rb = internal[..., :3, :3]
        Ta = external[..., :3, 3:4]
        Tb = internal[..., :3, 3:4]
        return mat3x4(torch.cat([Ra @ Rb, Ra @ Tb + Ta], dim=-1))

    def transposed(self):
        return mat4x3(self.transpose(dim0=-1, dim1=-2).contiguous())


class mat4x3(_GTensorBase):
    @staticmethod
    def trs(offset: vec3, axis: vec3, angle: Union[float, torch.Tensor], scale: vec3) -> 'mat4x3':
        offset, axis, angle, scale = broadcast_args_to_max_batch(
            (offset, (3,)),
            (axis, (3,)),
            (angle, ()),
            (scale, (3,))
        )
        ux = axis[..., 0:1]
        uy = axis[..., 1:2]
        uz = axis[..., 2:3]
        zeros = torch.zeros_like(ux)
        ones = torch.ones_like(ux)
        cos_theta = torch.cos(angle)
        sin_theta = torch.sin(angle)
        s = torch.cat([
            scale[..., 0:1], zeros, zeros, zeros,
            zeros, scale[..., 1:2], zeros, zeros,
            zeros, zeros, scale[..., 2:3], zeros,
            zeros, zeros, zeros, ones
        ], dim=-1).view(*offset.shape[:-1], 4, 4)
        m = torch.cat(
            [
                cos_theta + ux ** 2 * (1 - cos_theta),
                ux * uy * (1 - cos_theta) - uz * sin_theta,
                ux * uz * (1 - cos_theta) + uy * sin_theta,

                uy * ux * (1 - cos_theta) + uz * sin_theta,
                cos_theta + uy ** 2 * (1 - cos_theta),
                uy * uz * (1 - cos_theta) - ux * sin_theta,

                uz * ux * (1 - cos_theta) - uy * sin_theta,
                uz * uy * (1 - cos_theta) + ux * sin_theta,
                cos_theta + uz ** 2 * (1 - cos_theta),

                offset.repeat(len(ux), 1)
            ], dim=-1
        ).view(*offset.shape[:-1], 4, 3)
        T = s @ m
        return T

    @staticmethod
    def composite(internal: Optional[Union['mat4x3', torch.Tensor]], external: Optional[Union['mat4x3', torch.Tensor]]) -> 'mat4x3':
        if internal is None:
            return external
        if external is None:
            return internal
        Ra = external[..., :3, :3]
        Rb = internal[..., :3, :3]
        Ta = external[..., 3:4, :3]
        Tb = internal[..., 3:4, :3]
        return mat4x3(torch.cat([Ra @ Rb, Ra @ Tb + Ta], dim=-2))

    def transposed(self):
        return mat3x4(self.transpose(dim0=-1, dim1=-2).contiguous())


class mat4(_GTensorBase):
    @staticmethod
    def inv_look_at(ori: vec3, dir: vec3, nor: vec3):
        ori, dir, nor = broadcast_args_to_max_batch(
            (ori, (3,)),
            (dir, (3,)),
            (nor, (3,))
        )
        dev = ori.device
        zaxis = dir
        xaxis = vec3.normalize(vec3.cross(nor, zaxis))
        yaxis = vec3.cross(zaxis, xaxis)
        exp_xaxis = torch.cat([xaxis, torch.zeros(*xaxis.shape[:-1], 1).to(dev)], dim=-1).unsqueeze(-2)
        exp_yaxis = torch.cat([yaxis, torch.zeros(*xaxis.shape[:-1], 1).to(dev)], dim=-1).unsqueeze(-2)
        exp_zaxis = torch.cat([zaxis, torch.zeros(*xaxis.shape[:-1], 1).to(dev)], dim=-1).unsqueeze(-2)
        exp_ori = torch.cat([ori, torch.ones(*xaxis.shape[:-1], 1).to(dev)], dim=-1).unsqueeze(-2)
        return mat4(torch.cat([exp_xaxis, exp_yaxis, exp_zaxis, exp_ori], dim=-2))

    def inverse(self):
        return mat4(torch.linalg.inv(self))

    @staticmethod
    def trs(offset: vec3, axis: vec3, angle: Union[float, torch.Tensor], scale: vec3) -> 'mat4x3':
        offset, axis, angle, scale = broadcast_args_to_max_batch(
            (offset, (3,)),
            (axis, (3,)),
            (angle, ()),
            (scale, (3,))
        )
        ux = axis[..., 0:1]
        uy = axis[..., 1:2]
        uz = axis[..., 2:3]
        zeros = torch.zeros_like(ux)
        ones = torch.ones_like(ux)
        cos_theta = torch.cos(angle)
        sin_theta = torch.sin(angle)
        s = torch.cat([
            scale[..., 0:1], zeros, zeros, zeros,
            zeros, scale[..., 1:2], zeros, zeros,
            zeros, zeros, scale[..., 2:3], zeros,
            zeros, zeros, zeros, ones
        ], dim=-1).view(*offset.shape[:-1], 4, 4)
        m = torch.cat(
            [
                cos_theta + ux ** 2 * (1 - cos_theta),
                ux * uy * (1 - cos_theta) - uz * sin_theta,
                ux * uz * (1 - cos_theta) + uy * sin_theta,
                zeros,

                uy * ux * (1 - cos_theta) + uz * sin_theta,
                cos_theta + uy ** 2 * (1 - cos_theta),
                uy * uz * (1 - cos_theta) - ux * sin_theta,
                zeros,

                uz * ux * (1 - cos_theta) - uy * sin_theta,
                uz * uy * (1 - cos_theta) + ux * sin_theta,
                cos_theta + uz ** 2 * (1 - cos_theta),
                zeros,

                offset.repeat(len(ux), 1),
                ones
            ], dim=-1
        ).view(*offset.shape[:-1], 4, 3)
        T = s @ m
        return T


def broadcast_args_to_max_batch(*args):
    def valid_shape(t, shape):
        if isinstance(t, torch.Tensor):
            return t.shape[:-len(shape)] == shape
        return len(shape) == 0
    devices = [a.device for a in args if isinstance(a, torch.Tensor)]
    assert len(devices) > 0, 'At least one tensor should be provided'
    assert all(d == devices[0] for d in devices), 'All tensors has to be in same device.'
    def scalar_to_tensor(t):
        if not isinstance(t, torch.Tensor):
            assert isinstance(t, float) or isinstance(t, int)
            return torch.tensor([t], device=devices[0])
        return t
    max_batch = None
    batch_dimension = None
    assert all(valid_shape(t, shape) for (t, shape) in args)
    args = [scalar_to_tensor(t) for t in args]  # Convert all to tensor if necessary
    for (t,shape) in args:
        current_batch_dimension = len(t.shape) - len(shape)
        current_batch = list(t.shape[:-len(shape)])
        if current_batch_dimension > 0:  # is batched
            if max_batch is None:
                max_batch = current_batch
                batch_dimension = current_batch_dimension
            else:
                assert current_batch_dimension == batch_dimension, 'Can not have different batch dimensions'
                for i, d in enumerate(current_batch):
                    if d > 1:
                        if max_batch[i] == 1:
                            max_batch[i] = d
                        else:
                            assert max_batch[i] == d, 'Dimension of some argument missmatch same dimension in another'
    if max_batch is None:  # there is no argument batching
        return args
    def to_max_batch(t, shape):
        if len(t.shape) == len(shape):  # no batching
            for d in range(batch_dimension):
                t = t.unsqueeze(0)
        t = t.repeat(*(md // d for md, d in zip(max_batch, t.shape[:-len(shape)])) ,*([1]*len(shape)))
        return t

    return (to_max_batch(t, shape) for t, shape in args)

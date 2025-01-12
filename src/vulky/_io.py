"""
Home-made obj loader.
"""
import typing as _typing
import torch as _torch
from _rendering_internal import Image, Format, image_2D, Filter, graphics_manager

def _load_mtl_file(path, directory) -> dict:
    materials = {}
    current_material = {}
    current_material_name = None

    def close_current_material():
        nonlocal current_material
        if len(current_material) == 0:
            return
        materials[current_material_name] = current_material
        current_material = {}

    with open(directory + '/' + path, mode='r') as f:
        while True:
            line = f.readline()
            if line == '':  #eof
                break
            elements = line.split()
            if len(elements) == 0:  # empty line
                continue
            cmd = elements[0]
            if cmd == '#':
                continue
            if cmd == 'newmtl':
                close_current_material()
                current_material_name = ' '.join(elements[1:])
            else:
                assert current_material_name is not None, 'There should be newmtl command at least'
                if cmd.endswith('_map'):
                    # load texture
                    pass
                else:
                    current_material[cmd] = _torch.tensor([float(v) for v in elements[1:]])

    close_current_material()
    return materials


def load_obj(path: str) -> dict:
    '''
    Loads an obj file into a dict with separated vertex attributes and indices
    @number_of_objects
    @number_of_faces
    @number_of_materials
    @vertex_description: List[str] with all per-vertex elements. First attribute with be assumed as position always
    vertex_element_i: tensor(different_elements, element_components)
    ...
    vertex_element_i_indices: tensor(num_faces, 3) indices to vertex_element_i list for the element of a vertex
    ...
    @material_indices: List[int] with all per-faces materials
    '''
    current_object_name = 'default'
    current_mtl_dict = { }

    objects = { }  # name: tuple start_part_idx and stop_part_index
    parts = []  # tuples with start_face_idx, stop_face_index, materials

    face_of_last_closed_material = 0
    material_of_last_closed_object = 0

    current_used_material = None

    positions = []
    normals = []
    texcoords = []
    positions_indices = []
    normals_indices = []
    texcoords_indices = []

    def close_current_material():
        nonlocal face_of_last_closed_material
        current_face = len(positions_indices)
        if face_of_last_closed_material == current_face:
            return
        parts.append((face_of_last_closed_material, current_face, current_used_material))
        face_of_last_closed_material = current_face

    def close_current_object():
        close_current_material()
        nonlocal material_of_last_closed_object
        current_material = len(parts)
        if material_of_last_closed_object == current_material:
            return
        objects[current_object_name] = (material_of_last_closed_object, current_material)
        material_of_last_closed_object = current_material

    def add_triplets(t1: str, t2: str, t3: str):
        indices1 = t1.split('/')
        indices2 = t2.split('/')
        indices3 = t3.split('/')
        plen = len(positions)
        clen = len(texcoords)
        nlen = len(normals)
        if indices1[0] != '':
            positions_indices.append([
                (int(indices1[0]) - 1 + plen) % plen,
                (int(indices2[0]) - 1 + plen) % plen,
                (int(indices3[0]) - 1 + plen) % plen
            ])
        if len(indices1) > 1 and indices1[1] != '':
            texcoords_indices.append([
                (int(indices1[1]) - 1 + clen) % clen,
                (int(indices2[1]) - 1 + clen) % clen,
                (int(indices3[1]) - 1 + clen) % clen
            ])
        if len(indices1) > 2 and indices1[2] != '':
            normals_indices.append([
                (int(indices1[2]) - 1 + nlen) % nlen,
                (int(indices2[2]) - 1 + nlen) % nlen,
                (int(indices3[2]) - 1 + nlen) % nlen
            ])
    def add_face(triplets):
        for i in range(2, len(triplets)):
            t1 = triplets[0]
            t2 = triplets[i-1]
            t3 = triplets[i]
            add_triplets(t1, t2, t3)

    obj_directory = os.path.dirname(path)

    with open(path, mode='r') as f:
        while True:
            line = f.readline()
            if line == '':  # eof
                break
            elements = line.split()
            if len(elements) == 0:  # empty line
                continue
            cmd = elements[0]
            if cmd == 'v':  # new position
                positions.append([float(elements[1]), float(elements[2]), float(elements[3])])
            elif cmd == 'vn':  # new normal
                normals.append([float(elements[1]), float(elements[2]), float(elements[3])])
            elif cmd == 'vt':   # new texcoord
                texcoords.append([float(elements[1]), float(elements[2]), 0.0 if len(elements) < 4 else float(elements[3])])
            elif cmd == 'f':
                add_face(elements[1:])
            elif cmd == 'mtllib':
                assert len(current_mtl_dict) == 0, 'A mtl file was already loaded. Not supported several mtl files in obj.'
                current_mtl_dict = _load_mtl_file(elements[1], obj_directory)
            elif cmd == 'usemtl':
                close_current_material()
                mtl_name = ' '.join(elements[1:])
                current_used_material = current_mtl_dict[mtl_name]
            elif cmd == 'o' or cmd == 'g':  # create a new object
                close_current_object()
                current_object_name = elements[1]
        close_current_object()
    return {
        'objects': objects,
        'parts': parts,
        'buffers': dict(
            P=_torch.tensor(positions),
            N=_torch.tensor(normals),
            C=_torch.tensor(texcoords),
            P_indices=_torch.tensor(positions_indices, dtype=_torch.int32),
            N_indices=_torch.tensor(normals_indices, dtype=_torch.int32),
            C_indices=_torch.tensor(texcoords_indices, dtype=_torch.int32)
        )
    }


def create_mesh(obj_data: dict, mode: _typing.Literal['po', 'vo', 'fo']):
    """
    Creates the mesh information (vertices, indices) given an obj data.
    po: Position order - Each position determines a vertex, normals and coordinates belonging to the same vertex are blended
    vo: Vertex order - Each distinct triple determines a vertex.
    fo: Face order - Each face determines three new vertices.
    """
    positions = obj_data['buffers']['P']
    normals = obj_data['buffers']['N']
    coordinates = obj_data['buffers']['C']
    P_indices = obj_data['buffers']['P_indices']
    N_indices = obj_data['buffers']['N_indices']
    C_indices = obj_data['buffers']['C_indices']
    has_normals = len(normals) > 0
    if not has_normals:
        N_indices = _torch.zeros_like(P_indices)
    has_coordinates = len(coordinates) > 0
    if not has_coordinates:
        C_indices = _torch.zeros_like(P_indices)
    if mode == 'po':
        vertices = _torch.zeros(len(positions), 8)
        vertices[:, 0:3] = positions
        if has_normals:
            vertices[P_indices, 3:6] += normals[N_indices]
        if has_coordinates:
            vertices[P_indices, 6:8] += coordinates[C_indices, 0:2]
        # for fp, fn, fc in zip(P_indices, N_indices, C_indices):
        #     if has_normals:
        #         vertices[fp[0], 3:6] += normals[fn[0]]
        #         vertices[fp[1], 3:6] += normals[fn[1]]
        #         vertices[fp[2], 3:6] += normals[fn[2]]
        #     if has_coordinates:
        #         vertices[fp[0], 6:8] = coordinates[fc[0], 0:2]
        #         vertices[fp[1], 6:8] = coordinates[fc[1], 0:2]
        #         vertices[fp[2], 6:8] = coordinates[fc[2], 0:2]
        # normalize normals
        vertices[:,3:6] /= _torch.sqrt((vertices[:,3:6]**2).sum(dim=-1, keepdim=True)) + 0.00001
        return vertices, P_indices
    raise Exception()


def save_image(t: _torch.Tensor, filename: str):
    """
    t: (HxWx3)
    """
    if t.dtype == _torch.float:
        t = (_torch.clamp(t, 0.0, 1.0) * 255).to(_torch.uint8)
    try:
        ext = filename.split('.')[-1]
    except:
        raise Exception('Please refer to a valid extension [jpg, png]')

    t = t.permute(2, 0, 1)  # from HWC to CHW

    if t.device != 'cpu':
        t = t.to('cpu')

    import torchvision
    if ext.lower() == 'png':
        torchvision.io.write_png(t, filename)
    elif ext.lower() == 'jpg' or ext.lower() == 'jpeg':
        torchvision.io.write_jpeg(t, filename)
    else:
        raise Exception('Please refer to a valid extension [jpg, png]')


def save_video(t: _torch.Tensor, filename: str, fps: int = 20, **kwargs):

    kwargs = {k: str(v) for k,v in kwargs.items()}

    if t.dtype == _torch.float:
        t = (_torch.clamp(t, 0.0, 1.0) * 255).to(_torch.uint8)
    try:
        ext = filename.split('.')[-1]
    except:
        raise Exception('Please refer to a valid extension [webp, avi]')

    if t.device != 'cpu':
        t = t.to('cpu')

    import torchvision
    if ext.lower() == 'webp':
        codec = 'webp'
    else:
        codec = 'h264'
    torchvision.io.write_video(filename, t, fps, codec, options=kwargs)


def load_image(filename: str, with_alpha: bool = True) -> _torch.Tensor:
    import torchvision
    if with_alpha:
        t = torchvision.io.read_image(filename, mode=torchvision.io.ImageReadMode.RGB_ALPHA)
    else:
        t = torchvision.io.read_image(filename, mode=torchvision.io.ImageReadMode.RGB)
    t = (t.permute(1, 2, 0)/255.0).contiguous()  # from CHW to HWC
    return t


def load_texture(filename: str) -> Image:
    t = load_image(filename)
    width = t.shape[1]
    heigh = t.shape[0]
    im = image_2D(Format.VEC4, width, heigh)
    im.subresource().load(t)
    with graphics_manager() as man:
        for i in range(1, im.get_mip_count()):
            man.blit_image(im.subresource(i - 1, 0), im.subresource(i, 0), filter=Filter.LINEAR)
    return im


def load_video(filename: str):
    import torchvision.io
    return torchvision.io.read_video(filename)


import os as _os
import torch as _torch
from ._io import load_obj

__VULKY_DATA_REPO__ = "https://github.com/rendervous/vulky_data"
__DATASETS_PATH__ = _os.path.dirname(__file__).replace('\\','/')+ "/datasets"


def fetch_data(rewrite: bool = True):
    """
    Clone the Git repository of vulky_data into a specific folder in the library.
    """
    import shutil
    import stat
    import subprocess
    target_folder = __DATASETS_PATH__ + "/vulky_data"
    if _os.path.exists(target_folder):
        if not rewrite:
            return
        shutil.rmtree(target_folder,
                      onerror=lambda func, path, _: (_os.chmod(path, stat.S_IWRITE), func(path)))
    _os.makedirs(target_folder, exist_ok=False)
    try:
        subprocess.run(
            ["git", "clone", __VULKY_DATA_REPO__, target_folder],
            check=True,  # Raise an error if the command fails
            stdout=subprocess.PIPE,  # Capture output
            stderr=subprocess.PIPE,
        )
        print(f"Repository cloned into: {target_folder}")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e.stderr.decode('utf-8')}")


# fetch data when module is imported if not already downloaded
fetch_data(rewrite=False)


class Meshes:
    @classmethod
    @property
    def bunny(cls):
        return load_obj(__DATASETS_PATH__ + "/vulky_data/bunny.obj")

    @classmethod
    @property
    def dragon(cls):
        return load_obj(__DATASETS_PATH__ + "/vulky_data/dragon.obj")

    @classmethod
    @property
    def plate(cls):
        return load_obj(__DATASETS_PATH__ + "/vulky_data/plate.obj")


class Volumes:
    @classmethod
    @property
    def disney_cloud(cls):
        return _torch.load(__DATASETS_PATH__ + "/vulky_data/disney_reduced.pt", map_location=_torch.device('cpu'))


class Images:
    @classmethod
    @property
    def environment_example(cls):
        return _torch.load(__DATASETS_PATH__ + "/vulky_data/environment_0.pt", map_location=_torch.device('cpu'))


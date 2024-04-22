import shutil
import site
import os

def get_site_packages_path():
    return site.getsitepackages()[0]

def copy_files(files_to_copy):
    site_packages_path = get_site_packages_path()
    for src, dest in files_to_copy:
        full_dest_path = os.path.join(site_packages_path, dest)
        shutil.copyfile(src, full_dest_path)
        print(f"Copied {src} to {full_dest_path}")

def move_file(src, dest):
    site_packages_path = get_site_packages_path()
    full_dest_path = os.path.join(site_packages_path, dest)
    shutil.move(src, full_dest_path)
    print(f"Moved {src} to {full_dest_path}")

if __name__ == "__main__":
    files_to_copy = [
    ("./env_utils/Changed_file/att.yaml", "marllib/marl/models/configs/att.yaml"),
    ("./env_utils/Changed_file/centralized_critic.py", "marllib/marl/algos/utils/centralized_critic.py"),
    ("./env_utils/Changed_file/mappo.py", "marllib/marl/algos/core/CC/mappo.py"),
    ("./env_utils/Changed_file/matrpo.py", "marllib/marl/algos/core/CC/matrpo.py"),
    ("./env_utils/Changed_file/trust_regions.py", "marllib/marl/algos/utils/trust_regions.py"),
    ("./env_utils/Changed_file/mixing_critic.py", "marllib/marl/algos/utils/mixing_critic.py"),
    ("./env_utils/Changed_file/torch_ops.py", "ray/rllib/utils/torch_ops.py")
    ]

    copy_files(files_to_copy)
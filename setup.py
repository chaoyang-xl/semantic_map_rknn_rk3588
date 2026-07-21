import os
from glob import glob

from setuptools import find_packages, setup


package_name = "semantic_map_rknn"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md", "LICENSE"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
        (os.path.join("share", package_name, "docs"), glob("docs/*.md")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="weiyu",
    maintainer_email="1074793744@qq.com",
    description="RK3588 YOLO-World and MobileSAM semantic object mapping.",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "yolo_world_rknn_node = semantic_map_rknn.yolo_world_node:main",
            "sam_rknn_projector_node = semantic_map_rknn.sam_projector_node:main",
            "object_fusion_node = semantic_map_rknn.object_fusion_node:main",
        ],
    },
)

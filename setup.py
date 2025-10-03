from setuptools import setup, find_packages

package_name = "vlm_gist"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(include=[f"{package_name}*"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Bastian Pätzold",
    maintainer_email="paetzold@ais.uni-bonn.de",
    description="Official implementation of: 'Leveraging Vision-Language Models for Open-Vocabulary Instance Segmentation and Tracking' "
                "by Pätzold, Nogga & Behnke. IEEE Robotics and Automation Letters. 2025.",
    license_files=["LICENSE"],
    entry_points={
        "console_scripts": [
            f"fiftyone_download = {package_name}.fiftyone.fiftyone_download:main",
            f"fiftyone_label = {package_name}.fiftyone.fiftyone_label:main",
            f"fiftyone_edit = {package_name}.fiftyone.fiftyone_edit:main",
            f"fiftyone_describe = {package_name}.fiftyone.fiftyone_describe:main",
            f"fiftyone_detect = {package_name}.fiftyone.fiftyone_detect:main",
            f"fiftyone_validate = {package_name}.fiftyone.fiftyone_validate:main",
            f"fiftyone_baseline = {package_name}.fiftyone.fiftyone_baseline:main",
            f"fiftyone_label_match = {package_name}.fiftyone.fiftyone_label_match:main",
            f"fiftyone_eval = {package_name}.fiftyone.fiftyone_eval:main",
            f"fiftyone_show = {package_name}.fiftyone.fiftyone_show:main"
        ]
    }
)

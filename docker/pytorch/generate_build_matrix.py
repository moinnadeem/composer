# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""
Helper script to generate the build_matrix.yaml

Note: this script requires tabulate. Run `pip install tabulate` if not installed

To run: python generate_build_matrix.py

Also update the `README.md` in the docker folder with the resulting table.
"""

import itertools
import os
import sys

import packaging.version
import tabulate
import yaml


def get_pytorch_version(python_version: str):
    if python_version == "3.9":
        return "1.11.0"
    if python_version in "3.8":
        return "1.10.2"
    if python_version == "3.7":
        return "1.9.1"
    raise ValueError(f"Invalid python version: {python_version}")


def get_torchvision_version(pytorch_version: str):
    if pytorch_version == "1.10.2":
        return "0.11.3"
    if pytorch_version == "1.11.0":
        return "0.12.0"
    if pytorch_version == "1.9.1":
        return "0.10.1"
    raise ValueError(f"Invalid pytorch_version: {pytorch_version}")


def get_base_image(cuda_version: str):
    if cuda_version == "cpu":
        return "ubuntu:20.04"
    if cuda_version == "11.1.1":
        return "nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04"
    if cuda_version == "11.3.1":
        return "nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04"
    raise ValueError(f"Invalid cuda_version: {cuda_version}")


def get_cuda_version(pytorch_version: str, use_cuda: bool):
    if not use_cuda:
        return "cpu"
    if pytorch_version == "1.9.1":
        return "11.1.1"
    if pytorch_version in ("1.10.2", "1.11.0"):
        return "11.3.1"
    raise ValueError(f"Invalid pytorch_version: {str}")


def get_cuda_version_tag(cuda_version: str):
    if cuda_version == "cpu":
        return "cpu"
    if cuda_version == "11.1.1":
        return "cu111"
    if cuda_version == "11.3.1":
        return "cu113"
    raise ValueError(f"Invalid cuda_version: {cuda_version}")


def get_tags(python_version: str, pytorch_version: str, cuda_version_tag: str, cuda_version: str, stage: str):
    if stage == "pytorch_stage":
        base_image_name = "mosaicml/pytorch"
    elif stage == "vision_stage":
        base_image_name = "mosaicml/pytorch_vision"
    else:
        raise ValueError(f"Invalid stage: {stage}")
    tags = [f"{base_image_name}:{pytorch_version}_{cuda_version_tag}-python{python_version}-ubuntu20.04"]

    if python_version == "3.9":
        if cuda_version == "cpu":
            tags.append(f"{base_image_name}:latest_cpu")
        else:
            tags.append(f"{base_image_name}:latest")

    return tags


def main():
    python_versions = ["3.7", "3.8", "3.9"]
    cuda_options = [True, False]
    stages = ["pytorch_stage", "vision_stage"]

    entries = []

    for product in itertools.product(python_versions, cuda_options, stages):
        python_version, use_cuda, stage = product

        pytorch_version = get_pytorch_version(python_version)
        cuda_version = get_cuda_version(pytorch_version=pytorch_version, use_cuda=use_cuda)
        cuda_version_tag = get_cuda_version_tag(cuda_version)

        entry = {
            "BASE_IMAGE":
                get_base_image(cuda_version),
            "CUDA_VERSION":
                cuda_version,
            "CUDA_VERSION_TAG":
                cuda_version_tag,
            "LINUX_DISTRO":
                "ubuntu2004",
            "PYTHON_VERSION":
                python_version,
            "PYTORCH_VERSION":
                pytorch_version,
            "TARGET":
                stage,
            "TORCHVISION_VERSION":
                get_torchvision_version(pytorch_version),
            "TAGS":
                get_tags(
                    python_version=python_version,
                    pytorch_version=pytorch_version,
                    cuda_version_tag=cuda_version_tag,
                    cuda_version=cuda_version,
                    stage=stage,
                ),
        }

        if stage == "vision_stage":
            if python_version != "3.9":
                continue
            # only build the vision image on python 3.9
            entry["MMCV_TORCH_VERSION"] = f"torch{pytorch_version}"
            entry["MMCV_VERSION"] = "1.4.8"

        if cuda_version != "cpu":
            # Install the Mellanox drivers in the cuda images
            entry['MOFED_OS_VERSION'] = "ubuntu20.04-x86_64"
            entry['MOFED_VERSION'] = "5.5-1.0.3.2"

        entries.append(entry)

    with open(os.path.join(os.path.dirname(__name__), "build_matrix.yaml"), "w+") as f:
        f.write("# This file is automatically generated by generate_build_matrix.py. DO NOT EDIT!\n")
        yaml.safe_dump(entries, f)

    # Also print the table for the readme
    headers = ["Linux Distro", "Flavor", "PyTorch Version", "CUDA Version", "Python Version", "Docker Tags"]

    table = []
    for entry in entries:
        table.append([
            "Ubuntu 20.04",  # Linux distro
            "Base" if entry["TARGET"] == "pytorch_stage" else "Vision",  # Flavor
            entry["PYTORCH_VERSION"],  # Pytorch version
            entry["CUDA_VERSION"],  # Cuda version
            entry["PYTHON_VERSION"],  # Python version,
            ", ".join(reversed(list(f"`{x}`" for x in entry["TAGS"]))),  # Docker tags
        ])

    table.sort(key=lambda x: x[3])  # cuda version
    table.sort(key=lambda x: packaging.version.parse(x[4]), reverse=True)  # python version
    table.sort(key=lambda x: packaging.version.parse(x[2]), reverse=True)  # pytorch version
    table.sort(key=lambda x: x[1])  # flavor

    with open(os.path.join(os.path.dirname(__name__), "..", "README.md"), "r") as f:
        contents = f.read()

    begin_table_tag = "<!-- BEGIN_BUILD_MATRIX -->"
    end_table_tag = "<!-- END_BUILD_MATRIX -->"

    pre = contents.split(begin_table_tag)[0]
    post = contents.split(end_table_tag)[1]
    table = tabulate.tabulate(table, headers, tablefmt="github", floatfmt="", disable_numparse=True)
    new_readme = f"{pre}{begin_table_tag}\n{table}\n{end_table_tag}{post}"

    with open(os.path.join(os.path.dirname(__name__), "..", "README.md"), "w") as f:
        f.write(new_readme)

    print("Successfully updated `build_matrix.yaml` and `README.md`.", file=sys.stderr)


if __name__ == "__main__":
    main()

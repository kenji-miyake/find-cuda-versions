import argparse
import logging

import argcomplete  # pyright: ignore [reportMissingTypeStubs, reportUnknownVariableType]
from bs4 import BeautifulSoup
import requests

from find_cuda_versions import __version__
from find_cuda_versions import parser

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


def get_versions(os: str, arch: str):
    page_url = f"https://developer.download.nvidia.com/compute/cuda/repos/{os}/{arch}/"

    session = requests.Session()
    res = session.get(page_url)
    page = BeautifulSoup(res.content, "html.parser")

    cudnn_versions = parser.get_cudnn_versions(page)
    nvinfer_versions = parser.get_nvinfer_versions(page)
    tensorrt_versions = parser.get_tensorrt_versions(page)

    return cudnn_versions, nvinfer_versions, tensorrt_versions


def show_versions(cuda: str, os: str, arch_list: list[str]):
    versions: dict[str, dict[str, list[str]]] = {}
    arch: str
    for arch in arch_list:
        cudnn_versions, nvinfer_versions, tensorrt_versions = get_versions(os, arch)
        cuda_str = f"cuda{cuda}"

        versions[arch] = {}
        versions[arch]["cudnn"] = [v for v in cudnn_versions if cuda_str in v]
        versions[arch]["nvinfer"] = [v for v in nvinfer_versions if cuda_str in v]
        versions[arch]["tensorrt"] = [v for v in tensorrt_versions if cuda_str in v]

    for arch in arch_list:
        print(f"[cudnn {arch}]")
        for v in versions[arch]["cudnn"]:
            print(v)
        print("")

    for arch in arch_list:
        print(f"[nvinfer {arch}]")
        for v in versions[arch]["nvinfer"]:
            print(v)
        print("")

    for arch in arch_list:
        print(f"[tensorrt {arch}]")
        for v in versions[arch]["tensorrt"]:
            print(v)
        print("")


def main():
    def get_log_level_names():
        # TODO(Kenji Miyake): Use getLevelNamesMapping() in Python 3.11
        return logging._nameToLevel.keys()  # pyright: ignore [reportPrivateUsage]

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=str, default="11.7")
    parser.add_argument("--os", type=str, default="ubuntu2204")
    parser.add_argument("--arch", type=str, nargs="+", default=["x86_64", "sbsa"])
    parser.add_argument("--log-level", type=str, default="INFO", choices=get_log_level_names())
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()
    argcomplete.autocomplete(parser)  # pyright: ignore

    handler = logging.StreamHandler()
    handler.setLevel(args.log_level)
    logging.basicConfig(handlers=[handler])

    show_versions(args.cuda, args.os, args.arch)


if __name__ == "__main__":
    main()

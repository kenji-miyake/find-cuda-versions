import argparse
import logging

import argcomplete
from bs4 import BeautifulSoup
import requests

from find_cuda_versions import __version__
from find_cuda_versions import parser

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


def get_versions(os, arch):
    page_url = f"https://developer.download.nvidia.com/compute/cuda/repos/{os}/{arch}/"

    session = requests.Session()
    res = session.get(page_url)
    page = BeautifulSoup(res.content, "html.parser")

    cudnn_versions = parser.get_cudnn_versions(page)
    tensorrt_versions = parser.get_tensorrt_versions(page)

    return cudnn_versions, tensorrt_versions


def show_versions(args):
    versions = {}
    for arch in args.arch:
        cudnn_versions, tensorrt_versions = get_versions(args.os, arch)
        cuda_str = f"cuda{args.cuda}"

        versions[arch] = {}
        versions[arch]["cudnn"] = [v for v in cudnn_versions if cuda_str in v]
        versions[arch]["tensorrt"] = [v for v in tensorrt_versions if cuda_str in v]

    for arch in args.arch:
        print(f"[cudnn {arch}]")
        for v in versions[arch]["cudnn"]:
            print(v)
        print("")

    for arch in args.arch:
        print(f"[tensorrt {arch}]")
        for v in versions[arch]["tensorrt"]:
            print(v)
        print("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default="11.7")
    parser.add_argument("--os", default="ubuntu2204")
    parser.add_argument("--arch", nargs="+", default=["x86_64", "sbsa"])
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--log-level", default="INFO", choices=logging._nameToLevel.keys())
    args = parser.parse_args()
    argcomplete.autocomplete(parser)

    handler = logging.StreamHandler()
    handler.setLevel(args.log_level)
    logging.basicConfig(handlers=[handler])

    show_versions(args)


if __name__ == "__main__":
    main()

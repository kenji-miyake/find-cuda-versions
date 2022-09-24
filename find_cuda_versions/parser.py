import logging
import re

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


def trim(package_name):
    m = re.match(r"\w+_(.+)_\w+\.deb", package_name)
    return m[1]


def get_package_names(page, pattern):
    spans = page.find_all("span", {"class": "file"})
    return [t for span in spans if re.match(pattern, t := span.text)]


def get_cudnn_versions(page: BeautifulSoup):
    return [trim(name) for name in get_package_names(page, r"^libcudnn\d+_")]


def get_nvinfer_versions(page: BeautifulSoup):
    return [trim(name) for name in get_package_names(page, r"^libnvinfer\d+_")]


def get_tensorrt_versions(page: BeautifulSoup):
    return [trim(name) for name in get_package_names(page, r"^tensorrt+_")]

from pathlib import Path

from bs4 import BeautifulSoup

from find_cuda_versions import parser


def get_page(file_path: Path):
    return BeautifulSoup(file_path.read_text(encoding="utf8"), "html.parser")


def test_ubuntu2004__x86_64(shared_datadir: Path):
    page = get_page(shared_datadir / "ubuntu2004-x86_64.html")
    cudnn_versions = parser.get_cudnn_versions(page)
    assert cudnn_versions == [
        "8.0.5.39-1+cuda11.0",
        "8.0.5.39-1+cuda11.1",
        "8.1.0.77-1+cuda11.2",
        "8.1.1.33-1+cuda11.2",
        "8.2.0.53-1+cuda11.3",
        "8.2.1.32-1+cuda11.3",
        "8.2.2.26-1+cuda11.4",
        "8.2.4.15-1+cuda11.4",
        "8.3.0.98-1+cuda11.5",
        "8.3.1.22-1+cuda11.5",
        "8.3.2.44-1+cuda11.5",
        "8.3.3.40-1+cuda11.5",
        "8.4.0.27-1+cuda11.6",
        "8.4.1.50-1+cuda11.6",
    ]

    nvinfer_versions = parser.get_nvinfer_versions(page)
    assert nvinfer_versions == [
        "8.0.0-1+cuda11.0",
        "8.0.0-1+cuda11.3",
        "8.0.1-1+cuda11.3",
        "8.0.3-1+cuda11.3",
        "8.2.0-1+cuda11.4",
        "8.2.1-1+cuda11.4",
        "8.2.2-1+cuda11.4",
        "8.2.3-1+cuda11.4",
        "8.2.4-1+cuda11.4",
        "8.2.5-1+cuda11.4",
        "8.4.1-1+cuda11.6",
        "8.4.2-1+cuda11.6",
    ]

    tensorrt_versions = parser.get_tensorrt_versions(page)
    assert tensorrt_versions == [
        "8.4.1.5-1+cuda11.6",
        "8.4.2.4-1+cuda11.6",
    ]


def test_ubuntu2004__sbsa(shared_datadir: Path):
    page = get_page(shared_datadir / "ubuntu2004-sbsa.html")

    cudnn_versions = parser.get_cudnn_versions(page)
    assert cudnn_versions == [
        "8.1.0.77-1+cuda11.2",
        "8.1.1.33-1+cuda11.2",
        "8.2.0.53-1+cuda11.3",
        "8.2.1.32-1+cuda11.3",
        "8.2.2.26-1+cuda11.4",
        "8.2.4.12-1+cuda11.4",
        "8.3.0.98-1+cuda11.5",
        "8.3.1.22-1+cuda11.5",
        "8.3.2.44-1+cuda11.5",
        "8.3.3.40-1+cuda11.5",
        "8.4.0.27-1+cuda11.6",
        "8.4.1.50-1+cuda11.6",
    ]

    nvinfer_versions = parser.get_nvinfer_versions(page)
    assert nvinfer_versions == [
        "8.0.1-1+cuda11.3",
        "8.0.2-1+cuda11.3",
        "8.2.0-1+cuda11.4",
        "8.2.1-1+cuda11.4",
        "8.2.2-1+cuda11.4",
        "8.2.3-1+cuda11.4",
        "8.2.4-1+cuda11.4",
        "8.2.5-1+cuda11.4",
        "8.4.1-1+cuda11.6",
        "8.4.2-1+cuda11.6",
    ]

    tensorrt_versions = parser.get_tensorrt_versions(page)
    assert tensorrt_versions == [
        "8.4.1.5-1+cuda11.6",
        "8.4.2.4-1+cuda11.6",
    ]

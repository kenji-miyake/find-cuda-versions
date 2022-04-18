# find-cuda-versions

## Prerequisite

- Python 3.10

## Usage

```bash
$ pip install git+https://github.com/kenji-miyake/find-cuda-versions.git
$ find-cuda-versions --cuda 11.4 --os ubuntu2004 --arch x86_64 sbsa
[cudnn x86_64]
8.2.2.26-1+cuda11.4
8.2.4.15-1+cuda11.4

[cudnn sbsa]
8.2.2.26-1+cuda11.4
8.2.4.12-1+cuda11.4

[tensorrt x86_64]
8.2.0-1+cuda11.4
8.2.1-1+cuda11.4
8.2.2-1+cuda11.4
8.2.3-1+cuda11.4
8.2.4-1+cuda11.4

[tensorrt sbsa]
8.2.0-1+cuda11.4
8.2.1-1+cuda11.4
8.2.2-1+cuda11.4
8.2.3-1+cuda11.4
8.2.4-1+cuda11.4
```

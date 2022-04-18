# Surface Reconstruction Benchmark from point clouds: A Survey and a benchmark

![](teaser.png)

This repository contains the official experiment implementation to the the paper `Surface Reconstruction Benchmark from point clouds: A Survey and a benchmark`.

[[Paper]]() [[Dataset]](https://mailscuteducn-my.sharepoint.com/:f:/g/personal/201730254453_mail_scut_edu_cn/Em-Xehw0uHlMkj1XHRZWbT4BzWg4ssEV4aebZu6pOly2ew) [[Project Page]](https://Gorilla-Lab-SCUT.github.io/SurfaceReconstructionBenchmark)


## File Organization

```
SurfaceRecBenchamrk
    ├──data                     # Put data here
    |   ├──synthetic_object
    |   ├──synthetic_scene
    |   └──real_object 
    ├──build_dataset            # Methods to build our benchmark datasets
    |   ├──scan_and_synthesis
    |   └──preprocessing
    ├──reconstruction           # Reconstruction algorithms
    |
    └──metrics                  # Methods to evaluate the reconstructed surfaces 
        ├──vanilla_metric
        └──neural_metric
```

## How to use

```
    git clone https://github.com/Huang-ZhangJin/SurfaceRecBenchmark.git
    git submodule update --init --recursive
```
There is a `README.md` file in each subfolder that describes how to use each script.

### 1. Data
Download the [Dataset]() and put it in the `data` folder
- To synthetic point clouds yourself:
    - To perform object-level synthetic scanning, please follow [instructions](build_dataset\scan_and_synthesis\object_level\README.md)
    - To perform scene-level synthetic scanning, please follow [instructions](build_dataset\scan_and_synthesis\scene_level\README.md)
- Or use the point clouds provided by us
- To pre-processing the point clouds, please follow [instructions](build_dataset\preprocessing\README.md)

### 2. Reconstruction Methods
- Some surface reconstruction methods used in our paper
- Optionally, see our [3D Surface Reconstruction Library](https://github.com/Gorilla-Lab-SCUT/3DRecon)

### 3. Evaluation Metrics
To use the following evaluation metrics, please follow [instructions](metrics\README.md)
- Vanilla metrics
    - Chamfer Distance (CD)
    - F-score
    - Normal Consistency Score (NCS)
- Neural metrics
    - Neural Feature Similarity (NFS)

## Citation
If you find our work useful in your research, please consider citing:

    @incollection{xxx,
        author = {xxx},
        booktitle = {xxx},
        title = {xxx},
        year = {2021}
    }
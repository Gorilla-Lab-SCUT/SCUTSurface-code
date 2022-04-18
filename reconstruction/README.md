
# Reconstruction Methods

Run the following command to get the reconstruction methods:
```
    git submodule update --init --recursive
```

## [SAL: Sign Agnostic Learning of Shapes From Raw Data](https://github.com/matanatz/SAL)

## [IGR: Implicit Geometric Regularization for Learning Shapes](https://github.com/amosgropp/IGR)

## [Local Implicit Grid Representations for 3D Scenes (Tensorflow)](https://github.com/tensorflow/graphics/tree/master/tensorflow_graphics/projects/local_implicit_grid)

## [Local Implicit Grid Representations for 3D Scenes (Pytorch)](https://github.com/Gorilla-Lab-SCUT/3DRecon/tree/main/projects/LIG)

## [Points2Surf: Learning Implicit Surfaces from Point Clouds](https://github.com/ErlerPhilipp/points2surf)

## [Learning Delaunay Surface Elements for Mesh Reconstruction](https://github.com/mrakotosaon/dse-meshing)

## [DeepMLS: Deep Implicit Moving Least-Squares Functions for 3D Reconstruction](https://github.com/Andy97/DeepMLS)

## [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://github.com/facebookresearch/DeepSDF)

## [Occupancy Networks - Learning 3D Reconstruction in Function Space](https://github.com/autonomousvision/occupancy_networks)

## [3D Surface Reconstruction Library](https://github.com/Gorilla-Lab-SCUT/3DRecon)

#

## Resize the output to the original
Note that, some reconstruction algorithms need to resize the input before performing reconstruction.
You may need to scale it back to the input size.
```
    python resize.py --help
```
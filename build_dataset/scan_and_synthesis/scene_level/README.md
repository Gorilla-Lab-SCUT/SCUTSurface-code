
# Scene-level Synthetic-scanned Dataset

## Source surfaces
The source surfaces are obtained from three databases, including the SceneNet, the 3D-FRONT and the Replica. If you want to use the databases, please cite their works. 

<!-- We provide download links for surfaces of our choice xxx.  -->

## Generate the scanning viewpoints
```
python get_viewpoints.py --In_Dir MESH_DIRECTORY/meshes --Out_Dir MESH_DIRECTORY/viewpoints
```

## Use BlenSor to perform the synthetical scanning

##### The BlenSor software could be downloaded [here](https://www.blensor.org/pages/downloads.html)

<!-- ### Windows:
```
xxx/blender.exe --background -P camera_scan_vps.py xxx/meshes/xxx_scene.ply xxx/viewpoints/xxx_scene.xyz
```

### Linux: -->
```
Blensor-x64.AppImage --background -P camera_scan_vps.py MESH_DIRECTORY/meshes/xxx_scene.ply MESH_DIRECTORY/viewpoints/xxx_scene.xyz
```

After running the script, two folders would be generated in the `MESH_DIRECTORY` directory (i.e. `MESH_DIRECTORY/Nonuniform` and `MESH_DIRECTORY/Noise`) 

## Final synthesis
```
python synthetic_dataset_scene.py --In_Nonuniform_Dir MESH_DIRECTORY/Nonuniform --In_Noise_Dir MESH_DIRECTORY/Noise --Out_Noise_Normal_Dir MESH_DIRECTORY/Noise_Normal --Out_Dir MESH_DIRECTORY/Out --num_worker 8
```
#### The final output could be found in `MESH_DIRECTORY/Out`
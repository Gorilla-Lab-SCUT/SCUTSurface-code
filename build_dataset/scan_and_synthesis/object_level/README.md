
# Object-level Synthetic-scanned Dataset

## Source surfaces
The source surfaces are obtained from four databases, including the Thingi10k, the 3DNet Cat200 subset, the ABC and the Three D Scans. If you want to use the databases, please cite their works. 

<!-- We provide download links for surfaces of our choice `MESH_DIRECTORY`.  -->

## Generate the scanning list of objects name for batch scanning
```
python get_viewpoints.py --In_Dir MESH_DIRECTORY/meshes [--split 8]
```
Optionally, if you want to split the file list, just set a positive integer number after `--split`, and `-1` means not split. After running the script, a list of scanned file names will be generated (i.e. `blensor_scan_list.txt`).



## Use BlenSor to perform the synthetical scanning

##### The BlenSor software could be downloaded [here](https://www.blensor.org/pages/downloads.html)

```
Blensor-x64.AppImage --background -P camera_rotate_scan_batch.py blensor_scan_list.txt
```

After running the script, two subfolders will be generated in the `MESH_DIRECTORY` directory (i.e. `MESH_DIRECTORY/Nonuniform` and `MESH_DIRECTORY/Noise`) 


## Estimate normals for the synthetic Point Clouds
```
python estimate_normal_batch.py --In_Nonuniform_Dir MESH_DIRECTORY/Nonuniform --Out_Nonuniform_Normal_Dir MESH_DIRECTORY/Nonuniform_normal --In_Noise_Dir MESH_DIRECTORY/Noise --Out_Noise_Normal_Dir/Noise_Normal [--num_worker 8]
```
Optionally, set `--num_worker` to run with multithreading.

## Synthesis different challenges 
To generate data with different challenges, run the below scripts.
Note that, `Uniform` here is the same as `Perfect scanning` in the paper.
```
python synthetic_dataset_batch.py --Base_dir_in MESH_DIRECTORY -w Uniform --num_worker 8

python synthetic_dataset_batch.py --Base_dir_in MESH_DIRECTORY -w Nonuniform --num_worker 8

python synthetic_dataset_batch.py --Base_dir_in MESH_DIRECTORY -w Noise -sl 1 -an 0.001 --num_worker 8
python synthetic_dataset_batch.py --Base_dir_in MESH_DIRECTORY -w Noise -sl 2 -an 0.003 --num_worker 8
python synthetic_dataset_batch.py --Base_dir_in MESH_DIRECTORY -w Noise -sl 3 -an 0.006 --num_worker 8

python synthetic_dataset_batch.py --Base_dir_in MESH_DIRECTORY -w Outlier -sl 1 -on 0.001 -oi 0.1 --num_worker 8
python synthetic_dataset_batch.py --Base_dir_in MESH_DIRECTORY -w Outlier -sl 2 -on 0.003 -oi 0.1 --num_worker 8
python synthetic_dataset_batch.py --Base_dir_in MESH_DIRECTORY -w Outlier -sl 3 -on 0.006 -oi 0.1 --num_worker 8

python synthetic_dataset_batch.py --Base_dir_in MESH_DIRECTORY -w MissingData -sl 1 --num_worker 8 
python synthetic_dataset_batch.py --Base_dir_in MESH_DIRECTORY -w MissingData -sl 2 --num_worker 8
python synthetic_dataset_batch.py --Base_dir_in MESH_DIRECTORY -w MissingData -sl 3 --num_worker 8

python synthetic_dataset_batch.py --Base_dir_in MESH_DIRECTORY -w Misalignment -sl 1 -mi 0.005 -mma 0.5 --num_worker 8
python synthetic_dataset_batch.py --Base_dir_in MESH_DIRECTORY -w Misalignment -sl 2 -mi 0.01 -mma 1 --num_worker 8
python synthetic_dataset_batch.py --Base_dir_in MESH_DIRECTORY -w Misalignment -sl 3 -mi 0.02 -mma 2 --num_worker 8
```
The final output could be found in `MESH_DIRECTORY/Out`, and in TXT format.

# Pre-Processing

## Step 1: Outlier Removal
The [Point Cloud Library](https://pointclouds.org/) (version > 1.8) is required for this function. 
Follow the instructions below to install PCL (tested on Ubuntu 20.04 LTS):
```
    sudo apt-get update
    sudo apt-get install cmake cmake-gui 
    sudo apt-get install libflann-dev
    sudo apt-get install libeigen3-dev
    sudo apt-get install libboost-all-dev
    sudo apt-get install libvtk6-dev
    sudo apt-get install libpcap-dev
    cd ~ && mkdir PCL && cd PCL
    wget https://github.com/PointCloudLibrary/pcl/releases/download/pcl-1.11.1-rc2/source.tar.gz
    tar -zxvf source.tar.gz
    mv pcl pcl-1.11.1 && cd pcl-1.11.1 && mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j8
    sudo make -j8 install
```
Also, official [installation documentation](https://github.com/PointCloudLibrary/pcl) is available for reference.

When the environment is ready, please run the compile script:
```
    cd src/PCL
    bash install.sh
```
After that, a bin file `PCL_Preprocessing_Outlier_Removeo` could be found in `bin` folder.

Then modify `config.yaml` to specify the path to the data, and run 
```
    python batch_pcl_remove_outliers.py -conf config.yaml
```



## Step 2: De-noising
The [Computational Geometry Algorithms Library (CGAL)](https://www.cgal.org/index.html) (version > 5.1.2) is required for this function. Follow the instructions below to install CGAL (tested on Ubuntu 20.04 LTS):
```
    sudo sudo apt-get update
    sudo apt-get install cmake cmake-gui
    sudo apt-get install libgmp-dev libmpfr-dev

        # Note that, the above command may be fail. 
        # If failed, try to manual install the packages
            cd ~ && mkdir GMP && cd GMP
            wget https://ftp.gnu.org/gnu/gmp/gmp-6.2.1.tar.xz
            tar -xvJf gmp-6.2.1.tar.xz
            cd gmp-6.2.1
            ./configure
            make
            sudo make install

            cd ~ && mkdir MPFR && cd MPFR
            wget https://www.mpfr.org/mpfr-current/mpfr-4.1.0.tar.xz
            tar -xvJf mpfr-4.1.0.tar.xz
            cd mpfr-4.1.0
            ./configure
            make
            sudo make install

    cd ~ && mkdir CGAL && cd CGAL
    wget https://github.com/CGAL/cgal/releases/download/v5.3/CGAL-5.3-examples.tar.xz
    wget https://github.com/CGAL/cgal/releases/download/v5.3/CGAL-5.3.tar.xz
    tar -xvJf CGAL-5.3-examples.tar.xz
    tar -xvJf CGAL-5.3.tar.xz
    cd CGAL-5.3 && mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make
    sudo make install
```
When the environment is ready, please run the compile script:
```
    cd src/CGAL_Preprocess
    bash install.sh
```
After that, a bin file `cgal_Preprocessing` could be found in `bin` folder.

Then modify `config.yaml` to specify the path to the data, and run 
```
    python batch_cgal_prepocessing_smooth.py -conf config.yaml
```


## Step 3: Re-sampling
We use Farthest Point Sampling here, and Pytorch with CUDA environment is required here.

Modify `config.yaml` to specify the path to the data, and run 
```
    python batch_FPS_torch.py -conf config.yaml
```


## Step 4: Convert TXT to PLY format
```
    python txt2ply.py --In_txt_Dir INPUT_TXT_FILE_DIRECTORY --Out_ply_Dir OUTPUT_PLY_FILE_DIRECTORY
```
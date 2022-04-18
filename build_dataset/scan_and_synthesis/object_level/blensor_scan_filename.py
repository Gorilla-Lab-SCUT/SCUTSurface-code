import os
import numpy as np
import argparse
listfiles = lambda root : [os.path.join(base, f) for base, _, files in os.walk(root) if files for f in files]

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='!!!Get the meshes name list!!!')
    parser.add_argument('--In_Dir', '-id', type=str, default='../../../data/synthetic_object/meshes')
    parser.add_argument('--split', '-sp', type=int, default=-1)

    args = parser.parse_args()
    print(args)
    In_dir = os.path.abspath(args.In_Dir)
    splict_len = args.split

    # Step 1: Get all filename:
    if 1:
        In_dir = os.path.abspath(args.In_Dir)

        files = listfiles(In_dir)
        print(files)
        with open("blensor_scan_list.txt","w+") as f:
            for fileone in files:
                # print(fileone)
                if ".ply" in fileone:
                    f.write(fileone)
                    f.write('\n')
        f.close()

    # Step 2: Split filename:
    if splict_len != -1:
        sf = open("./blensor_scan_list.txt","r")
        lines = sf.readlines()
        print(lines)
        choice = np.arange(len(lines))
        np.random.shuffle(choice)
        choice_len = len(choice) // splict_len
        print('Each split file len is:', choice_len)
        for i in range(splict_len):
            if i == splict_len-1:
                Idx = choice[(splict_len-1)*choice_len:]
            else:
                Idx = choice[i*choice_len:(i+1)*choice_len]
            print(Idx)
            with open("./blensor_scan_list%d.txt"%i,"w+") as tmp:
                for idx in Idx:
                    tmp.write(lines[idx])
            tmp.close()
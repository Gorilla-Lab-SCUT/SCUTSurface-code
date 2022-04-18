import os
from tqdm import tqdm
from utils import load_config, ssubprocess
import concurrent.futures
import argparse

def main(cfg):
    main = cfg['pcl_outlier']['main']
    in_dir = cfg['pcl_outlier']['in_dir']
    out_dir = cfg['pcl_outlier']['out_dir']

    os.makedirs(out_dir,exist_ok=True)

    # StatisticalOutlierRemoval
    mean_k = cfg['pcl_outlier']['mean_k']
    std_dev_mul = cfg['pcl_outlier']['std_dev_mul']

    listfiles = lambda root : [os.path.join(base, f) for base, _, files in os.walk(root) if files for f in files]
    files = listfiles(in_dir)

    cmds = []
    for fileone in tqdm(files):
        if fileone.split('.')[-1] == 'txt':
            outfile = fileone.replace(in_dir, out_dir).replace('txt','txt')
            exe = str(main+' '+str(fileone)+' '+str(outfile)+' -method statistical ' + ' -mean_k  '+str(mean_k)+' -std_dev_mul ' +str(std_dev_mul))
            os.makedirs(os.path.dirname(outfile),exist_ok=True)
            cmds.append(exe)
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(cfg['num_worker'])) as executor:
        for i in range(len(cmds)):
            string = str(i) +'/'+ str(len(cmds))
            executor.submit(ssubprocess, string,cmds[i])
        executor.shutdown()
    print("##############################################################################################")
    print("Remove Outlier finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='!!!Reomve Outliers!!!')
    parser.add_argument('--configure', '-conf', type=str, default='config_example.yaml', help='the config yaml file')
    args = parser.parse_args()
    cfg = load_config(str(args.configure))
    main(cfg)
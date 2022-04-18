import yaml
import pandas as pd
import subprocess
import multiprocessing

###################################################
#multiprocess
###################################################
def mp_worker(call):
    """
    Small function that starts a new thread with a system call. Used for thread pooling.
    :param call:
    :return:
    """
    call = call.split(' ')
    verbose = call[-1] == '--verbose'
    if verbose:
        call = call[:-1]
        subprocess.run(call)
    else:
        #subprocess.run(call, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # suppress outputs
        subprocess.run(call, stdout=subprocess.DEVNULL)

def start_process_pool(worker_function, parameters, num_processes, timeout=None):
    if len(parameters) > 0:
        if num_processes <= 1:
            print('Running loop for {} with {} calls on {} workers'.format(
                str(worker_function), len(parameters), num_processes))
            results = []
            for c in parameters:
                results.append(worker_function(*c))
            return results
        print('Running loop for {} with {} calls on {} subprocess workers'.format(
            str(worker_function), len(parameters), num_processes))
        with multiprocessing.Pool(processes=num_processes, maxtasksperchild=1) as pool:
            results = pool.starmap(worker_function, parameters)
            return results
    else:
        return None


###################################################
#subprocess
###################################################
def ssubprocess(str,command):
    print(str,'---->',command)
    print("##############################################################################################")
    subproc = subprocess.Popen(command, stdout=subprocess.DEVNULL, shell=True)
    subproc.wait()


# General config
def load_config(path):
    ''' Loads config file.
    Args:  
        path (str): path to config file
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg = yaml.load(f)
    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.
    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used
    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

def writecsv(filename, data):
    pf = pd.DataFrame([data])
    pf = pf.fillna(-1)
    with open(filename, 'a+', encoding='utf-8') as f:
        try:
            pd.read_csv(filename, skiprows=1)
            flag =  0
        except Exception as e:
            flag = 1

        if flag:
            pf.to_csv(f, mode='a+')
        else:
            pf.to_csv(f, mode='a+',header=False)

def csv_removerepeat(filename):
    frame = pd.read_csv(filename, engine='python')
    data = frame.drop_duplicates(subset=['filename'], keep='last', inplace=False)
    data.to_csv(filename, encoding='utf-8')




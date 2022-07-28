import random
import os
import re
from pathlib import Path
from tqdm import tqdm 

def get_split_instances(path_to_data, test_ratio = 0.15):
    """
    Return a list of scans for testing and one for training
    Example return: ["CTP76_001_0000", "CTP17_002_0000", ...], [...]
    """
    all_identifiers = os.listdir(path_to_data)
    all_identifiers = [Path(identifier).stem for identifier in all_identifiers] # Strip out file extension.
    data_size = len(all_identifiers)
    
    if isinstance(test_ratio, int):
        num_test_instances = test_ratio
    elif isinstance(test_ratio, float):
        num_test_instances = int(data_size * test_ratio)
    else:
        raise TypeError
    num_train_instances = data_size - num_test_instances
    test_instances = random.choices(all_identifiers, k = num_test_instances) 
    train_instances = [file for file in all_identifiers if file not in test_instances]
    return train_instances, test_instances 

def split_data(path_to_data, identifier_pattern ,test_identifiers):
    """
    This function splits all data in a directory into two subdirectories "train" and "test" based on 
    some identifiers to recognize what instances belong to the test set.
    ----------
    Parameters
    test_identifiers: list of identifiers which distinguish test instances. Any file that has an identifier appear in its
    name will be put into the test set
    ----------
    """
    test_identifiers = set(test_identifiers) # It's faster to check for membership with set
    all_files = [f for f in os.listdir(path_to_data) if os.path.isfile(os.path.join(path_to_data, f))]
    for file in tqdm(all_files):
        try:
            scan_identifier = re.search(identifier_pattern, file).group(0) 
        except:
            print(f"cannot find pattern: {file}")
            break
        if scan_identifier in test_identifiers:
            try:
                os.rename(os.path.join(path_to_data, file),
                        os.path.join(path_to_data, 'test', file))
            except FileNotFoundError:
                os.mkdir(os.path.join(path_to_data, 'test'))
                os.rename(os.path.join(path_to_data, file),
                        os.path.join(path_to_data, 'test', file))
        else:
            try:
                os.rename(os.path.join(path_to_data, file),
                        os.path.join(path_to_data, 'train', file))
            except FileNotFoundError:
                os.mkdir(os.path.join(path_to_data, 'train'))
                os.rename(os.path.join(path_to_data, file),
                        os.path.join(path_to_data, 'train', file))

def split_patterns(path_to_data, path_to_destination, pattern_list):
    """
    This function move all files that have names fit in a pattern in a list into a file.
    ----------
    Parameters
    ----------
    path_to_data: Path to the directory containing files you want to move.
    path_to_destination: Path to the directory where you want to your files to go.
    pattern_list: A list containing all file identifiers. Any file that has its name matches a pattern in this list will be moved. 
    """
    all_files = [f for f in os.listdir(path_to_data) if os.path.isfile(os.path.join(path_to_data, f))]
    all_patterns = "|".join(list(map(lambda x: f"({x})", pattern_list)))
    
    for file in tqdm(all_files):
        if re.search(all_patterns, file):
            try:
                os.rename(os.path.join(path_to_data, file),
                        os.path.join(path_to_destination, file))
            except FileNotFoundError:
                os.mkdir(os.path.join(path_to_destination))
                os.rename(os.path.join(path_to_data, file),
                        os.path.join(path_to_destination, file))
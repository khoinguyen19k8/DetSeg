import re
import os
import pandas as pd

def k_split(a, n):
    """
    Given a list a with len(a) elements. Divide elements into n buckets with roughly equal size.
    Return a list with an inner list for each bucket.
    """
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def allocate_id_k_fold(path_to_data, path_to_csv, path_exclude ,id_pattern, k = 5):
    """
    Allocate CT scans id for k-fold validations.
    ----------
    Parameters
    ----------
    path_to_data: Path to where all the scans are located.
    path_to_csv: The csv file that you want to write the result in along with its path.
    path_exclude: The csv file contains Ids intended to be excluded from k-fold validation.
    id_pattern: regex expression to search for the unique id pattern in each file name.
    k: The number of folds
    """
    all_files = os.listdir(path_to_data)
    # Search for id_pattern in each file name. Make into a set for convenient set operations
    all_ids = set([re.search(id_pattern, file).group(0) for file in all_files if re.search(id_pattern, file)])

    # Instances which are excluded
    exclude = set(list(pd.read_csv(path_exclude).iloc[:,0])) 
    # Ids of all folds
    ids_all_folds = list(all_ids.difference(exclude))
    
    ids_k_folds = k_split(ids_all_folds, k)
    header = [f"fold_{i+1}" for i in range(k)]

    df = pd.DataFrame(columns = header)
    for fold, ids in zip(header, ids_k_folds):
        df.loc[:,fold] = pd.Series(ids)
    df.fillna('')
    df.to_csv(path_to_csv, index=False)







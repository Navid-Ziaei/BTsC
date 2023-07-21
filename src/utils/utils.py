import os
import pandas as pd

def append_dict_to_excel(filename, data_dict):
    """Append a dictionary to an Excel file.

    Args:
        filename (str): The path to the Excel file.
        data_dict (dict): The dictionary to be appended as rows in the Excel file.

    Raises:
        IOError: If the file cannot be accessed.

    Notes:
        - If the file already exists, the dictionary data will be appended without writing the headers.
        - If the file doesn't exist, a new file will be created with the dictionary data and headers.

    """
    df = pd.DataFrame(data_dict)

    if os.path.isfile(filename):
        # If it exists, append without writing the headers
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        # If it doesn't exist, create a new one with headers
        df.to_csv(filename, mode='w', header=True, index=False)


def print_directory_structure(startpath):
    """Print the directory structure recursively starting from a given path.

    Args:
        startpath (str): The path of the directory to print the structure from.

    """
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

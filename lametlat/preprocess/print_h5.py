from typing import Any, Callable
import h5py
import sys


def print_h5(group: h5py.Group, level=0, logger: Callable[[Any], Any] = print):
    """print h5 group to consol.

    Args:
        group (h5py.Group): _description_
        level (int, optional): _description_. Defaults to 0.
        logger (Callable[[Any], Any]): print like functions
    """
    if not isinstance(group, h5py.Group):
        logger(level * "\t", group)
        return
    else:
        for key in group.keys():
            logger(level * "\t" + key + ":")
            subgroup = group[key]
            print_h5(subgroup, level + 1)

# %%
if __name__=='__main__':
    if len(sys.argv)==1 or \
        (len(sys.argv)==2 and (sys.argv[1]=='-h' or sys.argv[1]=='--help')):
        print("""usage: \
to print the structure of HDF5 files.
\tshell>> python print_h5.py file1 [file2 ...]""")
        exit(0)

    for file in sys.argv[1:]:
        try:
            print(file+':')
            print_h5(h5py.File(file,'r'),1)
            pass
        except Exception:
            print('\tERROR')
            pass
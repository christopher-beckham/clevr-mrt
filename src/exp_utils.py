import json
import pprint
from haven import haven_utils as hu
import os

from .setup_logger import get_logger
logger = get_logger()

"""
def _filter_fn(val):
    if val == "true":
        return True
    elif val == "false":
        return False
    elif val == "none":
        return None
    else:
        return val
"""

def _filter_fn(val):
    if type(val) is list:
        val = [ _filter_fn(elem) for elem in val ]
        return val
    else:
        if val == "true":
            return True
        elif val == "false":
            return False
        elif val == "none":
            return None
        elif type(val) is str and val.startswith('$'):
            # assume it is an environment variable
            return os.environ[val[1:]]
        else:
            return val


def _traverse(dict_, keys, val):
    if len(keys) == 0:
        return
    else:
        # recurse
        if keys[0] not in dict_:
            if len(keys[1:]) == 0:
                dict_[keys[0]] = _filter_fn(val)
            else:
                dict_[keys[0]] = {}
        _traverse(dict_[keys[0]], keys[1:], val)

def unflatten(dict_):
    new_dict = {}
    for key, val in dict_.items():
        key_split = key.split(".")
        _traverse(new_dict, key_split, val)
    return new_dict

def enumerate_and_unflatten(filename):
    dict_ = json.loads(open(filename).read())
    exps = hu.cartesian_exp_group(dict_)
    return [unflatten(dd) for dd in exps]

def insert_defaults(exp_dict, defaults, only_valid_keys=True, verbose=False):
    """Inserts default values into the exp_dict.

    Will raise an exception if exp_dict contains
    a key that is not recognised.

    Args:
        exp_dict (dict): dictionary to be added to
    """
    for key in exp_dict.keys():
        if key not in defaults:
            if only_valid_keys:
                # Check if there are any unknown keys.
                raise Exception("Found key in exp_dict but is not recognised: {}".\
                    format(key))
            else:
                logger.warning("Found key in exp_dict but is not recognised: {}".\
                    format(key))
        else:
            if type(defaults[key]) == dict:
                # If this key maps to a dict, then apply
                # this function recursively
                insert_defaults(exp_dict[key], defaults[key])

    # insert defaults
    for k, v in defaults.items():
        if k not in exp_dict:
            exp_dict[k] = v
            if verbose:
                logger.info("Inserting default kwarg {} -> {}".format(k, v))

if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)
    # Shuriken-style config file
    dd = json.loads(open("test.json").read())
    # Just get one experiment from the cartesian gs
    from haven import haven_utils as hu
    exp1 = hu.cartesian_exp_group(dd)
    print("# exps,", len(exp1))
    print("test exps[0]...")
    print(pp.pprint(exp1[0]))
    exp1_unflat = unflatten(exp1[0])
    pp.pprint(exp1_unflat)
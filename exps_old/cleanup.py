import glob
import os
from pathlib import Path
from subprocess import check_output
import subprocess
import sys

RESULTS_DIR = "/results/holo_ae"


ignore = []
print("Reading backup.txt...")
with open("backup.txt") as f:
    for line in f:
        line = line.rstrip()
        if line == "":
            continue
        print(line)
        ignore.append(line)
ignore = set(ignore)
print("len of ignore set: %i" % len(ignore))

for seed in range(0,7):
    print("Searching through seed %i..." % seed)
    results_dir = "%s/s%i" % (RESULTS_DIR, seed)
    folders = glob.glob(results_dir + "/*")
    for folder in folders:
        owner = Path(folder).owner()
        if os.path.basename(folder) in ignore:
            print("  ignoring %s" % folder)
        elif owner == "root":
            print("  folder owner is root, ignoring...")
        else:
            print("  analysing %s" % folder)
            n_pkl = len(glob.glob("%s/*/*.pkl" % folder))
            print("    n chkpts: %i" % n_pkl)
            try:
                print("    rm %s/*/*.pkl" % folder)
                stdout = check_output(["rm -f %s/*/*.pkl" % folder],
                                      stderr=subprocess.STDOUT,
                                      shell=True).decode('utf-8')
                print("   " + stdout)
            except subprocess.CalledProcessError:
                print("    error: no pkls in this dir")

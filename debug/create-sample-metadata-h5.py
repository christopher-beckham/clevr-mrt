import os
import h5py
import sys

if 'DATASET_CLEVR_KIWI_META' not in os.environ:
    print("source env.sh first")
    sys.exit(1)

metadata_dir = os.environ['DATASET_CLEVR_KIWI_META']
new_metadata_dir = "/mnt/home/datasets/clevr-mrt-v2-sample/metadata"
subsample_N = 100

print("METADATA_DIR: {}".format(metadata_dir))
print("NEW_METADATA_DIR: {}".format(new_metadata_dir))

for split in ['train', 'valid']:
    print("processing: {}".format(split))
    f = h5py.File("{}/{}_questions.h5".format(metadata_dir, split), 'r')    
    sampled = { key:f[key][0:subsample_N] for key in f.keys() }
    f.close()
    
    subsampled_fnames = sampled['image_filenames']

    # open new file
    print("  writing new file...")
    g = h5py.File("{}/{}_questions.h5".format(new_metadata_dir, split), 'w')
    for key in sampled.keys():
        g[key] = sampled[key]
    g.close()

# Find what index folders the subsampled_fnames are inside

print("These indices should be copied over...")
for elem in set(subsampled_fnames.astype('str').tolist()):
    print(" ", elem.split("_")[2].replace("s",""))

#CLEVR_train-clevr-kiwi-spatial_s021708_cc

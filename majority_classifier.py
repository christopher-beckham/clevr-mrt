import h5py
import os
from collections import Counter
import numpy as np

meta_dir = os.environ['DATASET_CLEVR_KIWI_META']
print(meta_dir)

for filename in ['train_questions.h5', 'valid_questions.h5', 'test_questions.h5']:
    print(filename + "...")
    f = h5py.File("%s/%s" % (meta_dir, filename), 'r')

    answers = f['answers'][:]
    histogram = Counter(answers).most_common()
    maj_count = histogram[0][1]
    acc = (maj_count*1.0) / len(answers) * 100
    print("  total acc = %f" % acc, "%")

    template_names = np.asarray(list(Counter(f['template_filenames'][:]).keys()))
    tf = f['template_filenames'][:]
    print("  answers grouped by question template:")
    for temp_name in template_names:
        ans_subset = answers[ np.where(tf == temp_name)[0] ]
        histogram_subset = Counter(ans_subset).most_common()
        maj_count_subset = histogram_subset[0][1]
        acc_subset = (maj_count_subset*1.0) / len(ans_subset) * 100.
        print("    %s = %f" % (temp_name.decode('utf-8'), acc_subset), "%")

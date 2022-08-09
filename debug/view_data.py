import json
import numpy as np
import glob
from collections import Counter

q_files = glob.glob("/clevr_kiwi/*/questions.json")
s_files = glob.glob("/clevr_kiwi/*/scenes.json")


families = []
qs = []
answers = []
print("counting question family index...")
for file_ in q_files:
    dd = json.loads(open(file_).read())
    questions = dd['questions']
    for q in questions:
        families.append(q['question_family_index'])
        qs.append(q['question'])
        answers.append(q['answer'])
print(Counter(families))

print("histogram answer types...")
print("", Counter(answers))
print("baseline acc...")
print("", Counter(answers).most_common()[0][1]*1.0 / len(answers))


#print("scanning questions...")
#for q in qs:
#    if "left" not in q and "right" not in q and "front" not in q and "behind" not in q:
#        # Shouldn't be anything here.
#        print(q)


for file_ in s_files:
    print(file_)
    dd = json.loads(open(file_).read())
    scenes = dd['scenes']
    for scene in scenes:
        all_cams = [scene[key]['cam_params'] for key in scene.keys()]
        all_cams = np.asarray(all_cams)

        print(scene.keys())

        print(np.mean(all_cams, axis=0).tolist())
        print(np.std(all_cams, axis=0).tolist())
        break
    break

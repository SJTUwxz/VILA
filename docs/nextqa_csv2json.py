import os
import json
import pandas as pd

NEXTQA_dir = "/mnt/mir/datasets/vlm-evaluation-datasets/nextqa/annotations" 
nextqa_annotation = os.path.join(NEXTQA_dir, "val.csv")
output = os.path.join(NEXTQA_dir, "nextqa_val.json")
mapping = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E'}

question_list = []
df = pd.read_csv(nextqa_annotation)
for index, row in df.iterrows():
    video = str(row["video"])
    qid = row["qid"]
    qs = row["question"]
    a0 = row["a0"]
    a1 = row["a1"]
    a2 = row["a2"]
    a3 = row["a3"]
    a4 = row["a4"]
    
    more_options = f"A: {a0}. B: {a1}. C: {a2}. D: {a3}. E: {a4}.\n Select the correct answer from the options (A,B,C,D,E).\n"
    qs = qs +"?\n " + more_options

    answer = int(row["answer"])
    answer = mapping[answer]

    question_list.append({"q": qs, "a": answer, "video_id": video})

with open(output, 'w') as f:
    json.dump(question_list, f)

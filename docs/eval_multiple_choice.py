import os
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True, help="the folder that stores the experiments")

    args = parser.parse_args()


    with open(os.path.join(args.exp_dir, "merge.jsonl")) as json_file:
        json_list = list(json_file)

    num_questions = 0
    correct = 0
    for json_str in json_list:
        result = json.loads(json_str)
        answer = result['answer']
        options = result['options']
        pred = result['pred']
        if pred not in ['A', 'B', 'C', 'D', 'E']:
            pred = pred[0]
        assert pred in ['A', 'B', 'C', 'D', 'E'], f"prediction not valid: {result}"
        pred = ord(pred) - 65
        gt = options.index(answer)


        if gt == pred:
            correct += 1
        num_questions += 1

    print(correct / num_questions)

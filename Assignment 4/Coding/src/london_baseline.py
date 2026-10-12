# TODO: [part d]
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import argparse
import utils

def main():
    accuracy = 0.0

    # Compute accuracy in the range [0.0, 100.0]
    filepath = 'birth_dev.tsv'
    eval_corpus = open(filepath, encoding='utf-8').readlines()
    predictions = ['London'] * len(eval_corpus)
    total, correct = utils.evaluate_places(filepath, predictions)
    accuracy = (correct / total) * 100.0 if total > 0 else 0.0

    return accuracy

if __name__ == '__main__':
    accuracy = main()
    with open("london_baseline_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy}\n")

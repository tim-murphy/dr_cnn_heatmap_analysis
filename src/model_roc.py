#!/usr/bin/env python

import argparse
import csv
import matplotlib.pyplot as plt
import os
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import LabelBinarizer

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--accuracy_csv", type=str, required=True, nargs="+",
                           help="Path to accuracy CSV file")
    argparser.add_argument("--show_plot", action="store_true",
                           help="Display ROC graph")
    argparser.add_argument("--no_legend", action="store_true",
                           help="Do not add a legend to the graph")

    args = argparser.parse_args()

    plot_data = {} # format: [AUC] = (label, fpr, tpr, filename, category)
    one_vs_rest = False
    model_roc = {} # used to calculate the average ROC values.

    for c in args.accuracy_csv:
        # Validate command line arguments.
        if not os.path.isfile(c):
            raise ValueError("CSV file does not exist: " + c)

        # Load data from the CSV file.
        preds = []
        labels = []
        categories = []
        with open(c, 'r') as ifile:
            csvfile = csv.DictReader(ifile)
            # Get the possible categories.
            # The file format is image,[grades],ground_truth
            categories = [int(cat) for cat in csvfile.fieldnames[1:-1]]

            # If this is a binary classifier, we don't need cat zero.
            if len(categories) == 2:
                categories = [0]
            else:
                one_vs_rest = True

            for row in csvfile:
                pred_cats = []
                for cat in categories:
                    pred_cats.append(float(row[str(cat)]))

                labels.append(int(row['ground_truth']))
                preds.append(pred_cats)

        # Calculate the ROC values.
        for cat in categories:
            cat_labels = [1 if l == cat else 0 for l in labels]
            cat_preds = [p[cat] for p in preds]

            fpr, tpr, thresh = roc_curve(cat_labels, cat_preds)
            model_auc = auc(fpr, tpr)

            model_name = ""

            for i, m in enumerate(os.path.split(c)[1].split("+")):
                if i > 0:
                    model_name += " + "

                # Some models have a leading underscore. We don't want it.
                if m[0] == "_":
                    m = m[1:]
                model_name += m.split("~")[0]

                if len(categories) > 1:
                    model_name += " cat=" + str(cat)

            model_name += f" (AUC={model_auc:.3f})"

            # Very very very unlikely to happen. We can fudge this as it's
            # only used for sorting.
            while model_auc in plot_data:
                model_auc += 1e-10

            plot_data[model_auc] = (model_name, fpr, tpr, os.path.split(c)[1],
                                    cat if len(categories) > 1 else "binary")

    print("model,category,auc")
    for auc in reversed(sorted(plot_data.keys())):
        model_name = plot_data[auc][3]
        print(model_name, plot_data[auc][4], auc, sep=",")

        if model_name not in model_roc:
            model_roc[model_name] = []

        model_roc[model_name].append(auc)

    for model_name, rocs in model_roc.items():
        if len(rocs) > 1:
            print(model_name, "average", sum(rocs) / len(rocs), sep=",")

    if args.show_plot:
        plot_title = "Receiver Operating Characteristic Curves"

        if one_vs_rest:
            plot_title += " (OvR)"

        plt.title(plot_title)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("1 - Specificity")
        plt.ylabel("Sensitivity")

        for idx in reversed(sorted(plot_data.keys())):
            (model_name, fpr, tpr, _, _) = plot_data[idx]
            plt.plot(fpr, tpr, label=model_name)

        if not args.no_legend:
            plt.legend(loc='center right', bbox_to_anchor=(1.35, 0.50))

        plt.tight_layout()
        plt.show()

# EOF

#!/usr/local/bin/python3
import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


ADMIN_ID     = 0
ADMIN_DUR_ID = 1
INFO_ID      = 2
INFO_DUR_ID  = 3
PROD_ID      = 4
PROD_DUR_ID  = 5
B_RATE_ID    = 6
EXIT_RATE_ID =7
PAGE_ID      =8
SPEC_DAY_ID  =9
MONTH_ID     =10
OS_ID        =11
BROWSER_ID   =12
REG_ID       =13
TRAFF_ID     =14
VISISTOR_ID  =15
WEEKEND_ID   =16
REVENUE_ID   =17

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def convert_evidence(row):
    float_type_cells = [ ADMIN_DUR_ID, INFO_DUR_ID, PROD_DUR_ID, B_RATE_ID,
                         EXIT_RATE_ID, PAGE_ID, SPEC_DAY_ID ]
    month_to_int = {
        "Jan":0, "Feb":1, "Mar":2, "Apr":3, "May":4, "June":5,
        "Jul":6, "Aug":7, "Sep":8, "Oct":9, "Nov":10, "Dec":11
    }
    evidences = []
    for cell_id in range(len(row) - 1):
        #print(cell_id)
        if cell_id not in float_type_cells:
            if cell_id == MONTH_ID:
                evidences.append(int(month_to_int.get(row[cell_id])))
            elif cell_id == VISISTOR_ID:
                evidences.append(1 if row[cell_id] == "Returning_Visitor" else 0)
                #print(row[cell_id])
            elif cell_id == WEEKEND_ID:
                evidences.append(1 if row[cell_id] == "TRUE" else 0)
                #evidences.append(int(boolean_to_int.get(row[cell_id])))
            else:
                evidences.append(int(row[cell_id]))
                #print(row[cell_id])
        else:
            evidences.append(float(row[cell_id]))

    #print(evidences)
    return evidences

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    #print(filename)
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)

        evidences = []
        labels = []
        for row in reader:
            evidences.append(convert_evidence(row))
            labels.append(1 if row[REVENUE_ID] == "TRUE" else 0)

    #print(evidences)
    #print(labels)
    return evidences, labels

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)

    return model
    #raise NotImplementedError


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    correct_positive = 0
    correct_neg = 0
    total_positive = 0
    total_neg = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            total_positive += 1
            if labels[i] == predictions[i]:
                correct_positive += 1
        else:
            total_neg += 1
            if labels[i] == predictions[i]:
                correct_neg += 1

    #print(f"Total positive: {total_positive}")
    #print(f"Correct possitive: {correct_positive}")
    #print(f"Total negative: {total_neg}")
    #print(f"Correct negative: {correct_neg}")
    sensitivity = float(correct_positive / total_positive)
    #print(f"Sensitivity: {sensitivity:.2f}")
    specificity = float(correct_neg/total_neg)
    #print(f"specificity: {specificity:.2f}")

    return sensitivity, specificity

if __name__ == "__main__":
    main()

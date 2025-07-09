import parse
import node
import math
import pandas as pd
from collections import Counter
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)

def most_common_label(data, target_col_name):
    labels = [row[target_col_name] for row in data]
    counts = {label: labels.count(label) for label in set(labels)}
    return max(counts, key=counts.get)  

def check_homogeniety(data, target_col_name):
    flag = False
    labels = [row[target_col_name] for row in data]
    if len(set(labels)) == 1:
        flag = True
    return set(labels), flag

def extract_attributes(data, target_col_name):
    attributes = list(data[0].keys())  # Extract keys from the first row
    attributes.remove(target_col_name)
    # return the list of attributes extracted
    return attributes

def compute_entropy():
    
    pass

def compute_weighted_child_entropy(attribute):
    weights = {}
    final_entropy_val = []
    print("Attribute being tested is : ", attribute)
    values = [row[attribute] for row in data]
    counts = {label: values.count(label) for label in set(values)}
    print(f"Values are \n {values}")
    print(f"Counts are \n {counts}")

    # calculate the weight for each unique value, this will be multipled with the entropy
    for unique_value in counts.keys():
        probability = {}
        print(f"Unique value is {unique_value}")
        weights[unique_value] = counts[unique_value] / len(values) 
        # in this case class label does not have to be cleared
        class_label = [row["Class"] for row in data if row[attribute] == unique_value]
        print(f"Now we can compute the entropy for the unique value {unique_value}") 
        entropy_counts = Counter(class_label)
        for uni_val in entropy_counts.keys():
            probability[uni_val] = -(entropy_counts[uni_val]/ len(class_label) * math.log2(entropy_counts[uni_val]/ len(class_label)))
        print(probability)
        entropy_unique_val = sum(probability.values())

        # w0*h0
        mul_weights_entropy = weights[unique_value] * entropy_unique_val
        final_entropy_val.append(mul_weights_entropy)
        print(final_entropy_val)
    print(f"Final entropy compute is {sum(final_entropy_val)}")
    # return sum(final_entropy_val)
    # Now we can compute the entropy
    # pass

def stopping_criteria(data):
    # find out the most common label
    cl = most_common_label(data, "Class")
    print(f"Most common label in data is {cl}")
    # label the node with the most common label
    t = node.Node(label=cl)
    print(t.label)

    # check the target class
    label, is_homogenous = check_homogeniety(data, "Class")
    attributes = extract_attributes(data, "Class")
    print(f"Atttributes are {attributes}")

    # If all observations contain only positive or negative observations return
    if is_homogenous == True or not attributes: 
        print("Stopping condition met")
        if is_homogenous:
            # label with the homogenous class
            t = node.Node(label=label) 
        else:
            # Label with the most common class
            t = node.Node(label=cl)  
        return t
    else:
        print("Homogeniety not encountered, building the decision tree further")
        for att in attributes:
            compute_weighted_child_entropy(att)
            break
        # # compute the entropy
        # return t  
    
if __name__ == "__main__":
    data = parse.parse("candy.data")
    stopping_criteria(data)
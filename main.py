import parse
import node
import pandas as pd
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
    print(attributes)
    return attributes

def stopping_criteria(data):
    # find out the most common label
    cl = most_common_label(data, "Class")
    # label the node with the most common label
    t = node.Node(label=cl)
    print(t.label)
    # check the target class
    label, is_homogenous = check_homogeniety(data, "Class")
    attributes = extract_attributes(data, "Class")
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
        print("Build the decision tree further")
        # compute the entropy
        return t  
    
if __name__ == "__main__":
    data = parse.parse("candy.data")
    stopping_criteria(data)
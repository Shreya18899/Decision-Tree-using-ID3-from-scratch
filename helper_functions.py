import math
from collections import Counter
import node

def most_common_label(data, target_col_name):
    labels = [row[target_col_name] for row in data]
    counts = {label: labels.count(label) for label in set(labels)}
    return max(counts, key=counts.get)  

def get_categories_column(data, col):
    values = [row[col] for row in data]
    counts = {label: values.count(label) for label in set(values)}
    return values, counts

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

def compute_weighted_child_entropy(data, attribute):
    weights = {}
    final_entropy_val = []
    print("Attribute being tested is : ", attribute)
    values, counts = get_categories_column(data, attribute)
    print(f"Values are \n {values}")
    print(f"Counts are \n {counts}")

    # calculate the weight for each unique value, this will be multipled with the entropy
    for unique_value in counts.keys():
        probability = {}
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
    return sum(final_entropy_val)

def check_homogeniety_attribute_split(data, attribute):
    values, counts = get_categories_column(data, attribute)
    for unique_value in counts.keys():
        print(f"Unique value is {unique_value}")
        # get all the class labels associated with that unique value
        class_label = [row["Class"] for row in data if row[attribute] == unique_value]
        if len(set(class_label)) == 1:
            print("Homegenous condition reached")
        else:
            print("Homegenous condition not reached")

def build_tree(data, attributes):
    # find out the most common label
    cl = most_common_label(data, "Class")
    print(f"Most common label in data is {cl}")

    # label the node with the most common label in the data
    t = node.Node(label=cl)
    print(t.label)

    # check the target class
    label, is_homogenous = check_homogeniety(data, "Class")

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
        attribute_list_tested = []
        entropy_att = []
        # compute the entropy
        for att in attributes:
            attribute_list_tested.append(att)
            entropy = compute_weighted_child_entropy(data,att)
            entropy_att.append(entropy)
        att_entropy = dict(zip(attribute_list_tested, entropy_att))
        print(f"Entropy computed for all attributes {att_entropy}")    
        best_attribute = min(att_entropy, key=att_entropy.get)
        print(f"Attribute to split on is {best_attribute}")

        # create leaf node with the attribute with min entropy which becomes 
        # the root node
        root = node.Node(label=best_attribute)
        values, counts = get_categories_column(data, best_attribute)
        # children of this node would be the unique values of the attribute
        for category in counts.keys():
            subset = [row for row in data if row[best_attribute] == category]
            # if subset is empty
            if not subset:
                # No data for this branch → default to majority class in the data
                root.children[category] = node.Node(label=cl)
            else:
                class_label = [row["Class"] for row in data if row[best_attribute] == category]
                # if the category contains class labels which are homegenous
                if len(set(class_label) )== 1:
                    root.children[category] = node.Node(label=int(next(iter(class_label))))
                else:
                    # Implement recursive but only with the subset of data
                    # Attribute that was split on is getting removed ??
                    remaining_attrs = [a for a in attributes if a != best_attribute]
                    child_subtree = build_tree(subset, attributes=remaining_attrs) 
                    root.children[category] = child_subtree
        return root

def print_tree(node, depth=0):
    indent = "  " * depth
    if node.children:
        print(f"{indent}[Split on: {node.label}]")
        for val, child in node.children.items():
            print(f"{indent}-- {val} -->")
            print_tree(child, depth + 1)
    else:
        print(f"{indent}[Leaf: Predict {node.label}]")


from node import Node
from helper_functions import extract_attributes, build_tree, most_common_label

def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''
  print("Building the tree using the ID3 algorithm")
  # Returns the default label if no data is found
  if not examples:
    return Node(label=default)
  attributes = extract_attributes(examples, "Class")
  print(f"Attributes are: {attributes}")
  return build_tree(examples, attributes)

def examples_reaching_node(node, examples, path_from_root):
    """
    Returns a subset of examples that would reach the given node,
    based on the decisions along the path from the root.
    """
    filtered = []
    for ex in examples:
        reach = True
        for attr, val in path_from_root:
            if ex[attr] != val:
                reach = False
                break
        if reach:
            filtered.append(ex)
    return filtered

def prune(node, examples, root, path_from_root=[]):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''
  # Base case: if already leaf node → nothing to prune
  if not node.children:
      return node
  
  # Recurse on children first (post-order) - get the bottom internal node
  # Make the parent node point to the reassigned / pruned node and not the older version of the tree
  for val, child in list(node.children.items()):
      prune_path = path_from_root + [(node.label, val)]
      node.children[val] = prune(child, examples=examples, root=root, path_from_root=prune_path)

  # Filter validation examples that actually reach this node
  subset = examples_reaching_node(node, examples, path_from_root)
  if not subset: 
     return node

  # Compute the validation with the whole tree with no nodes pruned - root node passed
  val_accuracy_before = test(root, examples)
  print(f"Validation accuracy before the current node is pruned {val_accuracy_before}")  

  # Remove the tree connected to node {node} and recompute the accuracy 
  # pruned_node and node are two references to the exact same object in memory
  # Save the original label so that if val accuracy does not decrease we can pass the original tree
  saved_node_label = node.label
  saved_node_children = node.children
  print(f"Current node is {node}")
  # make current node - leaf node by getting the most common label in validation
  node.label = most_common_label(subset, "Class")
  # remove any children - pruning part
  node.children = {}
  print(f"Node of current children are {node.children}")

  print(f"Label of current node is {node.label}")

  # now compute accuracy after pruning
  val_accuracy_after = test(root, examples)
  print(f"Validation accuracy after the current node pruned is {val_accuracy_after}")  

  if val_accuracy_after < val_accuracy_before:
    print("Condition is true")
    # revert to original
    node.label = saved_node_label
    node.children = saved_node_children
  else:
    print("Accuracy increased after pruning")
  return node


def test_candy(node, examples, train, actuals):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''
  count=0
  all_predictions = []
  for example in examples:
    result = evaluate_single_observation(node, example, train)
    all_predictions.append(result)
  for i,j in zip(all_predictions, actuals):
    if int(i)==int(j):
      count=count+1
      print("True")
  accuracy = count / len(all_predictions)
  print(f"length of all predictions is {len(all_predictions)} and length of actuals is {len(actuals)}")
  return accuracy


def evaluate_single_observation(node, example, train):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''
  if not node.children:
    print("No more children found")
    # if leaf node is encountered - return the prediction else keep traversing
    prediction = node.label
    return int(prediction)
  else:
    attr = node.label
    value = example[attr]
    print("Root node feature is : ", attr)
    print(f"Value is : {value}")
    print(f"Children of {attr} are {node.children}")
    if value not in node.children:
      # if attribute is not found / value of attribute does not exist because it was not present in training data - fallback is label 
      # with the most common / majority class in the training data
      print("Returning the majority class in the training data for missing attributes")
      prediction = most_common_label(train, "Class")
      return int(prediction)
    else:
      # Choose the attribute value which is encountered to find the next node
      new_node = node.children[value]
      print(f"Next node in the tree based on value of attribute is {new_node.label}")
      return evaluate_single_observation(new_node, example=example, train=train)
    
def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''
  print(f"Reached evaluate with node {node} and example as {example}")
  if not node.children:
    print("No more children found")
    # if leaf node is encountered - return the prediction else keep traversing
    prediction = node.label
    return int(prediction)
  else:
    attr = node.label
    value = example[attr]
    print("Root node feature is : ", attr)
    print(f"Value is : {value}")
    print(f"Children of {attr} are {node.children}")
    if value not in node.children:
      # if attribute is not found / value of attribute does not exist because it was not present in training data assign a fallback label 
      print("Returning a random class for missing attributes")
      prediction = 1
      return int(prediction)
    else:
        # Choose the attribute value which is encountered to find the next node
        new_node = node.children[value]
        print(f"Next node in the tree based on value of attribute is {new_node.label}")
        return evaluate(new_node, example=example)
  
  
def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''
  count=0
  all_predictions = []
  actuals = []
  for example in examples:
    result = evaluate(node, example)
    y = example["Class"]
    all_predictions.append(result)
    actuals.append(y)
  for i,j in zip(all_predictions, actuals):
    # print(i, j)
    # print(type(i), type(j))
    if int(i)==int(j):
      count=count+1
  accuracy = count / len(all_predictions)
  print(f"length of all predictions is {len(all_predictions)} and length of actuals is {len(examples)}")
  print(f"Accuracy is {round(float(accuracy), 2)}")
  # Return a floating point value as determined in the unit test
  return round(float(accuracy), 2)
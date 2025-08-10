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


def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''

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
    print(i, j)
    print(type(i), type(j))
    if int(i)==int(j):
      count=count+1
  accuracy = count / len(all_predictions)
  print(f"length of all predictions is {len(all_predictions)} and length of actuals is {len(examples)}")
  print(f"Accuracy is {round(float(accuracy), 2)}")
  # Return a floating point value as determined in the unit test
  return round(float(accuracy), 2)
import parse
import helper_functions
import unit_tests
import ID3

if __name__ == "__main__":
    data = parse.parse("candy.data")
    train, test = helper_functions.split_train_test(data, 0.8, 0.2)
    actuals = [i["Class"] for i in test]
    tree = ID3.ID3(train, default=0)
    print("Finished building decision tree..Training complete")
    helper_functions.print_tree(tree)
    print("Now running evaluate with a single observation")
    features= ["chocolate", "fruity" , "caramel",
               "peanutyalmondy", "nougat",
               "crispedricewafer", "hard" , "bar",
               "pluribus"]

    values = ["1", "0" ,"0" ,"0" ,"0" ,"1" ,"0" ,"1" ,"1"]

    example = dict(zip(features, values))

    result = ID3.evaluate(tree, example, train=train)
    print(f"Final prediction is {result}")

    print("Now testing to check accuracy of the tree")
    accuracy = ID3.test(tree, examples=test, train=train, actuals=actuals)
    print(f"Testing accuracy for {len(test)} examples is {accuracy}")

import parse
import helper_functions
import unit_tests
import ID3

if __name__ == "__main__":
    data = parse.parse("candy.data")
    tree = ID3.ID3(data, default=0)
    print("Finished building decision tree")
    helper_functions.print_tree(tree)
    print("Now running evaluate")
    features= ["chocolate", "fruity" , "caramel",
               "peanutyalmondy", "nougat",
               "crispedricewafer", "hard" , "bar",
               "pluribus"]

    values = ["1", "0" ,"0" ,"0" ,"0" ,"1" ,"0" ,"1" ,"1"]

    example = dict(zip(features, values))

    label = ID3.evaluate(tree, example)
    # print("Now running unit tests")
    # unit_tests.testID3AndEvaluate()

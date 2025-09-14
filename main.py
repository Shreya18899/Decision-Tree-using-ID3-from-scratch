import parse
import helper_functions
import unit_tests
import ID3

if __name__ == "__main__":
    data = parse.parse("candy.data")

    # # data = parse.parse("house_votes_84.data")
    # # Giving all the data for training - 100 % allocation
    train, test_data = helper_functions.split_train_test(data, 0.9, 0.1)
    # print(len(train), len(test))
    # actuals = [i["Class"] for i in test]
    # tree = ID3.ID3(train, default=0)
    # print("Finished building decision tree..Training complete")
    # helper_functions.print_tree(tree)

    # # Run with my functions 
    # print("Now running evaluate with a single observation")
    # features= ["chocolate", "fruity" , "caramel",
    #            "peanutyalmondy", "nougat",
    #            "crispedricewafer", "hard" , "bar",
    #            "pluribus"]

    # values = ["1", "0" ,"0" ,"0" ,"0" ,"1" ,"0" ,"1" ,"1"]

    # example = dict(zip(features, values))

    # result = ID3.evaluate_single_observation(tree, example, train=train)
    # print(f"Final prediction is {result}")

    # pruned_tree = ID3.prune(tree, test, root=tree, path_from_root=[])
    # print("Printing the pruned tree")
    # helper_functions.print_tree(pruned_tree)

    # print("Now testing to check accuracy of the tree")
    # accuracy = ID3.test_candy(tree, examples=test, train=train, actuals=actuals)
    # print(f"Testing accuracy for {len(test)} examples is {accuracy}")

    # random forest implementation
    rf_predictions = ID3.randomforest(n_trees=2, train=train, test_data=test_data)
    rf_accuracy = ID3.RF_test_accuracy(test_data=test_data, preds=rf_predictions)

    # # # Run the unit test functions
    # # unit_tests.testID3AndEvaluate()
    # # unit_tests.testID3AndTest()
    # # unit_tests.testPruning()
    # # unit_tests.testPruningOnHouseData()

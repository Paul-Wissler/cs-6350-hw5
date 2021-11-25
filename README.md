# cs-6350-hw2
This is a machine learning library developed by Paul Wissler for CS 6350 in University of Utah

# Instructions
main.py can run most of this code, but bear in mind that many of the algorithms take a very long time to run, even with multiprocessing. All results can be reproduced by running main.py, with all questions being able to be reproduced by running the proper function from QuestionAnswers module (e.g. to get part 1 question 3a results, run QuestionAnswers.part1.q3a). However, for some of the results require far more processing power, and these are put into main.py directly under appropriate section headings (i.e. part 2 q1a is under q1a section, identified with a comment and followed by a long line of #'s). The following questions (all from part 2) are in these sections: q2a, q2c, q2d, q2e, q3 of the main function of main.py.

There are three modules of interest to the TA's: DecisionTree, EnsembleLearning, and LinearRegression. These are all packages, so all you need to do is import them like so:

```python
import DecisionTree as dtree
import EnsembleLearning as ensl
import LinearRegression as lr

```

Within each of these packages are classes which will learn the appropriate model. For each class, when you instantiate the model, it will automatically create the model as an attribute in the `__init__` method, then will delete the supplied X (but not y in some cases for implementation purposes) to help conserve memory. To test, simply do:

## Decision Tree Model
```python
tree = dtree.DecisionTreeModel(X.copy(), y.copy(), error_f=dtree.calc_entropy, 
            max_tree_depth=None, default_value_selection='subset_majority')
accuracy = tree.test(X_test, y_test)
```

There are optional kwargs you can pass in when you instantiate the model, these being error_f, max_tree_depth, and default_value_selection (the default values can be seen in the code block above). If you wish to change how information gain is calculated, you may use `dtree.calc_entropy`, `dtree.calc_majority_error`, or `dtree.calc_gini_index`. However, I do not think it will be necessary to change that kwarg. If you wish to set the maximum allowable tree depth, you may do so by changing the `max_tree_depth` kwarg to any value >0. Otherwise, it will go as deep as it can every single time. As for selecting default values, you may do so by changing the `default_value_selection` kwarg between 'subset_majority' and 'majority' (see the `default_value` method).

It should be noted that with all of my code, I generally assume that the user will input a pandas DataFrame for X and a pandas Series for y. To make sure there are no weird aliasing errors, please be sure to input `X.copy()` and `y.copy()`.

If, during a given evaluation post-training, the decision tree fails to find an output from any given input, it will return the mode of the training y as its best guess.

## Adaboost Model

To make an Adaboost model:

```python
model = ensl.AdaBoostModel(X, y, 
    sample_rate=100, 
    boost_rounds=500,
    max_tree_depth=1
)

```

Optional kwargs include: sample_rate, boost_rounds, error_f, max_tree_depth, default_value_selection, and reproducible_seed. Differences from DecisionTreeModel are boost_rounds (how many iterations the model will go through when adjusting weights) and reproducible_seed, which is a Boolean value that determines if random sampling can be reproduced, otherwise it will randomly sample as normal. *The max_tree_depth variable, the sample_rate variable, and reproducible_seed should all be removed from Adaboost, as they are not at all needed.*

**An important note is that AdaBoostModel.evaluate is not quite properly implemented, as it will return 1 or -1, as opposed to the original output values. This will be fixed in a future version. AdaBoostModel.test should still work fine, though.**

## Bagged Trees Model

To make a Bagged Trees model:

```python
ensl.BaggerModel(X.copy(), y.copy(), sample_rate=100, bag_rounds=500)

```

This has all the same input variables as AdaBoostModel, except it uses bag_rounds instead of boost_rounds. Also, sample_rate will determine per round how many X attributes are sampled for a given decision tree.

## Random Forest Model

To make a Random Forest Model:

```python
model = ensl.RandomForestModel(
    X.copy(), y.copy(), 
    sample_rate=10, 
    bag_rounds = 100,
    num_sample_attributes=n,
    reproducible_seed=False,
)

```

It has all the same input variables as BaggerModel except it also has num_sample_attributes, which determines how many unique attributes will be used to generate a decision tree each bagging round.

## Notes on EnsembleLearning Models

Each of the Ensemble Learners has a few methods that were useful for this assignment. One is test_cumulative_trees, which works like so:

```python
model.test_cumulative_trees(X_test, y_test, ix)
```
or like so for AdaBoostModel:
```python
model.test_cumulative_trees(X_test, y_test)

```
The ix variable can be used to supply specific indexes to test for (cumulatively). Each method then returns "single" results for each tree, as well as "cumulative" results as it works through the trees. For more details, see bagger.py or adaboost.py.

## Gradient Descent Models
Here is some example code for creating a BatchGradientDescentModel, as well as computing cost of the model:

```python
model = lr.BatchGradientDescentModel(
    X_train, y_train, max_rounds=10000, rate=0.01
)
cost = model.compute_cost(X_test, y_test, model.weights)

```

And for StochasticGradientDescentModel, which is a subclass of BatchGradientDescentModel:

```python
model = lr.StochasticGradientDescentModel(
    c_data[x_cols], c_data.Output, rate=0.005, max_rounds=10000, convergence_threshold=1e-8
)
cost = model.compute_cost(test_c_data[x_cols], test_c_data.Output, model.weights)

```

For both gradient descent models, you must supply an X and a y. You may optionally supply the following kwargs: rate, convergence_threshold, max_rounds, and bias. The rate variable tunes how much the weights change each round. The convergence_threshold variable sets some lower limit that the norm of the difference of subsequent weights must meet to converge. The max_rounds variable sets a maximum number of iterations that the weights will change during training. The bias variable adjusts the bias of the model before training. Bias and weights cannot currently be randomized (bias could be randomized outside the model technically, but that's not a supported feature).

The Gradient Descent Models track how much each step costs in the cost_of_each_step attribute, as well as the convergence of weights in the convergence_of_weights attribute.


## Notes on Perceptron Models

To use the Perceptron models, all can be instantiated the following ways:

```python
model = perc.PerceptronModel(X, y, random_seed=False, rate=.1, epochs=10, bias=0)
accuracy = model.test(X, y)
```

The random_seed variable will determine whether or not to use a deterministic random state for each shuffle per epoch, and epochs will determine how many epochs to loop throug, while rate will determine the learning rate and bias will instantiate the bias to some variable for the model.

For voted perceptron, the votes can be accessed with the votes_list attribute. For the averaged perceptron, the averaged weights can be accessed with the averaged_weights attribute.
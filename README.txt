## Files

- data.py contains functions to obtain the clean and noisy data provided in the wifi_db folder.

- decision_tree_algo.py contains the algorithm to create a decision tree using the information provided in the spec.

- evaluation.py contains the k-fold algorithm and functions to calculate performance metrics as given in step 3 of the spec.

- plot_tree.py contains an algorithm which uses depth first search to plot the tree that is generated.

- pruning.py contains the pruning algorithm and the k-fold algorithm as given in step 4 of the spec.

- main.py performs both step 3 and step 4 as given in the spec on both clean and noisy data using functions defined in evaluation.py and pruning.py.

## How to run the code?

After extracting the zip file, navigate to the extracted folder in the directory.

To run the code that we used to calculate the result that are in the report, run

```
python3 main.py
```

This will first run the 10-fold cross-validation on both clean and noisy datasets and then run the 10-fold cross-validation with pruning on the clean and noisy datasets. Results will print in the terminal while it calculates and averages displayed at the end. Note that the dataset is shuffled in a random order each time so results may not match exacly to what's in the report.


## To run cross-validation code on different data:

Move the file containing the data you want the algorithm to be run on and then run,

```
python3 step3.py <filename> or python step3.py <filename>
```
where filename is the .txt file with your data

## To run cross-validation code with pruning on different data:

Move the file containing the data to the extracted folder. Then run

```
python3 step4.py <filename> or python step4.py <filename>
```
where filename is the .txt file with your data

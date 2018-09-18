# `ciml`: Common Infrastructure for Machine Learning

*Alex Hagen*

## `spot` - Spot check

```python

XY = pd.load_txt('filename.txt')
X = XY.drop(col='labels')
Y = XY['labels']

problem = ciml.spot(data_type='tabular', problem_type='classification')
problem.load_data(X, Y)
problem.summarize()
problem.check()
problem.summarize_check()

```

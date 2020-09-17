# Nelder-Mead

## Learning a Model from Data

### Step 1 - specify the model

#### The Model Function

Adjust `model.py` to reflect the objective function that should be learned.
E.g., an (unnormalized, simplified) normal distribution with learnable mean and
variance:

```python
def function(self, x, mean, sigma):
  return np.exp(-1 * ((x - mean) / sigma)**2)

```

#### The Parameters to be Learned

The file `simplex_fork.py` will read the environmental variables specified with
`VAR{i}` (e.g., `VAR0`, `VAR1`, etc.) and search an optimal value for them based
on the loss in `Model.aggregate_loss()`. If you want to enforce restrictions on
a parameters, like non-negativity, you can specify this as a method of the class
`Model` and add this to the function `Model.learned_parameters()`. This latter
function takes the parameters as learned by `simplex_fork.py` and transforms
them to the parameters needed for the objective function `Model.function().`
E.g., for the model function specified above, we want to learn the mean and
sigma. Let's say we suspect the mean to be close to 10 and the sigma to 1:

```python
def mean(self, learned_mean):
  return 10 + learned_mean

def sigma(self, learned_sigma):
  return 1 * (1 + learned_sigma)

@property
def learned_parameters(self):
  return [float(os.environ["VAR0"]),
          float(os.environ["VAR1"])]

def parameters_from_learned(self, parameters):
    mean = parameters[0]
    sigma = parameters[1]
    return [self.mean(mean), self.sigma(sigma)]

def parameter_repr(self, parameters):
    parameters = tuple(self.parameters_from_learned(parameters))
    return "Mean: %.2f, Sigma: %.2f" % parameters
```

#### The Loss Function

The loss function looks at the actual data points and compares them with the
value predicted by `Model.function()`.

### Step 2 - Learn the Parameters

Go through the following steps to run optimization:

```bash
>> export VAR0=1
>> export VAR1=1
>> export BEST=undefined
>> ./simplex_fork.py ./loss.py 2 0.01 &> log.txt
```

This will cause `log.txt` to be populated with the learning progress.

### Step 3 - Visualize the Result

To extract the relevant information from the stdout in `log.txt` run:

```bash
>> mkdir logs
>> ./get_logs.sh log.txt logs/relevant_logs.txt
```

And to visualize this run:

```bash
>> python3 plot_learning.py --logs_dir=logs --logs_file=relevant_logs.txt
```

This will plot the learning development in a set of images and make a gif out of
them. It will also plot the loss curve over time.

# Contributing to `habmot`
All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.
We recommend going through the list of [`issues`](https://github.com/cr-crme/habmot/issues) to find issues that interest you, preferable those tagged with `good first issue`.
You can then get your development environment setup with the following instructions.

## Forking `habmot`

You will need your own fork to work on the code.
Go to the [habmot project page](https://github.com/cr-crme/habmot/) and hit the `Fork` button.
You will want to clone your fork to your machine:

```bash
git clone https://github.com/your-user-name/habmot.git
```

## Creating and activating conda environment

Before starting any development, we recommend that you create an isolated development environment. 
The easiest and most efficient way is to use an anaconda virtual environment. 

- Install [miniconda](https://conda.io/miniconda.html)
- `cd` to the `habmot` source directory
- Install `habmot` dependencies with:

```bash
conda env create -f environment.yml
```

## Testing your code

Adding tests are required to get your development merged to the master branch. 
Therefore, it is very good practice to get the habit of writing tests ahead of time so this is never an issue.
The `habmot` test suite runs automatically on GitHub every time a commit is submitted.
However, we strongly encourage running tests locally prior to submitting the pull-request.
To do so, simply run the tests folder in pytest (`pytest tests`).

## Commenting

Every function, class and module should have their respective proper docstrings completed.
The docstring convention used is NumPy. 
Moreover, if your new features is available to the lay user (i.e., it changes the API), the `ReadMe.md` should be modified accordingly.

## Convention of coding

`habmot` tries to follow as much as possible the PEP recommendations (https://www.python.org/dev/peps/). 
Unless you have good reasons to disregard them, your pull-request is required to follow these recommendations. 
I won't get into details here, if you haven't yet, you should read them :) 

All variable names that could be plural should be written as such.

Black is used to enforce the code spacing. 
`habmot` is linted with the 120-character max per line's option. 
This means that your pull-request tests on GitHub will appear to fail if black fails. 
The easiest way to make sure black is happy is to locally run this command:
```bash
black . -l120
```
If you need to install black, you can do it via conda using the conda-forge channel.


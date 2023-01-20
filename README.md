
# PML Exam 2023

## Code to replicate results

- The code to replicate the results of the paper are found inside [code](code/).
- In order to be able to replicate the results we have exported a `yml`-file with our environment inside [code](./code).
- See the [guide](create-environment-and-install-package) on how to install the environment and package


### Create environment and install package

- Change directory into [code](code/)
- Create a new conda environment by running 
```bash
conda env create -f pml.yml
```
- Activate the environment by running
```bash
conda activate pml
```
- Install the `pmldiku` package by running (from the top of the [code](./code) directory):

```bash
pip install -e .
```

- If everything went smooth the package you should now be able to open a REPL and write:
```bash
❯ python
Python 3.10.6 (main, Nov 14 2022, 16:10:14) [GCC 11.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import pmldiku
>>> pmldiku
<module 'pmldiku' (<_frozen_importlib_external._NamespaceLoader object at 0x7efe96c17a00>)>
```

ARENA verifier instantiated on ERAN
========

ARENA is a verifier that contains multiple-adverarial label elimination method and multi-ReLU network encoding. It is instantiated on DeepPoly domain in ETH Robustness Analyzer for Neural Networks ([ERAN](https://github.com/eth-sri/eran)). 


Requirements 
------------
GNU C compiler, ELINA, Gurobi,

python3.6 or higher (python3.6 might be preferred), tensorflow, numpy.


Installation
------------
Clone the ARENA repository via git as follows:
```
git clone https://github.com/arena-verifier/ARENA.git
cd ARENA
```

The dependencies can be installed step by step as follows (sudo rights might be required):
```
sudo ./install.sh
source gurobi_setup_path.sh
```

Note that to run the system with Gurobi one needs to obtain an academic license from https://user.gurobi.com/download/licenses/free-academic.

To install the remaining python dependencies (numpy and tensorflow), type:

```
pip3 install -r requirements.txt
```

Usage
-------------
To run the MNIST_6_200 experiment, please use the following script
```
cd tf_verify
bash ARENA_script.sh
```

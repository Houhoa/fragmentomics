# fragmentomics
Systematically Evaluating Cell-Free DNA Fragmentation Patterns for Cancer Diagnosis and Enhanced Cancer Detection via Integrating Multiple Fragmentation Patterns

## Table of Contents
1. [Quick start](#Quick-start)
2. [Usage](#Usage)
3. [Citation](#Citation)
4. [License](#License)
5. [Contact](#Contact)

## Quick start
1. Obtain files
	```
   git clone --recursive https://github.com/Houhoa/fragmentomics.git
	```
2. Extract data in folder
   	```
   cd fragmentation patterns
   cd model
	```
4. Configure experimental environment
	* Windows
	* python 3.8 
	* scikit-learn 1.1
5. Run the example at your Python build tools such as Pycharm, or in bash command line.  
	```
	python coverage.py
	```  
  
## Usage
1. After you have prepared the cell-free DNA sequencing file, you can use the code in "fragmentation patterns" to calculate the fragmentation pattern features.
2. After you have computed the fragmentation pattern, you can construct a model using the fragmentation pattern matrix and the class labels of the samples, with code similar to that in the "model" folder.

## Citation
```
Hou Y, Meng X & Zhou X. "Systematically Evaluating Cell-Free DNA Fragmentation Patterns for Cancer Diagnosis and Enhanced Cancer Detection via Integrating Multiple Fragmentation Patterns". In preparation.
```

## License
* For academic research, please refer to MIT license.
* For commerical usage, please contact the authors.

## Contact
* Yuying Hou <houou@webmail.hzau.edu.cn>
* Xionghui Zhou <zhouxionghui@mail.hzau.edu.cn>

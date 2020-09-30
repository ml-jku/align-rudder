# Align-RUDDER: Learning From Few Demonstrations by Reward Redistribution
_Vihang P. Patil<sup>1</sup>,
Markus Hofmarcher<sup>1</sup>,
Markus-Constantin Dinu<sup>1</sup>,
Matthias Dorfer<sup>3</sup>,
Patrick M. Blies<sup>3</sup>,
Johannes Brandstetter<sup>1</sup>,
Jose A. Arjona-Medina<sup>1</sup>,
Sepp Hochreiter<sup>1, 2</sup>_

<sup>1</sup> ELLIS Unit Linz and LIT AI Lab, Institute for Machine Learning, Johannes Kepler University Linz, Austria  
<sup>2</sup> Institute of Advanced Research in Artificial Intelligence (IARAI)  
<sup>3</sup> enliteAI, Vienna, Austria

---

##### Detailed blog post on this paper at [this link](https://ml-jku.github.io/align-rudder) and a video showcasing the MineCraft agent at [this link](https://www.youtube.com/watch?v=HO-_8ZUl-UY).

---

The full paper is available at [https://arxiv.org/abs/2009.14108](https://arxiv.org/abs/2009.14108)

# Implementation of Align-RUDDER 
This package contains an implementation of Align-RUDDER together with code to reproduce
the results of artificial tasks I & II as stated in the paper. 
For the sake of time the default settings include only 10 seeds per experiment instead of the 100 used for the results in the paper.

## Dependencies
To reproduce all results we provide an environment.yml file to setup a conda environment with the required packages.
Run the following command to create the environment:
```shell script
conda env create --file environment.yml
conda activate align-rudder
pip install -e .
```

## Usage
To recreate the results from the paper you can run the included run scripts for the
FourRooms and EightRooms environments and the respective method.
 
**Align-RUDDER**  
```
python align_rudder/run_four_alignrudder.py
python align_rudder/run_eight_alignrudder.py
```  
**Behavioral Cloning + Q-Learning**
```  
python align_rudder/run_four_bc.py
python align_rudder/run_eight_bc.py
```  
**DQFD (Deep Q-Learning from Demonstrations)**
```  
python align_rudder/run_four_dqfd.py
python align_rudder/run_eight_dqfd.py
```  

## Results
Once you ran all experiments you are interested in you can run the following script to get 
a summary of the results.
By default plots for all available environments will be generated.
```
python align_rudder/plot_results.py [--env "FourRooms"|"EightRooms"|"all"]
```

## LICENSE
MIT LICENSE

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

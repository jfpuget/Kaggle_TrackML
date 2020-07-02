# Kaggle_TrackML
Code for the TrackML competition on Kaggle: https://www.kaggle.com/c/trackml-particle-identification

That solution got ranked 9th in the competition.  Read the documentation before looking at the code ;)  The content below provides useful information if you want to run the code.

1. Hardware used. For EDA and model tuning on train events: an Intel box with a 4 core i7 at 4.2 GHZ and 64 GB of memory, running ubuntu 16.04. For computing tracks on test events, either a Dell T810 with 20 cores Xeon CPU at 2.4 GHZ, running ubuntu 14.04, and 64 GB of memory, or an IBM AC922 server with 40 P9 cores and 256 GB of memory running RHEL 7.5 (ppc64le).

2. The code consumes about 3GB per worker, hence memory is not an issue really.  One should favor a large number of cores as tracks are computed for a number of events in parallel.

3. We used various linux versions, depending on the machine used, see 1 above.

4. An environment.yml is provided but we use a much smaller set of packages than indicated.  The code can run with only numpy, pandas, pickle, and scikit-learn installed on top of Python 3.6.  EDA notebooks require anaconda, matplotlib, and seaborn. Version numbers are provided in the yaml file.

5. Running the code is rather simple:
- Complete the cloned repo with additional directories as follows:

  ./data/

  ./input/

  ./submissions/final/

  ./submissions/final_inner/

  ./submissions/merge_final/
  
  The test events data should then be downloaded into the ./input directory.
  
- Edit the base_path value in the scripts in src directory to match where you cloned the code.
- Edit the number of Pool workers in the scripts final_test.py, final_inner_test.py and merge_final.py to match the number of processors of your machine.

- Run the scripts in that order:

 data_prep_final.py
 
 final_test.py
 
 final_inner_test.py
 
 merge_final_test.py
 
The last script produces a file named merge_final.csv in the submissions/ directory.  This file can be submitted to Kaggle server to get a private LB score slightly above 0.800

Running only the first two scripts produce a simplified model file named final.csv in the submissions/ directory.  This file can be submitted to Kaggle server to get a private LB score slightly above 0.787. 

Running the above scripts can take days.  We provide final_test_150.py for faster runs. Edit the base_path and the number of workers in the pool before running it. Also create a directory final_150 in the submissions directory. That script runs in about 150 second per event with one i7 core and produces an output file named final_150.csv in the submissions/ directory. This file can be submitted to Kaggle server to yield a private LB score above 0.51.  

### latent shift adaptation using the Recognition Parametrized Model (RPM)

The RPM is described in "Unsupervised representation learning with recognition-parametrised probabilistic models" by Walker*, Soulat*, Yu, Sahani*. Available at https://proceedings.mlr.press/v206/walker23a.html


To run on example where X is a continuous vector and all other variables are discrete, first create the data using the code given in https://github.com/google-research/google-research/tree/master/latent_shift_adaptation with the appropriate parameters. In train_cont_vector_partial_RPM.py, change the folder_id and parent_folder to point to the data and output directories respectively. Change the paramDict (dictionary of parameters for the run) to suitable values. Then run 'rain_cont_vector_partial_RPM.py 0' where 0 can be any integer that determines the dataset to train the RPM on.

To run on example where X and W are CIFAR10 or MNIST images, first create an argument file with 'generate_arg_file.py'. Change the ARGUMENT_FILE to be where the file is saved and change the parameters for your run. Then run 'create_samples.py' to generate a set of discrete data to save such that all runs can use the same data. Again, change the 'ARG_FILE_NAME' to match the argument file and 'parent_folder' to the arg file directory. Then run 'train_partial_RPM.py 0' where 0 can be any integer that determines the dataset to train the RPM on.

# 🔥🔥🔥 Wildfire 🔥🔥🔥

This project was completed in a group of 5 students as part of Applications of Data Science module at Imperial College London for MSc Environmnetal Data Science & Machine Learning.  Detailed explanation of the project aim and structure can found in assignement.pdf. 

The overall aims are:

1. **Prediction**: use a neural network to train a forecasting model
2. **Correction**: perform data assimilation in a reduced space

## Data

1. Model data (already pre-processed):​
    - `Ferguson_fire_train`: training data​
      - Source: https://drive.google.com/file/d/11S0uXpT6rHOsph1OsMu5cGRNG4ieSjLW/view?usp=sharing​
    - `Ferguson_fire_test`: similar to `Ferguson_fire_train` with different simulations​
      - Source: https://drive.google.com/file/d/1K3q1g3gfacK7RL6VLHnRmK1QGEykws_m/view?usp=sharing​
    - `Ferguson_fire_background`: model data you will use for the data assimilation: ​
      - Source: https://drive.google.com/file/d/1DUvPaEnFTT1hQOHy0SHAZxfoKnxQKMid/view?usp=sharing​

2. Satellite data (already pre-processed):​
    - `Ferguson_fire_obs`: Observation data at different days after ignition (only one trajectory)​
      - Source: https://drive.google.com/file/d/1pK7W082NEpS5rL7e5_MbHmY_51mliu9N/view?usp=sharing​


## Installation Guide


Firstly, clone the repository:
```
git clone git@github.com:ese-msc-2021/ads-wildfire-team-creek.git
```
and `cd` into ads-wildfire-team-creek

Now create and activate the conda environment
```
conda env create -f environment.yml
conda activate creek
```

Install required packages
```
pip install -r requirements.txt
```

Activate the setup.py in order to create tools module with
```
pip install -e .
```

We can run pytests to see if everything is going alright
with the following command although there is an automated github workflow
```
pytest tools/tests/
```

## User instructions

How is repository structured and how to run the code?

### How is this repository structured?
#### For Notebooks:
The main notebooks are <b>Q1_Master.ipynb</b> and <b>Q2_Master.ipynb</b>. Please open these notebooks and run the notebooks from top to bottom.
You will find that the <b>'Q2_Master.ipynb'</b> has not been submitted with the outputs that we ran. This is because Google Colab does not
allow you to download .ipynb files with the outputs present. There is a corresponding pdf file in the main directory named <b>'Q2_Master.pdf'</b> which contains
evidence of the outputs that we ran (i.e it is the <b>'Q2_Master.ipynb'</b> notebook but with the outputs shown - but please look at the actual
<b>'Q2_Master.ipynb'</b> file for extra comments and documentation).


#### For Folders:

Inside the <b>'.github/workflows'</b> folder you can find four .yml files which consists of code that will automate PEP8 testing (both for .py and .ipynb notebooks), pytests and sphinx documentation everytime there is a a push to the repository.

Inside the <b>'docs'</b> folder you can find the documentation for how to use the functions inside the 'tools' folder inside docs/creek.pdf or open the html/index.html on your local computer. 

Inside the <b>'experiment'</b> folder you will find evidence of four extra methods for question 2 (each method has a .ipynb file and a corresponding
.pdf file). There is also a file named 'Question_1_Experiments.ipynb' in the experiment folder which includes some extra experiments/methods
for question 1. Also included is a notebook demonstrating the use of a holoviews slider widget for visualising the progression of the wildfires. And finally a notebook for saving reshaped data called Data_Shaping.ipynb.

Inside the <b>'research_parameters'</b> folder you can find two notebooks that are associated with question 2. These are named 'q2_vary_PCA_B_R_matrix.ipynb' and 'q2_vary_p_vs_mse.ipynb'. Both of these notebooks contain our investigation into optimising the paramters we used in 'Q2_Master.ipynb'.

Inside the <b>'tools'</b> folder you will find the functions that we have imported and used throughout our notebooks. They are under the names 'dataassimilation.py', 'loaddata.py' and 'visualisation.py'. Also, inside the 'tools' folder there is also a folder named 'tests' which
enables you to do pytests on the functions within the 'tools' folder.

##### Note on holoviews

A function to create an interactive widget that helps the user visualise the progression of the wildfires was built using the holoviews third party package. However, using these widgets inside jupyter notebooks leads to filesize increases that overstep gits limit. For this reason, we haven't included any in our master notebooks. We have, however, provided a demo in the "experiment/visualisation.ipynb" notebook. The user can run this locally to view some of the functionality of the visualisation widget.

#### For CSVs:
The values of the computed MSEs for each question are in .csv files in the main directory. For question 1 it is named 'Q1.csv' and for question 
2 it is named 'Q2.csv'.

### How to run the notebooks

In order to run the notebooks in this repository, you must have the 4 datasets in 4 separate folders INSIDE a
parent folder named 'data'. 
 
Within the 'data' folder you must have the training data, test data, background data and observation data saved
in 4 separate folders under the following names:

1. Training data must be in a folder named 'train'
    - Source: https://drive.google.com/file/d/11S0uXpT6rHOsph1OsMu5cGRNG4ieSjLW/view?usp=sharing​

2. Test data must be in a folder named 'test'
    - Source: https://drive.google.com/file/d/1K3q1g3gfacK7RL6VLHnRmK1QGEykws_m/view?usp=sharing​

3. Background data must be in a folder named 'background'
    - Source: https://drive.google.com/file/d/1DUvPaEnFTT1hQOHy0SHAZxfoKnxQKMid/view?usp=sharing​

4. Observation data must be in a folder named 'satellite'
    - Source: https://drive.google.com/file/d/1pK7W082NEpS5rL7e5_MbHmY_51mliu9N/view?usp=sharing​

ie. located in data/train, data/test, data/background, data/satellite

#### Important Note
The Q1_Master.ipyb was run on MacBook Pro with 16 GB RAM (16 GB 3733 MHz LPDDR4X). This was suficient to run model on full dataset without running it Kernel died issues. 
The compression files ie. Q2_Master.ipynb and all the .ipynb files in the experiment folder is ran on google colab with high RAM TPU.
However, from test runs on our local machine it seems like Q2_Master.ipynb, 'Q2_Linear_Autoencoder.ipynb', 'Q2_Nonlinear_Autoencoder.ipynb' works fine for a subset of the data. But, Q2_CAE.ipynb and Q2_CNN.ipynb only work on google colab.

In addition, all the .ipynb in research_parameters can be ran on the local machine



## Documentation

What documentation is available and where to find it?

You can find the documentation for how to use the functions inside the 'tools' folder inside docs/creek.pdf or open the html/index.html on your local computer. 

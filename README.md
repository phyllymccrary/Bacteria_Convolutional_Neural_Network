# Machine Learning Project

__**CAVEAT: THE DATA FILES THAT WERE USED TO TRAIN THE MODELS ARE NOT INCLUDED! But it's still interesting to present the code that was implemented and results obtained.**__

## Bacteria Neural Network

This project involves creating a Convolutional Neural Network (CNN) dedicated to learning the Pseudomonas aeruginosa bacteria. 

## Project Structure
Machine Learning Project: Base Project Directory  
&nbsp; |- Code: directory for hosting the code  
&nbsp;&nbsp;&nbsp;&nbsp; |- Jupyter: directory for Jupyter Notebooks  
&nbsp;&nbsp;&nbsp;&nbsp; |- Python: directory for Native Python files  
&nbsp; |- History: directory for the saved Loss values per Epoch of a pre-trained model  
&nbsp;&nbsp;&nbsp;&nbsp; |- History Plots: directory of History Plots for every column.  
&nbsp; |- Models: directory for the pre-trained Models of a specific column

## Code

Below are details on the different classes within this project. To successfully run the code, you will have to provide a few values, which are listed below. 

However, we provide pre-trained models for all of the columns for your evaluation. If you want to explore the results we obtained with our models, simply skip down to the [Bacteria_Network_Evaluation](###Bacteria_Network_Evaluation) section.

### Bacteria_Image_Preprocessing 
This class augments images within the __**RAW**__ image directory. It associates the strain id within the excel sheet with the number listed in the file name and moves the image into a newly created directory. Then the code applies several transformations (resizing, grayscaling, rotating, zooming, etc) and saves the augmented image to disk. With the image file path, a new excel sheet is created with the strain_id and image file path. This Images sheet is finally merged with the 'Total Database' sheet and forms a 1-to-Many relationship table, which is used for training the Convolutional Neural Network.

- Path location to the directory where images are stored
   - Variable name: base_directory
- Path location to the excel file
   - Variable name: bacteria_workbook

### Bacteria_Network_Training
This class trains a Convolutional Neural Network Model for every column within the 'Total Database' or 'CNN Dataset' worksheet. You can individually train a CNN per column or train every column. Once the model training is completed, its architecture, weights (as .h5 files), and history (as .json files) are saved to disk for future evaluation.

- Path location to the directory where images are stored
   - Variable name: base_directory
- Path location to the excel file
   - Variable name: bacteria_workbook
- Path location for saving model and history
   - Variable name: save_dir
- Column name extracted from excel sheet(s): Total Database or CNN Dataset 
   - Variable name: data_column

### Bacteria_Network_Evaluation
This class utilizes the saved trained models for evaluation. You can explore the history of the model's loss over a number of epochs or test the model on a new dataset.

__**If you would like to see the Model's History without running code, you can view the plot of a specific column by going to the /history/history_plots directory and explore.**__

- Path location for directory where you saved models and history
   - Variable name: save_dir
- Column name extracted from excel sheet(s): Total Database or CNN Dataset 
   - Variable name: data_column
- Path location for an image you are interested in evaluating
   - Variable name: img_path

Before you can run this code, you must download the required packages listed below.

## Required Packages

Need [Python 3 (we used 3.6.5)](https://www.python.org/downloads/release/python-365/)

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install:


[Augmentor](https://augmentor.readthedocs.io/en/master/): Image augmentation library for Machine Learning 
```bash
pip install Augmentor
```

[Pillow](https://pillow.readthedocs.io/en/5.3.x/): Python Imaging Library
```bash
pip install Pillow
```

[tqdm](https://tqdm.github.io/): Progress bar visualizer for iterables
```bash
pip install tqdm
```

[Pandas](https://pandas.pydata.org/): Powerful data structures for data analysis, time series, and statistics 
```bash
pip install pandas
```

[Openpyxl](https://openpyxl.readthedocs.io/en/stable/): openpyxl is a Python library to read/write Excel files.
```bash
pip install openpyxl
```

[Xlrd](https://xlrd.readthedocs.io/en/latest/): Extracts data from Excel spreadsheets on any platform
```bash
pip install xlrd
```

[Numpy](https://www.numpy.org/): a general-purpose array-processing package designed to efficiently manipulate large multi-dimensional arrays  
```bash
pip install numpy
```

[Keras](https://keras.io/):  a high-level neural networks API
```bash
pip install keras==2.1.3
```

[Matplotlib](https://matplotlib.org/): Python plotting package
```bash
pip install matplotlib
```

[Tensorflow](https://www.tensorflow.org): an open source software library for high performance numerical computation.
```bash
pip install tensorflow
```

[H5py](https://www.h5py.org/): a Pythonic interface to the HDF5 binary data format.
```bash
pip install h5py
```

[Scikit-Image](https://scikit-image.org/): an open-source image processing library.
```bash
pip install scikit-image
```

### __**If there are any errors about missing packages, please visit [PyPI](https://pypi.org/) to download the required package using [pip](https://pip.pypa.io/en/stable/). Thank you.**__
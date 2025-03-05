# Rationally Design the Ionic Liquid-based Absorbents for CO₂ Absorption using Machine Learning

## Project description

This study presents a method for rapid screening of ionic liquids(ILs) using Graph Neural Networks (GNNs). Approximately 40,000 experimental data points, including CO2 solubility, viscosity, melting point, and toxicity, were collected to train GNN regression models and XGBoost classification models. Various strategies for constructing molecular graphs based on IL structures were analyzed, and multiple GNN models were compared. A database of IL structures containing 200,000 data points was built, and screening threshold criteria were established. Through this process, ideal IL-based CO2 absorbents with low viscosity, non-toxicity, and liquid state at room temperature were identified.

## HOW TO

1.Add a column named Split to the clean datast or raw dataset and randomly divide it into Training or Testing according to a certain ratio, then change the file path in the code to your file path to run it.

2.Models are trained individually for different properties, look at the file name can distinguish each property of the corresponding data and models.

3.First open the Model folder and run the .py file to train the model, then open the Screening folder and run it according to the serial number to realize the screening.

4.The path of the model will be saved in the training process, when you run the code in the screening process, change it to the corresponding model path.

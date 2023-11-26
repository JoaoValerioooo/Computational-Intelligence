# Code for Visual Explainability in Image Classification using Grad-CAM in Fine-Tuned InceptionV3, ResNet50 and VGG16 Models 

## Team members

* João Valério
* Eirik Grytøyr
* Moritz Andres
* John Mocettini



## Running the scripts
The project is provided with for different .ipnb
1. Transfer learning with BO ResNet50.ipynb
2. Transfer learning with BO VGG16.ipynb
3. Transfer learning with BO InceptionV3
4. Baseline Models - Class Conversion.ipynb
5. GradCam Analysis.ipynb

The first three will use Bayesian optimization to find the best hyperparameters for each model and print the results from optimization, training, and testing.
To execute the code, the path in the first cell has to be changed to the position, where the models and optimization data should be stored.

The fourth code is used to test the base models without adaption or training, but with conversion between the Image net labels and the CIFAR 10.2 labels.
Execution will print out the test data from this.

The fifth code is used to obtain grad-CAM Heat maps for the best and worst predictions for the models.
To execute the code, change the path in the first cell to the position where the models are saved from the three first codes.
Then uncomment the model you want to analyze in the second last cell.
The number of best and worst predictions to show for each category can be selected by changing the n value in the last cell.
To execute and plot the results. execute the last cell.
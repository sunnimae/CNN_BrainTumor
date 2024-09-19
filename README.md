# CNN_BrainTumor
Project 4 - Creating a Convolutional Neural Network to predict if a scan image shows the presence of a brain tumor
Project 4: Brain Tumor Classification Using Machine Learning Background This project aims to classify MRI images as either containing a brain tumor or being normal, using a convolutional neural network (CNN) model. The goal is to create an automated tool that can assist in the early detection of brain tumors, which is crucial for timely treatment. Additionally, we present statistical information—via Tableau—on the most common types of cancer surgeries, top hospitals for brain cancer treatment, and the number of surgeries from 2012 to 2022.

Dataset The dataset consists of MRI brain scans, categorized into two classes:

Normal: MRI scans without any tumors. Tumor: MRI scans with identified brain tumors. Data Source: The dataset was sourced from Kaggle and consists of images split between normal and tumor classes. [Kaggle MRI Brain Scans Dataset](Dataset Link)

Project Objective The primary objective is to build a machine-learning model that can accurately classify MRI images as either normal or containing a tumor. This project involves the following steps:

Data Preprocessing and Augmentation Model Training using Convolutional Neural Networks (CNN) Model Improvement using Transfer Learning (VGG16) Performance Evaluation and Visualization Data Preprocessing Image Loading and Dataset Creation The tf.keras.utils.image_dataset_from_directory() function was used to load and preprocess images. This modern approach offers better integration with TensorFlow 2.x, optimizing the input pipeline for performance and consistency.

Data Augmentation Data augmentation techniques were applied to address the dataset's small size and prevent overfitting, including random flipping, brightness adjustment, and contrast variation.

Splitting Data The dataset was split into training and validation sets, with 20% of the data reserved for validation.

Model Architecture Convolutional Neural Network (CNN) A basic CNN model was initially developed with three convolutional layers followed by max pooling, a flattening layer, and dense layers leading to a final output layer with a sigmoid activation function for binary classification.

Transfer Learning with VGG16 Transfer learning was applied to improve the model’s accuracy using the pre-trained VGG16 model, which was fine-tuned for this specific task. The VGG16 model's layers were frozen, and custom layers were added for classification.

Early Stopping Early stopping was implemented to prevent overfitting by monitoring the validation loss and halting training when it stopped improving.

Model Training Initial CNN Model: Trained for 10 epochs, resulting in a validation accuracy of around 65%. Final Model: After implementing data augmentation and transfer learning, the model was re-trained for 70 epochs, with early stopping applied. Results and Evaluation Final Validation Accuracy: ~98% Loss Function: Binary cross-entropy was used as it is appropriate for binary classification tasks. Optimizer: Adam optimizer was used with a learning rate of 0.001 for the initial model and 0.0001 for the transfer learning model. Performance Visualization Training and validation accuracy and loss were plotted to visualize the model’s performance over the epochs. A histogram of prediction probabilities was generated to analyze the model's output distribution.

Ethical Considerations Data Privacy: Personal identifiers were removed from the dataset to protect patient privacy.

Check for Bias: Efforts were made to ensure that the model fairly represents both normal and tumor classes, with balanced training data. Model Interpretability: The model's predictions were analyzed for fairness and accuracy, considering the serious implications of false negatives in medical diagnoses.

Visualizations Visualized Images from the dataset

Tools and Libraries

Python

Matplotlib

TensorFlow

NumPy

Google Colab

Tableau - https://public.tableau.com/app/profile/brake.chias/viz/BrainTumorVisualization/Story1

How to Use Download the Dataset: Download the MRI brain scans dataset from Kaggle and place the images in the appropriate directories (Normal and Tumor). Run the Code: Execute the code in a Python environment such as Google Colab or Jupyter Notebook. Ensure TensorFlow, Keras, and other required libraries are installed. Training and Validation: The model will automatically train on the dataset and evaluate its performance on the validation set.

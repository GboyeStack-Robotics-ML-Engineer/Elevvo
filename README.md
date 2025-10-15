# Internship Projects at Eleevo Pathways

This repository contains the Jupyter notebooks for the four major tasks completed during my internship at Eleevo Pathways. Each notebook details a machine learning project, from data exploration and preprocessing to model building and evaluation.

## Projects

### Task 2: Customer Segmentation

* **Description**: This project focuses on segmenting customers based on their shopping behavior to identify distinct customer groups.
* **Dataset**: The "Mall_Customers.csv" dataset was used, which contains information about mall customers including age, gender, annual income, and spending score.
* **Approach**:
    * The notebook begins with exploratory data analysis to understand the customer data.
    * A custom K-Means clustering algorithm was implemented from scratch.
    * The `KMeans` algorithm from scikit-learn was also used for comparison and clustering the customers.
    * The silhouette score was calculated to evaluate the performance of the clustering models.
    * The final customer segments were visualized in a 3D scatter plot.

### Task 4: Loan Approval Prediction

* **Description**: This project involves building a model to predict whether a loan application will be approved or rejected.
* **Dataset**: The "loan_approval_dataset.csv" dataset was used for this task.
* **Approach**:
    * The notebook starts with data cleaning and preprocessing to handle categorical features and prepare the data for modeling.
    * The features were scaled using `StandardScaler` to normalize the data.
    * A Logistic Regression model was built and trained on the preprocessed data.
    * The model's performance was evaluated using an accuracy score, a classification report, and a confusion matrix.

### Task 6: Music Genre Classification

* **Description**: This project aims to classify the genre of a 30-second music clip.
* **Dataset**: The GTZAN dataset was used, which consists of images of spectrograms for different music genres.
* **Approach**:
    * The dataset was split into training, validation, and testing sets with an 80:9:10 ratio.
    * A Convolutional Neural Network (CNN) model was built using TensorFlow and Keras for image classification.
    * The model was trained on the spectrogram images to learn the features of each music genre.
    * The performance of the model was evaluated using accuracy and loss metrics, and visualized with a confusion matrix.

### Task 8: Traffic Sign Recognition

* **Description**: This project focuses on building a model to recognize and classify traffic signs from images.
* **Dataset**: The German Traffic Sign Recognition Benchmark (GTSRB) dataset was used.
* **Approach**:
    * The dataset was split into training and validation sets using stratified sampling.
    * A custom PyTorch `Dataset` class was created to load and preprocess the image data.
    * A custom Convolutional Neural Network (CNN) model was designed and built using PyTorch.
    * The model was trained for 20 epochs with an Adam optimizer and Cross-Entropy Loss.
    * The model's performance was evaluated using a confusion matrix and accuracy score.

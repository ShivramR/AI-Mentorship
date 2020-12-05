# AI Mentorship
Classifying EMG streams on the Myo armband with Keras and TensorFlow

_Virginia Commonwealth University, 2019-2020_

## Table of contents
* [Rationale](#Rationale)
* [An ML model structure](#ML-model-structure)
* [Accomplishments](#Accomplishments)
* [Future work](#Future-work)
* [Repository files](#Repository-files)

## Rationale
Using a [Myo armband](https://developerblog.myo.com/), I developed a wrist gesture classification algorithm in Python via machine learning models built in Keras and Google's TensorFlow. The rationale was to experiment with and develop a prototype solution for teleoperated robotics and haptic feedback systems, using this armband as a controller. "Telerobotics" thus deals with remote controls for potentially hazardous and distanced environments such as toxic waste sites, climate disaster zones, and underseas. My program is a proof of concept for a real-time system that can reach an accuracy of ~80% in classifying wrist gestures, and an ML model template for such time-based data.

## ML model structure
![A version of the ML model structure](https://github.com/ShivramR/AI-Mentorship/blob/main/model_graph.png)

## Accomplishments
* Developed a [complete dataset](https://github.com/ShivramR/AI-Mentorship/blob/main/emg-final.csv) of various wrist positions for ML model training
* Created a robust yet simple machine learning model from open libraries in Keras and TensorFlow
* Developed an algorithm that employs the model for real-time (200 Hz stream) prediction
* Explored the use of one-dimensional convolutional neural networks for time-linked data
* Used t-SNE and PCA plots to visually plot the overlap and distinctness of defined gestures

## Future work
* Develop an application that synthesizes datasets for individuals
  * A calibration procedure with which models can be readily trained and tested
* Experiment with different ML models and hyperparameter tuning to improve accuracy and robustness
* Determine a task-based accuracy metric (e.g. how closely a robot operated by the Myo adheres to the controls)
* Build an app with a visual interface that uses the Myo as a bluetooth control

## Repository files
* [Research paper](https://github.com/ShivramR/AI-Mentorship/blob/main/Research%20Findings.pdf)
* [Custom Myo EMG dataset](https://github.com/ShivramR/AI-Mentorship/blob/main/emg-final.csv)
* [EMG data collector](https://github.com/ShivramR/AI-Mentorship/blob/main/emg_collector.cpp)
  * A C++ script used to generate a .csv file for each assigned position
* [IPython ML notebook](https://github.com/ShivramR/AI-Mentorship/blob/main/model_creator.ipynb)
  * The notebook used for synthesizing the Keras model from the passed EMG dataset and testing accuracy
* [Final prediction model](https://github.com/ShivramR/AI-Mentorship/blob/main/model_normalized2.h5)
* [Real-time prediction program](https://github.com/ShivramR/AI-Mentorship/blob/main/prediction_util.py)
  * A Python script that prints real-time prediction values in the console from a passed model after armband initialization

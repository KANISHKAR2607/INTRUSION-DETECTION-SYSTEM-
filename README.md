# Intrusion-Detection-System-Using-Machine-Learning

This repository contains the code for the project "IDS-ML: Intrusion Detection System Development Using Machine Learning". The code and proposed Intrusion Detection System (IDSs) are general models that can be used in any IDS and anomaly detection applications. 


###  MTH-IDS: A Multi-Tiered Hybrid Intrusion Detection System for Internet of Vehicles
&emsp; Modern vehicles, including connected vehicles and autonomous vehicles, nowadays involve many electronic control units connected through intra-vehicle networks to implement various functionalities and perform actions. Modern vehicles are also connected to external networks through vehicle-to-everything technologies, enabling their communications with other vehicles, infrastructures, and smart devices. However, the improving functionality and connectivity of modern vehicles also increase their vulnerabilities to cyber-attacks targeting both intra-vehicle and external networks due to the large attack surfaces. To secure vehicular networks, many researchers have focused on developing intrusion detection systems (IDSs) that capitalize on machine learning methods to detect malicious cyber-attacks. In this paper, the vulnerabilities of intra-vehicle and external networks are discussed, and a multi-tiered hybrid IDS that incorporates a signature-based IDS and an anomaly-based IDS is proposed to detect both known and unknown attacks on vehicular networks. Experimental results illustrate that the proposed system can accurately detect various types of known attacks on the CAN-intrusion-dataset representing the intra-vehicle network data and the CICIDS2017 dataset illustrating the external vehicular network data.  
&emsp; The proposed MTH-IDS framework consists of two traditional ML stages (data pre-processing and feature engineering) and four tiers of learning models: 
1. Four tree-based supervised learners — decision tree (DT), random forest (RF), extra trees (ET), and extreme gradient boosting (XGBoost) — used as multi-class classifiers for known attack detection; 
2. A stacking ensemble model and a Bayesian optimization with tree Parzen estimator (BO-TPE) method for supervised learner optimization; 
3. A cluster labeling (CL) k-means used as an unsupervised learner for zero-day attack detection; 
4. Two biased classifiers and a Bayesian optimization with Gaussian process (BO-GP) method for unsupervised learner optimization. 

**<p align="center">Figure : The overview of the MTH-IDS model.</p>**
<p align="center">
<img src="https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-Machine-Learning/blob/main/Figures/MTH-IDS_Overview.png" width="700" />
</p>


###  LCCDE: A Decision-Based Ensemble Framework for Intrusion Detection in The Internet of Vehicles
&emsp; Modern vehicles, including autonomous vehicles and connected vehicles, have adopted an increasing variety of functionalities through connections and communications with other vehicles, smart devices, and infrastructures. However, the growing connectivity of the Internet of Vehicles (IoV) also increases the vulnerabilities to network attacks. To protect IoV systems against cyber threats, Intrusion Detection Systems (IDSs) that can identify malicious cyber-attacks have been developed using Machine Learning (ML) approaches. To accurately detect various types of attacks in IoV networks, we propose a novel ensemble IDS framework named Leader Class and Confidence Decision Ensemble (LCCDE). It is constructed by determining the best-performing ML model among three advanced ML algorithms (XGBoost, LightGBM, and CatBoost) for every class or type of attack. The class leader models with their prediction confidence values are then utilized to make accurate decisions regarding the detection of various types of cyber-attacks. Experiments on two public IoV security datasets (Car-Hacking and CICIDS2017 datasets) demonstrate the effectiveness of the proposed LCCDE for intrusion detection on both intra-vehicle and external networks. 

**<p align="center">Figure : The overview of the LCCCDE IDS model.</p>**
<p align="center">
<img src="https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-Machine-Learning/blob/main/Figures/LCCDE_Overview.jpg" width="800" />
</p>


## Implementation 
### Dataset 

CAN-intrusion dataset, a benchmark network security dataset for intra-vehicle intrusion detection
* Publicly available at: https://ocslab.hksecurity.net/Datasets/CAN-intrusion-dataset  
* Can be processed using the same code

### Machine Learning Algorithms  
* Decision tree (DT)
* Random forest (RF)
* Extra trees (ET)
* XGBoost  
* LightGBM  
* CatBoost  
* Stacking
* K-means

### Hyperparameter Optimization Methods  
* Bayesian Optimization with Gaussian Processes (BO-GP)
* Bayesian Optimization with Tree-structured Parzen Estimator (BO-TPE)  

If you are interested in hyperparameter tuning of machine learning algorithms, please see the code in the following link:  
https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

### Requirements & Libraries  
* Python 3.6+ 
* [scikit-learn](https://scikit-learn.org/stable/)  
* [Xgboost](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)
* [lightgbm](https://lightgbm.readthedocs.io/en/v3.3.2/Python-Intro.html)
* [catboost](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)
* [FCBF](https://github.com/SantiagoEG/FCBF_module)
* [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize)  
* [hyperopt](https://github.com/hyperopt/hyperopt)   
* [River](https://riverml.xyz/dev/)  

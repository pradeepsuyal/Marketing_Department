![Apache License](https://img.shields.io/hexpm/l/apa)  ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)  [![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)    ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)   ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)  ![Made with matplotlib](https://user-images.githubusercontent.com/86251750/132984208-76ce70c7-816d-4f72-9c9f-90073a70310f.png)  ![seaborn](https://user-images.githubusercontent.com/86251750/132984253-32c04192-989f-4ebd-8c46-8ad1a194a492.png)  ![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)  ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![tableau](https://img.shields.io/badge/Tableau-E97627?style=for-the-badge&logo=Tableau&logoColor=white) ![tensorflow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) ![keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white) ![medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white) ![coursera](https://img.shields.io/badge/Coursera-0056D2?style=for-the-badge&logo=Coursera&logoColor=white) ![udemy](https://img.shields.io/badge/Udemy-EC5252?style=for-the-badge&logo=Udemy&logoColor=white)

![Screenshot (212)](https://user-images.githubusercontent.com/86251750/146566666-3bed15fe-4122-41fb-97cd-fee238af321d.png)

## Marketing Department

* Marketing is crucial for the growth and sustainability of any business.

* Marketers can help build the company’s brand, engage customers, grow revenue, and increase sales.

* One of the key pain points for marketers is to know their customers and identify their needs.

* By understanding the customer, marketers can launch a targeted marketing campaign that is tailored for specific needs.

* If data about the customers is available, data science can be applied to perform market segmentation. 

![image](https://user-images.githubusercontent.com/86251750/146567041-c45f18dc-11b7-4fe6-8527-4b64a57c6ace.png)

## Acknowledgements

 - [python for ML and Data science, udemy](https://www.udemy.com/course/python-for-machine-learning-data-science-masterclass)
 - [ML A-Z, udemy](https://www.udemy.com/course/machinelearning/)
 - [ML by Stanford University ](https://www.coursera.org/learn/machine-learning)

## Appendix

* [Aim](#aim)
* [Dataset used](#data)
* [Run Locally](#run)
* [Exploring the Data](#viz)
   - [Matplotlib](#matplotlib)
   - [Seaborn](#seaborn)
* [solving the task](#fe)
* [prediction](#models)
* [conclusion](#conclusion)

## AIM:<a name="aim"></a>

The marketing team at the bank wants to launch a targeted ad marketing campaign by dividing their customers into at least 3 distinctive groups. So we need to cluster the data into different groups with similar properties.

## Dataset Used:<a name="data"></a>

The bank has extensive data on their customers for the past 6 months:

`CUSTID: Identification of Credit Card holder`

`BALANCE: Balance amount left in customer's account to make purchases`

`BALANCE_FREQUENCY: How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)`

`PURCHASES: Amount of purchases made from account`

`ONEOFFPURCHASES: Maximum purchase amount done in one-go`

`INSTALLMENTS_PURCHASES: Amount of purchase done in installment`
 
`CASH_ADVANCE: Cash in advance given by the user`
 
`PURCHASES_FREQUENCY: How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)`

`ONEOFF_PURCHASES_FREQUENCY: How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)`

`PURCHASES_INSTALLMENTS_FREQUENCY: How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)`

`CASH_ADVANCE_FREQUENCY: How frequently the cash in advance being paid`

`CASH_ADVANCE_TRX: Number of Transactions made with "Cash in Advance"`

`PURCHASES_TRX: Number of purchase transactions made`

`CREDIT_LIMIT: Limit of Credit Card for user`

`PAYMENTS: Amount of Payment done by user`

`MINIMUM_PAYMENTS: Minimum amount of payments made by user`

`PRC_FULL_PAYMENT: Percent of full payment paid by user`

`TENURE: Tenure of credit card service for user`

dataset can be found at [link](https://github.com/pradeepsuyal/Marketing_Department/tree/main/dataset)

## Run locally:<a name="run"></a>

Clone the project

```bash
  git clone https://github.com/pradeepsuyal/Marketing_Department.git
```

Go to the project directory

```bash
  cd Marketing_Department
```

Install dependencies

```bash
  npm install
```

Start the server

```bash
  npm run start
```

If you output `pip freeze` to a file with redirect >, you can use that file to install packages of the same version as the original environment in another environment.

First, output requirements.txt to a file.

```bash
  $ pip freeze > requirements.txt
```

Copy or move this `requirements.txt` to another environment and install with it.

```bash
  $ pip install -r requirements.txt
```

## Exploring the Data:<a name="viz"></a>

I have used pandas, matplotlib and seaborn visualization skills.

**Matplotlib:**<a name="matplotlib"></a>
--------
Matplotlib is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms. Matplotlib can be used in Python scripts, the Python and IPython shells, the Jupyter notebook, web application servers, and four graphical user interface toolkits.You can draw up all sorts of charts(such as Bar Graph, Pie Chart, Box Plot, Histogram. Subplots ,Scatter Plot and many more) and visualization using matplotlib.

Environment Setup==
If you have Python and Anaconda installed on your computer, you can use any of the methods below to install matplotlib:

    pip: pip install matplotlib

    anaconda: conda install matplotlib
    
    import matplotlib.pyplot as plt

![matplotlib](https://miro.medium.com/max/1000/0*DJ_mGzidt4WqWoVm)

for more information you can refer to [matplotlib](https://matplotlib.org/) official site

**Seaborn:**<a name="seaborn"></a>
------
Seaborn is built on top of Python’s core visualization library Matplotlib. Seaborn comes with some very important features that make it easy to use. Some of these features are:

**Visualizing univariate and bivariate data.**

**Fitting and visualizing linear regression models.**

**Plotting statistical time series data.**

**Seaborn works well with NumPy and Pandas data structures**

**Built-in themes for styling Matplotlib graphics**

**The knowledge of Matplotlib is recommended to tweak Seaborn’s default plots.**

Environment Setup==
If you have Python and Anaconda installed on your computer, you can use any of the methods below to install seaborn:

    pip: pip install seaborn

    anaconda: conda install seaborn
    
    import seaborn as sns
    
![seaborn](https://miro.medium.com/max/432/1*Ai7xvtO40nWx3pk1MIfxtg.gif)

for more information you can refer to [seaborn](https://seaborn.pydata.org/) official site.

**Screenshots from notebook**

![download](https://user-images.githubusercontent.com/86251750/146571598-a63ba603-8c05-4e0d-afe7-c2d96331bef9.png)

## approach for making prediction<a name="fe"></a>
-------

* My first step was to explore the data and gain insights from it.
* scaling the data.
* applying k-means and choose the optimal number of cluster with elbow method.
* applying Principal component analysis for dimensionality reduction.
* Than I perform dimensionality reduction using AutoEncoders
* Finally I make the prediction with the weights that provide me the lowest loss value.


## Prediction:<a name="models"></a>
------

**K-Means**

* K-means is an unsupervised learning algorithm (clustering).
* K-means works by grouping some data points together (clustering) in an unsupervised fashion.  
* The algorithm groups observations with similar attribute values together by measuring the Euclidian distance between points.

![image](https://user-images.githubusercontent.com/86251750/146573681-fff8b9a5-c02d-44aa-aa78-8ba5ee216979.png)

    K-MEANS ALGORITHM STEPS:
    
    1-Choose number of clusters “K”

    2-Select random K points that are going to be the centroids for each cluster

    3-Assign each data point to the nearest centroid, doing so will enable us to create “K” number of clusters 

    4-Calculate a new centroid for each cluster

    5-Reassign each data point to the new closest centroid

    6-Go to step 4 and repeat.
    
![image](https://user-images.githubusercontent.com/86251750/146574455-958a29b8-19e8-437a-9340-2166883059af.png)

    SELECT THE OPTIMAL NUMBER OF CLUSTERS (K) USING “ELBOW METHOD”
    
![image](https://user-images.githubusercontent.com/86251750/146574834-046b4d9d-55d1-4dee-9235-a7af0af5299b.png)

![image](https://user-images.githubusercontent.com/86251750/146575185-ad63f4bc-cc37-4be2-8ff3-3825b101c344.png)

![1](https://user-images.githubusercontent.com/86251750/146575611-ed56789e-99e9-4f44-b012-c25ed189b73c.PNG)

![image](https://user-images.githubusercontent.com/86251750/146575928-0c074902-6814-4ce5-9f61-3881f21b5e6a.png)

      for three clusters:
     
![2](https://user-images.githubusercontent.com/86251750/146576625-81b6477c-750a-44cd-a95a-b1054b1bdcf2.PNG)

![image](https://user-images.githubusercontent.com/86251750/146576811-f4603624-bda0-4f92-8dbb-d71c8d5b9359.png)

    selecting optimal number pf cluster using elbow method

![3](https://user-images.githubusercontent.com/86251750/146576941-6934ccda-88c9-4e52-9edb-5d18f71993ab.PNG)

[source](https://commons.wikimedia.org/wiki/File:Tennis_Elbow_Illustration.jpg)

**AutoEncoders**
       
* Auto encoders are a type of Artificial Neural Networks that are used to perform a task of data encoding (representation learning). 
* Auto encoders use the same input data for the input and output.

![image](https://user-images.githubusercontent.com/86251750/146577451-597a6c0d-c483-4aad-86d9-aa756cb90bb7.png)

photo credit [link1](https://commons.wikimedia.org/wiki/File:Autoencoder_structure.png), [link2](https://commons.wikimedia.org/wiki/File:Artificial_neural_network_image_recognition.png)

* Auto encoders work by adding a bottleneck in the network.
* This bottleneck forces the network to create a compressed (encoded) version of the original input
* Auto encoders work well if correlations exists between input data (performs poorly if the all input data is independent) 

![image](https://user-images.githubusercontent.com/86251750/146578297-410ba658-d6da-4a4a-bbe6-c0f743e55572.png)

    AUTOENCODER MATH
    
    ENCODER:`ℎ(𝑥)=𝑠𝑖𝑔𝑚𝑜𝑖𝑑(𝑊∗𝑥+𝑏)`
    
    DECODER:`𝑥 ̂=𝑠𝑖𝑔𝑚𝑜𝑖𝑑(𝑊^∗∗ℎ(𝑥)+𝑐)`

    TIED WEIGHTS: `Weights from input to hidden layer will be equal to the weights from hidden layer to output`
                   `𝑊^∗=𝑊^𝑇`
    
    
**PRINCIPAL COMPONENT ANALYSIS**

* PCA is an unsupervised machine learning algorithm.
* PCA performs dimensionality reductions while attempting at keeping the original information unchanged. 
* PCA works by trying to find a new set of features called components.
* Components are composites of the uncorrelated given input features.

![image](https://user-images.githubusercontent.com/86251750/146582085-18e10b0e-2da8-4877-8d35-ef45472e1304.png)

[photo credit](http://phdthesis-bioinformatics-maxplanckinstitute-molecularplantphys.matthias-scholz.de/)


## CONCLUSION:<a name="conclusion"></a>
-----
* with K-MEANS I was able to get 8 clusters(with the help of ELBOW METHOD) and than I applied PCA to saw the cluster

![download](https://user-images.githubusercontent.com/86251750/146580641-d259c976-c719-42b6-9cb9-676389d11355.png)

    as we can see that our model was able to seperate cluster with distinct property
    
* Than I applied AUTOENCODERS to perform dimensionality reduction

![download](https://user-images.githubusercontent.com/86251750/146581235-a6e09c24-b3d6-4f0c-9085-99b3147af80e.png)

    Here I got 4 cluster and we can clearly see that the clusters are more seperable from each other with respect to PCA .
    Thus we can conclude that AutoEncoders gives us better performance

    NOTE--> we can further improve the performance by using other clustering model such as Hierarchical Clustering, DBSCAN – Density-based Spatial Clustering,
            Agglomerative Hierarchical Clustering and many more. Further performance can be improved by using various hyperparameter optimization technique such as optuna,hyperpot, Grid Search, Randomized Search, etc.   


from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import graphviz

list_ = open("traceDMA.txt").read().split()
port=[]
size=[]
clss=[]
colour=[]
for i in range(0,len(list_)):
if i%3==0:
port.append(list_[i])
elif i%3==1:
size.append(list_[i])
else:
clss.append(list_[i])
for i in clss:
if i == "udp":
colour.append('b')
else:
colour.append('r')
port = list(map(int, port))
size = list(map(int, size))
portlog=[]
sizelog=[]
portlog=np.log(port)
sizelog=np.log(size)
figure=plt.figure()
frame=plt.gca()
plt.ylabel("SIZE")
plt.xlabel("PORT")
colors = ["r", "b"]
texts = ["TCP", "UDP"]
patches = [ plt.plot([],[], marker="o", ms=10, ls="", mec=None, color=colors[i],
label="{:s}".format(texts[i]) )[0] for i in range(len(texts)) ]
plt.scatter(port,size, c=colour,alpha = 0.4)
frame.axes.set_autoscale_on(True)
plt.show()
#Creating a log graph
figure1=plt.figure()
frame1=plt.gca()
plt.ylabel("SIZE")
plt.xlabel("PORT")
colors = ["r", "b"]
texts = ["TCP", "UDP"]
patches = [ plt.plot([],[], marker="o", ms=10, ls="", mec=None, color=colors[i],
label="{:s}".format(texts[i]) )[0] for i in range(len(texts)) ]
plt.scatter(portlog,sizelog, c=colour,alpha = 0.4)
plt.show()
X= np.vstack((port,size)).T
model=tree.DecisionTreeClassifier(criterion='gini',splitter='best',
min_impurity_decrease=0.002)
model.fit(X,clss)
dot_data=tree.export_graphviz(model,out_file=None,feature_names=['port','size'],class_na
mes=['tcp','udp'],filled=True,rounded=True,special_characters=True)
graph = graphviz.Source(dot_data)
graph
#Predicting the class for the full dataset
class_predict=model.predict(X)
print(confusion_matrix(clss, class_predict))
print(classification_report(clss, class_predict))
print(accuracy_score(clss,class_predict))

#Splitting the data into Train and Test in ordered
X_train=X[:7000]
X_test=X[7001:]
y_train=clss[:7000]
y_test=clss[7001:]
model2=tree.DecisionTreeClassifier(criterion='gini',splitter='best',
min_impurity_decrease=0.002)
model2.fit(X_train,y_train)
#Predicting the class for the test dataset
class_predict_2=model2.predict(X_test)
print(confusion_matrix(y_test, class_predict_2))
print(classification_report(y_test, class_predict_2))
print(accuracy_score(y_test,class_predict_2))
#Splitting the data into Train and Test in random
X_train, X_test, y_train, y_test = train_test_split(X, clss, test_size=3000,
random_state=41)
model1=tree.DecisionTreeClassifier(criterion='gini',splitter='best',
min_impurity_decrease=0.002)
model1.fit(X_train,y_train)
#Predicting the class for the test dataset
class_predict_1=model1.predict(X_test)
print(confusion_matrix(y_test, class_predict_1))
print(classification_report(y_test, class_predict_1))
print(accuracy_score(y_test,class_predict_1))
import networkx as nx
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import roc_curve as rc
import pandas


#construct unweighted and directed PPI network

ppin = nx.read_edgelist("./yeast.edgelist")
#2.1.1
print("Number of nodes:" + str(nx.number_of_nodes(ppin)))
print("Number of edges:" + str(len(ppin.edges)))
print("Network is connected:" + str(nx.is_connected(ppin)))


#2.1.2
#sort graph by node degree
degree_sequence = sorted([d for n, d in ppin.degree()], reverse=True)
dmax = max(degree_sequence)
dmin = min(degree_sequence)
davg = (sum(degree_sequence)/len(ppin.edges))
print("Max node degree:" + str(dmax))
print("Min node degree:" + str(dmin))
print("Avg node degree:" + str(davg))

#2.1.3 - answer is 6, takes hours to run
#print("Diameter:" + str(nx.diameter(ppin)))

yf = np.loadtxt('./yeast.emd', skiprows=1)
yf1 = yf[:,1:]

#There is no documentation on how to read EMD files, unless this is an electrom microscopy file - which it isn't
#first line tells us that there are 6536 entries, each with 32 dimensions
#arrayOfArrays = [] #each entry represents one point with 32 entries
#table = open('./yeast.emd')
#lines = table.readlines()
#for line in lines:
#    coordinates = []
#    for coordinate in line.split():
#        coordinates.append(float(coordinate))
#    arrayOfArrays.append(np.array(coordinates))

#2.2.1:
model = TSNE(n_components=2, perplexity=100)
twoD = model.fit_transform(yf1)
plt.scatter(twoD[:,0], twoD[:,1], sb.color_palette())
plt.show()

#2.2.2:
#testing perplexity
fig, perp = plt.subplots()
for i in range(5,30,5):
    model = TSNE(n_components=2, perplexity=i)
    twoD = model.fit_transform(yf1)
    perp.scatter(twoD[:,0], twoD[:,1], sb.color_palette(), label = i)

perp.legend()
plt.show()

#2.3.1

int_edgelist = nx.read_edgelist("./yeast_int.edgelist")

def hamprod(x,y):
    return x * y

print(hamprod(yf1[2],yf1[3]))

#2.3.2
#create adjacency matrix
am = nx.to_pandas_adjacency(ppin)

#print(am)

real_edges = np.argwhere(am.values)
fake_edges = np.argwhere(am.values != 1)

#print(len(real_edges))
#1062675

#truncate to get the same size - need to be 50/50
real = real_edges[:1062675]
fake = fake_edges[:1062675]

realun = real[:,0]
realdu = real[:,1]

realEmbeddings = []
for i in range(0, len(real)):
    realEmbeddings.append(hamprod(yf1[realun[i]], yf1[realdu[i]]))

realestEmbeddings = pandas.DataFrame(realEmbeddings)
realestEmbeddings["label"] = 1

fakeun = fake[:,0]
fakedu = fake[:,1]

fakeEmbeddings = []
for i in range(0, len(fake)):
    fakeEmbeddings.append(hamprod(yf1[fakeun[i]], yf1[fakedu[i]]))

fakestEmbeddings = pandas.DataFrame(fakeEmbeddings)
fakestEmbeddings["label"] = 0

finalSet = realestEmbeddings.append(fakestEmbeddings)

print(len(finalSet))
#its double the original size 2125350, which is what we want

#generate score - has 2 modes. By default generates the probabilities, but adding another argument
def gs(x, y = "prob"):
    #70/30 train test split
    x_train, x_test, y_train, y_test = tts(x, x.label , test_size = 0.3)
    data = x_train.iloc[:,:32]
    test_data = x_test.iloc[:,:32]

    #train model
    classifier = lr(random_state=0).fit(data, y_train)
    if y == "prob":
        pred = classifier.predict_proba(test_data)
    else:
        pred = classifier.predict(test_data)
    return pred, y_test.values

print(gs(finalSet))

#AUC Curve

a = gs(finalSet,"pred")
print( auc(a[0], a[1]) )


#Plot it
truePos, falsePos, thresholds = rc(a[1], a[0])
plt.plot(truePos,falsePos)
plt.show()
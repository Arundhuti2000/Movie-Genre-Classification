import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
import re
from nltk.corpus import stopwords
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from itertools import cycle
from scipy import interp
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
# nltk.download('punkt')
# nltk.download('stopwords')
## READING DATA FROM CSV FILE
data = pd.read_csv (r'D:\Movie-GenreClassification\data\movies_metadata.csv')
##D-3, F-5, G-6, J-9, U-20
features = ['genres']
for feature in features:
 data[feature] = data[feature].apply(literal_eval)
def get_list(x):
 # WORKED ON KEY AND VALUE PAIRS TO CREATE LIST
 if isinstance(x, list):
 names = [i['name'] for i in x]
 if len(names) > 3:
 names = names[:3]
 return names
data['genres'] = data['genres'].apply(get_list) 
column_names = ["id", "title", "overview", "genres"]
DF = pd.DataFrame(columns = column_names)
DF['id'] = data['id']
DF['title'] = data['title']
DF['overview'] = data['overview']
DF['genres'] = data['genres']
# print(DF)
## REMOVED NULL VALUES FROM DF
DF = DF.dropna()
## EXPORTED NEW DF TO CSV FILE
DF.to_csv('movies_genre_classification.csv',index=False)
## REMOVED NULL VALUES FROM DF
DF.dropna(inplace=True)
DF.reset_index(drop=True, inplace=True)
## REPLACED ALL THE EMPTY LITERALS WITH OTHERS CATEGORY
for i in range(len(DF)):
 if DF.iloc[i]['genres']==[]:
 DF.iloc[i]['genres'] = ['Others']
## print(DF)
## TOTAL COLUMNS DROPPED FROM ORIGINAL DF
## 45466-44506 = 960 => 960 + 954 => 1,914
## IMPLEMENTING NLP PIPELINE
DF['overview'] = DF['overview'].str.lower()
## print(DF['overview_text'])
stopwords = set(stopwords.words('english'))
movie_data = pd.DataFrame(columns = ['overview'])
## TOKENIZING ALL TEXT
def identify_tokens(row):
 review = row['overview']
 tokens = nltk.word_tokenize(review)
 # TAKEN ONLY WORDS (NO PUNCTUATIONS)
 token_words = [w for w in tokens if w.isalpha()]
 return token_words
movie_data['overview'] = DF.apply(identify_tokens, axis=1)
## print(movie_data['overview'])
movie_data_wo_stop = pd.DataFrame(columns = ['overview'])
## REMOVING STOPWORDS
def remove_stops(row):
 my_list = row['overview']
 meaningful_words = [w for w in my_list if not w in stopwords]
 return (" ".join(meaningful_words))
movie_data_wo_stop['overview'] = movie_data.apply(remove_stops, axis=1)
movie_data_wo_stop['title'] = DF ['title']
movie_data_wo_stop['genres'] = DF['genres']
print(movie_data_wo_stop)
## GRAPHS PLOTS
genres = movie_data_wo_stop.genres.tolist()
allGenres=sum(genres,[])
len(set(allGenres))
allGenres = nltk.FreqDist(allGenres)
myTags = nltk.FreqDist(allGenres)
## CREATE DATAFRAME
myTags_DF = pd.DataFrame({'Genre': list(allGenres.keys()), 'Count': list(allGenres.v
alues())})
#######################################################
#################### PLOTTING GRAPHS ###################
#######################################################
topGenres = myTags_DF.nlargest(columns="Count", n = 50)
plt.figure(figsize=(12,15))
ax = sns.barplot(data=topGenres, x= "Count", y = "Genre")
plt.xlabel("Count", size=20)
plt.ylabel("Genres", size=20)
# plt.show()
def freq_words(x, terms = 40):
 all_words = ' '.join([text for text in x])
 all_words = all_words.split()
 fdist = nltk.FreqDist(all_words)
 words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
 ## SELECTING TOP 20 MOST FREQUENT WORDS
 TopWords = words_df.nlargest(columns="count", n = terms)
 ## VISIUALIZE WORDS AND THIER FREQUENCY
 plt.figure(figsize=(12,15))
 ax = sns.barplot(data=TopWords, x= "count", y = "word")
 plt.xlabel("Count", size=20)
 plt.ylabel("Word", size=20)
 plt.rc('xtick', labelsize=16)
 plt.rc('ytick', labelsize=16)
# plt.show()
## PRINT 100 MOST FREQUENT WORDS
freq_words(movie_data_wo_stop['overview'], 20)
y_true_label = yval[:, 0]
y_pred_label = y_pred_new[:, 0]
cf_matrix = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)
sns.heatmap(cf_matrix, annot=True, fmt='g')
## MODEL ##
## MULTILABEL BINARIZER
multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(movie_data_wo_stop['genres'])
## TRANSFORM TARGET VARIABLE
y = multilabel_binarizer.transform(movie_data_wo_stop['genres'])
#######################################################
############# TF- IMPLEMENTATION #######################
#######################################################
tfidf_vectorizer = TfidfVectorizer(max_df=0.5)
xtrain, xval, ytrain, yval = train_test_split(movie_data_wo_stop['overview'], y, test_si
ze=0.2, random_state=5)
## CREATE TF-IDF FEATURES
xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)
lin_svc = svm.LinearSVC()
svc_clf = OneVsRestClassifier(lin_svc)
svc_clf.fit(xtrain_tfidf, ytrain)
y_pred = svc_clf.predict(xval_tfidf)
print(y_pred[3])
print(yval[3])
print(ytrain.shape)
print(xtrain_tfidf.shape)
#######################################################
#################### Evaluate Performance #################
#######################################################
target_names = ['History','Horror','Western','Comedy','Action','Mystery','Thriller',
 'Foreign','TV Movie','Adventure','Others','Fantasy','Documentary','Crime',
 'War','Drama','Music','Science Fiction','Animation','Family','Romance']
### PRECISION RECALL F1-SCORE SUPPORT ###
print(classification_report(yval, y_pred, target_names=target_names))
pred = multilabel_binarizer.inverse_transform(y_pred)
actaul = multilabel_binarizer.inverse_transform(yval)
#######################################################
#################### ROC CURVE PLOTTING ################
#######################################################
n_class = 21
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_class):
 fpr[i], tpr[i], _ = roc_curve(yval[:, i], y_pred[:, i])
 roc_auc[i] = auc(fpr[i], tpr[i])
### Compute micro-average ROC curve and ROC area ###
 
fpr["micro"], tpr["micro"], _ = roc_curve(yval.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
#######################################################
#######################################################
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_class):
 mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /=n_class
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
 label='micro-average ROC curve (area = {0:0.2f})'
 ''.format(roc_auc["micro"]),
 color='deeppink', linestyle=':', linewidth=4)
plt.plot(fpr["macro"], tpr["macro"],
 label='macro-average ROC curve (area = {0:0.2f})'
 ''.format(roc_auc["macro"]),
 color='navy', linestyle=':', linewidth=4)
colors = cycle(['aqua', 'darkorange', 'cornflowerblue','MAROON','LIME','FUCHSIA','N
AVY','OLIVE',
 'YELLOW','RED', 'GRAY', 'PURPLE', 'INDIANRED', '#1F618D','#138D75','#2
73746',
 'brown','#F1C40F','#512E5F','#9A7D0A','cyan'])
for i, color in zip(range(n_class), colors):
 plt.plot(fpr[i], tpr[i], color=color, lw=lw)
 print('ROC curve of class {0} (area = {1:0.2f})'
 ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
## OBTAINING HIT RATIO
hit = 0
hit_dict={}
for i in range (len(yval)):#len(yval):
 for j in range (len(yval[0])): #21
 if (yval[i][j]==1) and (yval[i][j]==y_pred[i][j]):
 hit = hit + 1
#######################################################
#######################################################
##1 TODO PIPELINE
##1) IMPLEMENTING COUNTER VECTORIZATION ✔
##2) IMPLEMENTING TD-IDF ALGORITHM ✔
##3) IMPLEMENTING CLASSFIER METHOD SVM ✔
##2 TODO DATA PREPROCESSING
##1) Cleaning up the dataset ✔
##2) GOING TO EXTRACT LETTERS/WORDS FROM OVERVIEW COLUMN✔
##3) SPLIT THEM ✔
##4) REMOVE THE STOPWORDS ✔
##5) THEN JOIN ALL THE MEANINGFUL WORDS ✔
##3 TODO IMPORTANT THINGS TO REMEMBER
##1) IMPLEMENTING THE TF-IDF ALGORITHM ✔
##2) FIGURING OUT MULTIPLE GENRES CLASSIFICATION ✔
##3) IMPLEMENTING SVM ALGORITHM ✔
##4) ADDING VECTIORIZATION ✔
##5) GRAPHICAL REPRESENTATION 
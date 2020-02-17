#!/usr/bin/env python
# -*- coding: utf-8 -*-

###################################################################################
# Programmname: k_means.py
# Autor: Steffen Freisinger
# Datum: 20.03.2019
# Beschreibung: Tf-Idf-basierter k-Means
# Ziel: thematisches Clustering von Textdokumenten
###################################################################################

# Imports
import argparse
import pickle
import numpy as np
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt

####################################################################
# Klasse k_Means
####################################################################

class K_Means():

    # Initialisiert k-Means Algorithmus
    def __init__(self, k, data):
        self.k = k
        self.data = data
        self.n_docs, self.n_feats = data.shape
        self.centroids = self.initialise()
        self.clusters = np.zeros((k, self.n_docs))

    # Initialisiert Centroids nach k-Means++ (Arthur/Vassilvitskii, 2006)
    def initialise(self):
        centroids = np.empty((self.k, self.n_feats))
        first = np.random.choice(self.n_docs)
        centroids[0] = self.data[first]
        for n in range(1, self.k):
            probs = np.empty(self.n_docs)
            for d in range(self.n_docs):
                doc = self.data[d]
                distances = np.linalg.norm(centroids-doc, axis=1)
                probs[d] = np.min(distances)**2
            probs = probs / np.sum(probs)
            indx = np.random.choice(np.arange(self.n_docs), 1, p=probs)[0]
            centroids[n] = self.data[indx]
        return centroids

    # Ordnet die Datenpunkte den Clustern zu
    def update_clusters(self):
        self.old_clusters = self.clusters
        self.clusters = np.zeros((self.k, self.n_docs))
        for d in range(self.n_docs):
            doc = self.data[d]
            distances = np.linalg.norm(self.centroids-doc, axis=1)
            c = np.argmin(distances)
            self.clusters[c][d] = 1
        if np.array_equal(self.old_clusters, self.clusters):
            return True
        else:
            return False

    # Berechnet die Centroids neu
    def update_centroids(self):
        for c in range(self.k):
            points = self.clusters[c]
            cluster = np.dot(np.diag(points), self.data)
            centroid = np.sum(cluster, axis = 0) / np.sum(points)
            self.centroids[c] = centroid

    # Berechnet den totalen SSE, sowie für jedes Cluster
    def get_loss(self):
        sse_total = 0.0
        sse_per_cluster = np.empty(self.k)
        for c in range(self.k):
            doc_ids = np.where(self.clusters[c,:] == 1)[0]
            dists = np.linalg.norm(self.data[doc_ids] - self.centroids[c], axis=1)
            sse = np.sum(dists**2)
            sse_total += sse
            sse_per_cluster[c] = sse
        return (sse_total, sse_per_cluster)

####################################################################
# Main
####################################################################

parser = argparse.ArgumentParser(description = "K-Means for Text-Documents")

parser.add_argument('--docfile', default='ted_transcripts.p', type=str,
                    help='Pickle-File containing a list of String_Docs')
parser.add_argument('--k', default=4, type=int,
                    help='Parameter k defining the number of Clusters')
parser.add_argument('--num_epochs', default=100, type=int,
                    help='Number of Epochs, if not stopped before')
parser.add_argument('--k_max', default=None, type=int,
                    help='If given k_max, k-Means will be ran for \
                    k-Values range(k, k_max) and print Elbow-Graph')
parser.add_argument('--min_df', default=5, type=int,
                    help='Min Doc-Frequency for the Terms used as Feature')
parser.add_argument('--max_df', default=0.4, type=float,
                    help='Max Doc-Frequency for the Terms (Percentage)')
parser.add_argument('--num_features', default=2000, type=int,
                    help='Max Number of Features (Terms for tf-idf)')

args = parser.parse_args()

# Laden der Documente und Vorverarbeitung
print("\n### Loading and Preprocessing Data\n")
data_raw = pickle.load(open(args.docfile, 'rb'))
data = []
lem = WordNetLemmatizer()
for doc in data_raw:
    data.append(' '.join([lem.lemmatize(t) for t in word_tokenize(doc)]))

# Berechnen der Doc-Feature-Matrix mit Tf-Idf-Werten
print("### Generating Features\n")
vectorizer = TfidfVectorizer(stop_words = 'english',
                            max_df = args.max_df,
                            min_df = args.min_df,
                            max_features = args.num_features)
X = vectorizer.fit_transform(data).toarray()

print("### Starting k-Means Clustering")
k_max = args.k_max if args.k_max else args.k
k2loss = defaultdict(lambda: tuple)

for k in range(args.k, k_max+1):
    # Initialisieren von k-Means
    k_means = K_Means(k, X)
    k_means.initialise()
    # Durchführen des Clustering
    for m in range(args.num_epochs):
        no_change = k_means.update_clusters()
        k_means.update_centroids()
        if no_change:
            print("### -> Early-Stop:", m+1, "Iterations needed for k =", k)
            break
    k2loss[k] = k_means.get_loss()

# Generating Output for single k-Value
if not args.k_max:
    (sse_total, sse_clusters) = k2loss[args.k]
    for c in range(args.k):
        doc_ids = np.where(k_means.clusters[c])[0]
        print("\n+ Cluster", c+1, "->", doc_ids)
        print("+ SSE for Cluster =", sse_clusters[c])
    print("\n### Total SSE for this Clustering =", sse_total)

# Generating Elbow-Graph for multiple k-Values
else:
    ks = []
    losses = []
    for k, loss in k2loss.items():
        ks.append(k)
        losses.append(loss[0])
    plt.figure()
    plt.plot(ks, losses, marker='o')
    plt.show()

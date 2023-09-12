import numpy as np

# Distance between two vectors
def dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:
    def __init__(self, k):
        self.k = k
        
    def fit(self, X, y):
        self.X = X
        self.y = y
    
    # Predicts outcomes of a set of vectors
    def predict(self, vecs):
        preds = []
        for vec in vecs:
            # Get distance between current vector and all vectors in X
            dists = [dist(vec, x) for x in self.X]
            # Sort to get indexes of first k closest vectors
            ks = np.argsort(dists)[:self.k]
            labels = []
            # Safety check bc this would go over array bounds at times
            l = len(self.y)
            for k in ks:
                if k<l:
                    labels.append(self.y[k])
            # Get most frequent prediction in labels, add it to preds list
            frequent = max(set(labels), key=labels.count)
            preds.append(frequent)
        return np.array(preds), labels


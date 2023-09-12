from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal


## EM algorithm for Gaussian Mixture Models, where flag indicates if the safer, improved version is to be used
def EMG(img, k, flag):
    image = io.imread(img)
    img = image/255
    # droppping the last column
    if(img.shape[2] == 4):
        img = np.delete(img, 3, 2)
    
    # reshape the mat from 3D to 2D
    img = np.reshape(img, (img.shape[0]*img.shape[1], 3))
    
    # kmeans with k clusters
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=2, n_init=1, random_state=0)
    
    # labels based on the distance from each centers
    label = kmeans.fit_predict(img)
    mean = kmeans.cluster_centers_
    #print('centers', mean)
    
    #data converted as color list
    colors = kmeans.cluster_centers_[label]
    cmap = (colors*255).astype('int')
    #print(len(label))
    #plt.imshow(np.reshape(cmap, (image.shape[0], image.shape[1], 3)))
    #plt.show()
    
    # prior calculation
    Ni = np.bincount(label)
    prior = Ni/len(label)
    # covariance calculation
    covariance = np.zeros((k,3,3))
    #print(img[0])
    for i in range(0, img.shape[0]):
        covariance[label[i]] += np.reshape((img[i]-mean[label[i]]),(-1,1)) @ np.reshape((img[i]-mean[label[i]]), (-1,1)).T
    
    for c in range(0, k):
        if(flag == 1):
                covariance[c] = (covariance[c] + (1e-6)*np.identity(covariance.shape[1]))
        covariance[c] = covariance[c]/Ni[c]
    #print(covariance)
    #Multivariate Probability calcuation for each class
    prob = np.zeros((img.shape[0],k))
    for c in range(0, k):
        prob[:,c] = multivariate_normal.pdf(img, mean=mean[c], cov=covariance[c])
    
    #print('covariance', covariance)
    #print('mean', mean)
    #print('Ni',Ni)
    #print('prior', prior)
    #print('prob', prob)
    
    
    iteration =10
    h = np.zeros((img.shape[0], k))
    Q = np.zeros((iteration, 2))
    for it in range(0,iteration):
        #E-step 
        
        #for c in range(0,k):
        #    if(np.linalg.det(covariance[c]) != 0):
        #        cov_inv = np.linalg.inv(covariance[c])
        #        det = np.linalg.det(covariance[c])
        #        for i in range(0, img.shape[0]):
        #            x_mu = np.reshape((img[i]-mean[c]),(-1,1))
        #            h[i][c] = prior[c]*(det**0.5)*np.exp(-0.5*(x_mu.T @ cov_inv @x_mu))
        
        #h = h/h.sum(axis=1, keepdims=True)

        px = prob@prior
        for i in range(k):
            h[:, i] = prior[i]*prob[:, i]/px
        #Q[iter_, 0] = np.sum(h@np.log(pc)) + np.sum(h*np.log(p+eps))

        idx = np.where(prob==0)
        prob[idx] = 0.0000001
        idx = np.where(prior==0)
        prior[idx] = 0.0000001
        eps = 0.0000001
        #Q[it][0] = np.sum(h, axis=0) @ np.log(prior).T + np.sum(np.sum(np.multiply(h,np.log(prob)), axis=1))
        Q[it][0] = np.sum(h@np.log(prior)) + np.sum(h*np.log(prob+eps))

        # M-Step

        Ni = np.sum(h, axis=0)
        mean = (h.T @ img)/Ni.reshape(k,1)
        
        for c in range(0,k):
            for i in range(0, img.shape[0]):
                x_mu = np.reshape((img[i]-mean[c]),(-1,1))
                covariance[c] += h[i][c]*(x_mu @ x_mu.T)
            if(flag == 1):
                covariance[c] = (covariance[c] + (1e-6)*np.identity(covariance.shape[1]))
            covariance[c] /= Ni[c] 
       # print(covariance)
        
        signal =0
        
        for c in range(0, k):
            try:
                prob[:,c] = multivariate_normal.pdf(img, mean=mean[c], cov=covariance[c])
            except:
                print("Sigular Covariance Matrix ")
                signal =1
                #print(mean)
                break
        #print (prob)
        if (signal==1):
            break
        idx = np.where(prob==0)
        prob[idx] = 0.0000001
        idx = np.where(prior==0)
        prior[idx] = 0.0000001
        #Q[it][1] = np.sum(h, axis=0) @ np.log(prior).T + np.sum(np.sum(np.multiply(h,np.log(prob)), axis=1))
        Q[it][1] = np.sum(h@np.log(prior)) + np.sum(h*np.log(prob+eps))
        
        print(Q)
        #print('covariance', covariance)
        #print('mean', mean)
        #print('Ni',Ni)
        #print('prior', prior)
        #print('prob', prob)
        ##print('Q', Q)
        #print('h', h)
    print(it)
    #print(Q)
    #plt.plot(Q[:it,0], 'r')
    #plt.plot(Q[:it,1], 'b')
    #plt.axis([0, 6])
    #plt.show()
    
    #print(mean)
    #new_img = mean[np.argmax(h, axis =1)]
    #print(np.bincount(np.argmax(h, axis =1)))
    #cmap = (new_img*255).astype('int')
    #plt.imshow(np.reshape(cmap, (image.shape[0], image.shape[1], 3)))
    #plt.show()


# Q part


if __name__ == '__main__':
    k=[4]#,8,12]
filename = './stadium.bmp'
for i in k:
    EMG (filename, i, 0)
"""
filename_goldy = './goldy.bmp'
image = io.imread(filename_goldy)
img = image/255
#droppping the last column
if(img.shape[2] == 4):
    img = np.delete(img, 3, 2)

#reshape the mat from 3D to 2D
img = np.reshape(img, (img.shape[0]*img.shape[1], 3))

#kmeans with k clusters
kmeans = KMeans(n_clusters=7, init='k-means++', max_iter=3, n_init=1, random_state=0)

#lables based on the distance from each centers
label = kmeans.fit_predict(img)
mean = kmeans.cluster_centers_
print('centers', mean)

#data converted as color list
colors = kmeans.cluster_centers_[label]
cmap = (colors*255).astype('int')
print(len(label))
plt.imshow(np.reshape(cmap, (image.shape[0], image.shape[1], 3)))
plt.show()

EMG (filename_goldy, 7, 0)

for i in k:
    EMG (filename, i, 1)

EMG (filename_goldy, 7, 1)
"""

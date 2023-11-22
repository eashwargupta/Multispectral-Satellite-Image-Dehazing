from flask import Flask, request, render_template, jsonify, send_file
import pandas as pd
import rasterio
import cv2
from flask_cors import CORS, cross_origin
import cv2
from itertools import combinations_with_replacement
from collections import defaultdict
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

app = Flask(__name__)

R, G, B = 0, 1, 2  # index for convenience

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])

        file_path = 'templates/S5P_NO2_India_2019.tif'

        with rasterio.open(file_path) as src:
            raster_data = src.read(1)

        city_df = pd.read_csv('Ind_City_Lat_Lon.csv')

        M = 46.0055 # Molar mass of NO2 in g/mol
        V = 0.0002  # Vertical column density of NO2 in mol/m2

        print(latitude)

        # perform some calculation here
        row, col = src.index(longitude, latitude)

        pixel_value = raster_data[row][col]
        NO2_concentration = (pixel_value * M) / V

        return jsonify({'sum': NO2_concentration})
    else:
        return render_template('index.html')
    
#dark channel prior
@app.route('/process_image', methods=['POST'])
def process_image():
    # Get the uploaded image file from the request object
    uploaded_file = request.files['image']
    
    # Save the uploaded image to a temporary file
    filename = 'tmp_image.jpg'
    uploaded_file.save(filename)

    class Node(object):
        def __init__(self,x,y,value):
            self.x = x
            self.y = y
            self.value = value

    def printInfo(self):
        print('%s:%s:%s' %(self.x,self.y,self.value))
        
    def getMinChannel(img):
        if len(img.shape)==3 and img.shape[2]==3:
            pass
        else:
            print("bad image shape, input must be color image")
            return None
        
        return np.min(img, axis=2)
   
    def getDarkChannel(img,blockSize = 3):
        if len(img.shape)==2:
            pass
        else:
            print("bad image shape, input image must be two demensions")
            return None
        if blockSize % 2 == 0 or blockSize < 3:
            print('blockSize is not odd or too small')
            return None

        A = int((blockSize-1)/2) #AddSize

        #New height and new width
        H = img.shape[0] + blockSize - 1
        W = img.shape[1] + blockSize - 1

        imgMiddle = 255 * np.ones((H,W))    

        imgMiddle[A:H-A, A:W-A] = img
        
        imgDark = np.zeros_like(img, np.uint8)    
        
        localMin = 255
        for i in range(A, H-A):
            for j in range(A, W-A):
                x = range(i-A, i+A+1)
                y = range(j-A, j+A+1)
                imgDark[i-A,j-A] = np.min(imgMiddle[x,y])                            
                
        return imgDark

    def getAtomsphericLight(darkChannel,img,meanMode = False, percent = 0.001):

        size = darkChannel.shape[0]*darkChannel.shape[1]
        height = darkChannel.shape[0]
        width = darkChannel.shape[1]

        nodes = []

        for i in range(0,height):
            for j in range(0,width):
                oneNode = Node(i,j,darkChannel[i,j])
                nodes.append(oneNode)	


        nodes = sorted(nodes, key = lambda node: node.value,reverse = True)

        atomsphericLight = 0

        if int(percent*size) == 0:
            for i in range(0,3):
                if img[nodes[0].x,nodes[0].y,i] > atomsphericLight:
                    atomsphericLight = img[nodes[0].x,nodes[0].y,i]
            return atomsphericLight

        if meanMode:
            sum = 0
            for i in range(0,int(percent*size)):
                for j in range(0,3):
                    sum = sum + img[nodes[i].x,nodes[i].y,j]
            atomsphericLight = int(sum/(int(percent*size)*3))
            return atomsphericLight

        for i in range(0,int(percent*size)):
            for j in range(0,3):
                if img[nodes[i].x,nodes[i].y,j] > atomsphericLight:
                    atomsphericLight = img[nodes[i].x,nodes[i].y,j]
        return atomsphericLight

    def getRecoverScene(img, omega=0.95, t0=0.1, blockSize=15, meanMode=False, percent=0.001, refine=True):

        imgGray = getMinChannel(img)
        imgDark = getDarkChannel(imgGray, blockSize = blockSize)
        atomsphericLight = getAtomsphericLight(imgDark,img,meanMode = meanMode,percent= percent)

        imgDark = np.float64(imgDark)
        transmission = 1 - omega * imgDark / atomsphericLight

        transmission[transmission<0.1] = 0.1     
        
        if refine:        
            normI = (img - img.min()) / (img.max() - img.min())  # normalize I
            transmission = guided_filter(normI, transmission, r=40, eps=1e-3)

        sceneRadiance = np.zeros(img.shape)
        img = np.float64(img)
        
        for i in range(3):        
            SR = (img[:,:,i] - atomsphericLight)/transmission + atomsphericLight
                
            SR[SR>255] = 255
            SR[SR<0] = 0                    
            sceneRadiance[:,:,i] = SR  
                
        sceneRadiance = np.uint8(sceneRadiance)

        return sceneRadiance
    

    def boxfilter(I, r):
   
        M, N = I.shape
        dest = np.zeros((M, N))

        # cumulative sum over Y axis
        sumY = np.cumsum(I, axis=0)
        # difference over Y axis
        dest[:r + 1] = sumY[r: 2 * r + 1]
        dest[r + 1:M - r] = sumY[2 * r + 1:] - sumY[:M - 2 * r - 1]
        dest[-r:] = np.tile(sumY[-1], (r, 1)) - sumY[M - 2 * r - 1:M - r - 1]

        # cumulative sum over X axis
        sumX = np.cumsum(dest, axis=1)
        # difference over Y axis
        dest[:, :r + 1] = sumX[:, r:2 * r + 1]
        dest[:, r + 1:N - r] = sumX[:, 2 * r + 1:] - sumX[:, :N - 2 * r - 1]
        dest[:, -r:] = np.tile(sumX[:, -1][:, None], (1, r)) - \
            sumX[:, N - 2 * r - 1:N - r - 1]

        return dest


    def guided_filter(I, p, r=40, eps=1e-3):    
        M, N = p.shape
        base = boxfilter(np.ones((M, N)), r)

        # each channel of I filtered with the mean filter
        means = [boxfilter(I[:, :, i], r) / base for i in range(3)]
        
        # p filtered with the mean filter
        mean_p = boxfilter(p, r) / base
        # filter I with p then filter it with the mean filter
        
        means_IP = [boxfilter(I[:, :, i] * p, r) / base for i in range(3)]
        # covariance of (I, p) in each local patch
        covIP = [means_IP[i] - means[i] * mean_p for i in range(3)]

        # variance of I in each local patch: the matrix Sigma in ECCV10 eq.14
        var = defaultdict(dict)
        for i, j in combinations_with_replacement(range(3), 2):
            var[i][j] = boxfilter(
                I[:, :, i] * I[:, :, j], r) / base - means[i] * means[j]

        a = np.zeros((M, N, 3))
        for y, x in np.ndindex(M, N):
            #         rr, rg, rb
            # Sigma = rg, gg, gb
            #         rb, gb, bb
            Sigma = np.array([[var[R][R][y, x], var[R][G][y, x], var[R][B][y, x]],
                            [var[R][G][y, x], var[G][G][y, x], var[G][B][y, x]],
                            [var[R][B][y, x], var[G][B][y, x], var[B][B][y, x]]])
            cov = np.array([c[y, x] for c in covIP])
            a[y, x] = np.dot(cov, inv(Sigma + eps * np.eye(3)))  # eq 14

        # ECCV10 eq.15
        b = mean_p - a[:, :, R] * means[R] - \
            a[:, :, G] * means[G] - a[:, :, B] * means[B]

        # ECCV10 eq.16
        q = (boxfilter(a[:, :, R], r) * I[:, :, R] + boxfilter(a[:, :, G], r) *
            I[:, :, G] + boxfilter(a[:, :, B], r) * I[:, :, B] + boxfilter(b, r)) / base

        return q
    
    # df_train = pd.read_csv('train_v2.csv')

    img = cv2.imread('tmp_image.jpg')
    dehazed_img2 = getRecoverScene(img, refine=False)
    fig = plt.figure()
    fig.set_size_inches(12, 4)


    # Save the processed image to a file
    filename1 = "output_image.jpg"
    cv2.imwrite(filename1, dehazed_img2)
    
    # Send the processed image file back to the client
    return send_file(filename1, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True)




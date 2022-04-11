from PIL import Image
from IPython.display import Image
from matplotlib import pyplot as plt
import pandas as pd, numpy as np
import io
pd.options.display.float_format = '{:,.2f}'.format
import warnings
warnings.simplefilter("ignore")
import os, cv2
import pathlib


# finds the horizontal line on page
def findHorizontalLines(img):
    img = cv2.imread(img) 
    
    #convert image to greyscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # set threshold to remove background noise
    thresh = cv2.threshold(gray,30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    
    # define rectangle structure (line) to look for: width 100, hight 1. This is a 
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200,1))
    
    # Find horizontal lines
    lineLocations = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    return lineLocations

# crops section 2 of page containing hand written text
def pageSegmentation1(img, w, df_SegmentLocations):
    outpathFile = img
    img = cv2.imread(img) 
    im2 = img.copy()
    segments = []

    y = df_SegmentLocations['SegmentStart'][2]
    h = df_SegmentLocations['Height'][2]

    cropped = im2[y:y + h, 0:w] 
    segments.append(cropped)

    #file that data is to going to be stored
    cv2.imwrite('C:/path/to/output/file'+outpathFile, cropped)
        
    return segments

def main():
    os.chdir(r'C:\Your\dir\here')
         
    fileList = [x for x in os.listdir() if 'png'  in x.lower()]
    print(fileList[:5])
    print("file count: "+str(len(fileList)))

    # try:
    #     Image(filename = fileList[0], width = 300)
    # except IOError:
    #     print("error opening image")

    i = 356
    while i < len(fileList):
        img = fileList[i]
        lineLocations = findHorizontalLines(img)
        plt.figure(figsize=(24,24))
        # plt.imshow(lineLocations, cmap='Greys')

        df_lineLocations = pd.DataFrame(lineLocations.sum(axis=1)).reset_index()
        df_lineLocations.columns = ['rowLoc', 'LineLength']
        df_lineLocations[df_lineLocations['LineLength'] > 0]

        df_lineLocations['line'] = 0
        df_lineLocations['line'][df_lineLocations['LineLength'] > 100] = 1

        df_lineLocations['cumSum'] = df_lineLocations['line'].cumsum()

        # df_lineLocations.head()

        import pandasql as ps

        query = '''
        select row_number() over (order by cumSum) as SegmentOrder
        , min(rowLoc) as SegmentStart
        , max(rowLoc) - min(rowLoc) as Height
        from df_lineLocations
        where line = 0
        --and CumSum !=0
        group by cumSum
        '''

        df_SegmentLocations  = ps.sqldf(query, locals())
        # df_SegmentLocations

        img = fileList[i]
        print(i, img)
        w = lineLocations.shape[1]
        segments = pageSegmentation1(img, w, df_SegmentLocations)
        i = i + 1

main()
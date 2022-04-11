import PIL.Image
from IPython.display import Image
import warnings

# import PIL
warnings.simplefilter("ignore")
import os

def blackWhitedithering(imagePath,
    dithering=True):
     
    fp = open(imagePath,"rb")
    colorImage = PIL.Image.open(fp)
    if dithering:
        bw = colorImage.convert('1')  
    else:
        bw = colorImage.convert('1', dither=Image.NONE)
    bw.save("..\\outputFile\\"+imagePath)


def main():

    fileList = os.listdir(r'C:\input\dir\here')

    print("file count: "+str(len(fileList)))
    os.chdir(r'C:\input\dir\here')

    i = 0
    while i < len(fileList):
        
        blackWhitedithering(fileList[i])
        print(i)
        i = i + 1

main()
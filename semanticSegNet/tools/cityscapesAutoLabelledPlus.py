from glob import glob
from os import mkdir
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import pathlib

def main():
    # autoLabeledPath = "/home/lasi/Downloads/datasets/cityscapes/autolabelled/train_extra/*/*leftImg8bit.png"
    # coarseLabelPath = "/home/lasi/Downloads/datasets/cityscapes/gtCoarse/gtCoarse/train_extra/*/*labelIds.png"
    autoLabeledPath = "/home/gama/Documentos/datasets/cityscapes/refinement_final_v0/refinement_final/train_extra/*/*leftImg8bit.png"
    coarseLabelPath = "/home/gama/Documentos/datasets/cityscapes/gtCoarse/gtCoarse/train_extra/*/*labelIds.png"
    alFiles = sorted(glob(autoLabeledPath))
    coarseFiles = sorted(glob(coarseLabelPath))
    # alFiles.remove("/home/lasi/Downloads/datasets/cityscapes/autolabelled/train_extra/troisdorf/troisdorf_000000_000073_leftImg8bit.png")
    alFiles.remove("/home/gama/Documentos/datasets/cityscapes/refinement_final_v0/refinement_final/train_extra/troisdorf/troisdorf_000000_000073_leftImg8bit.png")

    print(len(alFiles))
    print(len(coarseFiles))

    # saveFolderPath = "/home/lasi/Downloads/datasets/cityscapes/auto_coarse/"
    saveFolderPath = "/home/gama/Documentos/datasets/cityscapes/auto_coarse/"

    pathlib.Path(saveFolderPath + "augsburg").mkdir(parents=True, exist_ok=True)
    pathlib.Path(saveFolderPath + "bad-honnef").mkdir(parents=True, exist_ok=True)
    pathlib.Path(saveFolderPath + "bamberg").mkdir(parents=True, exist_ok=True)
    pathlib.Path(saveFolderPath + "bayreuth").mkdir(parents=True, exist_ok=True)
    pathlib.Path(saveFolderPath + "dortmund").mkdir(parents=True, exist_ok=True)
    pathlib.Path(saveFolderPath + "dresden").mkdir(parents=True, exist_ok=True)
    pathlib.Path(saveFolderPath + "duisburg").mkdir(parents=True, exist_ok=True)
    pathlib.Path(saveFolderPath + "erlangen").mkdir(parents=True, exist_ok=True)
    pathlib.Path(saveFolderPath + "freiburg").mkdir(parents=True, exist_ok=True)
    pathlib.Path(saveFolderPath + "heidelberg").mkdir(parents=True, exist_ok=True)
    pathlib.Path(saveFolderPath + "heilbronn").mkdir(parents=True, exist_ok=True)
    pathlib.Path(saveFolderPath + "karlsruhe").mkdir(parents=True, exist_ok=True)
    pathlib.Path(saveFolderPath + "konigswinter").mkdir(parents=True, exist_ok=True)
    pathlib.Path(saveFolderPath + "konstanz").mkdir(parents=True, exist_ok=True)
    pathlib.Path(saveFolderPath + "mannheim").mkdir(parents=True, exist_ok=True)
    pathlib.Path(saveFolderPath + "muhlheim-ruhr").mkdir(parents=True, exist_ok=True)
    pathlib.Path(saveFolderPath + "nuremberg").mkdir(parents=True, exist_ok=True)
    pathlib.Path(saveFolderPath + "oberhausen").mkdir(parents=True, exist_ok=True)
    pathlib.Path(saveFolderPath + "saarbrucken").mkdir(parents=True, exist_ok=True)
    pathlib.Path(saveFolderPath + "schweinfurt").mkdir(parents=True, exist_ok=True)
    pathlib.Path(saveFolderPath + "troisdorf").mkdir(parents=True, exist_ok=True)
    pathlib.Path(saveFolderPath + "wuppertal").mkdir(parents=True, exist_ok=True)
    pathlib.Path(saveFolderPath + "wurzburg").mkdir(parents=True, exist_ok=True)

    for alFile, coarseFile in tqdm(zip(alFiles, coarseFiles)):
        
        al = cv2.imread(alFile, cv2.IMREAD_UNCHANGED)
        coarse = cv2.imread(coarseFile, cv2.IMREAD_UNCHANGED)
        
        new = np.where(coarse[...]!= 0,coarse,al)
        newImg = Image.fromarray(new)
        # print(coarseFile[71:])
        newImg.save(saveFolderPath + coarseFile[71:])
        # break

    # al = cv2.resize(al, (1024, 512))
    # new = cv2.resize(new, (1024, 512))
    # cv2.imshow("orginal", al)
    # cv2.imshow("test", new)
    # cv2.waitKey(0)
    
    # print(new)


if __name__ == "__main__":
    main()

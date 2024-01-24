import numpy as np
from tqdm import tqdm

from utils import img_resolve as ir
import cv2
import os
metadata = {'description': ' Wayho Tech ', 'wavelength units': 'Nanometers',
            'band names': '{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26}',
            'wavelength': '{450.00,460.00,470.00,480.00,490.00,500.00,510.00,520.00,530.00,540.00,550.00,'
                          '560.00,570.00,580.00,590.00,600.00,610.00,620.00,630.00,640.00,650.00,'
                          '660.00,670.00,680.00,690.00,700.00}',
            'reflectance scale factor': 1.000000}
def run(folder_path,save_folderpath):
    files_list = os.listdir(folder_path)
    for per_file_name in tqdm(files_list):
        per_file_path = os.path.join(folder_path, per_file_name, per_file_name)
        per_img = np.zeros((512,512,26),dtype=np.uint8)
        for i in range(6,32):
            per_file_pngpath = os.path.join(per_file_path,per_file_name+f'_{i:02}.png')
            per_img[:,:,i-6] = cv2.imread(per_file_pngpath,cv2.IMREAD_GRAYSCALE)
        per_img = (per_img - np.min(per_img)) / (np.max(per_img) - np.min(per_img))
        per_img = per_img.astype(np.float32)
        per_save_imgpath = os.path.join(save_folderpath, per_file_name)
        if not os.path.exists(save_folderpath):
            os.makedirs(save_folderpath)
        ir.save_img(per_img, metadata, per_save_imgpath,dtype=np.float32)


if __name__ == '__main__':
    folder_path = r"D:\learn\condata\data\complete_ms_data"
    save_path = r'D:\learn\condata\data\out_cave'
    run(folder_path, save_path)
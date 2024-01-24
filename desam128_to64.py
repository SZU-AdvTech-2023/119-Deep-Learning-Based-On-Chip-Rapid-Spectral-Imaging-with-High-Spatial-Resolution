import numpy as np
from utils import img_resolve as ir
import os
metadata = {'description': ' Wayho Tech ', 'wavelength units': 'Nanometers',
            'band names': '{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26}',
            'wavelength': '{450.00,460.00,470.00,480.00,490.00,500.00,510.00,520.00,530.00,540.00,550.00,'
                          '560.00,570.00,580.00,590.00,600.00,610.00,620.00,630.00,640.00,650.00,'
                          '660.00,670.00,680.00,690.00,700.00}',
            'reflectance scale factor': 1.000000}
def run(folder_path,save_folderpath):
    files_list = os.listdir(folder_path)
    img_flies_list = [file for file in files_list if file.endswith('.img')]
    for per_file_name in img_flies_list:
        per_file_path = os.path.join(folder_path, per_file_name)
        per_img = ir.read_img(per_file_path)[:,:,:]
        # per_img = per_img[::4, ::4, :]
        for i in range(2):
            for j in range(2):
                out_img = per_img[i*256:(i+1)*256,j*256:(j+1)*256,:]
                per_save_imgpath = os.path.join(save_folderpath, per_file_name[:-4]+ f"_{i}_{j}")
                if not os.path.exists(save_folderpath):
                    os.makedirs(save_folderpath)
                ir.save_img(out_img, metadata, per_save_imgpath, dtype=np.float32)

if __name__ == '__main__':
    path = r"D:\learn\condata\dataset\valid_data"
    savepath = r"D:\learn\condata\dataset_256\valid_data"
    run(path, savepath)
import numpy as np
import spectral
from spectral import envi

def set_byte_order(hdr_file):
    lines = ""
    with open(hdr_file, mode='r', encoding='utf-8') as f:
        line = f.readline()
        while line != "":
            if line.startswith("Wayho"):
                line = "ENVI\n"
            if not line.startswith("ModuleName"):
                lines += line
            line = f.readline()
        if "byte order" not in lines:
            lines += "\nbyte order = 0"
    with open(hdr_file, mode='w', encoding='utf-8') as f:
        f.writelines(lines)


# 读取高光谱文件
def read_img(file_name, set_data_type=True, postfix='.img'):
    file_name = file_name[:-4]
    if set_data_type:
        set_byte_order(file_name + '.hdr')
    try:
        spectral.settings.envi_support_nonlowercase_params = True
        img = envi.open(file_name + '.hdr', file_name + postfix)
    except envi.EnviDataFileNotFoundError:
        print(file_name + ' can not be opened.')
    else:
        return img


# 保存高光谱文件
def save_img(data_set, metadata, file_name, dtype = np.uint16):
    if metadata and data_set.size != 0:
        if 'reflectance scale factor' in metadata:
            data_set = (data_set * int(metadata['reflectance scale factor'])).astype(dtype)
        else:
            data_set = data_set.astype(dtype)
        envi.save_image(file_name + '.hdr', data_set, force=True, interleave='bsq',dtype=dtype, metadata=metadata)
    else:
        print(file_name + ' is empty and cannot be saved.')


# 将高光谱数据准备成按行排列的数据集
def prepare_img(img):
    shape = img.shape # shape（图像行数， 图像每行像素数， 图像波段数量）

    data_set = np.zeros((shape[2], shape[0] * shape[1]), dtype=np.float32)

    for i in range(0, shape[2]):
        data_set[i, :] = img[:, :, i].reshape(1, -1).astype(np.float32)
    sample_mat = np.transpose(data_set)
    return sample_mat


if __name__ == "__main__":

    data = read_img()

    data1 = prepare_img(data)

    meta = data.metadata

    print('ss')

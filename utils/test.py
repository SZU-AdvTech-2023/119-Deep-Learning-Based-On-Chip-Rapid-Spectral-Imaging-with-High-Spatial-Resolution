import torch

# 创建一个形状为(3, 4, 4)的张量并填充0到47
tensor_array = torch.arange(48).reshape(3, 4, 4)

# 输出原始张量
print("原始张量:\n", tensor_array)

# 进行转置操作
transposed_tensor = torch.transpose(tensor_array, dim0=1, dim1=2)

out = transposed_tensor.contiguous().view(-1, 1)

a = out.view(3,4,4)
b = torch.transpose(a, dim0=1, dim1=2)
# 输出转置后的张量
print(out)
print(a)
print(b)



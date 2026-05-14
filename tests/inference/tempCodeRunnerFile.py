
import axion_cpp

x = axion_cpp.Tensor()

x.shape = [2, 4]

x.data = [
    1.0,
    -2.0,
    3.0,
    -4.0,
    5.0,
    -6.0,
    7.0,
    -8.0
]

q = axion_cpp.quantize_q8(x)

print("\nSCALE:")
print(q.scale)

print("\nQ8 DATA:")
print(q.data)

dq = axion_cpp.dequantize_q8(q)

print("\nDEQUANTIZED:")
print(dq.data)
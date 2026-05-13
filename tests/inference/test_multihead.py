
import axion_cpp

x = axion_cpp.Tensor()

x.shape = [2, 8]

x.data = [

    1,2,3,4,5,6,7,8,

    9,10,11,12,13,14,15,16
]

heads = axion_cpp.split_heads(
    x,
    2
)

print("\n=== SPLIT HEADS ===")

for i, h in enumerate(heads):

    print(f"\nHEAD {i}")

    print(h.shape)

    print(h.data)

merged = axion_cpp.merge_heads(
    heads
)

print("\n=== MERGED ===")

print(merged.shape)

print(merged.data)

import sys

sys.path.append(
    r"D:\LLM projects\Axion\inference\cpp\build"
)

import axion_cpp

loader = axion_cpp.GGUFLoader()

loader.load_file(
    r"D:\LLM projects\Axion\tests\inference\gguf\inferra-q4.gguf"
)
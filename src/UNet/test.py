import os

# 打印当前工作目录
print(os.path.join("model", max([f for f in os.listdir("model") if f.endswith(".pth")],
                                             key=lambda x: os.path.getctime(
                                                 os.path.join("model", x)))) )

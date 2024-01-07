from PIL import Image
import os


def resize_image(input_path, output_path, size):
    # 打开原始图片
    image = Image.open(input_path)
    # 调整图片尺寸
    resized_image = image.resize(size)
    # 导出调整后的图片
    resized_image.save(output_path)


def resize_all(input_folder, output_folder, size):
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".jpg") or file_name.endswith(".jpeg") or file_name.endswith(".png"):
            # 图片文件路径
            input_path = os.path.join(input_folder, file_name)
            # 输出图片文件路径
            output_path = os.path.join(output_folder, file_name.split(".")[0] + ".bmp")
            # 调整图片尺寸并导出
            resize_image(input_path, output_path, size)


if __name__ == "__main__":
    # 测试代码
    len = 512
    input_folder = "./Set14/Set14/"  # 原始图片路径
    output_folder = "./Set14/size_" + str(len) + "/"  # 输出图片路径
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    size = (len, len)  # 新的尺寸

    resize_all(input_folder, output_folder, size)

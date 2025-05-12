import shutil
import os


def copy_and_rename_file(source_path, destination_folder, new_filename):
    """
    将源文件复制到目标文件夹并重命名

    参数:
        source_path (str): 源文件的完整路径
        destination_folder (str): 目标文件夹路径
        new_filename (str): 新文件名(包含扩展名)

    返回:
        str: 新文件的完整路径
    """
    try:
        # 检查源文件是否存在
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"源文件 {source_path} 不存在")

        # 检查目标文件夹是否存在，不存在则创建
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
            print(f"创建了目标文件夹: {destination_folder}")

        # 构建新文件的完整路径
        destination_path = os.path.join(destination_folder, new_filename)

        # 复制并重命名文件
        shutil.copy2(source_path, destination_path)
        print(f"文件已成功复制并重命名: {destination_path}")

        return destination_path

    except Exception as e:
        print(f"操作失败: {str(e)}")
        return None

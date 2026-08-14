from ultralytics import YOLO

if __name__ == '__main__':
    # 加载已训练好的模型权重
    model = YOLO('/root/autodl-tmp/OCSSNet/output_dir/mscoco/mambayolo2/weights/best.pt')  # 已训练的模型权重 (.pt 文件)

    # 要预测的图片路径列表(TypeⅠ)
    image_paths = [
        '/root/autodl-tmp/OCSSNet/dataset/Type1/images/test2017/rail_1.jpg',
        # '/root/autodl-tmp/OCSSNet/dataset/Type1/images/test2017/rail_85.jpg',
        # '/root/autodl-tmp/OCSSNet/dataset/Type1/images/test2017/rail_97.jpg',
        # '/root/autodl-tmp/OCSSNet/dataset/Type1/images/test2017/rail_116.jpg',
        # '/root/autodl-tmp/OCSSNet/dataset/Type1/images/test2017/rail_127.jpg',
        # '/root/autodl-tmp/OCSSNet/dataset/Type1/images/test2017/rail_139.jpg',
        # '/root/autodl-tmp/OCSSNet/dataset/Type1/images/test2017/rail_145.jpg',
        # '/root/autodl-tmp/OCSSNet/dataset/Type1/images/test2017/rail_149.jpg',
        # '/root/autodl-tmp/OCSSNet/dataset/Type1/images/test2017/rail_156.jpg',
        # '/root/autodl-tmp/OCSSNet/dataset/Type1/images/test2017/rail_179.jpg',
    ]

    # # 要预测的图片路径列表(TypeⅡ)
    # image_paths = [
    #     '/tmp/pycharm_project_1/test_outputs/TypeⅡ/TypeⅡ_resized/rail_5.jpg',
    #     '/tmp/pycharm_project_1/test_outputs/TypeⅡ/TypeⅡ_resized/rail_29.jpg',
    #     '/tmp/pycharm_project_1/test_outputs/TypeⅡ/TypeⅡ_resized/rail_37.jpg',
    #     '/tmp/pycharm_project_1/test_outputs/TypeⅡ/TypeⅡ_resized/rail_256.jpg',
    #     '/tmp/pycharm_project_1/test_outputs/TypeⅡ/TypeⅡ_resized/rail_264.jpg',
    #     '/tmp/pycharm_project_1/test_outputs/TypeⅡ/TypeⅡ_resized/rail_500.jpg',
    #     '/tmp/pycharm_project_1/test_outputs/TypeⅡ/TypeⅡ_resized/rail_195.jpg',
    #     '/tmp/pycharm_project_1/test_outputs/TypeⅡ/TypeⅡ_resized/rail_187.jpg',
    #     '/tmp/pycharm_project_1/test_outputs/TypeⅡ/TypeⅡ_resized/rail_27.jpg',
    # ]

    # 循环处理每张图片
    for image_path in image_paths:
        results = model.predict(
            source=image_path,  # 当前图片路径
            imgsz=160,  # 图片尺寸
            conf=0.50,  # 置信度阈值
            device='0',  # 选择 GPU 或 CPU
            save=True,  # 是否保存输出图片
            project='/root/autodl-tmp/OCSSNet/Defect_Sample/Type1/',  # 保存输出文件的路径
            name='[predict_results]',  # 项目名称
            save_txt=True,  # 保存检测结果到文本文件
            line_thickness=1  # 框线的粗细
        )

        # 打印预测结果
        print(f"Results for {image_path}: {results}")


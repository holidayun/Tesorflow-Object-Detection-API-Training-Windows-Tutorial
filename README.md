# tesorflow-tools
tensorflow object detection api tools
训练前准备：
-----
1. 在训练自己的检查器之前，需要能通过官方tensorflow-object-detection-api的运行测试，详见上一篇博客：
https://blog.csdn.net/holidayun/article/details/82378201
2. 下载本教程提供的xml_to_csv.py 与 generate_tfrecord.py
将下载完成的文件放入到 
\tensorflow\models\research\object_detection 目录下
3. 在\tensorflow\models\research\object_detectio 下新建images文件夹
4. 在\tensorflow\models\research\object_detectio\imgaes  新建test文件夹与train文件夹
5. 模型下载，可根据自己的需求，下载对应的模型，
![这里写图片描述](https://github.com/holidayun/tesorflow-tools/raw/master/screenshots/model.png)
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

    SDD-MobileNet模型训练速度较快，但是精确度不太理想，本教程使用
    使用Faster-RCNN-Inception-V2模型。下载地址：
    
    http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
 下载完成之后，将faster_rcnn_inception_v2_coco_2018_01_28文件夹解压缩到tensorflow \ models \ research \ object_detection目录下。（注意：模型日期和版本将来可能会发生变化，但它仍应适用于本教程。）


1.收集图片
------
1. 收集的图片中包含检测目标。
2. 确保图片不是太大，每个应小于200KB，分辨率不应超过720x1280。图像越大，训练分类器所需的时间越长。
3. 将 收集到的图片放入\tensorflow\models\research\object_detectio\imgaes 目录下

2.标签图片
------
1. 收集完所有图片后，就可以在每张图片中标记所需的对象。LabelImg是一个很好的标记图像的工具，它的GitHub页面有关于如何安装和使用它的非常明确的说明：
https://github.com/tzutalin/labelImg

2. 使用LabelImg生成的xml文件，其中80%放入
\tensorflow\models\research\object_detectio\imgaes\train  目录下，用于训练使用。20%的放入\object_detection\imgaes\test  目录下，用于训练时测试使用。

3.生成培训数据
--------
1. 首先将生成的xml数据，转换为csv格式的数据。
使用编辑器打开 \object_detection 目录下的 xml_to_csv.py 文件，修改对应路径

```
def main():
	#image_path 指的是test文件夹路径
    image_path = r'D:\Image-GPU\tensorflow\models\research\object_detection\images\test'
    xml_df = xml_to_csv(image_path)
    #'test_labels.csv'生成的csv名称
    xml_df.to_csv('test_labels.csv', index=None)
    print('Successfully converted xml to csv.')
```

 执行：

```
D:\Image-GPU\tensorflow\models\research\object_detection> python xml_to_csv.py
```

  这样在object_detection 目录下生成了对应的test_labels.csv文件，再用同  样的办法，生成train_label.csv文件。
2. 将csv文件，转换为.record文件
接下来，在编辑器中打开generate_tfrecord.py文件。将自己的标签图片替换从第31行开始的标签图片，其中为每个对象分配一个ID编号。在接下来配置**labelmap.pbtxt**文件时，将使用相同的编号分配。

例如，假设您正在训练分类器来检测苹果，香蕉和橘子。您将在generate_tfrecord.py以下代码：

```
if row_label == 'collar':
        return 1
```
替换成：

```
if row_label == 'apple':
        return 1
if row_label == 'banana':
       return 2
if row_label == 'orange':
       return 3
```
接下来通过从\ object_detection文件夹发出以下命令来生成TFRecord文件

```
python generate_tfrecord.py --csv_input=train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=test_labels.csv --image_dir=images\test --output_path=test.record
```
它们在\ object_detection中生成train.record和test.record文件。这些将用于训练新的物体检测分类器。

5.创建标签映射
-------------
在  tensorflow  \ models \ research \ object_detection \ training目录下创建labelmap.pbtxt。例如，我需要检测的是苹果，香蕉，橘子，labelmap.pbtxt的内容如下：
***注：此处的id，name要与3-2中generate_tfrecord.py中标记的相同***

```
item {
  id: 1
  name: 'apple'
}

item {
  id: 2
  name: 'banana'
}

item {
  id: 3
  name: 'orange'
}
```

6.配置培训修改
--------
 配置对象检测训练管道。它定义了哪个模型以及将用于培训的参数。这是运行训练前的最后一步。

导航到 tensorflow1 \ models \ research \ object_detection \ samples \ configs，并将faster_rcnn_inception_v2_pets.config文件复制到\ object_detection \ training目录中。然后，使用编辑器打开文件。对.config文件进行了一些更改，主要是更改种类和示例的数量，以及将文件路径添加到训练数据中。

对faster_rcnn_inception_v2_pets.config文件进行以下更改。***注意：必须使用单个正斜杠（非反斜杠）输入路径，否则TensorFlow会在尝试训练模型时出现文件路径错误！此外，路径必须是双引号（“），而不是单引号（'）***。

第9行。将num_classes更改为您希望分类器检测的不同对象的数量。对于上面的苹果，香蕉，橘子探测器，它将是num_classes：3。

第106行。将fine_tune_checkpoint更改为：

```
fine_tune_checkpoint：“D:/Image-GPU/tensorflow/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt”
```

第123和125行。在train_input_reader部分中，将input_path和label_map_path更改为：

```
input_path：“D:/Image-GPU/tensorflow/models/research/object_detection/train.record”
label_map_path：“D:/Image-GPU/tensorflow/models/research/object_detection/training/labelmap.pbtxt”
```

第130行。将num_examples更改为\ images \ test目录中的图像数。

第135和137行。在eval_input_reader部分中，将input_path和label_map_path更改为：

```
input_path：“D:/Image-GPU/tensorflow/models/research/object_detection/test.record”
label_map_path：“D:/Image-GPU/tensorflow/models/research/object_detection/training/labelmap.pbtxt”
```

完成更改后保存文件。培训工作全部配置完毕并准备就绪。

7.运行培训
------

```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```

若是object-detection中没有train.py文件，可进入
到/research/object_detection/legacy 目录下，将train.py拷贝
到/research/object_detection目录下执行即可。

如果一切设置正确，TensorFlow将初始化培训。在实际训练开始之前，初始化可能需要几十秒。培训开始时，它将如下所示：
![这里写图片描述](https://github.com/holidayun/tesorflow-tools/raw/master/screenshots/training.png)

8.导出推理图
------

当训练已经完成，最后一步是生成.pb文件。从\ object_detection  目录下，发出以下命令，其中“model.ckpt-XXXX”中的“XXXX”应替换为training文件夹中编号最大的.ckpt文件：

```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```

这将在\ object_detection \ inference_graph文件夹中创建一个
frozen_inference_graph.pb文件。.pb文件包含对象检测分类器。

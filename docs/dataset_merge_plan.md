# 最终 16 类类别表与三数据集映射方案

## 最终 16 类类别表

1. apple
2. banana
3. orange
4. tomato
5. cucumber
6. potato
7. onion
8. carrot
9. green_pepper
10. egg
11. milk
12. bread
13. beverage
14. bottled_condiment
15. leafy_vegetable
16. packaged_food

## 设计原则

这 16 类不是按“商品最细粒度名称”设计，而是按“冰箱管理语义”设计：

1. 生鲜常见单品单独保留。
2. 果汁、汽水、茶饮等统一归到 `beverage`。
3. 酱油、沙拉酱、番茄酱等统一归到 `bottled_condiment`。
4. 白菜、生菜、菠菜等统一归到 `leafy_vegetable`。
5. 包装零食、黄油、酸奶、盒装食品等统一归到 `packaged_food`。

这样做的优点是：

1. 比 12 类更接近真实冰箱场景。
2. 比 30 类更稳，更适合课程项目答辩。
3. 检测结果更容易转成库存管理、补货提醒和空间整理建议。

## 三个数据集的使用优先级

### 1. Refrigerator Items

作为主数据集使用。

原因：

1. 场景最像真实冰箱。
2. 有堆叠、遮挡、多目标。
3. 最适合解决“整张冰箱图只检出 1 个”的问题。

### 2. Fridge Objects

作为冰箱环境补充数据使用。

原因：

1. 图少，但场景仍然是冰箱内部。
2. 能补充货架、冷光、摆放方式等背景分布。

### 3. Grocery Store Products

作为扩类辅助数据使用。

原因：

1. 能补饮料、包装食品、调料等类别覆盖。
2. 但场景偏超市，不适合作为主训练集。

## 推荐训练策略

1. 主训练数据：`Refrigerator Items`
2. 补充场景：`Fridge Objects`
3. 扩类辅助：`Grocery Store Products`
4. 模型建议：`yolov8s`
5. 图片尺寸建议：`640` 起步，冰箱整图可尝试 `800`
6. 优先先做 16 类稳定版，不建议直接上 30 类以上

## 下载方式

由于当前环境被 Roboflow 的 Cloudflare 校验拦住，无法稳定自动下载。建议你手动在浏览器中下载 YOLOv8 格式压缩包，然后放入下面目录：

```text
smart-fridge-vision/
└─ external_datasets/
   ├─ refrigerator_items/
   ├─ fridge_objects/
   └─ grocery_store_products/
```

每个目录中放对应数据集解压后的 YOLOv8 数据结构：

```text
dataset/
├─ data.yaml
├─ train/
├─ valid/ 或 val/
└─ test/
```

## 自动合并脚本

仓库已提供：

`scripts/merge_roboflow_yolo_datasets.py`

它会：

1. 读取 `config/dataset_merge_mapping.yaml`
2. 按映射关系重写标签
3. 合并三个数据集
4. 输出统一的 16 类 YOLO 数据集


"""
Thực hành cây quyết định - Phân loại Breast Cancer
    Step 1: Load dữ liệu
    Step 2: Chia tập train và tập test
    Step 3: Khởi tạo Tree
    Step 4: Fit tập train vào Tree 
    Step 5: Predict tập test
    Step 6: Tính accuracy của tập predict và tập test
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer()
print('cancer.keys():\n {}'.format(cancer.keys()))
print('Kích thước dữ liệu:\n {}'.format(cancer.data.shape))
print('Các thuộc tính: \n{}'.format(cancer.feature_names))
print('Các lớp: \n{}'.format(cancer.target_names))

x = cancer.data
y = cancer.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42)

tree = DecisionTreeClassifier(random_state=42)
tree.fit(x_train, y_train)
y_pred = tree.predict(x_test)

print('Độ chính xác tập huấn luyện: {:.4f}'.format(
    tree.score(x_train, y_train)))
print('Độ chính xác tập kiểm tra: {:.4f}'.format(tree.score(x_test, y_test)))

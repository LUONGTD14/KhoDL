import numpy as np
import pandas as pd
from pprint import pprint
from DecisionTree import decision_tree_algorithm, make_predictions, calculate_accuracy
from DataHandler import generate_data, create_plot, train_test_split

#chia dữ liệu sang 2 nhánh của cây
def filter_df(df, question):
    feature, comparison_operator, value = question.split()
    # thuộc tính liên tục
    if comparison_operator == "<=":
        df_yes = df[df[feature] <= float(value)]
        df_no = df[df[feature] > float(value)]
    # thuộc tính phân loại
    else:
        df_yes = df[df[feature].astype(str) == value]
        df_no = df[df[feature].astype(str) != value]
    return df_yes, df_no

#xác định nhãn của lá
def determine_leaf(df_train, ml_task):
    #bài toàn hồi quy: trung bình nhãn dự đoán
    if ml_task == "regression":
        return df_train.label.mean()
    # bài toàn phân loại: nhãn chiếm đa số
    else:
        return df_train.label.value_counts().index[0]

#xác định lỗi
def determine_errors(df_val, tree, ml_task):
    predictions = make_predictions(df_val, tree)
    actual_values = df_val.label
    #bài toán hồi quy sử dụng độ đo MSE = 1/n ( (y-y')^2)
    #bài toán phân loại sẽ là số lượng nhãn khác nhau dự đoán và thực tế
    if ml_task == "regression":
        return ((predictions - actual_values) ** 2).mean()
    else:
        return sum(predictions != actual_values)

def pruning_result(tree, df_train, df_val, ml_task):
    leaf = determine_leaf(df_train, ml_task)
    errors_leaf = determine_errors(df_val, leaf, ml_task)
    errors_decision_node = determine_errors(df_val, tree, ml_task)
    #nếu lỗi do lá xác định tốt hơn cây thì sử dụng nhãn của lá sẽ tốt hơn
    if errors_leaf <= errors_decision_node:
        return leaf
    else:
        return tree

#tỉa cây
def post_pruning(tree, df_train, df_val, ml_task):
    question = list(tree.keys())[0]
    yes_answer, no_answer = tree[question]
    #nút đã có nhãn thì chỉ kiểm tra lỗi
    if not isinstance(yes_answer, dict) and not isinstance(no_answer, dict):
        return pruning_result(tree, df_train, df_val, ml_task)
    else:
        df_train_yes, df_train_no = filter_df(df_train, question)
        df_val_yes, df_val_no = filter_df(df_val, question)
        #đệ quy trên 2 nhánh của cây tới khi ra nút cuối cùng
        if isinstance(yes_answer, dict):
            yes_answer = post_pruning(yes_answer, df_train_yes, df_val_yes, ml_task)
        if isinstance(no_answer, dict):
            no_answer = post_pruning(no_answer, df_train_no, df_val_no, ml_task)
        tree = {question: [yes_answer, no_answer]}
        return pruning_result(tree, df_train, df_val, ml_task)
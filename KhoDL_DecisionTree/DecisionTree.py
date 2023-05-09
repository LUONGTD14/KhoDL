# coding: utf-8
import numpy as np
import pandas as pd


# kiểm tra dữ liệu thuần khiết
def check_purity(data):
    label_column = data[:, -1]
    #lấy ra các giá trị khá cnhau của cột
    unique_classes = np.unique(label_column)
    if len(unique_classes) == 1:
        return True
    else:
        return False

# đưa ra quyết định của lá
def create_leaf(data, ml_task):
    label_column = data[:, -1]
    #bài toán hồi quy: trung bình nhãn dự đoán
    if ml_task == "regression":
        leaf = np.mean(label_column)
    # bài toàn phân loại: nhãn chiếm đa số
    else:
        unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
        index = counts_unique_classes.argmax()
        leaf = unique_classes[index]
    return leaf

# lấy ra tất cả các giá trị của cột để chia dữ liệu về 2 phía
def get_potential_splits(data):
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):  # bỏ đi cột nhãn
        values = data[:, column_index]
        unique_values = np.unique(values)
        potential_splits[column_index] = unique_values
    return potential_splits

# Entropy = sum(-p*log2(p))
def calculate_entropy(data):
    label_column = data[:, -1]
    #các giá trị khác nhau và số lượng
    _, counts = np.unique(label_column, return_counts=True)
    #xác suất của giá trị
    probabilities = counts / counts.sum()
    #tổng entropy
    entropy = sum(probabilities * -np.log2(probabilities))
    return entropy

#MSE = 1/n*sum((y-y')^2)
def calculate_mse(data):
    #kết qủa dự đoán trống hoặc không
    actual_values = data[:, -1]
    if len(actual_values) == 0:
        mse = 0
    else:
        prediction = np.mean(actual_values)
        mse = np.mean((actual_values - prediction) ** 2)
    return mse

# GAIN max -> tổng Entropy con min = sum(-p*log2(p))
#MSE min
def calculate_overall_metric(data_below, data_above, metric_function):
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n
    overall_metric = (p_data_below * metric_function(data_below)
                      + p_data_above * metric_function(data_above))
    return overall_metric

#giá trị phân chia tốt nhất
def determine_best_split(data, potential_splits, ml_task):
    first_iteration = True
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            #machine learning task: hồi quy thì tính tổng MSE
            if ml_task == "regression":
                current_overall_metric = calculate_overall_metric(data_below, data_above, metric_function=calculate_mse)
            # machine learning task: phân loại thì tính tổng Entropy
            else:
                current_overall_metric = calculate_overall_metric(data_below, data_above,
                                                                  metric_function=calculate_entropy)
            #min là tốt
            if first_iteration or current_overall_metric <= best_overall_metric:
                first_iteration = False

                best_overall_metric = current_overall_metric
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value


# chia dữ liệu theo điều kiện chia
def split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]
    type_of_feature = FEATURE_TYPES[split_column]
    #giá trị liên tục
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values > split_value]
    #giá trị phân loại có thể là string -> không thể sử dụng <=
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]
    return data_below, data_above


# xác định kiểu của thuộc tính
def determine_type_of_feature(df):
    feature_types = []
    n_unique_values_treshold = 15 # 1 thuộc tính có nhiều hơn 15 giá trị coi như là liên tục
    for feature in df.columns:
        if feature != "label":
            unique_values = df[feature].unique()
            example_value = unique_values[0]
            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    return feature_types


#thuật toán
def decision_tree_algorithm(df, ml_task, counter=0, min_samples=2, max_depth=5):
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES # biến toàn cục
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values
    else:
        data = df
    #trường hợp ngay khi bắt đầu dữ liệu đã tinh khiết
    #trường hợp lá không đử dữ liệu để phân tách
    #trường hợp đạt độ xâu tối đa
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        leaf = create_leaf(data, ml_task)
        return leaf
    else:
        counter += 1
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits, ml_task)
        data_below, data_above = split_data(data, split_column, split_value)
        #dừng khi dữ liệu trống
        if len(data_below) == 0 or len(data_above) == 0:
            leaf = create_leaf(data, ml_task)
            return leaf
        #tạo ra sub_tree = {question: điều kiện, {[đúng], [sai]}}
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)
        else:
            question = "{} = {}".format(feature_name, split_value)
        sub_tree = {question: []}
        # thuật toán tiếp tục trên 2 nhánh của cây
        yes_answer = decision_tree_algorithm(data_below, ml_task, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(data_above, ml_task, counter, min_samples, max_depth)
        #trường hợp 2 nhánh giống nhau
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        return sub_tree


# dự đoán nhãn bằng cách so sánh từng node
def predict_example(example, tree):
    #cây chỉ có 1 node
    if not isinstance(tree, dict):
        return tree
    #lấy ra các điều kiện của cây từ gốc tới ngọn
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")
    # so sánh
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    #đệ quy cho tới khi ra kết quả
    if not isinstance(answer, dict):
        return answer
    else:
        residual_tree = answer
        return predict_example(example, residual_tree)


#dự đoán trên tập
def make_predictions(df, tree):
    if len(df) != 0:
        predictions = df.apply(predict_example, args=(tree,), axis=1)
    else:
        predictions = pd.Series()
    return predictions

# độ chính xác
def calculate_accuracy(df, tree):
    predictions = make_predictions(df, tree)
    predictions_correct = predictions == df.label
    accuracy = predictions_correct.mean()
    return accuracy
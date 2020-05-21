import os
import time

import numpy as np
import pandas as pd

from dataset import NUM_PTS

column_names = ",".join([f"Point_M{i}_X,Point_M{i}_Y" for i in range(30)])
SUBMISSION_HEADER = f"file_name,{column_names}\n"


def time_measurer(point_time):
    time_diff = time.time() - point_time

    if time_diff >= 60:
        minutes = int(time_diff // 60)
        seconds = round(time_diff - minutes * 60)
        str_time = f"{minutes} minutes, {seconds} seconds"
    elif time_diff >= 1:
        seconds = round(time_diff, 1)
        str_time = f"{seconds} seconds"
    else:
        seconds = round(time_diff, 3)
        str_time = f"{seconds} seconds"

    print(f"done in {str_time}\n")

    return time.time()


def create_submission(path_to_data, test_predictions, output_file):
    with open(output_file, "w") as wf:
        wf.write(SUBMISSION_HEADER)

        mapping_path = os.path.join(path_to_data, "test/test_points.csv")
        mapping = pd.read_csv(mapping_path, delimiter="\t")

        n_rows_to_predict = mapping.shape[0]
        assert mapping.shape == (n_rows_to_predict, 2)
        assert test_predictions.shape == (n_rows_to_predict, NUM_PTS, 2)

        for i, row in mapping.iterrows():
            file_name = row[0]
            point_index_list = np.array(eval(row[1]))

            needed_points = test_predictions[i][point_index_list]
            assert isinstance(needed_points, np.ndarray)
            needed_points = np.around(needed_points).astype(np.int)
            needed_points = needed_points.reshape(-1)
            assert needed_points.shape == (2 * len(point_index_list),)

            wf.write(file_name + "," + ",".join(map(str, needed_points)) + "\n")

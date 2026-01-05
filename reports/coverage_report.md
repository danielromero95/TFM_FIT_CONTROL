........................................................................ [ 47%]
........................................................................ [ 94%]
........                                                                 [100%]
=============================== tests coverage ================================
______________ coverage: platform win32, python 3.10.19-final-0 _______________

Name                                                       Stmts   Miss  Cover   Missing
----------------------------------------------------------------------------------------
src\A_preprocessing\__init__.py                                0      0   100%
src\A_preprocessing\frame_extraction\__init__.py               4      0   100%
src\A_preprocessing\frame_extraction\core.py                  49     11    78%   72, 91-92, 124-142
src\A_preprocessing\frame_extraction\index_sampling.py        34      8    76%   30, 38, 41-44, 60-61
src\A_preprocessing\frame_extraction\preprocess.py            39      6    85%   40, 43, 49-51, 86
src\A_preprocessing\frame_extraction\state.py                 77     16    79%   36-37, 50, 52-60, 62, 77-80, 84, 104, 107
src\A_preprocessing\frame_extraction\time_sampling.py         86     34    60%   59, 74, 96, 99-122, 127, 134-146
src\A_preprocessing\frame_extraction\utils.py                 33     17    48%   28, 37-41, 48-55, 62-65, 71
src\A_preprocessing\frame_extraction\validation.py            16      3    81%   17, 27, 30
src\A_preprocessing\video_metadata.py                        212     56    74%   45-62, 67-85, 92, 99, 108, 122, 126, 148, 154-156, 207-213, 238-239, 247-267, 326-327, 365, 383-384, 424-425, 429-434, 446-447
src\A_preprocessing\video_utils.py                            21      2    90%   27-28
src\B_pose_estimation\__init__.py                              6      0   100%
src\B_pose_estimation\constants.py                            41      0   100%
src\B_pose_estimation\estimators\__init__.py                   3      0   100%
src\B_pose_estimation\estimators\base.py                      15      3    80%   24, 27-28
src\B_pose_estimation\estimators\mediapipe_estimators.py     419     91    78%   51, 96, 116, 139-140, 154, 285-296, 402, 488, 515-521, 528-533, 547-558, 606-609, 621-629, 735-787, 792-796, 850-859, 887-893
src\B_pose_estimation\estimators\mediapipe_pool.py            52     30    42%   24-30, 43-70, 74-75, 79-86
src\B_pose_estimation\geometry.py                             98     13    87%   45, 48, 68-71, 73-76, 134, 142-143
src\B_pose_estimation\metrics\__init__.py                      5      0   100%
src\B_pose_estimation\metrics\angles.py                       28     19    32%   13-19, 25-32, 38-45
src\B_pose_estimation\metrics\distances.py                    17     11    35%   15-26
src\B_pose_estimation\metrics\normalization.py                18     11    39%   16-35
src\B_pose_estimation\metrics\timeseries.py                   37      5    86%   24, 28, 39, 42, 57
src\B_pose_estimation\pipeline\__init__.py                     4      0   100%
src\B_pose_estimation\pipeline\compute.py                     80      8    90%   62, 150, 154, 157-158, 160, 188-189
src\B_pose_estimation\pipeline\extract.py                     66     15    77%   44, 46, 85, 98-103, 106-107, 123-127
src\B_pose_estimation\pipeline\postprocess.py                 88     78    11%   27-136
src\B_pose_estimation\roi_state.py                            77      8    90%   42-43, 46-51, 113
src\B_pose_estimation\signal.py                               86     18    79%   36, 45, 83, 87, 98, 109-120, 132, 134, 138
src\B_pose_estimation\types.py                                50      8    84%   32, 35, 38, 43-44, 51, 73, 82
src\C_analysis\__init__.py                                     4      0   100%
src\C_analysis\config_bridge.py                               26     21    19%   14-34
src\C_analysis\errors.py                                       3      0   100%
src\C_analysis\metrics.py                                    190    147    23%   52-57, 68-105, 111-122, 128-138, 144-155, 165-210, 225-279, 307, 320-322, 329-341, 356-365
src\C_analysis\overlay.py                                     75     59    21%   46-78, 103-170
src\C_analysis\pipeline.py                                   302     92    70%   52-55, 70, 73-77, 183-184, 217, 251-258, 273, 283-294, 328-330, 337, 340-344, 347, 369-446, 448, 471, 478, 509-511, 523-527, 537, 546-550, 553-554, 576, 642-645
src\C_analysis\repetition_counter.py                         181      7    96%   75, 79, 137, 158, 162, 216, 274
src\C_analysis\sampling.py                                    63     18    71%   39, 43, 52, 108-113, 122-136
src\C_analysis\streaming.py                                  235    205    13%   63-76, 88-142, 158-395, 407-421
src\D_visualization\__init__.py                                5      0   100%
src\D_visualization\landmark_drawing.py                       17      9    47%   30-36, 44-47
src\D_visualization\landmark_geometry.py                      74     37    50%   31-34, 39-40, 49-50, 52, 62-64, 66-71, 92-119
src\D_visualization\landmark_overlay_styles.py                18      0   100%
src\D_visualization\landmark_renderers.py                    201    141    30%   48-156, 191, 201-202, 213, 227, 230-282, 300-303, 307, 311, 315-318, 325-328
src\D_visualization\landmark_transforms.py                    13      3    77%   37-39
src\D_visualization\landmark_video_io.py                     106     40    62%   50-65, 81, 86-87, 96-98, 113-118, 133, 149-156, 163-164, 195-202, 208, 212-214, 217-219, 223-225
src\__init__.py                                                0      0   100%
src\config\__init__.py                                         4      0   100%
src\config\constants.py                                       12      0   100%
src\config\models.py                                          94     12    87%   110, 118, 143, 145, 153-161
src\config\settings.py                                        36      8    78%   13-22, 114
src\config\utils.py                                           11      4    64%   21-24
src\config\video_landmarks_visualization.py                    6      0   100%
src\core\types.py                                             39      6    85%   36, 45, 67, 69, 72-73
src\exercise_detection\__init__.py                             3      0   100%
src\exercise_detection\classification.py                     148     11    93%   97, 132, 143, 160, 168-169, 171, 235, 237, 239, 246
src\exercise_detection\classifier.py                         171     27    84%   268, 272, 288-291, 295, 300-323
src\exercise_detection\constants.py                          105      0   100%
src\exercise_detection\exercise_detector.py                   31     18    42%   22-50, 58-86
src\exercise_detection\extraction.py                         174    130    25%   46-139, 168-224, 258-318, 322-323, 330, 385-393
src\exercise_detection\features.py                            94     29    69%   57, 62-64, 77, 82-83, 101, 126, 132-134, 153-156, 182-196, 252, 256
src\exercise_detection\incremental.py                        182     65    64%   49, 54-56, 59-62, 108, 112-113, 116, 121, 128-143, 148-172, 196-197, 204-211, 217-218, 238-251, 296-316, 319-320, 330-332
src\exercise_detection\metrics.py                            122     16    87%   35, 49, 67, 74-80, 173, 175-176, 190, 200, 210
src\exercise_detection\segmentation.py                        94     12    87%   38, 44, 46, 50, 98, 101, 108, 133, 142-145
src\exercise_detection\smoothing.py                           39      5    87%   21, 28, 41, 49, 52
src\exercise_detection\stats.py                               37     11    70%   24-27, 33, 38-41, 47, 54
src\exercise_detection\types.py                               82      4    95%   26, 50-52
src\exercise_detection\view.py                               146      8    95%   63, 186, 213, 217-218, 220, 229, 238
src\pipeline_data.py                                          64      7    89%   80-105
----------------------------------------------------------------------------------------
TOTAL                                                       4968   1623    67%
152 passed in 4.66s

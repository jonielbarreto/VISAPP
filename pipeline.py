import os
from sahi_functions import SahiDetectionPipeline
from yolo_functions import YOLOPipeline
from yolo_coco_eval import COCOEvaluator
from ultralytics import YOLO

def yolo_pipeline(data_yaml, model_path, images_dir, labels_dir, to_train=True, video_path=None, visualize=False, img_size=640, conf_score=0.5, iou_match=0.5):
    
    pipeline = YOLOPipeline(model_path=model_path)
    # Train the model
    if to_train:
        train_results = pipeline.train(data_config_path=data_yaml, img_size=img_size, epochs=100)
        best_model_path = os.path.join(train_results.save_dir, 'weights', 'best.pt')
        print(f"\nLoading best model: {best_model_path}")
        pipeline.model = YOLO(best_model_path)
    # Test the model
    yolo_preds_json = pipeline.validate(
        data_config_path=data_yaml, 
        img_size=img_size, 
        conf_score=conf_score, 
        iou_thr=iou_match
    )
    if not os.path.exists(yolo_preds_json):
        print(f"\nError: predictions.json file not found in: {yolo_preds_json}")
        return
    
    txt_log_name = f"metrics_yolo_native_sz{img_size}.txt"
    txt_log_path = os.path.join(os.path.dirname(yolo_preds_json), txt_log_name)
    print(f"\nSaving analytical report to: {txt_log_path}")
    COCOEvaluator.evaluate_predictions(
        json_pred_file=yolo_preds_json, 
        orig_size=(1920, 1080), 
        target_size=(img_size, img_size), 
        gt_dir=labels_dir,
        output_txt_path=txt_log_path
    )

    # Get Precision/Recall/F1-score metrics
    txt_log_name = f"PR_yolo_native_sz{img_size}.txt"
    txt_log_path = os.path.join(os.path.dirname(yolo_preds_json), txt_log_name)
    COCOEvaluator.calculate_static_metrics(
        json_pred_file=yolo_preds_json,
        orig_size=(1920, 1080),
        target_size=(img_size, img_size),
        gt_dir=labels_dir,
        conf_threshold=conf_score,
        iou_threshold=iou_match,
        output_txt_path=txt_log_path
    )

    # Visual Analysis
    if visualize:
        pipeline.run_dataset_error_analysis(
            images_dir=images_dir, 
            labels_dir=labels_dir, 
            img_size=img_size, 
            confidence_threshold=conf_score, 
            iou_threshold=iou_match
        )
    if video_path is not None:pipeline.run_video_visualization(video_path=video_path, confidence_threshold=conf_score)

def sahi_pipeline(data_yaml, model_path, images_dir, labels_dir, to_train=True, video_path=None, visualize=False, conf_score=0.5, iou_match=0.5):
    
    # Train the model using YOLO
    pipeline = YOLOPipeline(model_path=model_path)
    if to_train:
        train_results = pipeline.train(data_config_path=data_yaml, img_size=640, epochs=100)
        best_model_path = os.path.join(train_results.save_dir, 'weights', 'best.pt')
        print(f"\nLoading best model: {best_model_path}")
        sahi_model_path=best_model_path
    else: sahi_model_path=model_path

    # SAHI only mode evaluation
    pipeline_sahi = SahiDetectionPipeline(model_path=sahi_model_path, mode="sahi_only", confidence_threshold=conf_score)
    log_file_sahi = os.path.join("C:/Users/Aluno-ITA/Documents/AuSSys/VISAPP/results", "metrics_sahi_only.txt")
    matrix_sahi = pipeline_sahi.run_dataset_benchmark(
        images_dir=images_dir, 
        labels_dir=labels_dir,
        output_txt_path=log_file_sahi
    )

    # Get Precision/Recall/F1-score metrics
    log_file_sahi = os.path.join("C:/Users/Aluno-ITA/Documents/AuSSys/VISAPP/results", "PR_sahi_only.txt")
    pipeline_sahi.calculate_static_metrics(
        images_dir=images_dir, 
        labels_dir=labels_dir,
        conf_score=conf_score, 
        iou_threshold=iou_match,
        output_txt_path=log_file_sahi
    )

    # Visual Analysis
    if visualize: 
        pipeline_sahi.run_dataset_error_analysis(
            images_dir=images_dir, 
            labels_dir=labels_dir,
            iou_threshold=iou_match
        )
    if video_path is not None:pipeline_sahi.run_video_visualization(video_path)
    """
    # SAHI + Temporal mode evaluation
    for seq in [1, 2, 3, 5]:
        print("\n" + "="*50)
        print(f"        Number of sequence: [{seq}]")
        print("="*50)
        
        pipeline_temporal = SahiDetectionPipeline(model_path=model_path, mode="temporal_consistency", confidence_threshold=conf_score)
        pipeline_temporal.max_memory_frames = 0        
        pipeline_temporal.edge_tolerance_px = 0
        pipeline_temporal.min_confirmation_frames = seq 
        pipeline_temporal.temporal.min_confirmation_frames = seq

        file_name = f"metrics_temporal_consistency_{seq}.txt"
        log_file_temporal = os.path.join("/home/jonba/Documents/AuSSys/VISAPP/results", file_name) 
        matrix_temporal = pipeline_temporal.run_dataset_benchmark(
            images_dir=images_dir, 
            labels_dir=labels_dir,
            output_txt_path=log_file_temporal
        )
        # Get Precision/Recall/F1-score metrics
        file_name = f"PR_temporal_consistency_{seq}.txt"
        log_file_temporal = os.path.join("/home/jonba/Documents/AuSSys/VISAPP/results", file_name) 
        pipeline_temporal.calculate_static_metrics(
            images_dir=images_dir, 
            labels_dir=labels_dir,
            conf_score=conf_score, 
            iou_threshold=iou_match,
        )

        # Visual Analysis
        if visualize: 
            pipeline_temporal.run_dataset_error_analysis(
                images_dir=images_dir, 
                labels_dir=labels_dir,
                iou_threshold=iou_match
            )
        if video_path is not None:pipeline_temporal.run_video_visualization(video_path)
    """

def main():
    video_dir = f"D:/Caragua_mission/DCIM/Wide/DJI_20231205101707_0002_W.mp4"

    for scene in ["Wide2"]:
        print("\n" + "="*50)
        print(f"EVALUATING : {scene}")
        print("\n" + "="*50)
        yaml_path = f"D:/MMOS_test_set_10fps/{scene}/data.yaml"
        imgs_path = f"D:/MMOS_test_set_10fps/{scene}/test/images"
        lbls_path = f"D:/MMOS_test_set_10fps/{scene}/test/labels"

        # YOLO: imgsz=640
        #model_path = "/home/jonba/Documents/AuSSys/Paper_Dataset/YOLO/wide/train/weights/best.pt"
        #yolo_pipeline(yaml_path, model_path, imgs_path, lbls_path, to_train=False, img_size=640, iou_match=0.5, visualize=False)

        # YOLO: imgsz=1920x1080
        #model_path = "/home/jonba/Documents/AuSSys/YOLO_detect/runs/detect/yolo_1920/weights/best.pt"
        #yolo_pipeline(yaml_path, model_path, imgs_path, lbls_path, to_train=False, img_size=1920, iou_match=0.5, visualize=False)

        # SAHI: imgsz=640
        model_path = "C:/Users/Aluno-ITA/Documents/AuSSys/Paper_Dataset/YOLO/wide/train/weights/best.pt"
        sahi_pipeline(yaml_path, model_path, imgs_path, lbls_path, to_train=False, iou_match=0.5, visualize=False)

if __name__ == "__main__":
    main()
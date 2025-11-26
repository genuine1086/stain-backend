from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os
import traceback

app = Flask(__name__)
# 모든 origin 허용 (Netlify, localhost 등에서 접근 가능하게)
CORS(app)

# 모델 경로 (app.py와 같은 폴더에 best.pt 있다고 가정)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
model = YOLO(MODEL_PATH)


@app.route("/", methods=["GET"])
def index():
    return "Stain Classifier API Running", 200


@app.route("/predict", methods=["POST"])
def predict():
    # 1) 파일 존재 확인
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    # 2) /tmp에 임시 저장 (Render에서도 항상 존재하는 경로)
    temp_path = os.path.join("/tmp", "upload.jpg")
    file.save(temp_path)

    try:
        # 3) 모델 추론
        results = model(temp_path)[0]

        cls_id = results.probs.top1
        cls_name = results.names[cls_id]
        conf = float(results.probs.top1conf)

        return jsonify({
            "class": cls_name,
            "confidence": conf
        }), 200

    except Exception as e:
        # 에러 로그 서버 콘솔에 찍기
        traceback.print_exc()
        return jsonify({
            "error": "inference failed",
            "detail": str(e)
        }), 500


if __name__ == "__main__":
    # 로컬에서 테스트할 때만 사용 (Render에서는 gunicorn으로 실행)
    app.run(host="0.0.0.0", port=10000, debug=True)

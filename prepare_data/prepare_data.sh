#!/usr/bin/env bash
# =============================================================================
# prepare_data.sh — Ditto Training: Video → Feature Extraction Pipeline
#
# Usage:
#   bash prepare_data.sh <data_info_json> <data_list_json> <data_preload_pkl>
#
# Arguments:
#   data_info_json   Path to input  data_info.json  (maps videos → feature paths)
#   data_list_json   Path to output data_list.json  (used during training)
#   data_preload_pkl Path to output data_preload.pkl (preloaded cache for training)
# =============================================================================

set -euo pipefail
SECONDS=0

# ---------------------------------------------------------------------------
# Paths — edit these two variables to match your RunPod workspace
# ---------------------------------------------------------------------------
DITTO_ROOT_DIR="/workspace/ditto-train-fixed-for-HDTF"
DITTO_PYTORCH_PATH="${DITTO_ROOT_DIR}/checkpoints/ditto_pytorch"

# Derived checkpoint paths (no need to edit)
HUBERT_ONNX="${DITTO_PYTORCH_PATH}/aux_models/hubert_streaming_fix_kv.onnx"
MP_FACE_LMK_TASK="${DITTO_PYTORCH_PATH}/aux_models/face_landmarker.task"

PYTHON="/workspace/miniconda/envs/ditto_train/bin/python"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
if [ "$#" -ne 3 ]; then
    echo "Usage: bash prepare_data.sh <data_info_json> <data_list_json> <data_preload_pkl>"
    exit 1
fi

data_info_json="$1"
data_list_json="$2"
data_preload_pkl="$3"

echo "============================================================"
echo " Ditto Data Preparation Pipeline"
echo "============================================================"
echo "  DITTO_ROOT_DIR   : ${DITTO_ROOT_DIR}"
echo "  DITTO_PYTORCH_PATH: ${DITTO_PYTORCH_PATH}"
echo "  data_info_json   : ${data_info_json}"
echo "  data_list_json   : ${data_list_json}"
echo "  data_preload_pkl : ${data_preload_pkl}"
echo "============================================================"

cd "${DITTO_ROOT_DIR}/prepare_data"

# ---------------------------------------------------------------------------
# Step 0: Verify checkpoint files exist
# ---------------------------------------------------------------------------
echo ""
echo "[Step 0] Checking checkpoints..."
${PYTHON} scripts/check_ckpt_path.py --ditto_pytorch_path "${DITTO_PYTORCH_PATH}"
echo "  Checkpoints OK."

# ---------------------------------------------------------------------------
# Step 1: Crop video (fps25_video_list → video_list)
# ---------------------------------------------------------------------------
echo ""
echo "[Step 1] Cropping videos via LP face detector..."
${PYTHON} scripts/crop_video_by_LP.py \
    -i "${data_info_json}" \
    --ditto_pytorch_path "${DITTO_PYTORCH_PATH}"

# ---------------------------------------------------------------------------
# Step 2: Extract audio (video_list → wav_list)
# ---------------------------------------------------------------------------
echo ""
echo "[Step 2] Extracting audio from videos..."
${PYTHON} scripts/extract_audio_from_video.py \
    -i "${data_info_json}"

# ---------------------------------------------------------------------------
# Step 3: Extract audio features via HuBERT (wav_list → hubert_aud_npy_list)
# ---------------------------------------------------------------------------
echo ""
echo "[Step 3] Extracting HuBERT audio features..."
${PYTHON} scripts/extract_audio_feat_by_Hubert.py \
    -i "${data_info_json}" \
    --Hubert_onnx "${HUBERT_ONNX}"

# ---------------------------------------------------------------------------
# Step 4: Extract motion features via LivePortrait
#         video_list → {LP_pkl_list, LP_npy_list}  (normal + flipped)
# ---------------------------------------------------------------------------
echo ""
echo "[Step 4a] Extracting LP motion features (normal)..."
${PYTHON} scripts/extract_motion_feat_by_LP.py \
    -i "${data_info_json}" \
    --ditto_pytorch_path "${DITTO_PYTORCH_PATH}"

echo ""
echo "[Step 4b] Extracting LP motion features (flipped)..."
${PYTHON} scripts/extract_motion_feat_by_LP.py \
    -i "${data_info_json}" \
    --ditto_pytorch_path "${DITTO_PYTORCH_PATH}" \
    --flip_flag

# ---------------------------------------------------------------------------
# Step 5: Extract eye features via MediaPipe
#         video_list → {MP_lmk_npy_list, eye_open_npy_list, eye_ball_npy_list}
#         (normal + flipped)
# ---------------------------------------------------------------------------
echo ""
echo "[Step 5a] Extracting eye ratio features (normal)..."
${PYTHON} scripts/extract_eye_ratio_from_video.py \
    -i "${data_info_json}" \
    --MP_face_landmarker_task_path "${MP_FACE_LMK_TASK}"

echo ""
echo "[Step 5b] Extracting eye ratio features (flipped)..."
${PYTHON} scripts/extract_eye_ratio_from_video.py \
    -i "${data_info_json}" \
    --MP_face_landmarker_task_path "${MP_FACE_LMK_TASK}" \
    --flip_lmk_flag

# ---------------------------------------------------------------------------
# Step 6: Extract emotion features (video_list → emo_npy_list)
# ---------------------------------------------------------------------------
echo ""
echo "[Step 6] Extracting emotion features..."
${PYTHON} scripts/extract_emo_feat_from_video.py \
    -i "${data_info_json}"

# ---------------------------------------------------------------------------
# Step 7: Gather data_list.json for training
# ---------------------------------------------------------------------------
echo ""
echo "[Step 7] Building data_list.json..."
${PYTHON} scripts/gather_data_list_json_for_train.py \
    -i "${data_info_json}" \
    -o "${data_list_json}" \
    --use_emo \
    --use_eye_open \
    --use_eye_ball \
    --with_flip

# ---------------------------------------------------------------------------
# Step 8: Preload training data into a .pkl cache (optional but recommended)
# ---------------------------------------------------------------------------
echo ""
echo "[Step 8] Preloading training data to pkl cache..."
${PYTHON} scripts/preload_train_data_to_pkl.py \
    --data_list_json "${data_list_json}" \
    --data_preload_pkl "${data_preload_pkl}" \
    --use_sc \
    --use_emo \
    --use_eye_open \
    --use_eye_ball \
    --motion_feat_dim 265

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
cd "${DITTO_ROOT_DIR}"

echo ""
echo "============================================================"
echo " [prepare_data] DONE"
echo "  data_list_json   : ${data_list_json}"
echo "  data_preload_pkl : ${data_preload_pkl}"
echo "  Elapsed time     : ${SECONDS} seconds"
echo "============================================================"

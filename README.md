# README — `l4_train_seq.py` (L4 TCN+CharCNN Sequence Classifier)

Tài liệu này rút thẳng từ code thật (`argparse` help text + logic training loop), không suy đoán. Nếu code đổi, README phải update theo — đừng tin README hơn code.
Link dataset : https://drive.google.com/drive/folders/1zfdOSe2ErAox1JSwvVr4IYX09U8eUTV9?usp=sharing
---

## 1. Nguyên tắc chạy tối thiểu cho kết quả DÙNG ĐƯỢC (không phải smoke-test)

Command tối thiểu để có 1 lần train **có thể báo cáo trong paper** (không phải chỉ debug):

```bash
python3 l4_train_seq.py \
    --stories stories_trace_e3_v2.jsonl \
    --labels  labels_trace_e3.csv \
    --eval_stories stories_theia_e3_v2.jsonl \
    --eval_labels  labels_theia_e3.csv \
    --out_dir l4_seq_model_losoA \
    --epochs 30 \
    --batch_size 4 \
    --max_seq_len 8192 \
    --hnm_epochs 10 \
    --device cuda
```

> ⚠️ **`--eval_stories`/`--eval_labels` KHÔNG bắt buộc theo code (`default=[]`), nhưng KHÔNG truyền = script tự in cảnh báo cuối chạy:**
> `[CANH BAO PHUONG PHAP] KHONG co --eval_stories/--eval_labels - toan bo con so loss/accuracy phia tren CHI la tren TAP TRAIN ... KHONG dung so nay lam ket qua danh gia trong paper.`
> Thiếu flag này thì log `train_loss` mỗi epoch **không nói lên được gì về generalization** — chỉ biết training có chạy/không nổ, không biết model có học đúng hay đang overfit.

> ⚠️ **Dù CÓ `--eval_stories`, `eval_acc` cũng KHÔNG phải số dùng cho paper** (script tự cảnh báo): accuracy thô trên class cực mất cân bằng (~1-2% malicious) bị lừa dễ dàng (dự đoán toàn 0 vẫn ra ~98% accuracy). `eval_loss`/`eval_acc` **chỉ để sanity-check model có đang học gì không qua từng epoch**. Số thật để báo cáo = chạy `.onnx` vừa xuất qua `avro_main.c` → đo bằng `19_uuid.py`/`20_uuid.py`, đúng phương pháp đã dùng cho THEIA E3/TRACE E3.

---

## 2. Giải thích từng tham số (đúng theo code, không thêm bớt)

### Bắt buộc (`required=True`)

| Tham số | Kiểu | Công dụng |
|---|---|---|
| `--stories` | append, lặp lại được | Story JSONL dùng để **train**. Lặp flag nhiều lần để gộp nhiều dataset vào 1 tập train (VD LOSO: `--stories theia.jsonl --stories trace.jsonl`). **Thứ tự phải khớp 1-1 với `--labels`** (file thứ i ghép với labels thứ i). |
| `--labels` | append, lặp lại được | Labels CSV (output từ `15.py`) dùng để train. Số lượng + thứ tự phải khớp `--stories`. |
| `--out_dir` | str | Thư mục ghi output: checkpoint `.pt`, vocab files, `.onnx`, norm file. |

### Có mặc định — tùy chỉnh để có kết quả tốt

| Tham số | Default | Công dụng |
|---|---|---|
| `--eval_stories` | `[]` (rỗng) | Story JSONL held-out dùng để **đánh giá** (KHÔNG đưa vào train). Lặp được nhưng thường chỉ 1 dataset còn lại. Rỗng = không có `eval_loss`/`eval_acc` in ra, chỉ có `train_loss`. Muốn LOSO cross-host thật: trỏ vào dataset **hoàn toàn khác** mọi file trong `--stories` (VD train THEIA E3+TRACE E3, eval THEIA E5). |
| `--eval_labels` | `[]` | Labels CSV cho `--eval_stories`, số lượng + thứ tự phải khớp 1-1. |
| `--epochs` | `30` | Số epoch train vòng chính (uniform sampling toàn dataset). |
| `--batch_size` | `4` | Batch size DataLoader. **Nhỏ = gradient noisy hơn**, đặc biệt nguy hiểm kết hợp `pos_weight` lớn (dữ liệu càng mất cân bằng, `pos_weight` càng lớn) → dễ thấy loss dao động mạnh vài epoch đầu (đã quan sát thực tế: epoch 1→2 loss tăng rồi mới hội tụ). Tăng batch_size giúp ổn định hơn nhưng tốn RAM/VRAM hơn — cần cân nhắc với giới hạn RAM Colab (~12GB) và VRAM T4 (~14.6GB). |
| `--lr` | `1e-3` | Learning rate cho Adam optimizer. Không có gradient clipping trong code — nếu thấy loss nổ (NaN/inf) chứ không chỉ dao động, đây là chỗ đầu tiên cần hạ (không tự ý sửa code thêm `clip_grad_norm_` khi chưa có xác nhận architect). |
| `--max_seq_len` | `2048` | Số event tối đa mỗi sample (dài hơn bị cắt ở cuối, mất label của các event bị cắt — script tự log rõ số event/label bị mất). **PHẢI khớp `L4_SEQ_LIVE_MAX_EVENTS` bên C** khi export `.onnx` để dùng với `avro_main.c`/`l4_infer.c`, đổi giá trị này bắt buộc recompile C tương ứng. |
| `--device` | `"auto"` | `"auto"` = dùng GPU nếu có, tự fallback CPU kèm cảnh báo. `"cuda"` = ép GPU, báo lỗi thoát ngay nếu không có (tránh âm thầm chạy CPU chậm mà không biết). `"cpu"` = ép CPU dù có GPU (debug/so sánh). Dataset luôn nằm trong CPU RAM bất kể giá trị này — chỉ từng batch được chuyển lên VRAM lúc train/eval, tránh OOM RAM 12GB. |

### Hard Negative Mining (OHEM) — chỉ có tác dụng nếu `--hnm_epochs > 0`

| Tham số | Default | Công dụng |
|---|---|---|
| `--hnm_epochs` | `0` (tắt) | Số epoch train THÊM sau vòng chính, tập trung vào benign "khó" (model tự chấm lại tập train, benign nào bị predict xác suất cao nhất = dễ nhầm malicious nhất). Đây là OHEM **in-sample** — mine từ chính tập train hiện có, KHÔNG sinh thêm benign mới từ C pipeline. Muốn pool benign lớn hơn để mine phải tăng `EDR_STORY_AUTOSEED_TOPK_C` bên `avro_main.c` + chạy lại `15.py` trước — **quyết định kiến trúc, không tự ý làm trong phạm vi script Python này**. |
| `--hnm_topk_frac` | `0.3` | Tỷ lệ benign "khó nhất" được oversample trong các epoch HNM (0.3 = 30% benign điểm predict cao nhất). |
| `--hnm_oversample_weight` | `5.0` | Hệ số tăng trọng số lấy mẫu (`WeightedRandomSampler`) cho hard negative so với sample thường (weight=1.0). Sample malicious KHÔNG bị loại, chỉ ít được lặp lại hơn hard negative — tránh catastrophic forgetting. |

---

## 3. Output ghi ra `--out_dir`

| File | Nội dung |
|---|---|
| `l4_seq_model_checkpoint.pt` | State dict model, **ghi đè sau MỖI epoch** (không phải 1 file/epoch) — nếu Colab bị ngắt giữa chừng, file này + vocab/norm là đủ để không mất tiến độ. |
| `l4_seq_cat_<name>.vocab` | Vocab cho từng categorical field (tag, syscall, errno, behavior). |
| `l4_seq_char.vocab` | Vocab ký tự cho CharCNN. |
| `l4_seq_numeric_norm.txt` | Norm stats cho 15 trường numeric. |
| `l4_seq_model.onnx` | Model cuối cùng, export sau khi xong cả `--epochs` + `--hnm_epochs` (nếu bật). Input 9 tensor đúng thứ tự `cat_tag/cat_syscall/cat_errno/cat_behavior/cat_is_seed/numeric/char_primary/char_secondary/mask`, output `prob_malicious_per_event` shape `[1, T]` — **1 xác suất/event, đã qua sigmoid sẵn** (không cần áp sigmoid lại phía C). |

---

## 4. Checklist trước khi tin số liệu là "kết quả tốt nhất"

1. ☐ Có `--eval_stories`/`--eval_labels` trỏ đúng dataset held-out (không trùng file nào trong `--stories`)?
2. ☐ Nếu mục tiêu là LOSO cross-host thật (Fold A: train THEIA E3+TRACE E3, test THEIA E5) — **THEIA E5 không nên đưa vào `--eval_stories` lúc train** (rủi ro OOM, không cần thiết vì metric chính thức đo qua `avro_main.c` sau khi export `.onnx`, không phải qua script này).
3. ☐ `--max_seq_len` đã khớp với `L4_SEQ_LIVE_MAX_EVENTS` bên C chưa, và đã cân nhắc số event/label bị cắt mất (script tự log số liệu này — nếu số malicious bị cắt lớn, cần architect quyết định tăng `max_seq_len` hay đổi cách cắt)?
4. ☐ `eval_loss`/`eval_acc` (nếu có) chỉ dùng để sanity-check, KHÔNG dùng làm Precision/Recall báo cáo trong paper.
5. ☐ Số liệu paper = chạy `l4_seq_model.onnx` qua `avro_main.c` (production C pipeline) → đo bằng `19_uuid.py`/`20_uuid.py` theo đúng phương pháp đã dùng cho THEIA E3/TRACE E3.
6. ☐ Kiểm tra dòng cảnh báo cuối log: nếu thấy `[CANH BAO PHUONG PHAP]` nghĩa là lần chạy đó **không đủ điều kiện báo cáo paper**, phải chạy lại có `--eval_stories`.

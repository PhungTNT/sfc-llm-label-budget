# UIT-VSFC: LLM Zero-shot, PhoBERT và ngân sách nhãn

Repository lưu trữ **mã nguồn notebook, dữ liệu đầu ra và kết quả thực nghiệm** cho nghiên cứu so sánh mô hình ngôn ngữ lớn (LLM) theo hướng zero-shot với PhoBERT và các phương pháp TF-IDF trên **UIT-VSFC**. Trọng tâm là phân loại cảm xúc (*sentiment*), số lượng nhãn thủ công cần thiết và khả năng huấn luyện mô hình nhỏ bằng nhãn giả do LLM tạo ra.

> **Trạng thái:** Kết quả thực nghiệm phục vụ báo cáo tiến độ/nghiên cứu. Chưa phải bộ mã tái lập tự động hoàn chỉnh hoặc bài báo đã công bố. Một số mô hình mở rộng trong kế hoạch vẫn chưa được chạy.

## 1. Dữ liệu và thiết lập

- **Bộ dữ liệu:** UIT-VSFC (phản hồi sinh viên tiếng Việt).
- **Chia tập:** train **11.426**; validation **1.583**; test **3.166**.
- **Tác vụ:** sentiment (3 lớp) và topic (4 lớp); các phân tích learning curve và pseudo-labeling dưới đây tập trung vào **sentiment**.
- **Chỉ số chính:** Macro-F1 trên tập test; báo cáo thêm Accuracy, độ lệch chuẩn giữa các seed và kiểm định khi có dữ liệu tương ứng.
- **Môi trường thực nghiệm:** notebook Jupyter chạy trên Kaggle/Colab; một số lượt chạy dùng GPU T4. Các file ZIP lưu chỉ số và dự đoán, không mặc định bao gồm trọng số mô hình.

## 2. Nội dung repository

| File | Nội dung |
| --- | --- |
| `th-c-nghi-m.ipynb` | Notebook thực nghiệm đã upload: mã lệnh, các cell thực thi và các output/log còn được lưu trong notebook. Đây là nhật ký thực nghiệm, **không đảm bảo Run All một lần sẽ tái lập toàn bộ**. |
| `archive.zip` | Backup thí nghiệm gốc, gồm dữ liệu và kết quả TF-IDF, Qwen2.5-7B zero-shot, PhoBERT baseline và các kết quả phụ trợ. |
| `UIT_VSFC_FINAL_RESEARCH_RESULTS.zip` | **67 file**: E3 learning curve (21 lượt), E5 PhoBERT học nhãn giả (6 lượt), bảng tổng hợp, thống kê và biểu đồ. |
| `E5_train_11426_dev_1583.zip` | Nhãn giả Qwen P2 calibrated cho toàn bộ train và validation (**13.009 câu**), kèm dữ liệu phụ trợ nếu có trong ZIP. |

**Cách xem kết quả:** tải và giải nén hai ZIP kết quả. Trong `UIT_VSFC_FINAL_RESEARCH_RESULTS.zip`, xem:

```text
E3_learning_curve/            # Metrics và dự đoán test của 21 lượt
E5_pseudo_student/            # Metrics và dự đoán test của 6 lượt
  E5_all_results.csv
  E5_summary_final.csv
  E5_bootstrap_comparison.csv
  E5_mcnemar_comparison.csv
final_analysis/
  table1_main_sentiment.csv
  learning_curve_final_summary.csv
  learning_curve_final.png
  learning_curve_final.pdf
```

Các file `metrics.csv` ghi chỉ số; `pred.csv` ghi dự đoán theo từng mẫu test. **Log huấn luyện chi tiết** chỉ có thể xem ở những cell output còn được lưu trong notebook; các ZIP metrics không thay thế cho log đầy đủ từng bước.

## 3. Các thí nghiệm đã thực hiện

| Nhóm | Nội dung | Kết quả lưu trữ |
| --- | --- | --- |
| E1 – Baseline | TF-IDF (SVM/LR) và PhoBERT supervised | `archive.zip` |
| E2 – Zero-shot | Qwen2.5-7B, prompt P1/P2/P3, bản raw và calibrated; sentiment và topic | `archive.zip` |
| E3 – Learning curve | PhoBERT sentiment ở 7 tỷ lệ nhãn (1%, 2%, 5%, 10%, 25%, 50%, 100%), 3 seed mỗi tỷ lệ | `UIT_VSFC_FINAL_RESEARCH_RESULTS.zip` |
| E5 – Pseudo-labeling | Qwen tạo nhãn giả cho train/dev; PhoBERT student với `all` và `conf80`, mỗi phương án 3 seed | Hai ZIP E5/kết quả tổng hợp |
| Phân tích | Bootstrap ghép cặp và McNemar cho các so sánh E5 | `E5_pseudo_student/` |

Các mô hình mở rộng như **Qwen3, Llama, SeaLLM** chưa được báo cáo là đã hoàn thành trong repository này.

## 4. Kết quả chính – Sentiment

| Phương pháp | Macro-F1 trên test |
| --- | ---: |
| TF-IDF + SVM | 0,74762 |
| PhoBERT supervised, 5 seed | **0,83484 ± 0,00500** |
| Qwen2.5-7B P2 raw, zero-shot | 0,68315 |
| Qwen2.5-7B P2 calibrated, zero-shot | 0,67234 |
| PhoBERT student, pseudo `all`, 3 seed | 0,67862 ± 0,01158 |
| PhoBERT student, pseudo `conf80`, 3 seed | **0,70925 ± 0,00455** |

### Learning curve: bao nhiêu nhãn là đủ?

- Ở mốc **329 nhãn** (train + validation), PhoBERT đạt Macro-F1 trung bình **0,67152**, dưới Qwen P2 raw (**0,68315**).
- Ở mốc **671 nhãn**, PhoBERT đạt **0,75220**: đây là **mốc được quan sát đầu tiên** trong các tỷ lệ đã thử vượt Qwen P2 raw.
- Nội suy theo trục log giữa hai mốc gần nhất cho ước lượng khoảng **365 nhãn**; **đây không phải** một lượt huấn luyện đã chạy tại đúng 365 nhãn.

### Học từ nhãn giả

- Lọc giữ các mẫu có độ tin cậy cao (`conf80`) đạt Macro-F1 **0,70925**, cao hơn Qwen P2 raw khoảng **0,02610**.
- Bootstrap ghép cặp với 2.000 lần lấy mẫu cho chênh lệch `conf80` − Qwen raw: **95% CI [0,00819; 0,04385]**.
- McNemar được áp dụng trên dự đoán đúng/sai (liên quan Accuracy), **không phải** kiểm định trực tiếp Macro-F1.
- Kết luận chỉ áp dụng cho thiết lập và tập test được báo cáo; cần thận trọng về nhiều phép so sánh, biến thiên seed và khả năng tổng quát hóa.

## 5. Lưu ý về khả năng tái lập

1. Dùng đúng cách chia train/validation/test và thứ tự mẫu; **không dùng nhãn test để huấn luyện hoặc chọn mô hình**.
2. Phân biệt **PhoBERT baseline supervised 5 seed** với **E3 learning curve 100% (3 seed)** vì quy trình thí nghiệm khác nhau.
3. Các kết quả `conf80` được tính từ nhãn giả và độ tin cậy Qwen, không phải từ nhãn sentiment thật trên train/dev.
4. Các file kết quả đã lưu là bằng chứng đầu ra; notebook có thể chứa cell thử nghiệm, cell sửa lỗi và phụ thuộc môi trường. Để tái lập hoàn toàn cần chuẩn hóa quy trình chạy và cài đúng phiên bản thư viện.
5. Nếu repository được công khai, cần rà soát quyền phân phối dữ liệu gốc và nội dung ZIP trước khi chia sẻ rộng rãi.

## 6. Phạm vi và trạng thái

Đây là kho lưu trữ phục vụ **giảng viên kiểm tra code, kết quả và tiến độ nghiên cứu**. Kết quả chưa bao phủ toàn bộ các LLM dự kiến trong kế hoạch, và báo cáo cuối vẫn cần chuẩn hóa tài liệu tham khảo, môi trường tái lập và hình thức trình bày.

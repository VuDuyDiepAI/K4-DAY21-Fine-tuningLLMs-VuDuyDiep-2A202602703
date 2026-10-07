# Lab 21 — Fine-tuning thắng tác vụ nhưng chưa vượt cổng hồi quy

**Họ tên:** Vũ Duy Điệp  
**MSSV:** 2A202602703  
**Ngày thực hiện:** 07/10/2026  
**Môi trường:** Google Colab, Tesla T4 (14,6 GB), fp16  
**Base model:** unsloth/Qwen3.5-4B

## 1. Câu hỏi và thiết kế thí nghiệm

Thí nghiệm kiểm tra liệu LoRA có cải thiện phân loại ticket chăm sóc khách hàng tiếng Việt so với cùng base model đã được prompt tối ưu, đồng thời giữ năng lực phổ thông hay không. Tôi sử dụng model và corpus mặc định để có phép so sánh theo đúng khung của lab. Model 4B phù hợp tier T4; corpus có nhãn khách quan cho bốn trường intent, urgency, product và sentiment, nên không cần LLM judge.

Corpus huấn luyện có 250 mẫu, chia 225 train / 25 validation với seed 42 theo mã notebook. Tập đánh giá độc lập gồm 50 ticket target và 15 câu hỏi regression. NB2 đóng băng baseline trước huấn luyện; không đổi prompt tối ưu hoặc tập eval để làm fine-tune trông tốt hơn. baselines_frozen.json ghi smoke_mode=false, eval_limit=null và optimized_prompt_sha=719e74d3b6232053. Output verify trên Colab xác nhận checksum dữ liệu không đổi và baseline (b) tốt hơn (a).

| Thiết lập | Giá trị |
|---|---|
| Tier | T4 |
| MASK_MODE | assistant-only |
| Epochs | 2 theo cấu hình mặc định; ngân sách thực tế 30 optimizer steps |
| Batch / gradient accumulation | 1 / 16; batch hiệu dụng 16 |
| max_length thực tế theo tier trong notebook | 1024 |
| Độ dài đo ở NB1 | mean 93,1; p50 93; p95 98; p99 100; max 101 |
| suggested_max_length | 256 |

Độ dài thực tế 1024 là cấu hình có sẵn của tier T4 trong bản notebook đã chạy, không phải kết quả suy ra từ p95. Số đo cho thấy 256 đã đủ cho corpus này; giữ 1024 không cắt mất mẫu vì max chỉ 101, nhưng là trần dư thừa. Tôi khai báo độ lệch này thay vì ghi rằng đã train với 256. Lượt tối ưu tiếp theo nên đặt trần 256 sau khi kiểm tra lại chuỗi render, không cần thay đổi số liệu của lượt đã chạy.

## 2. Bằng chứng mask và chat template

Nguồn: results/mask_proof.json và results/template_check.json.

| Kiểm tra | Kết quả |
|---|---|
| Token supervised / tổng token trong mẫu proof | 39 / 94 |
| supervised_fraction | 0,4149 |
| answer_is_supervised | true |
| question_is_masked | true |
| Template giữ thẻ mở và nội dung reasoning | true / true |
| Phán quyết template | reasoning preserved — safe to train on traces |

Đoạn được tính loss trong mẫu proof:

~~~text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
~~~

Prompt system và câu hỏi user nằm trong masked_preview; chúng không nằm trong phần supervised. Tỷ lệ 41,49% là của mẫu proof, không phải tỷ lệ toàn corpus. Mã NB3 truyền input_ids và labels đã được dựng bằng mask này cho trainer. Đây là bằng chứng cụ thể hơn việc chỉ tin vào cờ assistant_only_loss của thư viện.

Template có thể giữ reasoning trong mẫu kiểm tra, nhưng corpus JSON mặc định không cung cấp reasoning trace. valid_trace_rate=0,0 trong verdict vì lượt sinh dùng enable_thinking=False và không huấn luyện trên trace. Con số này không đủ để kết luận reasoning-trace collapse, và tôi không nhận điểm thưởng B3.

## 3. Ba baseline và bốn nhóm đánh giá

Nguồn: results/verdict.json; baseline ban đầu đối chiếu với results/baselines_frozen.json.

| Run | target | regression | format | latency ms/mẫu |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0,0000 | 0,7911 | 0,0000 | 3176,7 |
| (b) base + optimized prompt | 0,7650 | 0,7911 | 1,0000 | 1019,0 |
| (c) LoRA fine-tune | 0,9700 | 0,6111 | 1,0000 | 1357,7 |

Target là trung bình độ chính xác của bốn trường, không phải tỷ lệ ticket đúng toàn bộ. Regression là trung bình keyword recall trên 15 câu hỏi, không phải accuracy trắc nghiệm. Format đo các khóa bắt buộc có trong object JSON được parser chấp nhận; không chứng minh toàn bộ giá trị đều đúng. Latency dùng greedy decoding theo hàm generate_batch của lab.

Prompt tối ưu nâng target từ 0 lên 0,765 và format lên 1,0, đồng thời giảm latency khoảng 67,9% so với prompt đơn giản. Đây là một baseline mạnh và không bị chỉnh yếu đi. Fine-tune được đánh giá bằng prompt ngắn giống dạng prompt huấn luyện; baseline (b) sử dụng prompt có schema và ví dụ theo thiết kế của lab.

## 4. Đối chứng: vị trí adapter, learning rate và lượng tử hóa

Nguồn: results/runs.csv và results/autopsy.json. Tất cả các run có 30 steps. alpha/r=2 ở các cấu hình; số tham số của attn_only lệch correct khoảng 0,0252%, thấp hơn ngưỡng 5%.

| Run | Vị trí | r / alpha | LR | Trainable | Loss ghi trong CSV | target | format | Train s | Peak VRAM GB |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| correct (dòng cuối) | text-linear, 12 suffix | 16 / 32 | 1e-4 | 32464896 | 0,6260 | 0,9700 | 1,0000 | 390,7 | 8,78 |
| attn_only | q,v, 2 suffix | 283 / 566 | 1e-4 | 32456704 | 0,5366 | 0,9700 | 1,0000 | 262,5 | 8,79 |
| wrong_lr | text-linear | 16 / 32 | 1e-5 | 32464896 | 1,5702 | 0,0000 | 0,0000 | 391,7 | 8,78 |
| qlora | text-linear, base 4-bit | 16 / 32 | 1e-4 | 32464896 | 0,7058 | 0,9400 | 1,0000 | 455,2 | 3,86 |

**Lịch sử correct:** CSV giữ hai lần train: dòng đầu loss 0,6258, thời gian 397,1 s; dòng cuối loss 0,6260, thời gian 390,7 s. Mã save_pretrained lưu vào cùng adapters/correct trước khi append CSV, nên bảng sử dụng dòng cuối theo cơ chế ghi đè của notebook. ZIP không lưu run ID hoặc hash liên kết adapter với lượt eval; vì vậy đây là quy ước dựa vào mã và thứ tự CSV, không phải chứng minh độc lập về lịch sử chạy. Tôi giữ nguyên cả hai dòng CSV để không xóa lịch sử. Hai lượt có cùng tham số và số step.

**Lưu ý tên loss:** Cột có tên final_loss, nhưng mã notebook ghi result.training_loss từ trainer. Đây là giá trị loss huấn luyện do trainer tổng hợp; không có log từng step trong ZIP để dựng đường loss hoặc khẳng định loss của step cuối.

### 4.1. Vị trí so với rank

Xếp hạng theo target là correct = attn_only (0,97), tiếp theo qlora (0,94), cuối cùng wrong_lr (0,00). attn_only có loss thấp hơn correct nhưng target chỉ hòa, nên loss tốt hơn không đồng nghĩa với chất lượng tác vụ tốt hơn. Rank 283 được tăng để khớp ngân sách khi chỉ gắn q,v; thí nghiệm này không phải rank sweep có kiểm soát. Kết quả chưa cho phép nói text-linear thắng attention-only trên target ở corpus này. attn_only còn có latency target 873,7 ms, thấp hơn correct 1357,7 ms; tuy nhiên không có regression sweep của attn_only để kết luận nó an toàn hơn.

### 4.2. Learning rate

wrong_lr chỉ đổi LR từ 1e-4 xuống 1e-5 trong thiết kế đối chứng. Loss tổng hợp tăng lên 1,5702, target và format đều về 0; latency tăng lên 5130,9 ms. Không có log từng step nên tôi không mô tả một đường loss chưa được lưu. Nếu chỉ nhìn loss, có thể nhận ra cấu hình này học kém hơn nhưng không thấy mức thất bại đầu ra JSON. Trong những biến đã đo, LR tạo ra chênh lệch target lớn nhất: giảm 97 điểm phần trăm so với correct. Đây là kết luận trong ngân sách 30 steps này, không phải kết luận mọi LR 1e-5 đều vô dụng.

### 4.3. QLoRA

QLoRA giảm peak VRAM từ 8,78 xuống 3,86 GB, tiết kiệm 4,92 GB, tương đương khoảng 56,0%. Đổi lại target giảm 3 điểm phần trăm, thời gian train tăng từ 390,7 lên 455,2 s (khoảng 16,5%), latency target tăng lên 1740,6 ms (khoảng 28,2%). Format vẫn đạt 1,0. Số đo phù hợp với một đánh đổi bộ nhớ/chất lượng, nhưng một corpus và một lượt chạy không đủ để khẳng định khuyến nghị của nhà cung cấp đúng cho mọi bài toán. Khi LoRA 16-bit đã vừa T4, correct là lựa chọn hợp lý hơn về target; khi thiếu VRAM, QLoRA vẫn là phương án cần cân nhắc. Regression của QLoRA chưa được đo trong autopsy.

## 5. Phán quyết và giới hạn triển khai

**Verdict: FAILED. target delta=+0,205; regression delta=-0,180; tolerance=0,020.**

Fine-tune thực sự tốt hơn prompt tối ưu trên target: điểm tăng từ 0,765 lên 0,970, tương đương tăng 20,5 điểm phần trăm. Tuy nhiên lợi ích này đi kèm suy giảm regression từ 0,7911 xuống 0,6111. Mức giảm 18 điểm phần trăm lớn gấp chín lần ngưỡng 2 điểm cho phép. Vì cổng yêu cầu vừa tăng target vừa giữ năng lực phổ thông, kết luận FAILED đúng theo số đo, dù output JSON đã hoàn chỉnh. Kết quả phù hợp với hiện tượng quên năng lực ngoài miền sau huấn luyện chuyên biệt, nhưng chỉ số keyword recall không tự xác định cơ chế gây ra mọi lỗi. Tôi chưa có dự đoán regression từng câu trong ZIP để kiểm tra sâu nguyên nhân. Tôi không nới ngưỡng hoặc thay eval để đổi verdict. Hướng thử nghiệm tiếp theo là thêm 1–5% replay dữ liệu phổ thông theo gợi ý trong verdict, giữ tập đánh giá và prompt baseline đã đóng băng, rồi đo lại bốn nhóm trước khi quyết định triển khai. Hiện tại chưa nên thay base model bằng adapter này cho một trợ lý đa nhiệm.

## 6. Định tính: ca đúng và ca sai thực tế

Nguồn dự đoán: results/qualitative.json. Nhãn và ticket đầy đủ được đối chiếu với corpus eval_target.jsonl mặc định trong repo. File do NB5 xuất cắt ticket ở 70 ký tự và dự đoán ở 90 ký tự; bản xem trước không phải JSON đầy đủ. Format=1,0 được chấm trên output đầy đủ trước khi cắt, nên không coi đoạn preview cụt là lỗi parse.

| Index eval (từ 0) | Ticket rút gọn | Nhãn urgency | FT urgency nhìn thấy trong preview | FT score | Phân tích |
|---|---|---|---|---:|---|
| 0 | Chuột không dây; cho tôi trả lại; gấp; shop hỗ trợ tốt | cao | cao | 1,00 | Đúng 4/4 trường theo scorer; ví dụ đúng |
| 27 | Bàn phím cơ; cho tôi trả lại; ngay lập tức | cao | cao | 1,00 | Đúng 4/4; preview lưu đủ object của mẫu này |
| 3 | Bình giữ nhiệt; chưa thấy tiền; khi nào tiện | thap | trung_binh | 0,75 | Sai urgency; nhận đúng intent hoan_tien |
| 5 | Nồi chiên không dầu; thiếu phụ kiện; khi nào tiện | thap | trung_binh | 0,75 | Sai urgency; nhận đúng intent san_pham_loi |
| 12 | Áo khoác gió; bị lỗi; khi nào tiện | thap | trung_binh | 0,75 | Sai urgency dù ticket chỉ rõ mức không gấp |
| 39 | Nồi chiên không dầu; hoàn tiền; khi nào tiện | thap | trung_binh | 0,75 | Sai urgency trên một intent khác |
| 41 | Đèn bàn LED; giao hàng chậm; khi nào tiện | thap | trung_binh | 0,75 | Lặp lại nhầm mức thấp thành trung bình |
| 46 | Đèn bàn LED; sai màu; khi nào tiện | thap | trung_binh | 0,75 | Cùng lỗi urgency |

Cả sáu ca sai đều có nhãn urgency=thap nhưng dự đoán trung_binh, cùng chứa cụm “Khi nào tiện”. Đây là mẫu lỗi cụ thể đáng kiểm tra trong dữ liệu train; chưa đủ để kết luận do thiếu mẫu hoặc do model bỏ qua cụm đó nếu chưa đếm phân bố và xem output đầy đủ. 44 mẫu còn lại có score 1,0; 6 mẫu score 0,75, tạo trung bình (44+6×0,75)/50=0,97, khớp verdict.

**Giới hạn bằng chứng:** ZIP không có output của baseline (b) trên từng ticket. Các ca sai ở trên là FT thua nhãn chuẩn, chưa chứng minh FT thua prompt (b) trên chính các ca đó. Tôi không gán nhãn “FT thắng/thua (b)” khi chưa có dự đoán đối chiếu. Mục rubric yêu cầu ít nhất hai ca thua so với baseline cần bổ sung bằng file qualitative_paired.json từ ô Colab đi kèm. Phần này khai báo thiếu bằng chứng thay vì dựng câu trả lời baseline.

## 7. Kết luận và điều rút ra

Kết quả trả lời hai câu hỏi chính của lab bằng bằng chứng riêng biệt. Mask proof cho thấy câu trả lời được tính loss và câu hỏi được loại khỏi loss, nên quá trình huấn luyện có cơ sở đúng ở mức token. Đánh giá sau huấn luyện cho thấy model học được tác vụ JSON: target đạt 0,97 và format đạt 1,0. Tuy vậy, chỉ nhìn hai chỉ số này sẽ dẫn tới quyết định triển khai quá sớm. Regression giảm 18 điểm phần trăm và latency tăng khoảng 33,2% so với prompt tối ưu. Đối với một trợ lý cần cả triage và trả lời phổ thông, adapter hiện tại chưa đáp ứng yêu cầu giữ năng lực. Kết luận này không phủ nhận lợi ích chuyên biệt của fine-tuning; nó đặt lợi ích ấy cạnh chi phí đã đo.

Đối chứng cũng giúp tránh lấy giả thuyết từ slide làm kết quả của mình. Attention-only có loss thấp hơn nhưng chỉ hòa correct về target, nên không thể nói vị trí text-linear thắng trong lượt này. LR quá thấp mới là biến gây tác động lớn nhất trên target ở ngân sách hiện tại. QLoRA tiết kiệm đáng kể VRAM nhưng có chi phí về chất lượng và tốc độ. Tôi ưu tiên sửa mẫu lỗi urgency, bổ sung replay dữ liệu phổ thông và thu thập dự đoán từng câu trước khi thử tăng rank. Nếu có thêm hai giờ, tôi sẽ thiết kế một lượt replay 1–5%, giữ cùng eval, kiểm tra lại mask cho dữ liệu hỗn hợp, rồi so sánh target, regression, format và latency. Chỉ khi vượt cổng hồi quy và đã xem các ca thất bại, mới cân nhắc dùng adapter trong hệ thống thực tế.

Ba điều rút ra từ số đo:

1. Loss thấp hơn không chứng minh target cao hơn: attn_only 0,5366 nhưng target vẫn hòa correct.
2. Target tăng không đảm bảo model đa nhiệm tốt hơn: +20,5 điểm target đi kèm -18 điểm regression.
3. Prompt tối ưu là baseline bắt buộc: nó đạt 0,765, format 1,0 và latency 1019 ms trước fine-tuning.

AI assistant hỗ trợ đọc hướng dẫn, giải thích cấu hình Colab, đối chiếu artefact và soạn báo cáo. Các kết luận định lượng dựa trên file kết quả, không dựa vào số mẫu trong tài liệu. Trải nghiệm cá nhân như chỗ mất thời gian nhất hoặc điều bất ngờ nhất chưa được người thực hiện cung cấp, nên không tự thêm vào báo cáo.

## 8. Tình trạng kiểm tra và điểm thưởng

Output verify được cung cấp sau lượt Colab: 119 tests passed; 25 checks passed, 1 warning và 1 failure do report còn là template. Report này đã được điền; các file results gốc và verdict FAILED được giữ nguyên. Chưa có một lượt chạy lại đầy đủ scripts/verify.py sau khi thay report trong môi trường Colab, nên không tuyên bố đã có kết quả verify mới. Không thực hiện NB6, dataset riêng, reasoning-trace experiment, rank sweep hoặc push Hub; không nhận điểm thưởng chưa có artefact.

Báo cáo đã có kết quả chính và ví dụ lỗi, nhưng phần so sánh định tính theo từng mẫu vẫn cần bổ sung để đáp ứng đủ rubric. Chạy notebook bổ sung đi kèm không train lại và không sửa baseline đã đóng băng.

## 9. Bổ sung đối chiếu dự đoán thật giữa baseline (b) và fine-tune

Phần này bổ sung bằng chứng từng mẫu bằng cách sinh lại output; không sửa các số liệu chính đã đóng băng. Full output nằm trong results/qualitative_paired.json. Các ca regression dùng keyword recall, các ca target dùng độ chính xác bốn trường; không cộng hai thang điểm này thành một metric.

| Nhóm | FT thắng | FT thua | Hòa |
|---|---:|---:|---:|
| target | 33 | 0 | 17 |
| regression | 1 | 5 | 9 |

### Ví dụ 1: FT thắng (b) — target index 6

Xin chào, mình đặt balo laptop mã đơn DH863123. Đổi size. Hỏi cho biết thôi. Lần cuối mua ở đây.

Nhãn chuẩn / keyword: {"intent": "doi_tra", "urgency": "thap", "product": "balo laptop", "sentiment": "tieu_cuc"}

Baseline (b):
~~~text
{"intent": "hoan_tien", "urgency": "cao", "product": "balo laptop", "sentiment": "tieu_cuc"}
~~~

Fine-tune: 
~~~text
{"intent": "doi_tra", "urgency": "thap", "product": "balo laptop", "sentiment": "tieu_cuc"}
~~~

Điểm (b)=0.5000; FT=1.0000; delta=+0.5000.

### Ví dụ 2: FT thắng (b) — target index 7

Alo shop, mình đặt máy xay sinh tố mã đơn OD126693. Muốn đổi. Đã 3 ngày rồi. Bực mình.

Nhãn chuẩn / keyword: {"intent": "doi_tra", "urgency": "trung_binh", "product": "máy xay sinh tố", "sentiment": "tieu_cuc"}

Baseline (b):
~~~text
{"intent": "van_chuyen", "urgency": "cao", "product": "máy xay sinh tố", "sentiment": "tieu_cuc"}
~~~

Fine-tune: 
~~~text
{"intent": "doi_tra", "urgency": "trung_binh", "product": "máy xay sinh tố", "sentiment": "tieu_cuc"}
~~~

Điểm (b)=0.5000; FT=1.0000; delta=+0.5000.

### Ví dụ 3: FT thua (b) — regression index 2

1 km bằng bao nhiêu mét?

Nhãn chuẩn / keyword: ["1000"]

Baseline (b):
~~~text
Trong hệ đo lường quốc tế, tiền tố "kilo" (ký hiệu là **km**) có nghĩa là **1000**. Do đó, 1 kilômét tương đương với 1000 mét.

Kết quả: **1 km = 1000 m**.
~~~

Fine-tune: 
~~~text
{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh"}
~~~

Điểm (b)=1.0000; FT=0.0000; delta=-1.0000.

### Ví dụ 4: FT thua (b) — regression index 3

Viết một câu chúc mừng sinh nhật bằng tiếng Việt.

Nhãn chuẩn / keyword: ["sinh nhật"]

Baseline (b):
~~~text
Chúc bạn một ngày sinh nhật thật vui vẻ, tràn đầy niềm vui và sức khỏe, cùng với những điều may mắn và hạnh phúc nhất trong năm mới!
~~~

Fine-tune: 
~~~text
{"intent": "chuc_mung_sinh_nhat", "urgency": "trung_tinh", "tone": "trung_thinh", "product": null, "sentiment": "tich_cuc"}
~~~

Điểm (b)=1.0000; FT=0.0000; delta=-1.0000.

### Ví dụ 5: FT thắng (b) — target index 0

Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt.

Nhãn chuẩn / keyword: {"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}

Baseline (b):
~~~text
{"intent": "hoan_tien", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}
~~~

Fine-tune: 
~~~text
{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}
~~~

Điểm (b)=0.7500; FT=1.0000; delta=+0.2500.

Đã ghi ít nhất hai ca FT thắng và hai ca FT thua so với baseline bằng output thực tế. Nếu ca thua thuộc regression, chúng là bằng chứng suy giảm năng lực phổ thông, không phải ticket target bị phân loại kém hơn.

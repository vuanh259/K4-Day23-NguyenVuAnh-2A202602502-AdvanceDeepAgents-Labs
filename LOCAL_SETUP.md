# Chạy lab bằng API hoặc Ollama và Docker

## Trạng thái lần chạy

Đã lưu đủ năm bộ báo cáo; `self_check.py` đánh dấu tất cả `OK`, bao gồm kiểm
tra trích dẫn và Git/bí mật. Mỗi bộ có báo cáo, nguồn JSON và metadata thực tế.
Mỗi báo cáo có ít nhất ba họ nguồn và ba khẳng định được citation-checker kiểm tra.

- [World model](reports/survey-about-world-model.md): 8 nguồn.
- [RL for LLM reasoning](reports/survey-about-reinforcement-learning-for-llm-reasoning.md): 9 nguồn.
- [LLM agents and tool use](reports/survey-about-llm-agents-and-tool-use.md): 8 nguồn.
- [Video and multimodal generation](reports/survey-about-video-and-multimodal-generation.md): 8 nguồn.
- [Efficient inference and small language models](reports/survey-about-efficient-inference-and-small-language-models.md): 6 nguồn.

`run_all.py` bỏ qua các bộ đã đạt; không dùng `--force` nếu muốn giữ chúng.
Nội dung báo cáo và nguồn được tải nguyên byte từ Docker sau khi finalizer và
validator chạy trong sandbox. Không chỉnh tay báo cáo sau khi tải về.

## Cấu hình API đang dùng

Key Codex của bên cung cấp đã qua phép thử gọi công cụ bằng Responses API
tại endpoint cấu hình thực tế `modelapi.vn`. Chat Completions từng gặp lỗi 502.
Cấu hình hiện tại dùng Responses API:

```dotenv
LAB_MODEL=openai:gpt-6-sol
LAB_TEMPERATURE=1
OPENAI_API_KEY=<điền trực tiếp tại máy>
LAB_USE_RESPONSES_API=1
LAB_REASONING_EFFORT=low
SANDBOX=docker
SANDBOX_IMAGE=python:3.12-slim
WEB_FETCH_DIRECT_FALLBACK=1
```

`agents.py` bật Responses API khi `LAB_USE_RESPONSES_API=1`, giới hạn đầu ra
ở 8192 token và timeout mỗi yêu cầu 120 giây. Endpoint có thể được đặt bằng
`OPENAI_API_BASE` theo adapter LangChain, hoặc `LAB_BASE_URL` cùng `LAB_MODEL`
là tên model không có tiền tố `openai:` theo factory gốc.
Không đưa `.env` vào Git. API có tính phí hoặc quota theo tài khoản của bạn.
Hai báo cáo đầu được sinh bằng `gpt-6-luna`; do tuyến đó tiếp tục lỗi 502,
cấu hình chuyển sang `gpt-6-sol`, đã kiểm tra gọi công cụ thành công.

Gemini cũng đã qua phép thử gọi công cụ, nhưng key hiện tại hết quota miễn phí
20 yêu cầu/ngày của `gemini-2.5-flash`. Muốn dùng lại, đặt
`LAB_MODEL=google_genai:gemini-2.5-flash`, `GOOGLE_API_KEY=${GEMINI_API_KEY}`,
và bỏ `LAB_BASE_URL`. Nhịp gọi chung mặc định 13 giây giữa các yêu cầu Gemini;
điều chỉnh bằng `LAB_GOOGLE_REQUEST_INTERVAL` theo quota thực tế của tài khoản.

```powershell
.\.venv\Scripts\python.exe preflight.py
.\.venv\Scripts\python.exe run_all.py
```

## Phương án Ollama không cần API key

Ollama chạy LLM trên máy; Docker là sandbox Linux biệt lập để agent ghi chú,
viết báo cáo và chạy trình kiểm tra. Công cụ arXiv, Hugging Face, Exa vẫn cần
Internet và chạy trên host. Exa không khóa có thể bị giới hạn tốc độ.
Thiết lập local giới hạn Exa ở 3 lần thử, mỗi lần chờ tối đa 10 giây. Khi Exa không truy cập được,
`web_fetch` đọc trực tiếp cùng URL qua HTTP và ghi rõ phương thức trong kết quả;
`web_search` vẫn báo lỗi trung thực. Khi đó báo cáo có thể dùng ba họ nguồn
arxiv, hf-daily và hf-search nếu có đủ tài liệu liên quan.

## Cài đặt trên Windows PowerShell

Mở Docker Desktop và Ollama, sau đó chạy trong thư mục repo:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen2.5:7b-instruct
ollama create deepresearch-qwen2.5:7b -f Modelfile
Copy-Item .env.ollama.example .env
docker pull python:3.12-slim
.\start_ollama.ps1
```

`LAB_MODEL=ollama:deepresearch-qwen2.5:7b` dùng adapter Ollama trực tiếp,
không cần biến API key. `model.py` và `sandbox.py` gốc không cần sửa.
`LAB_OLLAMA_URL` trỏ tới dịch vụ local tại cổng 11435 do `start_ollama.ps1` mở.
Dịch vụ này dùng Flash Attention, cache q8 và một lượt sinh tại một thời điểm.
Trên máy hiện tại, đã kiểm tra mô hình tải đủ 29/29 lớp lên GPU; gọi công cụ và
đọc/ghi/chạy Python trong Docker đều đạt. Đóng mô hình đang dùng ở Ollama mặc định
trước khi chạy lab để tránh hai dịch vụ cùng giữ VRAM.
Không copy `.env` nếu đã có cấu hình riêng cần giữ.

## Kiểm tra và chạy

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tools.py
.\.venv\Scripts\python.exe research.py "survey about world model"
.\.venv\Scripts\python.exe run_all.py
.\.venv\Scripts\python.exe self_check.py
```

`run_all.py` chạy lần lượt năm chủ đề và bỏ qua kết quả đã hợp lệ. Nếu một
chủ đề thất bại, script dừng với mã khác 0; chạy lại để tiếp tục. Dùng `--force`
để sinh lại mọi chủ đề. Báo cáo tiếng Anh, nguồn và thống kê lưu trong `reports/`.
Token trong metadata chỉ tính lead, không tính researcher.

Mô hình 7B instruction được chọn trên máy RAM 16 GB / RTX 4050 6 GB. Chất lượng
và tốc độ phải được đánh giá bằng lần chạy thật. Modelfile đặt context 16384;
nếu tràn bộ nhớ hoặc chậm do offload CPU, giảm context hoặc dùng máy mạnh hơn.
Mô hình nhỏ có thể gọi sai công cụ hoặc hết hạn mức; sửa prompt/mã rồi chạy lại,
không sửa tay báo cáo hoặc làm giả metadata.

Qwen3 4B cũng đã được thử: template thinking của bản Ollama đang cài khiến
quá trình sinh kéo dài, còn template tùy chỉnh bị lỗi gọi công cụ. Cấu hình cuối
dùng Qwen2.5 Instruct và template gốc để tránh lỗi này. Mô hình 7B vẫn cần kiểm
tra chất lượng báo cáo thực tế; có thể đổi LAB_MODEL sang mô hình mạnh hơn.

## Nộp bài

Chỉ nộp khi đủ 5 bộ `.md`, `.sources.json`, `.meta.json`, `self_check.py` đạt,
và đã đọc kiểm tra các trích dẫn mẫu. Commit mã và `reports/` vào public repo
theo hướng dẫn VLearn. `.env` và `.venv` được gitignore.

Tài liệu: [Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling),
[OpenAI-compatible API](https://docs.ollama.com/api/openai-compatibility),
[Qwen2.5](https://ollama.com/library/qwen2.5).
